"""Data ingestion, zero-leakage preprocessing, and 29-feature engineering.

Matches agy_19okt_28godean_v2.ipynb Cells 2-6 verbatim so all row counts,
chronological splits, physics equations, and autoregressive lags stay identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

LATITUDE = -6.1944
LONGITUDE = 106.8229
TIMEZONE = "Asia/Jakarta"
START_DATE = "2023-10-04"
END_DATE = "2026-10-03"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT = 90

TARGET = "pm2_5"
RAW_LAGS = ["pm2_5_lag_1", "pm2_5_lag_2", "pm2_5_lag_3", "pm2_5_roll_mean_6h"]
RAW_WEATHER = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]
TEMPORAL_COLS = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "is_rush_hour",
    "is_weekend",
    "rush_and_weekend",
]
PHYSICS_COLS = [
    "wind_u",
    "wind_v",
    "dew_point",
    "vpd",
    "temperature_2m_roll_mean_6h",
    "temperature_2m_roll_std_6h",
    "relative_humidity_2m_roll_mean_6h",
    "relative_humidity_2m_roll_std_6h",
    "wind_speed_10m_roll_mean_6h",
    "wind_speed_10m_roll_std_6h",
]

SET_1_RAW = RAW_LAGS + RAW_WEATHER
SET_2_TEMPORAL = SET_1_RAW + TEMPORAL_COLS
SET_3_CLIMATE = SET_1_RAW + PHYSICS_COLS
SET_4_FULL = SET_1_RAW + TEMPORAL_COLS + PHYSICS_COLS

FEATURE_GROUPS = {
    "Lag Autoregresif (4)": RAW_LAGS,
    "Cuaca Mentah (6)": RAW_WEATHER,
    "Siklik Temporal & Komuter (9)": TEMPORAL_COLS,
    "Fisika Iklim (10)": PHYSICS_COLS,
}


@dataclass(frozen=True)
class ChronologicalSplits:
    """Container for chronological Train / Validation / Test dataframes."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def categorize_feature(name: str) -> str:
    """Map predictor column name to its ablation study feature category."""
    if name in RAW_LAGS:
        return "Lag Autoregresif"
    if name in RAW_WEATHER:
        return "Cuaca Mentah"
    if name in PHYSICS_COLS:
        return "Fisika Iklim"
    return "Siklik Temporal"


def _fetch_api_hourly(url: str, params: dict, label: str) -> dict:
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "hourly" not in data:
        raise ValueError(f'Key "hourly" tidak ditemukan pada respons {label}')
    return data["hourly"]


def load_raw_data(cache_path: Path, allow_api_fallback: bool = True) -> tuple[pd.DataFrame, str]:
    """Load cached hourly CAMS + ERA5 dataset, fetching from Open-Meteo if absent."""
    if cache_path.exists():
        df_raw = pd.read_csv(cache_path, index_col="time")
        df_raw.index = pd.to_datetime(df_raw.index, utc=True).tz_convert(TIMEZONE)
        df_raw.index.name = "time"
        return df_raw, "cache_lokal"

    if not allow_api_fallback:
        raise FileNotFoundError(f"Cache dataset tidak ditemukan di {cache_path}")

    aq_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm2_5",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
    }
    weather_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,surface_pressure",
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
    }
    hourly_aq = _fetch_api_hourly(AIR_QUALITY_URL, aq_params, "Air Quality (CAMS/Open-Meteo)")
    hourly_w = _fetch_api_hourly(WEATHER_URL, weather_params, "Weather Historical Archive")

    df_aq = pd.DataFrame({"time": hourly_aq["time"], "pm2_5": hourly_aq["pm2_5"]})
    weather_keys = ["time", *RAW_WEATHER]
    df_w = pd.DataFrame({k: hourly_w[k] for k in weather_keys})

    df_raw = pd.merge(df_aq, df_w, on="time", how="inner")
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw = df_raw.set_index("time").sort_index()
    if df_raw.index.tz is None:
        df_raw.index = df_raw.index.tz_localize(TIMEZONE)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(cache_path)
    return df_raw, "open_meteo_api"


def clean_and_split_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply segmented forward-only interpolation without crossing chronological boundaries."""
    df_clean = df_raw.copy()
    n_total = len(df_clean)
    idx_train, idx_val = int(n_total * 0.70), int(n_total * 0.85)
    segments = [df_clean.iloc[:idx_train], df_clean.iloc[idx_train:idx_val], df_clean.iloc[idx_val:]]
    df_clean = pd.concat(
        [seg.interpolate(method="linear", limit_direction="forward").ffill() for seg in segments]
    )
    df_clean.iloc[:idx_train] = df_clean.iloc[:idx_train].bfill()
    if df_clean.isna().sum().sum() != 0:
        raise ValueError("Masih terdapat missing value setelah pembersihan segmen.")
    return df_clean


def compute_cams_diagnostics(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Diagnose CAMS 3-hourly temporal resolution via centered second differences."""
    pm = df_raw[TARGET]
    d2c = (pm.shift(-1) - 2 * pm + pm.shift(1)).abs()
    diag = d2c.groupby(pm.index.hour % 3).agg(["mean", "median"])
    diag["frac |d2| <= 0.25"] = d2c.groupby(pm.index.hour % 3).apply(
        lambda s: float((s.dropna() <= 0.25).mean())
    )
    diag.index.name = "jam WIB mod 3"
    return diag


def engineer_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Compute all 29 domain-engineered predictors matching notebook Cells 4-5."""
    df_feat = df_clean.copy()

    wind_rad = np.radians(df_feat["wind_direction_10m"])
    df_feat["wind_u"] = -df_feat["wind_speed_10m"] * np.sin(wind_rad)
    df_feat["wind_v"] = -df_feat["wind_speed_10m"] * np.cos(wind_rad)

    a, b = 17.27, 237.7
    alpha = ((a * df_feat["temperature_2m"]) / (b + df_feat["temperature_2m"])) + np.log(
        df_feat["relative_humidity_2m"] / 100.0
    )
    df_feat["dew_point"] = (b * alpha) / (a - alpha)

    es = 0.61078 * np.exp((17.27 * df_feat["temperature_2m"]) / (df_feat["temperature_2m"] + 237.3))
    ea = es * (df_feat["relative_humidity_2m"] / 100.0)
    df_feat["vpd"] = es - ea

    for col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]:
        df_feat[f"{col}_roll_mean_6h"] = df_feat[col].rolling(window=6, min_periods=6).mean()
        df_feat[f"{col}_roll_std_6h"] = df_feat[col].rolling(window=6, min_periods=6).std()

    df_feat["pm2_5_lag_1"] = df_feat[TARGET].shift(1)
    df_feat["pm2_5_lag_2"] = df_feat[TARGET].shift(2)
    df_feat["pm2_5_lag_3"] = df_feat[TARGET].shift(3)
    df_feat["pm2_5_roll_mean_6h"] = df_feat[TARGET].shift(1).rolling(6).mean()

    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat.index.hour / 24.0)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat.index.hour / 24.0)
    df_feat["dayofweek_sin"] = np.sin(2 * np.pi * df_feat.index.dayofweek / 7.0)
    df_feat["dayofweek_cos"] = np.cos(2 * np.pi * df_feat.index.dayofweek / 7.0)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat.index.month / 12.0)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat.index.month / 12.0)

    hours = df_feat.index.hour
    df_feat["is_rush_hour"] = (((hours >= 6) & (hours <= 9)) | ((hours >= 16) & (hours <= 19))).astype(int)
    df_feat["is_weekend"] = (df_feat.index.dayofweek >= 5).astype(int)
    df_feat["rush_and_weekend"] = df_feat["is_rush_hour"] * df_feat["is_weekend"]

    df_feat = df_feat.dropna()
    if set(df_feat.columns) - {TARGET} != set(SET_4_FULL):
        raise ValueError("Himpunan kolom fitur rekayasa tidak cocok dengan SET_4_FULL.")
    return df_feat


def split_chronological(df_feat: pd.DataFrame) -> ChronologicalSplits:
    """Partition cleaned feature dataframe chronologically 70% / 15% / 15%."""
    n = len(df_feat)
    train_idx, val_idx = int(n * 0.70), int(n * 0.85)
    train_df = df_feat.iloc[:train_idx]
    val_df = df_feat.iloc[train_idx:val_idx]
    test_df = df_feat.iloc[val_idx:]
    if not (train_df.index.max() < val_df.index.min() < test_df.index.min()):
        raise ValueError("Urutan waktu partisi Train/Val/Test tidak monoton!")
    return ChronologicalSplits(train=train_df, val=val_df, test=test_df)
