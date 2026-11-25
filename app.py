"""Streamlit Multi-Tab Dashboard: Evaluasi Model Prediksi PM2.5 DKI Jakarta.

Aligned 1:1 with BAB III Subbab 3.8 (Gambar 3.5-3.8 & Tabel 3.9-3.10).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.features import (
    clean_and_split_raw,
    engineer_features,
    load_raw_data,
    split_chronological,
)
from src.pipeline import (
    LEADERBOARD_ORDER,
    MODEL_SPECS,
    RUSH_HOURS,
    artifacts_ready,
    load_trained_artifact,
    predict_with_model,
)
from src.tabs.tab_dataset import render_dataset_tab
from src.tabs.tab_leaderboard import render_leaderboard_tab
from src.tabs.tab_prediction import render_prediction_tab
from src.theme import inject_custom_css

APP_DIR = Path(__file__).resolve().parent
DATA_CACHE = APP_DIR / "data" / "cache_jakarta_raw.csv"
ARTIFACTS_DIR = APP_DIR / "artifacts"


@st.cache_data(show_spinner="Memuat dataset historis & menghitung 29 fitur...")
def load_pipeline_data():
    df_raw, source_tag = load_raw_data(DATA_CACHE, allow_api_fallback=True)
    df_clean = clean_and_split_raw(df_raw)
    df_feat = engineer_features(df_clean)
    splits = split_chronological(df_feat)
    return df_feat, splits, source_tag


@st.cache_data
def load_static_artifacts():
    lb = pd.read_csv(ARTIFACTS_DIR / "leaderboard_v2.csv")
    fi = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    diag = pd.read_csv(ARTIFACTS_DIR / "cams_diagnostics.csv", index_col="jam WIB mod 3")
    subseg = pd.read_csv(ARTIFACTS_DIR / "subsegment_eval.csv")
    tuning = json.loads((ARTIFACTS_DIR / "tuning_summary.json").read_text(encoding="utf-8"))
    split_meta = json.loads((ARTIFACTS_DIR / "split_metadata.json").read_text(encoding="utf-8"))
    return lb, fi, diag, subseg, tuning, split_meta


@st.cache_resource
def get_cached_model(model_code: str):
    return load_trained_artifact(model_code, ARTIFACTS_DIR)


def main() -> None:
    st.set_page_config(
        page_title="Dashboard Evaluasi Prediksi PM2.5 DKI Jakarta",
        page_icon="🌫️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    missing = artifacts_ready(ARTIFACTS_DIR)
    if missing:
        st.error(
            f"Artefak model terlatih atau rekap multi-seed belum lengkap ({len(missing)} berkas hilang: {', '.join(missing[:3])}...). "
            "Jalankan `python scripts/build_artifacts.py` terlebih dahulu sesuai arsitektur sistem (Gambar 3.5)."
        )
        st.stop()

    try:
        df_feat, splits, _source_tag = load_pipeline_data()
        leaderboard_df, fi_df, cams_diag, subseg_df, tuning_summary, split_meta = load_static_artifacts()
    except Exception as exc:
        st.error(f"Gagal memuat dataset atau artefak evaluasi: {exc}")
        st.stop()

    test_df = splits.test
    p95_val = float(split_meta["p95_threshold"])
    min_date = test_df.index.min().date()
    max_date = test_df.index.max().date()

    # Sidebar Panel Konfigurasi (Gambar 3.6-3.8)
    with st.sidebar:
        st.header("Panel Konfigurasi")

        model_options = list(reversed(LEADERBOARD_ORDER))  # F' (Model G) on top
        main_code = st.selectbox(
            "Model Evaluasi Utama",
            options=model_options,
            index=0,
            format_func=lambda c: MODEL_SPECS[c].display_name,
        )
        default_cmp_idx = model_options.index("C") if "C" in model_options else 0
        cmp_code = st.selectbox(
            "Model Pembanding",
            options=model_options,
            index=default_cmp_idx,
            format_func=lambda c: MODEL_SPECS[c].display_name,
        )

        preset = st.radio(
            "Preset Rentang Data Uji",
            ["14 Hari Pertama (Seperti Laporan Fig. 2)", "Seluruh Data Uji (3.945 Jam)", "Pilih Rentang Tanggal"],
            index=0,
        )
        if preset.startswith("14 Hari"):
            d_start = min_date
            d_end = min(max_date, (test_df.index.min() + pd.Timedelta(days=13)).date())
        elif preset.startswith("Seluruh"):
            d_start, d_end = min_date, max_date
        else:
            date_picked = st.date_input(
                "Rentang Tanggal Data Uji",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if isinstance(date_picked, tuple) and len(date_picked) == 2:
                d_start, d_end = date_picked
            else:
                d_start, d_end = min_date, max_date

        st.markdown("**Filter Sub-Segmen Kritis**")
        only_peak = st.checkbox(f"Puncak Polusi (>= P95: {p95_val:.1f} µg/m³)", value=False)
        only_rush = st.checkbox("Jam Sibuk (06–09 & 16–19 WIB)", value=False)

    main_model = get_cached_model(main_code)
    cmp_model = get_cached_model(cmp_code)
    y_pred_main = predict_with_model(main_code, main_model, test_df)
    y_pred_cmp = predict_with_model(cmp_code, cmp_model, test_df)

    idx_dates = test_df.index.date
    mask_filtered = pd.Series((idx_dates >= d_start) & (idx_dates <= d_end), index=test_df.index)
    if only_peak:
        mask_filtered &= test_df["pm2_5"] >= p95_val
    if only_rush:
        mask_filtered &= test_df.index.hour.isin(RUSH_HOURS)

    st.markdown(
        """
        <div class="header-banner">
            <div class="header-title">Dashboard Evaluasi Model Prediksi PM2.5 DKI Jakarta</div>
            <p class="header-sub">
                XGBoost Multi-Skenario · Data Per Jam Open-Meteo (CAMS &amp; ERA5) · 26.298 Baris × 30 Kolom (Uji: 3.945 Jam)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Tab 1: Visualisasi Prediksi",
            "Tab 2: Eksplorasi Dataset",
            "Tab 3: Leaderboard & XAI",
        ]
    )

    with tab1:
        render_prediction_tab(
            df_feat=df_feat,
            test_df=test_df,
            main_code=main_code,
            cmp_code=cmp_code,
            y_pred_main=y_pred_main,
            y_pred_cmp=y_pred_cmp,
            mask_filtered=mask_filtered,
            p95_threshold=p95_val,
            leaderboard_df=leaderboard_df,
            subseg_df=subseg_df,
        )
    with tab2:
        render_dataset_tab(
            df_feat=df_feat,
            splits=splits,
            split_meta=split_meta,
            cams_diag=cams_diag,
        )
    with tab3:
        render_leaderboard_tab(
            leaderboard_df=leaderboard_df,
            fi_df=fi_df,
            tuning_summary=tuning_summary,
        )


if __name__ == "__main__":
    main()
