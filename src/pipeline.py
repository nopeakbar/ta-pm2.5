"""Model specs, metric calculators, artifact training, and live inference loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid

from src.features import (
    SET_1_RAW,
    SET_2_TEMPORAL,
    SET_3_CLIMATE,
    SET_4_FULL,
    TARGET,
    ChronologicalSplits,
    categorize_feature,
    clean_and_split_raw,
    compute_cams_diagnostics,
    engineer_features,
    load_raw_data,
    split_chronological,
)

SEED = 42
SEEDS = [42, 7, 13, 21, 99, 100, 123, 2024, 777, 888]
BASE_STD = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8],
}
MAX_TREES = 2000
ES_ROUNDS = 50
RUSH_HOURS = (6, 7, 8, 9, 16, 17, 18, 19)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata describing each of the 11 ablation & baseline models."""

    code: str
    display_name: str
    feature_set_name: str
    features: list[str]
    regime: str
    model_type: str  # "persistence" | "linear" | "xgb"
    artifact_relpath: str | None


MODEL_SPECS: dict[str, ModelSpec] = {
    "F'": ModelSpec(
        code="F'",
        display_name="Model G (F') — XGBoost Full Engineered (Tuned)",
        feature_set_name="Set 4 Full (29 Fitur)",
        features=SET_4_FULL,
        regime="Rezim 2 (Tuned)",
        model_type="xgb",
        artifact_relpath="models/model_F_tuned.json",
    ),
    "D'": ModelSpec(
        code="D'",
        display_name="Model D' — XGBoost Raw + Temporal (Tuned)",
        feature_set_name="Set 2 Temporal (19 Fitur)",
        features=SET_2_TEMPORAL,
        regime="Rezim 2 (Tuned)",
        model_type="xgb",
        artifact_relpath="models/model_D_tuned.json",
    ),
    "E'": ModelSpec(
        code="E'",
        display_name="Model E' — XGBoost Raw + Iklim (Tuned)",
        feature_set_name="Set 3 Iklim (20 Fitur)",
        features=SET_3_CLIMATE,
        regime="Rezim 2 (Tuned)",
        model_type="xgb",
        artifact_relpath="models/model_E_tuned.json",
    ),
    "C'": ModelSpec(
        code="C'",
        display_name="Model C' — XGBoost Raw (Tuned)",
        feature_set_name="Set 1 Raw (10 Fitur)",
        features=SET_1_RAW,
        regime="Rezim 2 (Tuned)",
        model_type="xgb",
        artifact_relpath="models/model_C_tuned.json",
    ),
    "F": ModelSpec(
        code="F",
        display_name="Model F — XGBoost Full Engineered (Default)",
        feature_set_name="Set 4 Full (29 Fitur)",
        features=SET_4_FULL,
        regime="Rezim 1 (Default)",
        model_type="xgb",
        artifact_relpath="models/model_F_default.json",
    ),
    "D": ModelSpec(
        code="D",
        display_name="Model D — XGBoost Raw + Temporal (Default)",
        feature_set_name="Set 2 Temporal (19 Fitur)",
        features=SET_2_TEMPORAL,
        regime="Rezim 1 (Default)",
        model_type="xgb",
        artifact_relpath="models/model_D_default.json",
    ),
    "E": ModelSpec(
        code="E",
        display_name="Model E — XGBoost Raw + Iklim (Default)",
        feature_set_name="Set 3 Iklim (20 Fitur)",
        features=SET_3_CLIMATE,
        regime="Rezim 1 (Default)",
        model_type="xgb",
        artifact_relpath="models/model_E_default.json",
    ),
    "C": ModelSpec(
        code="C",
        display_name="Model C — XGBoost Raw (Default Baseline)",
        feature_set_name="Set 1 Raw (10 Fitur)",
        features=SET_1_RAW,
        regime="Rezim 1 (Default)",
        model_type="xgb",
        artifact_relpath="models/model_C_default.json",
    ),
    "B2": ModelSpec(
        code="B2",
        display_name="Model B2 — Linear Regression (Raw)",
        feature_set_name="Set 1 Raw (10 Fitur)",
        features=SET_1_RAW,
        regime="Deterministik",
        model_type="linear",
        artifact_relpath="models/model_B2_linear_raw.joblib",
    ),
    "B": ModelSpec(
        code="B",
        display_name="Model B — Linear Regression (Full)",
        feature_set_name="Set 4 Full (29 Fitur)",
        features=SET_4_FULL,
        regime="Deterministik",
        model_type="linear",
        artifact_relpath="models/model_B_linear_full.joblib",
    ),
    "A": ModelSpec(
        code="A",
        display_name="Model A — Naive Persistence (y_t = y_{t-1})",
        feature_set_name="Lag 1 (1 Fitur)",
        features=["pm2_5_lag_1"],
        regime="Deterministik",
        model_type="persistence",
        artifact_relpath=None,
    ),
}

LEADERBOARD_ORDER = ["A", "B", "B2", "C", "D", "E", "F", "C'", "D'", "E'", "F'"]
XGB_SCENARIOS = {
    "C": ("XGBoost Skenario 1 (Raw)", SET_1_RAW),
    "D": ("XGBoost Skenario 2 (Raw + Temporal)", SET_2_TEMPORAL),
    "E": ("XGBoost Skenario 3 (Raw + Iklim)", SET_3_CLIMATE),
    "F": ("XGBoost Skenario 4 (Full Engineered)", SET_4_FULL),
}


def compute_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float | None]:
    """Calculate RMSE, MAE, MAPE (%), and R2 safely for arbitrary slice sizes."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) == 0:
        return {"rmse": None, "mae": None, "mape": None, "r2": None}

    mask = yt > 0
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0) if np.any(mask) else None
    r2 = float(r2_score(yt, yp)) if len(yt) >= 2 and float(np.var(yt)) > 0 else None
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "mape": mape,
        "r2": r2,
    }


def fmt_ms(values: pd.Series | np.ndarray, decimals: int = 3) -> str:
    """Format mean ± sample std string matching notebook Cell 7/8b."""
    arr = np.asarray(values, dtype=float)
    return f"{np.mean(arr):.{decimals}f} ± {np.std(arr, ddof=1):.{decimals}f}"


def paired_delta(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> str:
    """Format paired relative RMSE reduction (%) across 10 seeds."""
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    d = (arr_a - arr_b) / arr_b * 100.0
    return f"{d.mean():+.2f} ± {d.std(ddof=1):.2f} (lebih baik di {(d < 0).sum()}/{len(d)} seed)"


def load_trained_artifact(model_code: str, artifacts_dir: Path):
    """Load persisted model object from artifacts/models directory."""
    spec = MODEL_SPECS[model_code]
    if spec.model_type == "persistence" or spec.artifact_relpath is None:
        return None
    full_path = artifacts_dir / spec.artifact_relpath
    if not full_path.exists():
        raise FileNotFoundError(f"Artefak model {model_code} tidak ditemukan: {full_path}")
    if spec.model_type == "xgb":
        mdl = xgb.XGBRegressor()
        mdl.load_model(str(full_path))
        return mdl
    return joblib.load(full_path)


def predict_with_model(model_code: str, model_obj, df_eval: pd.DataFrame) -> np.ndarray:
    """Execute live prediction on df_eval using loaded model artifact."""
    spec = MODEL_SPECS[model_code]
    if spec.model_type == "persistence":
        return df_eval["pm2_5_lag_1"].to_numpy(dtype=float)
    return np.asarray(model_obj.predict(df_eval[spec.features]), dtype=float)


def artifacts_ready(artifacts_dir: Path) -> list[str]:
    """Return list of missing required artifact files (empty when complete)."""
    required = [
        "leaderboard_v2.csv",
        "feature_importance.csv",
        "tuning_summary.json",
        "cams_diagnostics.csv",
        "subsegment_eval.csv",
        "split_metadata.json",
    ]
    for spec in MODEL_SPECS.values():
        if spec.artifact_relpath:
            required.append(spec.artifact_relpath)
    return [rel for rel in required if not (artifacts_dir / rel).exists()]


def build_all_artifacts(cache_path: Path, artifacts_dir: Path) -> dict:
    """Run full notebook training, tuning, multi-seed eval, XAI & persist all artifacts."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df_raw, source_tag = load_raw_data(cache_path, allow_api_fallback=True)
    df_clean = clean_and_split_raw(df_raw)
    cams_diag = compute_cams_diagnostics(df_clean)
    cams_diag.to_csv(artifacts_dir / "cams_diagnostics.csv")

    df_feat = engineer_features(df_clean)
    splits: ChronologicalSplits = split_chronological(df_feat)
    train_df, val_df, test_df = splits.train, splits.val, splits.test
    y_test = test_df[TARGET].to_numpy(dtype=float)
    y_val = val_df[TARGET].to_numpy(dtype=float)

    preds_seed42: dict[str, np.ndarray] = {}
    preds_seed42["A"] = test_df["pm2_5_lag_1"].to_numpy(dtype=float)

    lr_full = LinearRegression().fit(train_df[SET_4_FULL], train_df[TARGET])
    joblib.dump(lr_full, models_dir / "model_B_linear_full.joblib")
    preds_seed42["B"] = lr_full.predict(test_df[SET_4_FULL])

    lr_raw = LinearRegression().fit(train_df[SET_1_RAW], train_df[TARGET])
    joblib.dump(lr_raw, models_dir / "model_B2_linear_raw.joblib")
    preds_seed42["B2"] = lr_raw.predict(test_df[SET_1_RAW])

    def _run_seeds(feats: list[str], params: dict) -> pd.DataFrame:
        seed_rows = []
        for s in SEEDS:
            mdl = xgb.XGBRegressor(**params, random_state=s, n_jobs=-1).fit(train_df[feats], train_df[TARGET])
            m = compute_metrics(y_test, mdl.predict(test_df[feats]))
            seed_rows.append(
                {
                    "MAE (ug/m3)": m["mae"],
                    "RMSE (ug/m3)": m["rmse"],
                    "MAPE (%)": m["mape"],
                    "R2 Score": m["r2"],
                }
            )
        return pd.DataFrame(seed_rows)

    ms_default: dict[str, pd.DataFrame] = {}
    for code, (_, feats) in XGB_SCENARIOS.items():
        mdl_42 = xgb.XGBRegressor(**BASE_STD, random_state=SEED, n_jobs=-1).fit(train_df[feats], train_df[TARGET])
        mdl_42.save_model(str(models_dir / f"model_{code}_default.json"))
        preds_seed42[code] = mdl_42.predict(test_df[feats])
        ms_default[code] = _run_seeds(feats, BASE_STD)

    grid = list(ParameterGrid(PARAM_GRID))
    best_cfg: dict[str, dict] = {}
    tune_top5: dict[str, list[dict]] = {}
    for code, (_, feats) in XGB_SCENARIOS.items():
        t_rows = []
        for p in grid:
            mdl = xgb.XGBRegressor(
                **p,
                n_estimators=MAX_TREES,
                early_stopping_rounds=ES_ROUNDS,
                random_state=SEED,
                n_jobs=-1,
            )
            mdl.fit(train_df[feats], train_df[TARGET], eval_set=[(val_df[feats], y_val)], verbose=False)
            bi = int(mdl.best_iteration)
            pv = mdl.predict(val_df[feats], iteration_range=(0, bi + 1))
            mv = compute_metrics(y_val, pv)
            t_rows.append(
                {
                    **p,
                    "n_estimators": bi + 1,
                    "Val RMSE": round(float(mv["rmse"]), 6),
                    "Val MAE": round(float(mv["mae"]), 6),
                    "Val MAPE (%)": round(float(mv["mape"]), 6),
                }
            )
        t_df = pd.DataFrame(t_rows).sort_values("Val RMSE").reset_index(drop=True)
        tune_top5[code] = t_df.head(5).to_dict(orient="records")
        top1 = t_df.iloc[0]
        best_cfg[code] = {
            "n_estimators": int(top1["n_estimators"]),
            "max_depth": int(top1["max_depth"]),
            "learning_rate": float(top1["learning_rate"]),
            "subsample": float(top1["subsample"]),
            "colsample_bytree": float(top1["colsample_bytree"]),
        }

    tuned_models: dict[str, xgb.XGBRegressor] = {}
    ms_tuned: dict[str, pd.DataFrame] = {}
    for code, (_, feats) in XGB_SCENARIOS.items():
        mdl_t = xgb.XGBRegressor(**best_cfg[code], random_state=SEED, n_jobs=-1).fit(
            train_df[feats], train_df[TARGET]
        )
        mdl_t.save_model(str(models_dir / f"model_{code}_tuned.json"))
        tuned_models[code] = mdl_t
        preds_seed42[f"{code}'"] = mdl_t.predict(test_df[feats])
        ms_tuned[code] = _run_seeds(feats, best_cfg[code])

    lb_rows = []
    for code_, label_ in [
        ("A", "Naive Persistence"),
        ("B", "Linear Regression (Full)"),
        ("B2", "Linear Regression (Raw)"),
    ]:
        m = compute_metrics(y_test, preds_seed42[code_])
        lb_rows.append(
            {
                "Kode": code_,
                "Model": label_,
                "RMSE": f"{m['rmse']:.3f}",
                "MAE": f"{m['mae']:.3f}",
                "MAPE (%)": f"{m['mape']:.2f}",
                "R2": f"{m['r2']:.3f}",
                "Delta vs Raw (sama regime)": "-",
                "Delta efek tuning (set sama)": "-",
            }
        )

    for code, (nm, _) in XGB_SCENARIOS.items():
        d = ms_default[code]
        lb_rows.append(
            {
                "Kode": code,
                "Model": f"{nm} - default",
                "RMSE": fmt_ms(d["RMSE (ug/m3)"]),
                "MAE": fmt_ms(d["MAE (ug/m3)"]),
                "MAPE (%)": fmt_ms(d["MAPE (%)"], 2),
                "R2": fmt_ms(d["R2 Score"]),
                "Delta vs Raw (sama regime)": "-"
                if code == "C"
                else paired_delta(d["RMSE (ug/m3)"], ms_default["C"]["RMSE (ug/m3)"]),
                "Delta efek tuning (set sama)": "-",
            }
        )

    for code, (nm, _) in XGB_SCENARIOS.items():
        d = ms_tuned[code]
        tag = " (= Model G)" if code == "F" else ""
        lb_rows.append(
            {
                "Kode": f"{code}'",
                "Model": f"{nm} - TUNED{tag}",
                "RMSE": fmt_ms(d["RMSE (ug/m3)"]),
                "MAE": fmt_ms(d["MAE (ug/m3)"]),
                "MAPE (%)": fmt_ms(d["MAPE (%)"], 2),
                "R2": fmt_ms(d["R2 Score"]),
                "Delta vs Raw (sama regime)": "-"
                if code == "C"
                else paired_delta(d["RMSE (ug/m3)"], ms_tuned["C"]["RMSE (ug/m3)"]),
                "Delta efek tuning (set sama)": paired_delta(
                    d["RMSE (ug/m3)"], ms_default[code]["RMSE (ug/m3)"]
                ),
            }
        )

    leader_df = pd.DataFrame(lb_rows)
    leader_df.to_csv(artifacts_dir / "leaderboard_v2.csv", index=False)

    model_G = tuned_models["F"]
    pi = permutation_importance(
        model_G,
        test_df[SET_4_FULL],
        test_df[TARGET],
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,
    )
    fi_df = pd.DataFrame(
        {
            "Fitur": SET_4_FULL,
            "PermImp (naik RMSE)": pi.importances_mean,
            "Std": pi.importances_std,
            "Gain Importance": model_G.feature_importances_,
        }
    )
    fi_df["Kategori"] = fi_df["Fitur"].map(categorize_feature)
    fi_df = fi_df.sort_values("PermImp (naik RMSE)", ascending=False).reset_index(drop=True)
    fi_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)

    p95 = float(np.percentile(y_test, 95))
    peak_mask = y_test >= p95
    rush_mask = np.isin(test_df.index.hour, RUSH_HOURS)

    sub_rows = []
    for code in LEADERBOARD_ORDER:
        pred = preds_seed42[code]
        all_mask = np.ones_like(peak_mask, dtype=bool)
        sub_rows.append(
            {
                "Kode": code,
                "Model": MODEL_SPECS[code].display_name,
                "RMSE total": float(np.sqrt(np.mean((y_test[all_mask] - pred[all_mask]) ** 2))),
                "RMSE puncak (>=P95)": float(np.sqrt(np.mean((y_test[peak_mask] - pred[peak_mask]) ** 2))),
                "Bias puncak (pred-aktual)": float(np.mean(pred[peak_mask] - y_test[peak_mask])),
                "RMSE jam sibuk": float(np.sqrt(np.mean((y_test[rush_mask] - pred[rush_mask]) ** 2))),
                "RMSE di luar jam sibuk": float(np.sqrt(np.mean((y_test[~rush_mask] - pred[~rush_mask]) ** 2))),
                "MAE jam sibuk": float(np.mean(np.abs(y_test[rush_mask] - pred[rush_mask]))),
            }
        )
    pd.DataFrame(sub_rows).to_csv(artifacts_dir / "subsegment_eval.csv", index=False)

    tuning_payload = {
        "best_cfg": best_cfg,
        "top5": tune_top5,
        "seeds": SEEDS,
        "grid_count": len(grid),
        "max_trees": MAX_TREES,
        "early_stopping_rounds": ES_ROUNDS,
    }
    (artifacts_dir / "tuning_summary.json").write_text(
        json.dumps(tuning_payload, indent=2), encoding="utf-8"
    )

    split_meta = {
        "source": source_tag,
        "raw_rows": int(len(df_raw)),
        "clean_rows": int(len(df_feat)),
        "feature_cols": int(len(SET_4_FULL)),
        "p95_threshold": round(p95, 3),
        "peak_count": int(peak_mask.sum()),
        "rush_count": int(rush_mask.sum()),
        "splits": {
            "Train": {
                "rows": int(len(train_df)),
                "start": str(train_df.index.min()),
                "end": str(train_df.index.max()),
                "mean_pm25": round(float(train_df[TARGET].mean()), 2),
                "std_pm25": round(float(train_df[TARGET].std()), 2),
            },
            "Validation": {
                "rows": int(len(val_df)),
                "start": str(val_df.index.min()),
                "end": str(val_df.index.max()),
                "mean_pm25": round(float(val_df[TARGET].mean()), 2),
                "std_pm25": round(float(val_df[TARGET].std()), 2),
            },
            "Test": {
                "rows": int(len(test_df)),
                "start": str(test_df.index.min()),
                "end": str(test_df.index.max()),
                "mean_pm25": round(float(test_df[TARGET].mean()), 2),
                "std_pm25": round(float(test_df[TARGET].std()), 2),
            },
        },
    }
    (artifacts_dir / "split_metadata.json").write_text(
        json.dumps(split_meta, indent=2), encoding="utf-8"
    )
    return split_meta
