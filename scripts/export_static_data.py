"""Export authentic notebook/Streamlit artifacts and predictions into compact static JSON files."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WEB_DIR = Path(__file__).resolve().parents[1]
STREAMLIT_DIR = WEB_DIR.parent / "streamlit_app"
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))

from src.features import (  # noqa: E402
    FEATURE_GROUPS,
    TARGET,
    clean_and_split_raw,
    engineer_features,
    load_raw_data,
    split_chronological,
)
from src.pipeline import (  # noqa: E402
    LEADERBOARD_ORDER,
    MODEL_SPECS,
    load_trained_artifact,
    predict_with_model,
)

ARTIFACTS_DIR = STREAMLIT_DIR / "artifacts"
CACHE_PATH = STREAMLIT_DIR / "data" / "cache_jakarta_raw.csv"
PLOTLY_SRC = Path(r"D:\Career\TA-Fword\env\Lib\site-packages\plotly\package_data\plotly.min.js")


def main() -> None:
    data_dir = WEB_DIR / "data"
    assets_dir = WEB_DIR / "assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy local bundled plotly.min.js so app never depends on flaky CDN
    if not PLOTLY_SRC.exists():
        raise FileNotFoundError(f"plotly.min.js not found at {PLOTLY_SRC}")
    shutil.copy2(PLOTLY_SRC, assets_dir / "plotly.min.js")
    print(f"Copied plotly.min.js ({(assets_dir / 'plotly.min.js').stat().st_size:,} B)")

    # 2. Load raw & engineered splits identically to Streamlit/ipynb
    df_raw, _ = load_raw_data(CACHE_PATH, allow_api_fallback=False)
    df_clean = clean_and_split_raw(df_raw)
    df_feat = engineer_features(df_clean)
    splits = split_chronological(df_feat)
    test_df = splits.test

    # 3. Export test set timestamps, actuals, and all 11 model predictions
    preds_dict: dict[str, list[float]] = {}
    hourly_mae: dict[str, list[float]] = {}
    y_true = test_df[TARGET].to_numpy(dtype=float)
    test_hours = test_df.index.hour

    for code in LEADERBOARD_ORDER:
        mdl = load_trained_artifact(code, ARTIFACTS_DIR)
        pred = predict_with_model(code, mdl, test_df)
        preds_dict[code] = [round(float(v), 4) for v in pred]
        abs_err = pd.Series(np.abs(y_true - pred), index=test_hours)
        h_means = abs_err.groupby(level=0).mean().reindex(range(24)).fillna(0.0)
        hourly_mae[code] = [round(float(v), 4) for v in h_means.tolist()]

    predictions_payload = {
        "times": [ts.strftime("%Y-%m-%d %H:%M") for ts in test_df.index],
        "dates": [ts.strftime("%Y-%m-%d") for ts in test_df.index],
        "hours": [int(h) for h in test_df.index.hour],
        "actual": [round(float(v), 3) for v in y_true],
        "preds": preds_dict,
    }
    pred_out = data_dir / "predictions_test.json"
    pred_out.write_text(json.dumps(predictions_payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {pred_out.name} ({pred_out.stat().st_size:,} B)")

    # 4. Export summary artifacts & precomputed charts/stats
    lb_df = pd.read_csv(ARTIFACTS_DIR / "leaderboard_v2.csv")
    fi_df = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    diag_df = pd.read_csv(ARTIFACTS_DIR / "cams_diagnostics.csv")
    subseg_df = pd.read_csv(ARTIFACTS_DIR / "subsegment_eval.csv")
    tuning = json.loads((ARTIFACTS_DIR / "tuning_summary.json").read_text(encoding="utf-8"))
    split_meta = json.loads((ARTIFACTS_DIR / "split_metadata.json").read_text(encoding="utf-8"))

    cat_totals_df = (
        fi_df.groupby("Kategori")[["PermImp (naik RMSE)", "Gain Importance"]]
        .sum()
        .sort_values("PermImp (naik RMSE)", ascending=False)
        .reset_index()
    )

    hourly_prof = df_feat.groupby(df_feat.index.hour)[["pm2_5", "wind_speed_10m", "dew_point"]].mean().reindex(range(24))
    monthly_prof = df_feat.groupby(df_feat.index.month)["pm2_5"].mean().reindex(range(1, 13))
    months_lbl = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

    all_cols = [TARGET] + [c for c in df_feat.columns if c != TARGET]
    seg_map = {
        "test": splits.test,
        "val": splits.val,
        "train": splits.train,
        "all": df_feat,
    }
    desc_stats: dict[str, dict[str, dict]] = {}
    samples: dict[str, dict] = {}
    for seg_key, seg_df in seg_map.items():
        d = seg_df[all_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
        desc_stats[seg_key] = {
            var: {
                "count": int(row["count"]),
                "mean": round(float(row["mean"]), 3),
                "std": round(float(row["std"]), 3),
                "min": round(float(row["min"]), 3),
                "median": round(float(row["50%"]), 3),
                "max": round(float(row["max"]), 3),
            }
            for var, row in d.iterrows()
        }
        head_df = seg_df[all_cols].head(250)
        samples[seg_key] = {
            "times": [ts.strftime("%Y-%m-%d %H:%M WIB") for ts in head_df.index],
            "rows": [
                [round(float(val), 4) for val in row]
                for row in head_df.to_numpy(dtype=float)
            ],
        }

    summary_payload = {
        "model_specs": [
            {
                "code": MODEL_SPECS[c].code,
                "display_name": MODEL_SPECS[c].display_name,
                "feature_set_name": MODEL_SPECS[c].feature_set_name,
                "regime": MODEL_SPECS[c].regime,
            }
            for c in reversed(LEADERBOARD_ORDER)
        ],
        "leaderboard": lb_df.to_dict(orient="records"),
        "feature_importance": fi_df.to_dict(orient="records"),
        "category_totals": cat_totals_df.round(3).to_dict(orient="records"),
        "subsegment_eval": subseg_df.round(3).to_dict(orient="records"),
        "tuning_summary": tuning,
        "cams_diagnostics": diag_df.round(3).to_dict(orient="records"),
        "split_metadata": split_meta,
        "feature_groups": FEATURE_GROUPS,
        "all_columns": all_cols,
        "diurnal_profile": {
            "hours": list(range(24)),
            "pm2_5": [round(float(v), 3) for v in hourly_prof["pm2_5"].tolist()],
            "wind_speed_10m": [round(float(v), 3) for v in hourly_prof["wind_speed_10m"].tolist()],
            "dew_point": [round(float(v), 3) for v in hourly_prof["dew_point"].tolist()],
        },
        "monthly_profile": {
            "months": months_lbl,
            "pm2_5": [round(float(v), 3) for v in monthly_prof.tolist()],
        },
        "hourly_mae": hourly_mae,
        "descriptive_stats": desc_stats,
    }
    sum_out = data_dir / "summary_artifacts.json"
    sum_out.write_text(json.dumps(summary_payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {sum_out.name} ({sum_out.stat().st_size:,} B)")

    samp_out = data_dir / "dataset_samples.json"
    samp_out.write_text(json.dumps({"columns": all_cols, "segments": samples}, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {samp_out.name} ({samp_out.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
