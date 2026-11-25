"""Black-Box & Notebook Parity tests aligned with Tabel 3.10 (Skenario 1-5)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.features import (  # noqa: E402
    SET_1_RAW,
    SET_2_TEMPORAL,
    SET_3_CLIMATE,
    SET_4_FULL,
    clean_and_split_raw,
    compute_cams_diagnostics,
    engineer_features,
    load_raw_data,
    split_chronological,
)
from src.pipeline import (  # noqa: E402
    LEADERBOARD_ORDER,
    artifacts_ready,
    compute_metrics,
    load_trained_artifact,
    predict_with_model,
)

CACHE_PATH = APP_DIR / "data" / "cache_jakarta_raw.csv"
ARTIFACTS_DIR = APP_DIR / "artifacts"
REF_LEADERBOARD = Path(r"D:\Career\TA-Fword\19 october\leaderboard_v2.csv")


def test_scenario1_raw_data_loaded_without_missing():
    df_raw, src = load_raw_data(CACHE_PATH, allow_api_fallback=False)
    assert src == "cache_lokal"
    assert df_raw.shape == (26304, 7)
    assert int(df_raw.isna().sum().sum()) == 0


def test_scenario2_feature_engineering_and_cams_diagnostics():
    df_raw, _ = load_raw_data(CACHE_PATH, allow_api_fallback=False)
    df_clean = clean_and_split_raw(df_raw)
    diag = compute_cams_diagnostics(df_clean)
    assert list(diag.index) == [0, 1, 2]
    df_feat = engineer_features(df_clean)
    assert df_feat.shape == (26298, 30)
    assert int(df_feat.isna().sum().sum()) == 0
    assert [len(s) for s in (SET_1_RAW, SET_2_TEMPORAL, SET_3_CLIMATE, SET_4_FULL)] == [10, 19, 20, 29]


def test_scenario3_chronological_split_zero_overlap():
    df_raw, _ = load_raw_data(CACHE_PATH, allow_api_fallback=False)
    splits = split_chronological(engineer_features(clean_and_split_raw(df_raw)))
    assert (len(splits.train), len(splits.val), len(splits.test)) == (18408, 3945, 3945)
    assert splits.train.index.max() < splits.val.index.min() < splits.test.index.min()


def test_scenario4_artifacts_and_notebook_parity():
    assert artifacts_ready(ARTIFACTS_DIR) == []
    lb_app = pd.read_csv(ARTIFACTS_DIR / "leaderboard_v2.csv")
    assert len(lb_app) == 11
    if REF_LEADERBOARD.exists():
        lb_ref = pd.read_csv(REF_LEADERBOARD)
        pd.testing.assert_frame_equal(lb_app, lb_ref)

    df_raw, _ = load_raw_data(CACHE_PATH, allow_api_fallback=False)
    test_df = split_chronological(engineer_features(clean_and_split_raw(df_raw))).test
    for code in LEADERBOARD_ORDER:
        mdl = load_trained_artifact(code, ARTIFACTS_DIR)
        preds = predict_with_model(code, mdl, test_df)
        assert len(preds) == len(test_df)
        m = compute_metrics(test_df["pm2_5"], preds)
        assert m["rmse"] is not None and 5.0 < m["rmse"] < 9.0


def test_scenario5_streamlit_apptest_tabs_and_filters():
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 3

    # Switch primary model to D' and enable rush-hour + peak filters simultaneously
    at.sidebar.selectbox[0].set_value("D'").run()
    assert not at.exception
    at.sidebar.checkbox[1].set_value(True).run()
    assert not at.exception
    at.sidebar.checkbox[0].set_value(True).run()
    assert not at.exception

    # Switch to full test range and clear filters
    at.sidebar.radio[0].set_value("Seluruh Data Uji (3.945 Jam)").run()
    assert not at.exception
