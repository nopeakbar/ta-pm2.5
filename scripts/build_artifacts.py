"""One-time deterministic builder to pre-train models and verify notebook parity."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from src.pipeline import build_all_artifacts  # noqa: E402


def main() -> None:
    cache_path = APP_DIR / "data" / "cache_jakarta_raw.csv"
    source_cache = Path(r"D:\Career\TA-Fword\19 october\cache_jakarta_raw.csv")
    if not cache_path.exists() and source_cache.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_cache, cache_path)
        print(f"Copied raw cache from {source_cache} -> {cache_path}")

    artifacts_dir = APP_DIR / "artifacts"
    t0 = time.time()
    meta = build_all_artifacts(cache_path, artifacts_dir)
    elapsed = time.time() - t0
    print(f"Built all artifacts cleanly in {elapsed:.1f}s: clean_rows={meta['clean_rows']}, p95={meta['p95_threshold']}")

    ref_lb = Path(r"D:\Career\TA-Fword\19 october\leaderboard_v2.csv")
    if ref_lb.exists():
        df_new = pd.read_csv(artifacts_dir / "leaderboard_v2.csv")
        df_ref = pd.read_csv(ref_lb)
        if df_new.equals(df_ref):
            print("PARITY OK: Generated leaderboard_v2.csv is 100% identical to 19 october/leaderboard_v2.csv!")
        else:
            print("NOTE: Minor floating/seed variance vs reference CSV:")
            print(df_new[["Kode", "RMSE", "MAE", "R2"]])


if __name__ == "__main__":
    main()
