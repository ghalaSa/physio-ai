"""Phase 3: build the manifest and participant-level splits, then run the leakage test.

Usage:
    python scripts/build_manifest.py
Outputs:
    data/manifest/manifest.csv, outputs/splits/splits.json, outputs/metrics/dataset_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio import config  # noqa: E402
from physio.data import manifest as M  # noqa: E402
from physio.data import splits as S  # noqa: E402


def main() -> int:
    config.ensure_dirs()
    df = M.build_manifest()
    M.save_manifest(df)
    summary = M.summarize(df)
    print(json.dumps(summary, indent=2))

    sp = S.make_splits(df)
    sp.to_json()
    df_split = S.attach_split(df, sp)
    S.assert_no_leakage(df_split)
    M.save_manifest(df_split)          # manifest now carries split + cv_fold columns
    print("\nSplit summary (videos per exercise):")
    print(S.split_summary(df_split).to_string())
    print(f"\ntrain={len(sp.train)} val={len(sp.val)} test={len(sp.test)} participants; seed={sp.seed}")

    (config.METRICS_DIR / "dataset_summary.json").write_text(json.dumps({
        "dataset": summary,
        "splits": {"train": sp.train, "val": sp.val, "test": sp.test, "seed": sp.seed},
        "split_table": S.split_summary(df_split).to_dict(),
    }, indent=2, default=str))
    print(f"\nmanifest -> {config.MANIFEST_PATH}\nsplits   -> {config.SPLITS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
