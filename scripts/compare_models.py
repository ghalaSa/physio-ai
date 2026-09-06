"""Phase 8: compare all logged runs on validation metrics and select the deployed models."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio.models.compare import load_runs, save_best, select_best, write_report  # noqa: E402


def main() -> int:
    df = load_runs()
    if df.empty:
        print("no runs logged yet")
        return 1
    df = df[~df["run_name"].str.startswith("cv_")]
    best = select_best(df)
    path = write_report(df, best)
    save_best(best)
    print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
