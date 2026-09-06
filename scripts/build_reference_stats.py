"""Phase 9: compute deterministic movement features for every TRAINING clip and store
per-exercise reference percentiles used by the heuristic observations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio import config  # noqa: E402
from physio.analysis.features import compute_features  # noqa: E402
from physio.analysis.observations import REFERENCE_PATH, build_reference_stats  # noqa: E402
from physio.data.manifest import load_manifest  # noqa: E402
from physio.pose.extractor import PoseSequence, pose_cache_path  # noqa: E402
from physio.pose.normalize import normalize_sequence  # noqa: E402


def main() -> int:
    man = load_manifest()
    rows = []
    for r in man.itertuples(index=False):
        p = pose_cache_path(r.video_id)
        if not p.exists():
            continue
        ps = PoseSequence.load(p)
        ok, _ = ps.quality_gate()
        if not ok:
            continue
        norm = normalize_sequence(ps.landmarks, ps.detected, ps.meta["width"], ps.meta["height"])
        mf = compute_features(norm["xy_full"], float(ps.meta.get("sampled_fps") or config.TARGET_FPS), r.exercise)
        rows.append({"video_id": r.video_id, "exercise": r.exercise, "split": r.split, "participant": r.participant,
                     "quality_score": r.quality_score, "primary_angle": mf.primary_angle, **mf.flat()})
    df = pd.DataFrame(rows)
    df.to_csv(config.METRICS_DIR / "movement_features_all.csv", index=False)
    train_rows = df[df["split"] == "train"].drop(columns=["video_id", "split", "participant", "quality_score", "primary_angle"])
    ref = build_reference_stats(train_rows.to_dict("records"))
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_PATH.write_text(json.dumps(ref, indent=2))
    # correlation of each measurement with the physiotherapist score (train split) for the report
    tr = df[(df["split"] == "train") & df["quality_score"].notna()]
    corr = {c: round(float(tr[c].corr(tr["quality_score"])), 3) for c in mf.flat() if tr[c].notna().sum() > 10}
    (config.METRICS_DIR / "feature_score_correlation.json").write_text(json.dumps(corr, indent=2))
    print(f"{len(df)} clips featurized; reference stats for {list(ref)} -> {REFERENCE_PATH}")
    print("feature/score correlations (train):", corr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
