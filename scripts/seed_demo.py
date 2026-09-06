"""Seed the session database with analyzed sessions from cached TEST-split clips so the
dashboards have realistic content for the end-to-end demonstration.

Creates two demo patients, each with several sessions per exercise spread over past
weeks (so quality trends and repeated observations are visible). Uses the selected
models and the configured feedback provider.

Usage:
    python scripts/seed_demo.py [--per-exercise 4] [--reset]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio import config  # noqa: E402
from physio.data.manifest import load_manifest  # noqa: E402
from physio.db import repo  # noqa: E402
from physio.db.models import Base, make_engine, make_session_factory  # noqa: E402
from physio.pipeline import Analyzer, store_result  # noqa: E402
from physio.pose.extractor import PoseSequence, pose_cache_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-exercise", type=int, default=4)
    ap.add_argument("--reset", action="store_true", help="drop existing tables first")
    args = ap.parse_args()

    if args.reset:
        Base.metadata.drop_all(make_engine())
    SessionLocal = make_session_factory()
    db = SessionLocal()
    an = Analyzer()
    man = load_manifest()
    test = man[man["split"] == "test"]
    test_participants = sorted(test["participant"].unique())
    demo = {"Demo Patient A": test_participants[0], "Demo Patient B": test_participants[1]}
    now = datetime.now(timezone.utc)
    n_done = 0
    for name, pid in demo.items():
        patient = repo.get_or_create_patient(db, name)
        clips = test[test["participant"] == pid].sort_values(["exercise", "variation", "angle"])
        for ex, grp in clips.groupby("exercise"):
            chosen = [v for v in grp["video_id"] if pose_cache_path(v).exists()][: args.per_exercise]
            for i, vid in enumerate(chosen):
                ps = PoseSequence.load(pose_cache_path(vid))
                when = now - timedelta(days=7 * (len(chosen) - i), hours=i)
                hist = repo.history_for(db, patient.id, ex, before=when)
                res = an.analyze_pose_sequence(ps, history=hist)
                s = store_result(db, patient.id, f"{vid}.mp4", res)
                s.created_at = when
                db.commit()
                n_done += 1
                print(f"{name}: {vid} -> {res.status} {res.exercise_code} conf={res.exercise_confidence} score={res.quality_score}")
    print(f"seeded {n_done} sessions into {config.DEFAULT_DB_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
