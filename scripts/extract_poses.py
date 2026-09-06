"""Phase 4: streaming pose extraction.

For each video in the manifest (in a balanced priority order): download from
Dataverse -> run MediaPipe pose -> save data/poses/<video_id>.npz -> delete the video.
Resumable: videos with an existing .npz (or a recorded permanent failure) are skipped.

Usage:
    python scripts/extract_poses.py --workers 8                # everything
    python scripts/extract_poses.py --max-videos 200           # quick subset
    python scripts/extract_poses.py --participants P01 P02     # specific people
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio import config  # noqa: E402
from physio.data.manifest import load_manifest  # noqa: E402

PROGRESS_LOG = config.POSES_DIR / "_progress.jsonl"


def priority_order(df):
    """Round-robin over (participant, exercise) cells so that any prefix of the list is
    balanced across people and exercises; within a cell prefer front/full-light first."""
    var_rank = {"FL": 0, "ML": 1, "LL": 2, "LJ": 3, "HJ": 4, "O": 5, "LR": 6}
    ang_rank = {"F": 0, "L": 1, "R": 2}
    df = df.assign(_v=df["variation"].map(var_rank).fillna(9), _a=df["angle"].map(ang_rank))
    df = df.sort_values(["participant", "exercise", "_v", "_a"])
    df["_rank"] = df.groupby(["participant", "exercise"]).cumcount()
    return df.sort_values(["_rank", "participant", "exercise"]).drop(columns=["_v", "_a", "_rank"])


def load_progress() -> dict[str, dict]:
    done = {}
    if PROGRESS_LOG.exists():
        for line in PROGRESS_LOG.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["video_id"]] = rec
    return done


def process_one(job: dict) -> dict:
    """Runs in a worker process: download, extract, delete."""
    from physio.data.download import download_file
    from physio.pose.extractor import extract_pose_sequence, pose_cache_path
    vid, fid, size = job["video_id"], job["file_id"], job["size_bytes"]
    tmp = config.TMP_DIR / f"{vid}.mp4"
    t0 = time.time()
    rec = {"video_id": vid, "file_id": fid}
    try:
        download_file(fid, tmp, expected_size=size)
        t1 = time.time()
        ps = extract_pose_sequence(tmp, video_id=vid)
        ps.save(pose_cache_path(vid))
        ok, reason = ps.quality_gate()
        rec.update(status="ok", n_frames=ps.n_frames, n_detected=ps.n_detected, detection_rate=round(ps.detection_rate, 3),
                   gate_ok=ok, gate_reason=reason, download_s=round(t1 - t0, 1), pose_s=round(time.time() - t1, 1))
    except Exception as exc:  # noqa: BLE001 - recorded per video, never aborts the batch
        rec.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
    finally:
        tmp.unlink(missing_ok=True)
        tmp.with_suffix(".mp4.part").unlink(missing_ok=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 0))
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--exercises", nargs="*", default=None)
    ap.add_argument("--retry-errors", action="store_true", help="re-attempt videos that previously errored")
    args = ap.parse_args()

    config.ensure_dirs()
    df = load_manifest()
    if args.participants:
        df = df[df["participant"].isin(args.participants)]
    if args.exercises:
        df = df[df["exercise"].isin(args.exercises)]
    df = priority_order(df)

    progress = load_progress()
    have = {p.stem for p in config.POSES_DIR.glob("*.npz")}
    jobs = []
    for r in df.itertuples(index=False):
        if r.video_id in have:
            continue
        prev = progress.get(r.video_id)
        if prev and prev.get("status") == "error" and not args.retry_errors:
            continue
        jobs.append({"video_id": r.video_id, "file_id": int(r.file_id), "size_bytes": int(r.size_bytes)})
    if args.max_videos:
        jobs = jobs[: max(0, args.max_videos - len(have))]
    print(f"{len(have)} cached, {len(jobs)} to process, workers={args.workers}", flush=True)
    if not jobs:
        return 0

    t0 = time.time()
    n_ok = n_err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(PROGRESS_LOG, "a") as log:
        futures = {ex.submit(process_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            rec["ts"] = time.time()
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if rec["status"] == "ok":
                n_ok += 1
            else:
                n_err += 1
            if i % 10 == 0 or i == len(jobs):
                el = time.time() - t0
                rate = i / el
                print(f"[{i}/{len(jobs)}] ok={n_ok} err={n_err} {el/60:.1f} min elapsed, "
                      f"~{(len(jobs)-i)/max(rate,1e-6)/60:.0f} min remaining", flush=True)
    print(f"done: ok={n_ok} err={n_err} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
