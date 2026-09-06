"""Assemble model-ready arrays from the cached pose sequences and the manifest.

Nothing here reads video. It joins manifest rows with the .npz cache, applies the
quality gate, normalizes, and produces:
    X_seq   (N, SEQ_LEN, 165)  temporal features
    X_stat  (N, 660)           summary statistics for the baseline
    y_cls   (N,)               exercise index
    y_reg   (N,)               quality score 0-100 (NaN if unscored)
    groups  (N,)               participant id (for grouped CV)
    index   DataFrame          manifest rows that made it in, with 'gate_reason'
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from physio import config
from physio.pose.extractor import PoseSequence, pose_cache_path
from physio.pose.normalize import normalize_sequence, summary_statistics


@dataclass
class Arrays:
    X_seq: np.ndarray
    X_stat: np.ndarray
    y_cls: np.ndarray
    y_reg: np.ndarray
    groups: np.ndarray
    index: pd.DataFrame
    classes: list[str]

    def subset(self, mask: np.ndarray) -> "Arrays":
        return Arrays(self.X_seq[mask], self.X_stat[mask], self.y_cls[mask], self.y_reg[mask],
                      self.groups[mask], self.index[mask].reset_index(drop=True), self.classes)

    def for_split(self, split: str) -> "Arrays":
        return self.subset((self.index["split"] == split).to_numpy())

    def scored(self) -> "Arrays":
        return self.subset(~np.isnan(self.y_reg))

    def __len__(self) -> int:
        return int(self.X_seq.shape[0])


def available_pose_ids(poses_dir: Path = config.POSES_DIR) -> set[str]:
    return {p.stem for p in Path(poses_dir).glob("*.npz")}


def load_arrays(manifest: pd.DataFrame, poses_dir: Path = config.POSES_DIR,
                seq_len: int = config.SEQ_LEN, verbose: bool = True) -> Arrays:
    classes = sorted(manifest["exercise"].unique())
    cls_index = {c: i for i, c in enumerate(classes)}
    have = available_pose_ids(poses_dir)
    rows, seqs, stats, gated = [], [], [], []
    for r in manifest.itertuples(index=False):
        if r.video_id not in have:
            continue
        ps = PoseSequence.load(pose_cache_path(r.video_id, poses_dir))
        ok, reason = ps.quality_gate()
        if not ok:
            gated.append((r.video_id, reason))
            continue
        norm = normalize_sequence(ps.landmarks, ps.detected, ps.meta["width"], ps.meta["height"], seq_len)
        seqs.append(norm["features"])
        stats.append(summary_statistics(norm["features"]))
        rows.append(r._asdict())
    if verbose:
        print(f"[dataset] {len(rows)} clips loaded, {len(gated)} failed the pose quality gate, "
              f"{len(manifest) - len(rows) - len(gated)} not yet extracted")
    index = pd.DataFrame(rows)
    index["gate_reason"] = "ok"
    X_seq = np.stack(seqs) if seqs else np.zeros((0, seq_len, 165), np.float32)
    X_stat = np.stack(stats) if stats else np.zeros((0, 660), np.float32)
    y_cls = index["exercise"].map(cls_index).to_numpy() if len(index) else np.zeros(0, int)
    y_reg = index["quality_score"].to_numpy(dtype=float) if len(index) else np.zeros(0)
    groups = index["participant"].to_numpy() if len(index) else np.zeros(0, object)
    return Arrays(X_seq, X_stat, y_cls, y_reg, groups, index, classes)


def gated_report(manifest: pd.DataFrame, poses_dir: Path = config.POSES_DIR) -> pd.DataFrame:
    """Per-video pose quality diagnostics (used for the evaluation report)."""
    have = available_pose_ids(poses_dir)
    out = []
    for vid in manifest["video_id"]:
        if vid not in have:
            continue
        ps = PoseSequence.load(pose_cache_path(vid, poses_dir))
        ok, reason = ps.quality_gate()
        out.append({"video_id": vid, "n_frames": ps.n_frames, "n_detected": ps.n_detected,
                    "detection_rate": ps.detection_rate, "mean_visibility": ps.mean_visibility,
                    "gate_ok": ok, "gate_reason": reason})
    return pd.DataFrame(out)
