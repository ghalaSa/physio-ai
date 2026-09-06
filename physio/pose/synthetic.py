"""Synthetic pose sequences for tests and demos (no video, no MediaPipe needed).

Generates a plausible standing skeleton in normalized image coordinates with one
limb oscillating, so the whole downstream pipeline (normalization, angles, features,
models, feedback, storage) can be exercised deterministically.
"""
from __future__ import annotations

import numpy as np

from physio.pose import normalize as N
from physio.pose.extractor import PoseSequence

# base skeleton (x, y) in normalized image coords for a 720x1280 portrait frame, facing camera
_BASE = {
    0: (0.50, 0.15),                             # nose
    11: (0.42, 0.28), 12: (0.58, 0.28),          # shoulders
    13: (0.40, 0.42), 14: (0.60, 0.42),          # elbows
    15: (0.39, 0.55), 16: (0.61, 0.55),          # wrists
    17: (0.385, 0.58), 18: (0.615, 0.58), 19: (0.38, 0.585), 20: (0.62, 0.585), 21: (0.39, 0.575), 22: (0.61, 0.575),
    23: (0.45, 0.55), 24: (0.55, 0.55),          # hips
    25: (0.45, 0.75), 26: (0.55, 0.75),          # knees
    27: (0.45, 0.93), 28: (0.55, 0.93),          # ankles
    29: (0.44, 0.95), 30: (0.56, 0.95), 31: (0.46, 0.97), 32: (0.54, 0.97),
}
for _i in (1, 2, 3, 7, 9):
    _BASE[_i] = (0.47, 0.14)
for _i in (4, 5, 6, 8, 10):
    _BASE[_i] = (0.53, 0.14)


def synthetic_sequence(n_frames: int = 120, fps: float = 10.0, reps: float = 4.0, amplitude_deg: float = 80.0,
                       side: str = "right", noise: float = 0.002, dropout_frames: int = 0, seed: int = 0,
                       width: int = 720, height: int = 1280, video_id: str = "synthetic") -> PoseSequence:
    """Right (or left) arm abduction-like motion: the arm rotates about the shoulder."""
    rng = np.random.default_rng(seed)
    T = n_frames
    lm = np.zeros((T, 33, 4), np.float32)
    for i in range(33):
        lm[:, i, 0], lm[:, i, 1] = _BASE[i]
    lm[:, :, 2] = 0.0
    lm[:, :, 3] = 0.95
    t = np.arange(T) / fps
    phase = 0.5 * (1 - np.cos(2 * np.pi * reps * t / (T / fps)))       # 0..1..0 per rep
    theta = np.radians(amplitude_deg) * phase
    sh, el, wr = (12, 14, 16) if side == "right" else (11, 13, 15)
    sign = 1.0 if side == "right" else -1.0
    sx, sy = _BASE[sh]
    upper, fore = 0.14, 0.13
    aspect = width / height
    for k in range(T):
        # rotate the arm outward from hanging straight down; x scaled by aspect ratio
        ex = sx + sign * upper * np.sin(theta[k]) / aspect
        ey = sy + upper * np.cos(theta[k])
        wx = ex + sign * fore * np.sin(theta[k]) / aspect
        wy = ey + fore * np.cos(theta[k])
        lm[k, el, :2] = (ex, ey)
        lm[k, wr, :2] = (wx, wy)
        for h in ((17, 19, 21) if side == "left" else (18, 20, 22)):
            lm[k, h, :2] = (wx + sign * 0.005, wy + 0.02)
    lm[:, :, :2] += rng.normal(0, noise, size=(T, 33, 2)).astype(np.float32)
    det = np.ones(T, bool)
    if dropout_frames:
        idx = rng.choice(T, size=min(dropout_frames, T), replace=False)
        det[idx] = False
        lm[idx] = np.nan
    world = np.concatenate([(lm[:, :, :2] - 0.5) * 1.7, lm[:, :, 2:3]], axis=2).astype(np.float32)
    meta = {"video_id": video_id, "fps": 30.0, "width": width, "height": height, "n_frames": T * 3,
            "sampled_fps": fps, "duration_s": float(T / fps), "target_fps": fps, "n_sampled": T}
    return PoseSequence(landmarks=lm, world=world, detected=det, timestamps=t.astype(np.float32), meta=meta)
