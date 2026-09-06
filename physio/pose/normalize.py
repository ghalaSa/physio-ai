"""Pose normalization and temporal resampling (proposal section 7.1).

Steps
  1. Convert normalized image coordinates to pixel space (x*width, y*height) so the
     aspect ratio is not distorted.
  2. Fill short gaps of undetected frames by linear interpolation in time.
  3. Translate so the hip midpoint is the origin.
  4. Scale by the median torso length (mid-shoulder to mid-hip) so body size and
     camera distance do not matter.
  5. Resample to a fixed number of frames (config.SEQ_LEN) by linear interpolation.
  6. Build the model input: [x, y] for 33 landmarks + visibility + frame-to-frame
     velocity of [x, y]  ->  33*2 + 33 + 33*2 = 165 features per frame.

`flip_left_right` is a training-time augmentation that mirrors the body (swaps
left/right landmark indices and negates x).
"""
from __future__ import annotations

import numpy as np

from physio import config

# MediaPipe pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32
L_INDEX, R_INDEX = 19, 20
L_PINKY, R_PINKY = 17, 18
L_THUMB, R_THUMB = 21, 22

LEFT_RIGHT_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
                    (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
FLIP_INDEX = np.arange(33)
for _a, _b in LEFT_RIGHT_PAIRS:
    FLIP_INDEX[_a], FLIP_INDEX[_b] = _b, _a

N_FEATURES = 33 * 2 + 33 + 33 * 2


def to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """(T,33,4) normalized -> (T,33,2) pixel x,y."""
    xy = landmarks[:, :, :2].astype(np.float32).copy()
    xy[:, :, 0] *= float(width)
    xy[:, :, 1] *= float(height)
    return xy


def interpolate_gaps(arr: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Linearly interpolate frames where detected is False (any trailing/leading gaps are
    filled with the nearest detected frame). arr: (T, ...)."""
    out = arr.copy()
    T = arr.shape[0]
    idx = np.arange(T)
    good = np.asarray(detected, dtype=bool)
    if good.sum() == 0:
        return out
    flat = out.reshape(T, -1)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        flat[:, j] = np.interp(idx, idx[good], col[good])
    return flat.reshape(arr.shape)


def center_and_scale(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Hip-centre and torso-scale a (T,33,2) sequence. Returns (normalized, torso_len_px)."""
    hip_mid = (xy[:, L_HIP] + xy[:, R_HIP]) / 2.0                     # (T,2)
    shoulder_mid = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2.0
    torso = np.linalg.norm(shoulder_mid - hip_mid, axis=1)            # (T,)
    torso_len = float(np.nanmedian(torso))
    if not np.isfinite(torso_len) or torso_len < 1e-3:
        torso_len = 1.0
    centred = xy - hip_mid[:, None, :]
    return centred / torso_len, torso_len


def resample(arr: np.ndarray, n_out: int = config.SEQ_LEN) -> np.ndarray:
    """Resample the first axis of arr (T, ...) to n_out samples by linear interpolation."""
    T = arr.shape[0]
    if T == n_out:
        return arr.copy()
    src = np.linspace(0.0, 1.0, T)
    dst = np.linspace(0.0, 1.0, n_out)
    flat = arr.reshape(T, -1)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst, src, flat[:, j])
    return out.reshape((n_out,) + arr.shape[1:])


def normalize_sequence(landmarks: np.ndarray, detected: np.ndarray, width: int, height: int,
                       seq_len: int = config.SEQ_LEN) -> dict:
    """Full normalization. Returns dict with 'xy' (L,33,2), 'vis' (L,33), 'features' (L,165),
    'torso_len_px', and the un-resampled normalized 'xy_full' (T,33,2) for the analysis module."""
    xy = to_pixels(landmarks, width, height)
    vis = np.nan_to_num(landmarks[:, :, 3].astype(np.float32), nan=0.0)
    xy = interpolate_gaps(xy, detected)
    xy_norm, torso_len = center_and_scale(xy)
    xy_norm = np.nan_to_num(xy_norm, nan=0.0)
    xy_r = resample(xy_norm, seq_len)
    vis_r = resample(vis, seq_len)
    feats = build_features(xy_r, vis_r)
    return {"xy": xy_r, "vis": vis_r, "features": feats, "torso_len_px": torso_len, "xy_full": xy_norm}


def build_features(xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """(L,33,2) + (L,33) -> (L,165) coords, visibility, velocity."""
    L = xy.shape[0]
    vel = np.zeros_like(xy)
    if L > 1:
        vel[1:] = xy[1:] - xy[:-1]
    return np.concatenate([xy.reshape(L, -1), vis.reshape(L, -1), vel.reshape(L, -1)], axis=1).astype(np.float32)


def flip_left_right(xy: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror augmentation: swap left/right landmarks and negate x."""
    xy_f = xy[:, FLIP_INDEX].copy()
    xy_f[:, :, 0] *= -1.0
    return xy_f, vis[:, FLIP_INDEX].copy()


def summary_statistics(features: np.ndarray) -> np.ndarray:
    """Per-clip statistical descriptor for the non-temporal baseline: mean, std, min, max
    over time for every feature -> (4*165,)."""
    return np.concatenate([features.mean(0), features.std(0), features.min(0), features.max(0)]).astype(np.float32)
