"""Deterministic joint-angle geometry from 2D pose landmarks.

All angles are planar (image-plane) angles in degrees computed from the hip-centred,
torso-scaled 2D coordinates. They are geometric measurements, not goniometry, and
are labelled as such in every user-facing output.
"""
from __future__ import annotations

import numpy as np

from physio.pose import normalize as N

# angle name -> (point A, vertex B, point C)
ANGLE_DEFS: dict[str, tuple[int, int, int]] = {
    "left_elbow": (N.L_SHOULDER, N.L_ELBOW, N.L_WRIST),
    "right_elbow": (N.R_SHOULDER, N.R_ELBOW, N.R_WRIST),
    "left_shoulder": (N.L_HIP, N.L_SHOULDER, N.L_ELBOW),
    "right_shoulder": (N.R_HIP, N.R_SHOULDER, N.R_ELBOW),
    "left_hip": (N.L_SHOULDER, N.L_HIP, N.L_KNEE),
    "right_hip": (N.R_SHOULDER, N.R_HIP, N.R_KNEE),
    "left_knee": (N.L_HIP, N.L_KNEE, N.L_ANKLE),
    "right_knee": (N.R_HIP, N.R_KNEE, N.R_ANKLE),
    "left_wrist": (N.L_ELBOW, N.L_WRIST, N.L_INDEX),
    "right_wrist": (N.R_ELBOW, N.R_WRIST, N.R_INDEX),
}
BILATERAL_PAIRS = [("left_elbow", "right_elbow"), ("left_shoulder", "right_shoulder"),
                   ("left_hip", "right_hip"), ("left_knee", "right_knee"), ("left_wrist", "right_wrist")]

# Which geometric angles matter most for each exercise. This is a heuristic mapping
# from the exercise names in the dataset README to the joints visible in 2D; it is
# used to pick the "primary" signal for range-of-motion, tempo and repetition
# analysis. It is NOT a clinical definition of the exercise.
EXERCISE_FOCUS: dict[str, dict] = {
    "E01": {"primary": ["left_shoulder", "right_shoulder"], "note": "shoulder abduction: arm-to-trunk angle"},
    "E02": {"primary": ["left_shoulder", "right_shoulder"], "note": "shoulder adduction: arm-to-trunk angle"},
    "E03": {"primary": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
            "note": "lateral rotation: forearm sweep seen through elbow/shoulder angles in 2D"},
    "E04": {"primary": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
            "note": "medial rotation: forearm sweep seen through elbow/shoulder angles in 2D"},
    "E05": {"primary": ["left_shoulder", "right_shoulder"], "note": "circumduction: shoulder angle cycles"},
    "E06": {"primary": ["left_wrist", "right_wrist"], "note": "wrist extension: forearm-to-hand angle"},
    "E07": {"primary": ["left_hip", "right_hip"], "note": "hip flexion: trunk-to-thigh angle"},
    "E08": {"primary": ["left_knee", "right_knee", "left_hip", "right_hip"], "note": "both-leg flexion: knee and hip angles"},
    "E09": {"primary": ["left_hip", "right_hip", "trunk_lean"], "note": "back extension: trunk inclination and hip angle"},
}


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for arrays of 2D points (T,2)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.einsum("...i,...i->...", v1, v2) / np.maximum(n1 * n2, 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def trunk_lean(xy: np.ndarray) -> np.ndarray:
    """Angle of the hip->shoulder vector from the image vertical, degrees (0 = upright)."""
    hip = (xy[:, N.L_HIP] + xy[:, N.R_HIP]) / 2
    sh = (xy[:, N.L_SHOULDER] + xy[:, N.R_SHOULDER]) / 2
    v = sh - hip
    # image y grows downward; upright torso points to negative y
    return np.degrees(np.arctan2(np.abs(v[:, 0]), np.maximum(-v[:, 1], 1e-8)))


def compute_angles(xy: np.ndarray) -> dict[str, np.ndarray]:
    """xy: (T,33,2) normalized coordinates -> {angle_name: (T,) degrees}."""
    out = {name: angle_between(xy[:, a], xy[:, b], xy[:, c]) for name, (a, b, c) in ANGLE_DEFS.items()}
    out["trunk_lean"] = trunk_lean(xy)
    return out


def primary_angles(exercise_code: str) -> list[str]:
    return EXERCISE_FOCUS.get(exercise_code, {}).get("primary", ["left_shoulder", "right_shoulder"])
