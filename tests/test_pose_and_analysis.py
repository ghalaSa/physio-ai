import numpy as np
import pytest

from physio import config
from physio.analysis.angles import angle_between, compute_angles, primary_angles
from physio.analysis.features import compute_features, detect_reps, sparc
from physio.analysis.observations import build_reference_stats, make_observations
from physio.pose import normalize as N
from physio.pose.extractor import PoseSequence
from physio.pose.synthetic import synthetic_sequence


def test_pose_sequence_roundtrip(tmp_path):
    ps = synthetic_sequence(n_frames=40)
    p = ps.save(tmp_path / "x.npz")
    ps2 = PoseSequence.load(p)
    assert ps2.landmarks.shape == (40, 33, 4)
    assert ps2.meta["video_id"] == "synthetic"
    assert ps2.quality_gate()[0]


def test_quality_gate_flags_bad_sequences():
    ok, reason = synthetic_sequence(n_frames=10).quality_gate()
    assert not ok and "frames" in reason
    ok, reason = synthetic_sequence(n_frames=60, dropout_frames=40).quality_gate()
    assert not ok and "detected" in reason
    ps = synthetic_sequence(n_frames=60)
    ps.landmarks[:, :, 3] = 0.2
    ok, reason = ps.quality_gate()
    assert not ok and "visibility" in reason


def test_normalization_is_centered_scaled_and_fixed_length():
    ps = synthetic_sequence(n_frames=90, dropout_frames=5)
    out = N.normalize_sequence(ps.landmarks, ps.detected, ps.meta["width"], ps.meta["height"])
    assert out["features"].shape == (config.SEQ_LEN, N.N_FEATURES)
    xy = out["xy"]
    hip_mid = (xy[:, N.L_HIP] + xy[:, N.R_HIP]) / 2
    assert np.allclose(hip_mid, 0, atol=1e-5)
    torso = np.linalg.norm((xy[:, N.L_SHOULDER] + xy[:, N.R_SHOULDER]) / 2 - hip_mid, axis=1)
    assert abs(np.median(torso) - 1.0) < 0.05
    assert np.isfinite(out["features"]).all()


def test_normalization_invariant_to_scale_and_translation():
    ps = synthetic_sequence(n_frames=50)
    a = N.normalize_sequence(ps.landmarks, ps.detected, 720, 1280)["xy"]
    lm2 = ps.landmarks.copy()
    lm2[:, :, :2] = lm2[:, :, :2] * 0.5 + 0.2          # shrink and shift in image space
    b = N.normalize_sequence(lm2, ps.detected, 720, 1280)["xy"]
    assert np.allclose(a, b, atol=1e-4)


def test_flip_left_right_swaps_sides():
    ps = synthetic_sequence(n_frames=20)
    out = N.normalize_sequence(ps.landmarks, ps.detected, 720, 1280)
    xy_f, vis_f = N.flip_left_right(out["xy"], out["vis"])
    assert np.allclose(xy_f[:, N.L_SHOULDER, 0], -out["xy"][:, N.R_SHOULDER, 0])
    assert np.allclose(xy_f[:, N.L_SHOULDER, 1], out["xy"][:, N.R_SHOULDER, 1])


def test_angle_geometry():
    a = np.array([[1.0, 0.0]]); b = np.array([[0.0, 0.0]]); c = np.array([[0.0, 1.0]])
    assert abs(angle_between(a, b, c)[0] - 90) < 1e-6
    assert abs(angle_between(a, b, -a)[0] - 180) < 1e-6


def test_synthetic_abduction_features_are_sensible():
    ps = synthetic_sequence(n_frames=120, fps=10, reps=4, amplitude_deg=80, side="right")
    out = N.normalize_sequence(ps.landmarks, ps.detected, 720, 1280)
    angles = compute_angles(out["xy_full"])
    assert set(angles) >= {"right_shoulder", "left_shoulder", "trunk_lean"}
    mf = compute_features(out["xy_full"], 10.0, "E01")
    assert mf.primary_angle == "right_shoulder"
    assert 60 <= mf.range_of_motion_deg.value <= 95
    assert mf.repetitions.value == 4
    assert abs(mf.tempo_reps_per_s.value - 4 / 12) < 0.08
    assert mf.trajectory_consistency.value > 0.9
    assert mf.symmetry_ratio.value is not None and mf.symmetry_ratio.value < 0.5   # one-sided movement
    assert mf.trunk_lean_range_deg.value < 5
    assert "E01" in mf.to_dict()["exercise_code"]


def test_detect_reps_and_sparc():
    fs = 10.0
    t = np.arange(0, 12, 1 / fs)
    sig = 60 + 30 * np.sin(2 * np.pi * 0.5 * t)
    peaks, dom = detect_reps(sig, fs)
    assert 5 <= len(peaks) <= 7
    assert abs(dom - 0.5) < 0.1
    smooth = sparc(np.gradient(sig) * fs, fs)
    jerky = sparc(np.gradient(sig + 15 * np.random.default_rng(0).normal(size=len(sig))) * fs, fs)
    assert smooth > jerky


def test_observations_use_reference_and_flags():
    ps = synthetic_sequence(n_frames=120, amplitude_deg=30)
    out = N.normalize_sequence(ps.landmarks, ps.detected, 720, 1280)
    mf = compute_features(out["xy_full"], 10.0, "E01")
    ref = build_reference_stats([{"exercise": "E01", "range_of_motion_deg": 80 + i, "tempo_reps_per_s": 0.3,
                                  "smoothness_sparc": -1.5} for i in range(10)])
    obs = make_observations(mf, "E01", exercise_conf=0.4, quality_score=50.0, reference=ref)
    codes = {o.code for o in obs}
    assert {"low_exercise_confidence", "low_predicted_quality", "low_range_of_motion"} <= codes
    assert all(o.kind in ("model", "heuristic") for o in obs)
    assert not any("diagnos" in o.text.lower() for o in obs)
    assert primary_angles("E07") == ["left_hip", "right_hip"]
