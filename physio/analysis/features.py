"""Interpretable movement measurements (proposal section 8).

Every value is a deterministic function of the pose sequence. Each measurement
carries a `reliable` flag so the feedback layer never reports a number computed
from too little data (e.g. repetition consistency with a single repetition).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import signal

from physio.analysis.angles import BILATERAL_PAIRS, compute_angles, primary_angles


@dataclass
class Measurement:
    value: float | None
    unit: str
    reliable: bool
    note: str = ""


@dataclass
class MovementFeatures:
    exercise_code: str
    primary_angle: str
    fps: float
    duration_s: float
    range_of_motion_deg: Measurement
    rom_per_angle: dict[str, float]
    smoothness_sparc: Measurement
    tempo_reps_per_s: Measurement
    repetitions: Measurement
    rep_duration_s: Measurement
    rep_duration_cv: Measurement
    symmetry_ratio: Measurement
    trajectory_consistency: Measurement
    trunk_lean_range_deg: Measurement
    angle_series: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self, include_series: bool = False) -> dict:
        d = asdict(self)
        if not include_series:
            d.pop("angle_series", None)
        return d

    def flat(self) -> dict[str, float | None]:
        """Scalar values only, for tables and reference statistics."""
        return {
            "range_of_motion_deg": self.range_of_motion_deg.value,
            "smoothness_sparc": self.smoothness_sparc.value,
            "tempo_reps_per_s": self.tempo_reps_per_s.value,
            "repetitions": self.repetitions.value,
            "rep_duration_s": self.rep_duration_s.value,
            "rep_duration_cv": self.rep_duration_cv.value,
            "symmetry_ratio": self.symmetry_ratio.value,
            "trajectory_consistency": self.trajectory_consistency.value,
            "trunk_lean_range_deg": self.trunk_lean_range_deg.value,
        }


def robust_range(x: np.ndarray) -> float:
    return float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5))


def sparc(movement: np.ndarray, fs: float, padlevel: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral arc length smoothness (Balasubramanian et al. 2015). Closer to 0 = smoother.
    Applied to the angular-velocity profile of the primary angle."""
    x = np.asarray(movement, dtype=float)
    x = x - x.mean()
    if len(x) < 4 or np.allclose(x, 0):
        return float("nan")
    nfft = int(2 ** (np.ceil(np.log2(len(x))) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    mag = np.abs(np.fft.fft(x, nfft))
    mag = mag / max(mag.max(), 1e-12)
    sel = f <= fc
    f, mag = f[sel], mag[sel]
    above = np.where(mag >= amp_th)[0]
    if len(above) == 0:
        return float("nan")
    f, mag = f[: above[-1] + 1], mag[: above[-1] + 1]
    if len(f) < 2:
        return float("nan")
    arc = -np.sum(np.sqrt((np.diff(f) / (f[-1] - f[0])) ** 2 + np.diff(mag) ** 2))
    return float(arc)


def detect_reps(angle: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Find repetition peaks of a smoothed angle signal. Returns (peak_indices, dominant_freq_hz)."""
    x = np.asarray(angle, dtype=float)
    if len(x) < 8:
        return np.array([], dtype=int), float("nan")
    win = max(3, int(round(fs * 0.3)) | 1)
    xs = signal.savgol_filter(x, window_length=min(win, len(x) - (1 - len(x) % 2)), polyorder=2) if len(x) > win else x
    rom = robust_range(xs)
    if rom < 5.0:
        return np.array([], dtype=int), float("nan")
    # dominant frequency from the detrended spectrum
    d = xs - xs.mean()
    freqs = np.fft.rfftfreq(len(d), 1.0 / fs)
    spec = np.abs(np.fft.rfft(d))
    spec[freqs < 0.1] = 0
    dom = float(freqs[np.argmax(spec)]) if spec.max() > 0 else float("nan")
    min_dist = max(2, int(fs * 0.5))
    peaks, _ = signal.find_peaks(xs, prominence=0.3 * rom, distance=min_dist)
    return peaks, dom


def trajectory_consistency(angle: np.ndarray, peaks: np.ndarray, n_points: int = 20) -> float:
    """Mean pairwise correlation between rep-to-rep angle trajectories (peak to peak)."""
    if len(peaks) < 3:
        return float("nan")
    segs = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        seg = np.asarray(angle[a:b + 1], dtype=float)
        if len(seg) < 4:
            continue
        segs.append(np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(seg)), seg))
    if len(segs) < 2:
        return float("nan")
    S = np.stack(segs)
    S = S - S.mean(1, keepdims=True)
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    S = S / np.maximum(norms, 1e-8)
    C = S @ S.T
    iu = np.triu_indices(len(segs), 1)
    return float(np.clip(C[iu].mean(), -1, 1))


def compute_features(xy: np.ndarray, fps: float, exercise_code: str) -> MovementFeatures:
    """xy: (T,33,2) hip-centred torso-scaled coordinates (NOT resampled). fps: sampling rate."""
    angles = compute_angles(xy)
    T = xy.shape[0]
    duration = T / fps if fps > 0 else float("nan")
    prim_names = primary_angles(exercise_code)
    rom_all = {k: robust_range(v) for k, v in angles.items()}
    # primary signal = the primary angle with the largest range (the moving side)
    prim = max(prim_names, key=lambda k: rom_all.get(k, 0.0))
    sig = angles[prim]
    rom = rom_all[prim]

    vel = np.gradient(sig) * fps
    sm = sparc(vel, fps)
    peaks, dom = detect_reps(sig, fps)
    n_reps = int(len(peaks))
    rep_dur = np.diff(peaks) / fps if n_reps >= 2 else np.array([])
    cons = trajectory_consistency(sig, peaks)

    # symmetry: ratio of ROM on the smaller vs larger side for the primary bilateral pair
    sym, sym_note = None, "no bilateral pair for primary angle"
    for l, r in BILATERAL_PAIRS:
        if prim in (l, r):
            lo, hi = sorted([rom_all[l], rom_all[r]])
            sym = float(lo / hi) if hi > 1e-6 else None
            sym_note = f"{l} vs {r} range of motion"
            break
    unilateral = sym is not None and sym < 0.5

    return MovementFeatures(
        exercise_code=exercise_code, primary_angle=prim, fps=float(fps), duration_s=float(duration),
        range_of_motion_deg=Measurement(round(rom, 1), "deg", T >= 10, f"5th-95th percentile range of {prim}"),
        rom_per_angle={k: round(v, 1) for k, v in rom_all.items()},
        smoothness_sparc=Measurement(None if np.isnan(sm) else round(sm, 3), "sparc", not np.isnan(sm) and T >= 20,
                                     "spectral arc length of angular velocity; closer to 0 is smoother"),
        tempo_reps_per_s=Measurement(None if np.isnan(dom) else round(dom, 3), "Hz", n_reps >= 2,
                                     "dominant frequency of the primary angle"),
        repetitions=Measurement(n_reps, "count", T >= 20, "peaks of the primary angle"),
        rep_duration_s=Measurement(round(float(rep_dur.mean()), 2) if len(rep_dur) else None, "s", len(rep_dur) >= 2),
        rep_duration_cv=Measurement(round(float(rep_dur.std() / rep_dur.mean()), 3) if len(rep_dur) >= 2 and rep_dur.mean() > 0 else None,
                                    "ratio", len(rep_dur) >= 3, "coefficient of variation of rep duration"),
        symmetry_ratio=Measurement(None if sym is None else round(sym, 3), "ratio", sym is not None and not unilateral,
                                   sym_note + (" (single-side exercise, symmetry not applicable)" if unilateral else "")),
        trajectory_consistency=Measurement(None if np.isnan(cons) else round(cons, 3), "corr", not np.isnan(cons),
                                           "mean correlation between successive repetitions"),
        trunk_lean_range_deg=Measurement(round(rom_all["trunk_lean"], 1), "deg", T >= 10, "range of trunk inclination"),
        angle_series={k: [round(float(x), 2) for x in v] for k, v in angles.items()},
    )
