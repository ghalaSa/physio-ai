"""Heuristic observations derived from the deterministic measurements.

These are rule-based flags compared against reference statistics computed from the
TRAINING split (outputs/models/reference_stats.json). They are explicitly labelled
`kind="heuristic"`; model predictions are `kind="model"`; measurements are
`kind="measurement"`. No clinical interpretation is generated here or anywhere in the
system (proposal sections 8 and 16).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from physio import config
from physio.analysis.features import MovementFeatures

REFERENCE_PATH = config.MODELS_DIR / "reference_stats.json"

# Fallback thresholds used only when no reference statistics exist yet.
DEFAULTS = {
    "symmetry_ratio_min": 0.75,
    "consistency_min": 0.6,
    "rep_duration_cv_max": 0.35,
    "min_reps": 2,
    "classifier_confidence_min": 0.6,
    "quality_review_threshold": 60.0,
    "trunk_lean_upper_limb_max_deg": 15.0,
}
UPPER_LIMB = {"E01", "E02", "E03", "E04", "E05", "E06"}


@dataclass
class Observation:
    code: str
    kind: str                # "heuristic" | "model" | "measurement" | "system"
    severity: str            # "info" | "attention" | "review"
    text: str
    evidence: dict


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def _ref(reference: dict, exercise: str, key: str, stat: str):
    return reference.get(exercise, {}).get(key, {}).get(stat)


def make_observations(features: MovementFeatures, exercise_code: str, exercise_conf: float | None,
                      quality_score: float | None, reference: dict | None = None) -> list[Observation]:
    ref = reference if reference is not None else load_reference()
    obs: list[Observation] = []
    f = features

    # --- model-derived --------------------------------------------------------
    if exercise_conf is not None and exercise_conf < DEFAULTS["classifier_confidence_min"]:
        obs.append(Observation("low_exercise_confidence", "model", "review",
                               "The exercise could not be identified confidently from the video.",
                               {"confidence": round(exercise_conf, 3)}))
    if quality_score is not None and quality_score < DEFAULTS["quality_review_threshold"]:
        obs.append(Observation("low_predicted_quality", "model", "review",
                               "The predicted movement-quality score is low for this session.",
                               {"quality_score": round(quality_score, 1)}))

    # --- heuristic, reference-compared ---------------------------------------
    rom = f.range_of_motion_deg
    p25 = _ref(ref, exercise_code, "range_of_motion_deg", "p25")
    if rom.reliable and rom.value is not None and p25 is not None and rom.value < p25:
        obs.append(Observation("low_range_of_motion", "heuristic", "attention",
                               f"Range of motion of the {f.primary_angle.replace('_', ' ')} was smaller than in most reference sessions of this exercise.",
                               {"range_of_motion_deg": rom.value, "reference_p25": round(p25, 1)}))

    sym = f.symmetry_ratio
    if sym.reliable and sym.value is not None and sym.value < DEFAULTS["symmetry_ratio_min"]:
        obs.append(Observation("asymmetric_movement", "heuristic", "attention",
                               "One side moved through a noticeably smaller range than the other.",
                               {"symmetry_ratio": sym.value, "threshold": DEFAULTS["symmetry_ratio_min"]}))

    reps = f.repetitions
    if reps.reliable and reps.value is not None and reps.value < DEFAULTS["min_reps"]:
        obs.append(Observation("few_repetitions_detected", "heuristic", "info",
                               "Fewer than two repetitions were detected, so tempo and consistency could not be measured.",
                               {"repetitions": reps.value}))

    cons = f.trajectory_consistency
    if cons.reliable and cons.value is not None and cons.value < DEFAULTS["consistency_min"]:
        obs.append(Observation("inconsistent_repetitions", "heuristic", "attention",
                               "Successive repetitions followed noticeably different paths.",
                               {"trajectory_consistency": cons.value, "threshold": DEFAULTS["consistency_min"]}))

    cv = f.rep_duration_cv
    if cv.reliable and cv.value is not None and cv.value > DEFAULTS["rep_duration_cv_max"]:
        obs.append(Observation("irregular_tempo", "heuristic", "attention",
                               "Repetition timing varied a lot from one repetition to the next.",
                               {"rep_duration_cv": cv.value, "threshold": DEFAULTS["rep_duration_cv_max"]}))

    tempo = f.tempo_reps_per_s
    t_p25, t_p75 = _ref(ref, exercise_code, "tempo_reps_per_s", "p25"), _ref(ref, exercise_code, "tempo_reps_per_s", "p75")
    if tempo.reliable and tempo.value is not None and t_p25 is not None and t_p75 is not None:
        if tempo.value > t_p75 * 1.5:
            obs.append(Observation("fast_tempo", "heuristic", "info", "Repetitions were faster than in most reference sessions.",
                                   {"tempo_reps_per_s": tempo.value, "reference_p75": round(t_p75, 3)}))
        elif tempo.value < t_p25 * 0.5:
            obs.append(Observation("slow_tempo", "heuristic", "info", "Repetitions were slower than in most reference sessions.",
                                   {"tempo_reps_per_s": tempo.value, "reference_p25": round(t_p25, 3)}))

    sm = f.smoothness_sparc
    s_p25 = _ref(ref, exercise_code, "smoothness_sparc", "p25")
    if sm.reliable and sm.value is not None and s_p25 is not None and sm.value < s_p25:
        obs.append(Observation("less_smooth_movement", "heuristic", "info",
                               "The movement was less smooth than in most reference sessions.",
                               {"smoothness_sparc": sm.value, "reference_p25": round(s_p25, 3)}))

    tl = f.trunk_lean_range_deg
    if exercise_code in UPPER_LIMB and tl.reliable and tl.value is not None and tl.value > DEFAULTS["trunk_lean_upper_limb_max_deg"]:
        obs.append(Observation("trunk_movement_during_arm_exercise", "heuristic", "attention",
                               "The trunk tilted noticeably during an arm exercise.",
                               {"trunk_lean_range_deg": tl.value, "threshold": DEFAULTS["trunk_lean_upper_limb_max_deg"]}))
    return obs


def observations_to_dicts(obs: list[Observation]) -> list[dict]:
    return [asdict(o) for o in obs]


def build_reference_stats(rows: list[dict]) -> dict:
    """rows: [{'exercise': 'E01', **features.flat()}] from the TRAINING split only."""
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(rows)
    out: dict = {}
    for ex, g in df.groupby("exercise"):
        out[ex] = {}
        for col in g.columns:
            if col == "exercise":
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals) < 5:
                continue
            out[ex][col] = {"n": int(len(vals)), "p25": float(np.percentile(vals, 25)),
                            "median": float(np.median(vals)), "p75": float(np.percentile(vals, 75))}
    return out
