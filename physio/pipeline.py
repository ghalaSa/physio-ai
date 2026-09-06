"""End-to-end analysis pipeline (proposal section 4):

video -> pose -> quality gate -> exercise classification -> movement-quality regression
      -> deterministic measurements -> heuristic observations -> LLM feedback -> storage.

`Analyzer` loads the selected models once (outputs/models/best_models.json) and is
shared by the API. `analyze_pose_sequence` is the video-free entry point used by
tests and by the batch evaluation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from physio import config
from physio.analysis.features import compute_features
from physio.analysis.observations import REFERENCE_PATH, load_reference, make_observations, observations_to_dicts
from physio.llm.interface import FeedbackProvider, FeedbackRequest, HistoryItem, SessionSummary, get_provider
from physio.models.compare import BEST_PATH
from physio.pose.extractor import PoseSequence, extract_pose_sequence
from physio.pose.normalize import normalize_sequence, summary_statistics


@dataclass
class AnalysisResult:
    status: str
    unable_reason: str | None = None
    exercise_code: str | None = None
    exercise_name: str | None = None
    exercise_confidence: float | None = None
    class_probabilities: dict[str, float] = field(default_factory=dict)
    quality_score: float | None = None
    measurements: dict = field(default_factory=dict)
    observations: list[dict] = field(default_factory=list)
    pose_quality: dict = field(default_factory=dict)
    feedback: str | None = None
    feedback_provider: str | None = None
    feedback_warnings: list[str] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    model_versions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class _TorchModel:
    def __init__(self, path: Path):
        from physio.models.train import load_temporal
        self.net, self.ck = load_temporal(path)
        self.classes = self.ck["classes"]

    def predict(self, features: np.ndarray) -> dict:
        import torch
        x = (features - self.ck["feat_mu"]) / self.ck["feat_sd"]
        with torch.no_grad():
            out = self.net(torch.tensor(x[None], dtype=torch.float32))
        res = {}
        if "logits" in out:
            res["probs"] = torch.softmax(out["logits"], 1)[0].numpy()
        if "score" in out:
            res["score"] = float(np.clip(out["score"].item() * self.ck["reg_sd"] + self.ck["reg_mu"], 0, 100))
        return res


class _SklearnModel:
    def __init__(self, path: Path):
        import joblib
        bundle = joblib.load(path)
        self.model, self.classes, self.task = bundle["model"], bundle["classes"], bundle["task"]

    def predict(self, features: np.ndarray) -> dict:
        x = summary_statistics(features)[None]
        if self.task == "cls":
            return {"probs": self.model.predict_proba(x)[0]}
        return {"score": float(np.clip(self.model.predict(x)[0], 0, 100))}


def _load_model(entry: dict):
    art = config.MODELS_DIR / entry["artifact"]
    return _TorchModel(art) if art.suffix == ".pt" else _SklearnModel(art)


class Analyzer:
    def __init__(self, best_path: Path = BEST_PATH, provider: FeedbackProvider | None = None,
                 reference_path: Path = REFERENCE_PATH):
        if not Path(best_path).exists():
            raise FileNotFoundError(f"{best_path} not found: train models and run scripts/compare_models.py first")
        best = json.loads(Path(best_path).read_text())
        self.best = best
        self.classifier = _load_model(best["classification"]) if "classification" in best else None
        self.regressor = _load_model(best["regression"]) if "regression" in best else None
        if self.classifier is None:
            raise ValueError("no classification model selected in best_models.json")
        self.classes = list(self.classifier.classes)
        self.codebook = config.load_codebook()
        self.reference = load_reference(reference_path)
        self.provider = provider or get_provider()
        self.model_versions = {k: v.get("run_name") for k, v in best.items()}

    # ---------------------------------------------------------------- core
    def analyze_pose_sequence(self, ps: PoseSequence, history: list[dict] | None = None,
                              audience: str = "patient", generate_feedback: bool = True) -> AnalysisResult:
        ok, reason = ps.quality_gate()
        pq = {"n_frames": ps.n_frames, "n_detected": ps.n_detected, "detection_rate": round(ps.detection_rate, 3),
              "mean_visibility": round(ps.mean_visibility, 3), "gate_ok": ok, "gate_reason": reason}
        if not ok:
            res = AnalysisResult(status="unable_to_analyze", unable_reason=reason, pose_quality=pq,
                                 needs_review=False, model_versions=self.model_versions)
            if generate_feedback:
                self._feedback(res, history or [], audience)
            return res

        norm = normalize_sequence(ps.landmarks, ps.detected, ps.meta["width"], ps.meta["height"])
        feats = norm["features"]
        c = self.classifier.predict(feats)
        probs = c["probs"]
        k = int(np.argmax(probs))
        code = self.classes[k]
        conf = float(probs[k])
        score = None
        if self.regressor is not None:
            r = self.regressor.predict(feats)
            score = r.get("score")
        elif "score" in c:
            score = c["score"]

        fps = float(ps.meta.get("sampled_fps") or config.TARGET_FPS)
        mf = compute_features(norm["xy_full"], fps, code)
        obs = make_observations(mf, code, conf, score, self.reference)
        obs_d = observations_to_dicts(obs)
        review_reasons = [o["text"] for o in obs_d if o["severity"] == "review"]

        res = AnalysisResult(status="analyzed", exercise_code=code, exercise_name=self.codebook.name(code),
                             exercise_confidence=round(conf, 4),
                             class_probabilities={self.classes[i]: round(float(p), 4) for i, p in enumerate(probs)},
                             quality_score=None if score is None else round(float(score), 1),
                             measurements=mf.to_dict(include_series=True), observations=obs_d, pose_quality=pq,
                             needs_review=bool(review_reasons), review_reasons=review_reasons,
                             model_versions=self.model_versions)
        if generate_feedback:
            self._feedback(res, history or [], audience)
        return res

    def analyze_video(self, video_path: Path, history: list[dict] | None = None, audience: str = "patient") -> AnalysisResult:
        try:
            ps = extract_pose_sequence(Path(video_path))
        except Exception as exc:  # noqa: BLE001 - surfaced as an explicit unable state
            res = AnalysisResult(status="unable_to_analyze", unable_reason=f"video could not be decoded ({type(exc).__name__})",
                                 model_versions=self.model_versions)
            self._feedback(res, history or [], audience)
            return res
        return self.analyze_pose_sequence(ps, history, audience)

    # ---------------------------------------------------------------- feedback
    def _feedback(self, res: AnalysisResult, history: list[dict], audience: str) -> None:
        meas = {k: v for k, v in res.measurements.items()
                if isinstance(v, dict) and {"value", "unit", "reliable"} <= set(v)}
        summary = SessionSummary(status=res.status, exercise_code=res.exercise_code, exercise_name=res.exercise_name,
                                 exercise_confidence=res.exercise_confidence, quality_score=res.quality_score,
                                 measurements=meas, observations=res.observations, unable_reason=res.unable_reason,
                                 session_date=datetime.now(timezone.utc).isoformat(timespec="minutes"))
        req = FeedbackRequest(session=summary, audience=audience,
                              history=[HistoryItem(h["session_date"], h.get("quality_score"), h.get("observation_codes", []))
                                       for h in history])
        fb = self.provider.generate(req)
        res.feedback, res.feedback_provider, res.feedback_warnings = fb.text, fb.provider, fb.warnings


def store_result(db, patient_id: int, video_name: str, res: AnalysisResult):
    from physio.db.repo import add_session
    return add_session(db, patient_id=patient_id, video_name=video_name, status=res.status,
                       unable_reason=res.unable_reason, exercise_code=res.exercise_code, exercise_name=res.exercise_name,
                       exercise_confidence=res.exercise_confidence, class_probabilities=res.class_probabilities,
                       quality_score=res.quality_score, measurements=res.measurements, observations=res.observations,
                       pose_quality=res.pose_quality, feedback=res.feedback, feedback_provider=res.feedback_provider,
                       needs_review=res.needs_review, review_reasons=res.review_reasons, model_versions=res.model_versions)
