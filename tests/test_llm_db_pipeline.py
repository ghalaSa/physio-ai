import json

import numpy as np
import pytest

from physio import config
from physio.db import repo
from physio.llm.interface import FeedbackRequest, HistoryItem, SessionSummary, get_provider, violates_constraints
from physio.llm.template_provider import TemplateProvider
from physio.pose.synthetic import synthetic_sequence


def _summary(status="analyzed", score=72.0, conf=0.9):
    return SessionSummary(status=status, exercise_code="E01", exercise_name="Abduction", exercise_confidence=conf,
                          quality_score=score,
                          measurements={"range_of_motion_deg": {"value": 78.2, "unit": "deg", "reliable": True, "note": ""},
                                        "repetitions": {"value": 4, "unit": "count", "reliable": True, "note": ""},
                                        "symmetry_ratio": {"value": 0.3, "unit": "ratio", "reliable": False, "note": ""}},
                          observations=[{"code": "low_range_of_motion", "kind": "heuristic", "severity": "attention",
                                         "text": "Range of motion was smaller than reference.", "evidence": {}}])


def test_template_provider_reports_only_reliable_measurements():
    fb = TemplateProvider().generate(FeedbackRequest(session=_summary()))
    assert "Abduction" in fb.text and "78.2" in fb.text and "4 repetitions" in fb.text
    assert "symmetry" not in fb.text.lower()            # unreliable measurement is not mentioned
    assert "not a medical assessment" in fb.text
    assert not violates_constraints(fb.text)


def test_template_provider_history_and_repeats():
    hist = [HistoryItem("2026-09-01T10:00", 60.0, ["low_range_of_motion"]),
            HistoryItem("2026-08-25T10:00", 58.0, ["low_range_of_motion"])]
    fb = TemplateProvider().generate(FeedbackRequest(session=_summary(score=72.0), history=hist))
    assert "higher than" in fb.text
    assert "several sessions" in fb.text and "low range of motion" in fb.text


def test_template_provider_unable_state():
    fb = TemplateProvider().generate(FeedbackRequest(session=_summary(status="unable_to_analyze")))
    assert "could not analyze" in fb.text.lower()
    assert "score" not in fb.text.lower()


def test_constraint_checker_and_provider_selection(monkeypatch):
    assert violates_constraints("This suggests a rotator cuff tear.") == ["tear"]
    monkeypatch.setenv("PHYSIO_LLM_PROVIDER", "template")
    assert get_provider().name == "template"


def test_db_roundtrip_and_progress(tmp_db):
    db = tmp_db()
    p = repo.get_or_create_patient(db, "Test Patient")
    assert repo.get_or_create_patient(db, "Test Patient").id == p.id
    obs = [{"code": "asymmetric_movement", "kind": "heuristic", "severity": "attention", "text": "x", "evidence": {}}]
    for i, score in enumerate([55.0, 62.0, 70.0]):
        repo.add_session(db, patient_id=p.id, video_name=f"v{i}.mp4", status="analyzed", exercise_code="E01",
                         exercise_name="Abduction", exercise_confidence=0.9, quality_score=score,
                         measurements={}, observations=obs, needs_review=(i == 0), review_reasons=["low"] if i == 0 else [])
    repo.add_session(db, patient_id=p.id, video_name="bad.mp4", status="unable_to_analyze", unable_reason="dark",
                     needs_review=False)
    trend = repo.quality_trend(db, p.id)
    assert [x["score"] for x in trend["E01"]] == [55.0, 62.0, 70.0]
    rep = repo.repeated_observations(db, p.id)
    assert rep and rep[0]["code"] == "asymmetric_movement" and rep[0]["count"] == 3
    assert len(repo.sessions_requiring_review(db)) == 1
    hist = repo.history_for(db, p.id, "E01")
    assert hist[0]["quality_score"] == 70.0 and len(hist) == 3
    ov = repo.patient_overview(db)
    assert ov[0]["n_sessions"] == 4 and ov[0]["n_needing_review"] == 1
    sid = repo.sessions_requiring_review(db)[0].id
    repo.set_clinician_note(db, sid, "seen")
    assert repo.sessions_requiring_review(db) == []
    d = repo.session_to_dict(repo.get_session(db, sid))
    assert d["clinician_note"] == "seen"


@pytest.fixture()
def fake_models(tmp_path, monkeypatch):
    """A tiny sklearn classifier + regressor trained on synthetic clips so the pipeline runs without torch."""
    import joblib
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from physio.pose.normalize import normalize_sequence, summary_statistics
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    X, y = [], []
    for side, label in (("right", 0), ("left", 1)):
        for seed in range(6):
            ps = synthetic_sequence(n_frames=80, side=side, seed=seed, amplitude_deg=60 + 5 * seed)
            X.append(summary_statistics(normalize_sequence(ps.landmarks, ps.detected, 720, 1280)["features"]))
            y.append(label)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(np.stack(X), y)
    joblib.dump({"model": clf, "classes": ["E01", "E02"], "task": "cls"}, tmp_path / "clf.joblib")
    reg = DummyRegressor(strategy="constant", constant=77.0).fit(np.stack(X), np.full(len(X), 77.0))
    joblib.dump({"model": reg, "classes": ["E01", "E02"], "task": "reg"}, tmp_path / "reg.joblib")
    best = tmp_path / "best_models.json"
    best.write_text(json.dumps({"classification": {"run_name": "clf", "artifact": "clf.joblib"},
                                "regression": {"run_name": "reg", "artifact": "reg.joblib"}}))
    return best


def test_pipeline_end_to_end_synthetic(fake_models, tmp_db):
    from physio.pipeline import Analyzer, store_result
    an = Analyzer(best_path=fake_models, provider=TemplateProvider(), reference_path=fake_models.parent / "none.json")
    res = an.analyze_pose_sequence(synthetic_sequence(n_frames=120, side="right", seed=99))
    assert res.status == "analyzed"
    assert res.exercise_code == "E01" and res.exercise_confidence > 0.5
    assert res.quality_score == 77.0
    assert res.measurements["repetitions"]["value"] == 4
    assert res.feedback and "Abduction" in res.feedback
    assert res.feedback_provider == "template"
    assert res.model_versions == {"classification": "clf", "regression": "reg"}

    bad = an.analyze_pose_sequence(synthetic_sequence(n_frames=8))
    assert bad.status == "unable_to_analyze" and bad.exercise_code is None and bad.feedback

    db = tmp_db()
    p = repo.get_or_create_patient(db, "P")
    s = store_result(db, p.id, "synthetic.mp4", res)
    assert s.id and s.exercise_code == "E01"
    assert repo.session_to_dict(s)["measurements"].get("angle_series") is None
    assert "angle_series" in repo.session_to_dict(s, include_series=True)["measurements"]


def test_api_with_fake_models(fake_models, tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PHYSIO_LLM_PROVIDER", "template")
    monkeypatch.setattr(config, "DEFAULT_DB_URL", f"sqlite:///{tmp_path / 'api.db'}")
    from physio.models import compare
    monkeypatch.setattr(compare, "BEST_PATH", fake_models)
    import physio.pipeline as pl
    monkeypatch.setattr(pl, "BEST_PATH", fake_models)
    monkeypatch.setattr(config, "POSES_DIR", tmp_path)
    synthetic_sequence(n_frames=100, side="left", seed=3, video_id="E02_P99_AF_VFL_GM").save(tmp_path / "E02_P99_AF_VFL_GM.npz")
    from physio.api.main import app
    with TestClient(app) as c:
        assert c.get("/health").json()["models_loaded"] is True
        r = c.post("/sessions/analyze-cached", data={"video_id": "E02_P99_AF_VFL_GM", "patient_name": "Amal"})
        assert r.status_code == 200, r.text
        s = r.json()
        assert s["status"] == "analyzed" and s["exercise_code"] == "E02"
        r2 = c.post("/sessions/analyze-cached", data={"video_id": "E02_P99_AF_VFL_GM", "patient_name": "Amal"})
        assert "previous session" in r2.json()["feedback"]
        pid = s["patient_id"]
        assert len(c.get(f"/patients/{pid}/sessions").json()) == 2
        assert "E02" in c.get(f"/patients/{pid}/progress").json()["quality_trend"]
        assert c.get("/patients").json()[0]["n_sessions"] == 2
        assert c.get(f"/sessions/{s['id']}").status_code == 200
        assert c.post(f"/sessions/{s['id']}/note", json={"note": "ok"}).json()["clinician_note"] == "ok"
        assert c.post("/sessions/analyze-cached", data={"video_id": "nope", "patient_name": "x"}).status_code == 404
