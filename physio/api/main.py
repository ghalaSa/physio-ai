"""FastAPI backend (proposal section 11).

Run:  uvicorn physio.api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from physio import config
from physio.db import repo
from physio.db.models import make_session_factory
from physio.pipeline import Analyzer, store_result
from physio.pose.extractor import PoseSequence, pose_cache_path

STATE: dict = {"analyzer": None, "analyzer_error": None, "SessionLocal": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    STATE["SessionLocal"] = make_session_factory(config.DEFAULT_DB_URL)   # resolved at startup, so tests can redirect it
    try:
        STATE["analyzer"] = Analyzer()
    except Exception as exc:  # noqa: BLE001 - API stays up; analysis endpoints report 503
        STATE["analyzer_error"] = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="AI Home Physiotherapy Assistant", version="1.0.0", lifespan=lifespan)


def get_db():
    db = STATE["SessionLocal"]()
    try:
        yield db
    finally:
        db.close()


def get_analyzer() -> Analyzer:
    if STATE["analyzer"] is None:
        raise HTTPException(503, f"models not available: {STATE['analyzer_error']}")
    return STATE["analyzer"]


class PatientIn(BaseModel):
    name: str


class NoteIn(BaseModel):
    note: str
    mark_reviewed: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": STATE["analyzer"] is not None, "analyzer_error": STATE["analyzer_error"]}


@app.get("/models")
def models(an: Analyzer = Depends(get_analyzer)):
    return {"selected": an.best, "classes": an.classes,
            "exercise_names": {c: an.codebook.name(c) for c in an.classes}, "feedback_provider": an.provider.name}


@app.post("/patients")
def create_patient(body: PatientIn, db=Depends(get_db)):
    p = repo.get_or_create_patient(db, body.name.strip())
    return {"patient_id": p.id, "name": p.name}


@app.get("/patients")
def patients(db=Depends(get_db)):
    return repo.patient_overview(db)


@app.get("/patients/{patient_id}/sessions")
def patient_sessions(patient_id: int, exercise_code: str | None = None, db=Depends(get_db)):
    if repo.get_patient(db, patient_id) is None:
        raise HTTPException(404, "patient not found")
    return [repo.session_to_dict(s) for s in repo.list_sessions(db, patient_id, exercise_code)]


@app.get("/patients/{patient_id}/progress")
def patient_progress(patient_id: int, db=Depends(get_db)):
    if repo.get_patient(db, patient_id) is None:
        raise HTTPException(404, "patient not found")
    return {"quality_trend": repo.quality_trend(db, patient_id),
            "repeated_observations": repo.repeated_observations(db, patient_id)}


@app.get("/sessions/{session_id}")
def session(session_id: int, include_series: bool = True, db=Depends(get_db)):
    s = repo.get_session(db, session_id)
    if s is None:
        raise HTTPException(404, "session not found")
    return repo.session_to_dict(s, include_series=include_series)


@app.post("/sessions/{session_id}/note")
def add_note(session_id: int, body: NoteIn, db=Depends(get_db)):
    s = repo.set_clinician_note(db, session_id, body.note, body.mark_reviewed)
    if s is None:
        raise HTTPException(404, "session not found")
    return repo.session_to_dict(s)


@app.get("/review")
def review_queue(db=Depends(get_db)):
    return [repo.session_to_dict(s) for s in repo.sessions_requiring_review(db)]


def _run_and_store(db, an: Analyzer, patient_name: str, video_name: str, audience: str, run):
    p = repo.get_or_create_patient(db, patient_name.strip())
    res = run()
    # history is looked up after classification so it matches the detected exercise
    if res.status == "analyzed":
        hist = repo.history_for(db, p.id, res.exercise_code)
        if hist:
            an._feedback(res, hist, audience)
    s = store_result(db, p.id, video_name, res)
    return repo.session_to_dict(s)


@app.post("/sessions/analyze")
async def analyze(file: UploadFile = File(...), patient_name: str = Form(...), audience: str = Form("patient"),
                  db=Depends(get_db), an: Analyzer = Depends(get_analyzer)):
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp = config.TMP_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        return _run_and_store(db, an, patient_name, file.filename or tmp.name, audience,
                              lambda: an.analyze_video(tmp, audience=audience))
    finally:
        tmp.unlink(missing_ok=True)


@app.post("/sessions/analyze-cached")
def analyze_cached(video_id: str = Form(...), patient_name: str = Form(...), audience: str = Form("patient"),
                   db=Depends(get_db), an: Analyzer = Depends(get_analyzer)):
    """Analyze a video whose pose sequence is already cached (demo / evaluation without re-downloading)."""
    path = pose_cache_path(video_id)
    if not path.exists():
        raise HTTPException(404, f"no cached pose sequence for {video_id}")
    ps = PoseSequence.load(path)
    return _run_and_store(db, an, patient_name, f"{video_id}.mp4", audience,
                          lambda: an.analyze_pose_sequence(ps, audience=audience))
