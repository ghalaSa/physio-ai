"""Repository functions and progress-tracking queries (proposal sections 10 and 11)."""
from __future__ import annotations

from collections import Counter
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from physio.db.models import ExerciseSession, Patient


# ---------------------------------------------------------------- patients
def get_or_create_patient(db: Session, name: str) -> Patient:
    p = db.scalar(select(Patient).where(Patient.name == name))
    if p is None:
        p = Patient(name=name)
        db.add(p)
        db.commit()
        db.refresh(p)
    return p


def list_patients(db: Session) -> list[Patient]:
    return list(db.scalars(select(Patient).order_by(Patient.name)))


def get_patient(db: Session, patient_id: int) -> Patient | None:
    return db.get(Patient, patient_id)


# ---------------------------------------------------------------- sessions
def add_session(db: Session, **fields) -> ExerciseSession:
    s = ExerciseSession(**fields)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def get_session(db: Session, session_id: int) -> ExerciseSession | None:
    return db.get(ExerciseSession, session_id)


def list_sessions(db: Session, patient_id: int | None = None, exercise_code: str | None = None,
                  limit: int | None = None, newest_first: bool = True) -> list[ExerciseSession]:
    q = select(ExerciseSession)
    if patient_id is not None:
        q = q.where(ExerciseSession.patient_id == patient_id)
    if exercise_code is not None:
        q = q.where(ExerciseSession.exercise_code == exercise_code)
    q = q.order_by(ExerciseSession.created_at.desc() if newest_first else ExerciseSession.created_at.asc())
    if limit:
        q = q.limit(limit)
    return list(db.scalars(q))


def set_clinician_note(db: Session, session_id: int, note: str, reviewed: bool = True) -> ExerciseSession | None:
    s = db.get(ExerciseSession, session_id)
    if s is None:
        return None
    s.clinician_note = note
    if reviewed:
        s.needs_review = False
    db.commit()
    db.refresh(s)
    return s


# ---------------------------------------------------------------- progress
def history_for(db: Session, patient_id: int, exercise_code: str, before: datetime | None = None,
                limit: int = 5) -> list[dict]:
    """Earlier analyzed sessions of the same exercise, newest first."""
    q = (select(ExerciseSession).where(ExerciseSession.patient_id == patient_id,
                                       ExerciseSession.exercise_code == exercise_code,
                                       ExerciseSession.status == "analyzed"))
    if before is not None:
        q = q.where(ExerciseSession.created_at < before)
    q = q.order_by(ExerciseSession.created_at.desc()).limit(limit)
    return [{"session_id": s.id, "session_date": s.created_at.isoformat(timespec="minutes"),
             "quality_score": s.quality_score,
             "observation_codes": [o["code"] for o in (s.observations or [])]} for s in db.scalars(q)]


def quality_trend(db: Session, patient_id: int) -> dict[str, list[dict]]:
    """exercise_code -> [{date, score, session_id}] oldest first."""
    out: dict[str, list[dict]] = {}
    for s in list_sessions(db, patient_id, newest_first=False):
        if s.status != "analyzed" or s.exercise_code is None:
            continue
        out.setdefault(s.exercise_code, []).append({"date": s.created_at.isoformat(timespec="minutes"),
                                                   "score": s.quality_score, "session_id": s.id,
                                                   "exercise_name": s.exercise_name})
    return out


def repeated_observations(db: Session, patient_id: int, exercise_code: str | None = None,
                          last_n: int = 10, min_count: int = 2) -> list[dict]:
    sessions = list_sessions(db, patient_id, exercise_code, limit=last_n)
    counter: Counter = Counter()
    texts: dict[str, str] = {}
    for s in sessions:
        for o in s.observations or []:
            if o.get("kind") in ("heuristic", "model"):
                counter[(s.exercise_code, o["code"])] += 1
                texts[o["code"]] = o["text"]
    return [{"exercise_code": ex, "code": code, "count": n, "of_sessions": len(sessions), "text": texts[code]}
            for (ex, code), n in counter.most_common() if n >= min_count]


def sessions_requiring_review(db: Session, limit: int = 50) -> list[ExerciseSession]:
    q = (select(ExerciseSession).where(ExerciseSession.needs_review.is_(True))
         .order_by(ExerciseSession.created_at.desc()).limit(limit))
    return list(db.scalars(q))


def patient_overview(db: Session) -> list[dict]:
    rows = []
    for p in list_patients(db):
        ss = list_sessions(db, p.id)
        scored = [s.quality_score for s in ss[:5] if s.quality_score is not None]
        rows.append({"patient_id": p.id, "name": p.name, "n_sessions": len(ss),
                     "last_session": ss[0].created_at.isoformat(timespec="minutes") if ss else None,
                     "mean_recent_score": round(sum(scored) / len(scored), 1) if scored else None,
                     "n_needing_review": sum(1 for s in ss if s.needs_review),
                     "exercises": sorted({s.exercise_code for s in ss if s.exercise_code})})
    return rows


def session_to_dict(s: ExerciseSession, include_series: bool = False) -> dict:
    d = {c.name: getattr(s, c.name) for c in s.__table__.columns}
    d["created_at"] = s.created_at.isoformat(timespec="seconds")
    if not include_series and d.get("measurements"):
        d["measurements"] = {k: v for k, v in d["measurements"].items() if k != "angle_series"}
    return d
