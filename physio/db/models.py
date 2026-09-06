"""SQLAlchemy schema. SQLite by default (PHYSIO_DB_URL switches to PostgreSQL)."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from physio import config


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    sessions: Mapped[list["ExerciseSession"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class ExerciseSession(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    video_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32))                 # analyzed | unable_to_analyze
    unable_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    exercise_code: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    exercise_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    exercise_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    class_probabilities: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    measurements: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    observations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    pose_quality: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    feedback_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    clinician_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    review_reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)
    model_versions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    patient: Mapped[Patient] = relationship(back_populates="sessions")


def make_engine(url: str = config.DEFAULT_DB_URL):
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, **kw)


def make_session_factory(url: str = config.DEFAULT_DB_URL):
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
