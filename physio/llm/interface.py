"""Provider-independent feedback interface (proposal sections 9 and 11).

The LLM only ever sees the structured, validated output of the CV pipeline:
predicted exercise + confidence, predicted quality score, deterministic measurements
with their reliability flags, heuristic observations, and prior-session summaries.
It never sees video. Providers must implement `FeedbackProvider`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

# Words that indicate the model drifted into diagnosis or medical advice. Output
# containing any of them is rejected and replaced by the template provider.
FORBIDDEN_TERMS = (
    "diagnos", "tendinitis", "tendonitis", "bursitis", "tear", "rupture", "fracture", "arthritis",
    "impingement", "frozen shoulder", "sciatica", "herniat", "you have ", "you are suffering",
    "medication", "painkiller", "ibuprofen", "prescri",
)


@dataclass
class SessionSummary:
    """Structured result of one analyzed session, as handed to the LLM."""
    status: str                                  # "analyzed" | "unable_to_analyze"
    exercise_code: str | None
    exercise_name: str | None
    exercise_confidence: float | None
    quality_score: float | None
    measurements: dict = field(default_factory=dict)   # name -> {value, unit, reliable, note}
    observations: list[dict] = field(default_factory=list)
    unable_reason: str | None = None
    session_date: str | None = None


@dataclass
class HistoryItem:
    session_date: str
    quality_score: float | None
    observation_codes: list[str]


@dataclass
class FeedbackRequest:
    session: SessionSummary
    history: list[HistoryItem] = field(default_factory=list)
    audience: str = "patient"                    # "patient" | "physiotherapist"


@dataclass
class FeedbackResult:
    text: str
    provider: str
    model: str | None = None
    warnings: list[str] = field(default_factory=list)


class FeedbackProvider(Protocol):
    name: str

    def generate(self, request: FeedbackRequest) -> FeedbackResult: ...


def violates_constraints(text: str) -> list[str]:
    low = text.lower()
    return [t for t in FORBIDDEN_TERMS if t in low]


def get_provider(kind: str | None = None) -> FeedbackProvider:
    """Choose the provider: PHYSIO_LLM_PROVIDER = anthropic | template | auto (default)."""
    from physio.llm.template_provider import TemplateProvider
    kind = (kind or os.environ.get("PHYSIO_LLM_PROVIDER", "auto")).lower()
    if kind == "template":
        return TemplateProvider()
    if kind in ("anthropic", "auto"):
        try:
            from physio.llm.anthropic_provider import AnthropicProvider
            if kind == "anthropic" or AnthropicProvider.credentials_available():
                return AnthropicProvider()
        except ImportError:
            if kind == "anthropic":
                raise
    return TemplateProvider()
