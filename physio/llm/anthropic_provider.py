"""Anthropic implementation of the feedback provider.

The model receives only the structured JSON summary. The system prompt fixes its
role to explaining measured results; any output that drifts into diagnosis is
discarded and replaced by the deterministic template (see interface.FORBIDDEN_TERMS).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from physio.llm.interface import FeedbackRequest, FeedbackResult, violates_constraints
from physio.llm.template_provider import TemplateProvider

MODEL = os.environ.get("PHYSIO_LLM_MODEL", "claude-opus-5")

SYSTEM_PROMPT = """You write short feedback for a home physiotherapy monitoring tool.

You receive a JSON object describing ONE recorded exercise session. It contains:
- a predicted exercise and a confidence (from a machine-learning classifier),
- a predicted movement-quality score from 0 to 100 (a model estimate of how physiotherapists rated similar executions),
- deterministic measurements from 2D pose landmarks, each with a "reliable" flag,
- heuristic observations already produced by rules, each with a "kind" (model / heuristic) and a "severity",
- optionally a short history of earlier sessions of the same exercise.

Rules you must follow:
1. Only restate what is in the JSON. Never invent measurements, causes, or trends.
2. Never diagnose, name conditions or injuries, suggest treatment, medication, or exercise changes.
   You may only suggest recording tips (visibility, lighting, camera steadiness) and talking to the physiotherapist.
3. Ignore measurements whose "reliable" flag is false.
4. Keep model predictions, measurements, and heuristic observations distinguishable
   ("the system predicted", "the measured range was", "a rule flagged").
5. If history is present, compare the quality score with the most recent earlier session and mention
   observations that repeat across sessions.
6. If the session status is not "analyzed", explain that it could not be analyzed and give recording tips only.
7. Audience "patient": plain language, 90-160 words, warm but factual, second person.
   Audience "physiotherapist": concise clinical-note style, 60-120 words, third person, numbers included.
8. End with one sentence stating that this is an automated summary of measurements, not a medical assessment.
Output plain text only, no headings, no lists, no markdown."""


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = MODEL):
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self._fallback = TemplateProvider()

    @staticmethod
    def credentials_available() -> bool:
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return True
        # `ant auth login` profiles are picked up by the SDK automatically
        return (os.path.expanduser("~/.config/anthropic") and os.path.isdir(os.path.expanduser("~/.config/anthropic")))

    def _payload(self, request: FeedbackRequest) -> str:
        return json.dumps({"audience": request.audience, "session": asdict(request.session),
                           "history": [asdict(h) for h in request.history]}, indent=1, default=str)

    def generate(self, request: FeedbackRequest) -> FeedbackResult:
        a = self._anthropic
        warnings: list[str] = []
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                thinking={"type": "adaptive"},
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": self._payload(request)}],
            )
            if response.stop_reason == "refusal":
                warnings.append("model refused the request")
                text = ""
            else:
                text = "".join(b.text for b in response.content if b.type == "text").strip()
        except a.RateLimitError:
            warnings.append("rate limited")
            text = ""
        except a.APIStatusError as e:
            warnings.append(f"api error {e.status_code}")
            text = ""
        except a.APIConnectionError:
            warnings.append("connection error")
            text = ""

        bad = violates_constraints(text) if text else []
        if bad:
            warnings.append(f"constraint violation: {bad}")
        if not text or bad:
            fb = self._fallback.generate(request)
            fb.warnings = warnings + ["fell back to template provider"]
            return fb
        return FeedbackResult(text=text, provider=self.name, model=self.model, warnings=warnings)
