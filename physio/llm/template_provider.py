"""Deterministic, provider-free feedback. Used as the default when no LLM credentials
are configured and as the fallback whenever an LLM response is unavailable or violates
the non-diagnostic constraints."""
from __future__ import annotations

from physio.llm.interface import FeedbackRequest, FeedbackResult

CLOSING = ("This summary describes what was measured in the video. It is not a medical assessment. "
           "Please discuss any concerns with your physiotherapist.")


def _fmt(m: dict | None, digits: int = 1) -> str | None:
    if not m or m.get("value") is None or not m.get("reliable", False):
        return None
    v = m["value"]
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


class TemplateProvider:
    name = "template"

    def generate(self, request: FeedbackRequest) -> FeedbackResult:
        s = request.session
        lines: list[str] = []
        if s.status != "analyzed":
            lines.append("We could not analyze this video reliably."
                         + (f" Reason: {s.unable_reason}." if s.unable_reason else ""))
            lines.append("Try recording again with your whole body visible, good lighting, and the camera held still.")
            lines.append(CLOSING)
            return FeedbackResult(text=" ".join(lines), provider=self.name)

        conf = s.exercise_confidence or 0.0
        if conf >= 0.6:
            lines.append(f"Detected exercise: {s.exercise_name} (confidence {conf:.0%}).")
        else:
            lines.append(f"The exercise looked most like {s.exercise_name}, but the system was not confident ({conf:.0%}). "
                         "Please check that the correct exercise was recorded.")
        if s.quality_score is not None:
            lines.append(f"Predicted movement-quality score: {s.quality_score:.0f} out of 100 "
                         "(an estimate of how a physiotherapist might rate the execution, not a clinical judgement).")

        m = s.measurements
        parts = []
        if (v := _fmt(m.get("range_of_motion_deg"))):
            parts.append(f"range of motion about {v} degrees")
        if (v := _fmt(m.get("repetitions"), 0)):
            parts.append(f"{v} repetitions detected")
        if (v := _fmt(m.get("rep_duration_s"))):
            parts.append(f"about {v} seconds per repetition")
        if (v := _fmt(m.get("symmetry_ratio"), 2)):
            parts.append(f"left/right symmetry ratio {v}")
        if parts:
            lines.append("Measurements: " + ", ".join(parts) + ".")

        obs = [o for o in s.observations if o.get("kind") in ("heuristic", "model")]
        if obs:
            lines.append("Observations: " + " ".join(o["text"] for o in obs))
        else:
            lines.append("No specific observations were flagged in this session.")

        if request.history:
            prev = [h for h in request.history if h.quality_score is not None]
            if prev and s.quality_score is not None:
                last = prev[0].quality_score
                delta = s.quality_score - last
                trend = "higher than" if delta > 3 else "lower than" if delta < -3 else "similar to"
                lines.append(f"Compared with your previous session ({last:.0f}), today's score is {trend} last time.")
            codes = [o["code"] for o in obs]
            repeated = [c for c in codes if sum(c in h.observation_codes for h in request.history) >= 2]
            if repeated:
                lines.append("The following observations have now appeared in several sessions: "
                             + ", ".join(c.replace("_", " ") for c in sorted(set(repeated)))
                             + ". It may be worth mentioning them to your physiotherapist.")
        if any(o.get("severity") == "review" for o in obs):
            lines.append("This session has been flagged for your physiotherapist to review.")
        lines.append(CLOSING)
        return FeedbackResult(text=" ".join(lines), provider=self.name)
