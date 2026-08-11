"""Agent 2 — Critic: Static analysis of NetworkPolicy YAML."""

import json
from pathlib import Path
from pydantic import ValidationError
from policy_agent import ollama_client
import policy_agent.config as _cfg
from policy_agent.models import CriticError, CriticReport, PolicyIntent

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "critic.txt").read_text()


def _parse_report(raw: str) -> CriticReport:
    """Parse LLM output into a CriticReport; gracefully handle bad JSON.

    Critic is advisory-only (see pipeline.py) — it never gates pass/fail —
    so a weak model hallucinating a wrong field name (e.g. "kindinsecurity"
    instead of "kind") should degrade that one error entry, not crash the
    whole row with an unhandled pydantic ValidationError.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()

    try:
        data = json.loads(text)
        errors = []
        for e in data.get("errors", []):
            try:
                errors.append(CriticError(**e))
            except (ValidationError, TypeError):
                errors.append(CriticError(kind="unknown", message=str(e)[:200]))
        return CriticReport(
            has_errors=bool(data.get("has_errors", len(errors) > 0)),
            errors=errors,
            confidence=float(data.get("confidence", 0.8)),
            raw_response=raw,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
        # If we can't parse JSON, treat it as a syntax-level parse failure
        return CriticReport(
            has_errors=True,
            errors=[CriticError(kind="syntax", message=f"Critic JSON parse error: {raw[:200]}")],
            confidence=0.5,
            raw_response=raw,
        )


def analyze(intent: PolicyIntent, yaml_policy: str) -> CriticReport:
    """Statically analyze YAML against the intent and return a CriticReport."""
    user_prompt = (
        f"Intent: {intent.description}\n\n"
        f"YAML to analyze:\n{yaml_policy}\n\n"
        "Return JSON analysis:"
    )
    raw = ollama_client.generate(_cfg.CRITIC_MODEL, user_prompt, system=_SYSTEM_PROMPT)
    return _parse_report(raw)
