"""Agent 4 — Refiner: Fix YAML using semantic diff + critic/verifier errors."""

import re
from pathlib import Path
from policy_agent import ollama_client
import policy_agent.config as _cfg
from policy_agent.models import CriticReport, PolicyIntent, SemanticReport, VerifierReport

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "refiner.txt").read_text()

# Actionable lines worth pulling out of a verbose Kubernetes API error dump —
# a small Refiner model reacts far more reliably to one concise line than to
# a full traceback-shaped blob (this was root cause #2 of the correction
# loop not converging: the same metadata.name violation survived 5 iterations
# despite the raw error being present, just buried).
_CONCISE_ERROR_PATTERNS = [
    re.compile(r'Invalid value: "[^"]*"[^\n]*'),
    re.compile(r'unknown field "[^"]+"'),
    re.compile(r'cannot unmarshal \w+ into Go struct field [^\n]+'),
]


def _extract_concise_error(raw_error: str) -> str:
    matches: list[str] = []
    for pat in _CONCISE_ERROR_PATTERNS:
        matches.extend(pat.findall(raw_error))
    if matches:
        return "; ".join(matches)
    lines = [l.strip() for l in raw_error.strip().splitlines() if l.strip()]
    return lines[-1] if lines else raw_error


def refine(
    intent: PolicyIntent,
    yaml_policy: str,
    critic_report: CriticReport,
    verifier_report: VerifierReport,
    semantic_report: SemanticReport | None = None,
    blind: bool = False,
    critic_only: bool = False,
) -> str:
    """Return a corrected NetworkPolicy YAML.

    `blind=True` is the Checkpoint 5 no-refiner-feedback ablation: the
    Refiner is told only that the previous attempt failed, with none of
    the specific semantic/verifier/critic detail — tests whether the
    loop's benefit comes from the SPECIFIC feedback, or just from getting
    more inference-time attempts at the same problem.

    `critic_only=True` is the self_refine baseline (Madaan et al.): the
    Refiner sees ONLY the Critic's own self-review, never the deterministic
    Verifier/Semantic Verifier detail — matching genuine self-refine, where
    the model corrects itself using nothing but its own critique.
    """
    if blind:
        user_prompt = (
            f"Intent: {intent.description}\n\n"
            f"Current YAML:\n{yaml_policy}\n\n"
            "The previous attempt did not pass verification (no further detail "
            "provided). Generate an improved YAML that better satisfies the intent:"
        )
        raw = ollama_client.generate(_cfg.REFINER_MODEL, user_prompt, system=_SYSTEM_PROMPT)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return text

    error_lines = []

    if critic_only:
        if critic_report.has_errors:
            for e in critic_report.errors:
                error_lines.append(f"[{e.kind.upper()}] {e.message}")
        error_summary = "\n".join(error_lines) if error_lines else "No specific errors — improve clarity and security."
        user_prompt = (
            f"Intent: {intent.description}\n\n"
            f"Current YAML:\n{yaml_policy}\n\n"
            f"Errors to fix:\n{error_summary}\n\n"
            "Return corrected YAML only:"
        )
        raw = ollama_client.generate(_cfg.REFINER_MODEL, user_prompt, system=_SYSTEM_PROMPT)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return text

    # Primary signal: the deterministic semantic verifier's precise diff
    # against what was actually asked for — ground truth, not an LLM's
    # opinion. Listed first so the Refiner treats it as the thing to fix.
    if semantic_report is not None and not semantic_report.matches_intent:
        error_lines.append("=== SEMANTIC MISMATCH (fix this first) ===")
        error_lines.append(semantic_report.reason)

    if not verifier_report.dry_run_passed:
        error_lines.append("\n=== API ERROR (fix exactly this) ===")
        error_lines.append(_extract_concise_error(verifier_report.dry_run_error))

    if verifier_report.failed_tests:
        error_lines.append("\n=== FAILED CONNECTIVITY TESTS ===")
        for t in verifier_report.failed_tests:
            expected = "ALLOWED" if t.expected_allowed else "BLOCKED"
            actual   = "ALLOWED" if t.actual_allowed else "BLOCKED"
            error_lines.append(
                f"  {t.description}: expected {expected}, got {actual}"
            )

    # Secondary, supplementary only — the Critic is an LLM opinion and can be
    # wrong; it no longer gates acceptance (see pipeline.py), so its notes
    # are listed last and flagged as such rather than treated as authoritative.
    if critic_report.has_errors:
        error_lines.append("\n=== STATIC ANALYSIS NOTES (secondary; may be imprecise) ===")
        for e in critic_report.errors:
            error_lines.append(f"[{e.kind.upper()}] {e.message}")

    error_summary = "\n".join(error_lines) if error_lines else "No specific errors — improve clarity and security."

    user_prompt = (
        f"Intent: {intent.description}\n\n"
        f"Current YAML:\n{yaml_policy}\n\n"
        f"Errors to fix:\n{error_summary}\n\n"
        "Return corrected YAML only:"
    )

    raw = ollama_client.generate(_cfg.REFINER_MODEL, user_prompt, system=_SYSTEM_PROMPT)

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()

    return text
