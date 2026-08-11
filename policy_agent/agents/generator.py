"""Agent 1 — Generator: Natural language intent → NetworkPolicy YAML."""

from pathlib import Path
from policy_agent import ollama_client
import policy_agent.config as _cfg
from policy_agent.models import PolicyIntent

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "generator.txt").read_text()


def generate(intent: PolicyIntent, previous_yaml: str = "") -> str:
    """Return a NetworkPolicy YAML string for the given intent."""
    if previous_yaml:
        user_prompt = (
            f"Intent: {intent.description}\n\n"
            f"Previous attempt (improve this):\n{previous_yaml}\n\n"
            "Generate an improved NetworkPolicy YAML:"
        )
    else:
        user_prompt = (
            f"Intent: {intent.description}\n\n"
            "Generate a NetworkPolicy YAML:"
        )

    raw = ollama_client.generate(_cfg.GENERATOR_MODEL, user_prompt, system=_SYSTEM_PROMPT)

    # Strip markdown fences if the model wraps output
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()

    return text
