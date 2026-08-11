"""Thin wrapper around the Ollama /api/generate endpoint."""

import json
import requests
from policy_agent.config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT


def generate(model: str, prompt: str, system: str = "") -> str:
    """Call Ollama and return the full text response."""
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
        },
    }
    if system:
        payload["system"] = system

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def embed(model: str, text: str) -> list[float]:
    """Call Ollama's embedding endpoint and return the embedding vector."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": model, "input": text},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def generate_json(model: str, prompt: str, system: str = "") -> dict:
    """Call Ollama and parse the response as JSON.
    Strips markdown fences if present."""
    raw = generate(model, prompt, system)
    # Strip optional ```json ... ``` fences
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # remove first and last fence lines
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()
    return json.loads(text)
