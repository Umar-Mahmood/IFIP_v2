"""
Agent 0 — Extractor: free-text intent → structured fields.

This is the piece that makes semantic_verifier.py (and, downstream,
verifier.py's test derivation) usable outside this benchmark. In production
there is no CSV row to read policy_type/destination_cidr/allowed/port from —
a real user just types a sentence. This agent extracts exactly those same
fields from the intent's own natural-language description, so the rest of
the pipeline can be fed EXTRACTED structured intent instead of the dataset's
own ground truth.

Extraction is itself an LLM call, so it can be wrong — swapping "read it off
a CSV" for "ask an LLM to guess it" doesn't solve anything on its own unless
we can also tell when it's wrong. round_trip_check() is that self-check: it
rebuilds a canonical sentence from the extracted fields (via
`batch.to_nl_intent`) and compares it against the ORIGINAL intent.

First attempt was embedding cosine similarity — cheap, no extra LLM call.
calibrate_round_trip_threshold.py's non-circular test (paraphrase the
original + separately corrupt the extraction, check similarity separates
the two) showed it DOESN'T work well enough: a single flipped allow/deny
word or a single differing CIDR octet barely moves cosine similarity when
the surrounding sentence is topically near-identical, so no threshold gave
clean separation (best case ~68% positive-accept vs ~48% negative-reject —
close to a coin flip). Kept as a logged diagnostic
(`round_trip_similarity`) but it does not gate anything.

The actual check is an LLM judge (prompts/round_trip_judge.txt): given A
(original) and B (round-trip sentence), explicitly compare only the
load-bearing details — direction, CIDR value, allow/deny — and ignore
wording. This can reason about a specific digit or a specific polarity word
the way embeddings can't. See calibrate_round_trip_threshold.py for the
same non-circular evidence applied to this method.

measure_extractor_accuracy.py is the other half — measuring extraction
accuracy against the CSV's real fields at full dataset scale, which is only
possible here because we happen to have labels for this benchmark.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from policy_agent import ollama_client
import policy_agent.config as _cfg
from policy_agent.models import ExtractedIntent, PortSpec

_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "extractor.txt").read_text()
_JUDGE_SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "round_trip_judge.txt").read_text()

_CIDR_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b")

# The six classes semantic_verifier.py/verifier.py know how to check —
# anything else fails closed (parse_ok=False) rather than being guessed at.
_KNOWN_CLASSES = {
    "Reachability", "Isolation", "NamespaceSelector",
    "CombinedIngressEgress", "MultiPort", "MultiLabelSelector",
}


def _require_cidr(data: dict, key: str) -> str:
    val = data.get(key)
    if not val or not _CIDR_RE.match(str(val)):
        raise ValueError(f"missing/invalid {key} {val!r}")
    return val


def _require_bool(data: dict, key: str) -> bool:
    val = data.get(key)
    if not isinstance(val, bool):
        raise ValueError(f"{key} must be bool, got {val!r}")
    return val


def _require_str(data: dict, key: str) -> str:
    val = data.get(key)
    if not val or not isinstance(val, str):
        raise ValueError(f"missing/invalid {key} {val!r}")
    return val


def _require_labels(data: dict, key: str) -> dict:
    val = data.get(key)
    if (not isinstance(val, dict) or not val
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in val.items())):
        raise ValueError(f"missing/invalid {key} {val!r}")
    return val


def _require_ports(data: dict, key: str) -> list[PortSpec]:
    val = data.get(key)
    if not isinstance(val, list) or not val:
        raise ValueError(f"missing/invalid {key} {val!r}")
    out = []
    for p in val:
        if not isinstance(p, dict) or "port" not in p:
            raise ValueError(f"malformed port entry {p!r} in {key}")
        out.append(PortSpec(protocol=str(p.get("protocol", "TCP")).upper(), port=int(p["port"])))
    return out


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract(description: str) -> ExtractedIntent:
    """Pull structured fields out of a free-text intent — no CSV involved."""
    user_prompt = f"Request: \"{description}\"\n\nExtract the structured fields as JSON:"
    raw = ollama_client.generate(_cfg.EXTRACTOR_MODEL, user_prompt, system=_SYSTEM_PROMPT)

    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()

    try:
        data = json.loads(text)
        policy_type = data.get("policy_type")
        if policy_type not in _KNOWN_CLASSES:
            raise ValueError(f"unrecognized policy_type {policy_type!r}")

        common = dict(
            policy_type=policy_type,
            port=int(data.get("port", 80)),
            protocol=str(data.get("protocol", "TCP")).upper(),
            parse_ok=True,
            raw_response=raw,
        )

        if policy_type in ("Reachability", "Isolation"):
            return ExtractedIntent(
                destination_cidr=_require_cidr(data, "destination_cidr"),
                allowed=_require_bool(data, "allowed"),
                **common,
            )
        if policy_type == "NamespaceSelector":
            return ExtractedIntent(
                peer_namespace=_require_str(data, "peer_namespace"),
                allowed=_require_bool(data, "allowed"),
                **common,
            )
        if policy_type == "CombinedIngressEgress":
            return ExtractedIntent(
                allowed=_require_bool(data, "allowed"),
                egress_destination_cidr=_require_cidr(data, "egress_destination_cidr"),
                egress_allowed=_require_bool(data, "egress_allowed"),
                **common,
            )
        if policy_type == "MultiPort":
            return ExtractedIntent(
                destination_cidr=_require_cidr(data, "destination_cidr"),
                ports=_require_ports(data, "ports"),
                **common,
            )
        # policy_type == "MultiLabelSelector"
        return ExtractedIntent(
            target_labels=_require_labels(data, "target_labels"),
            peer_labels=_require_labels(data, "peer_labels"),
            **common,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return ExtractedIntent(parse_ok=False, raw_response=f"{raw}\n[extractor error: {e}]")


def _llm_same_meaning(description: str, round_trip_sentence: str) -> tuple[bool, str]:
    """Ask an LLM to compare the two sentences on load-bearing details only
    (direction, CIDR value, allow/deny) — see prompts/round_trip_judge.txt.
    Fails closed (same_meaning=False) if the judge's output doesn't parse,
    since an unreadable verdict is not evidence of a match."""
    user_prompt = f'A (original): "{description}"\nB (reconstructed): "{round_trip_sentence}"\n\nCompare:'
    raw = ollama_client.generate(_cfg.EXTRACTOR_MODEL, user_prompt, system=_JUDGE_SYSTEM_PROMPT)
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()
    try:
        data = json.loads(text)
        same = bool(data["same_meaning"])
        reason = str(data.get("reason", ""))
        return same, reason
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False, f"judge output unparseable: {raw[:200]!r}"


def _bool_str(val) -> str:
    if val is None:
        return ""
    return "true" if val else "false"


def _extracted_to_row(extracted: ExtractedIntent, namespace: str) -> dict:
    """Build a batch.to_nl_intent()-compatible row dict from an
    ExtractedIntent, for round-trip reconstruction across all 6 classes."""
    return {
        "policy_type": extracted.policy_type,
        "namespace": namespace,
        "destination_cidr": extracted.destination_cidr or "",
        "allowed": _bool_str(extracted.allowed),
        "port": str(extracted.port),
        "protocol": extracted.protocol,
        "peer_namespace": extracted.peer_namespace or "",
        "egress_destination_cidr": extracted.egress_destination_cidr or "",
        "egress_allowed": _bool_str(extracted.egress_allowed),
        "ports_json": json.dumps([p.model_dump() for p in extracted.ports]) if extracted.ports else "",
        "target_labels_json": json.dumps(extracted.target_labels) if extracted.target_labels else "",
        "peer_labels_json": json.dumps(extracted.peer_labels) if extracted.peer_labels else "",
    }


def round_trip_check(description: str, extracted: ExtractedIntent, namespace: str) -> ExtractedIntent:
    """Rebuild a canonical sentence from the extracted fields, then check it
    against the ORIGINAL intent two ways:
      - round_trip_similarity: embedding cosine similarity. Logged as a
        diagnostic only — calibration showed it can't reliably tell "same
        meaning, different words" from "one flipped word/digit" apart, so
        it does not gate anything (see module docstring).
      - round_trip_match / round_trip_reason: an LLM judge comparing only
        the load-bearing details (direction, CIDR, allow/deny). This is the
        actual trust signal, and the one that works on arbitrary real-world
        phrasing, not just this dataset's own template."""
    # Deferred import: batch.py imports pipeline.py, which imports this
    # module — a module-level import here would be circular whenever
    # batch.py is the first thing loaded (batch.py -> pipeline -> agents ->
    # extractor -> batch, before batch finishes defining to_nl_intent). By
    # the time this function actually runs, batch.py is always fully loaded.
    from policy_agent.batch import to_nl_intent

    if not extracted.parse_ok or extracted.policy_type is None:
        return extracted.model_copy(update={
            "round_trip_sentence": "", "round_trip_similarity": 0.0,
            "round_trip_match": False, "round_trip_reason": "extraction did not parse",
        })

    round_trip_sentence = to_nl_intent(_extracted_to_row(extracted, namespace))

    emb_original = ollama_client.embed(_cfg.EMBEDDING_MODEL, description)
    emb_round_trip = ollama_client.embed(_cfg.EMBEDDING_MODEL, round_trip_sentence)
    similarity = cosine_similarity(emb_original, emb_round_trip)

    match, reason = _llm_same_meaning(description, round_trip_sentence)

    return extracted.model_copy(update={
        "round_trip_sentence": round_trip_sentence,
        "round_trip_similarity": similarity,
        "round_trip_match": match,
        "round_trip_reason": reason,
    })
