"""Pydantic data models shared across all agents."""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PortSpec(BaseModel):
    """One (protocol, port) pair — used by the MultiPort class, where a
    single rule must allow more than one port/protocol simultaneously."""
    protocol: str = "TCP"
    port: int


class PolicyIntent(BaseModel):
    """Natural language intent provided by the user."""
    description: str
    source_namespace: Optional[str] = None
    target_namespace: Optional[str] = None
    # Structured ground truth, carried straight from the dataset row when
    # available, so semantic_verifier can check the generated YAML against
    # what was actually asked for — never against the YAML's own shape.
    # "Reachability" | "Isolation" | "NamespaceSelector" |
    # "CombinedIngressEgress" | "MultiPort" | "MultiLabelSelector"
    policy_type: Optional[str] = None
    destination_cidr: Optional[str] = None
    allowed: Optional[bool] = None
    port: int = 80
    protocol: str = "TCP"

    # NamespaceSelector class: the other namespace referenced via
    # namespaceSelector (matched on the auto-populated
    # kubernetes.io/metadata.name label) instead of an ipBlock CIDR.
    peer_namespace: Optional[str] = None

    # CombinedIngressEgress class: the policy's ingress side reuses
    # destination_cidr/allowed/port/protocol above (same semantics as the
    # Isolation class); these carry the independent egress side (same
    # semantics as the Reachability class).
    egress_destination_cidr: Optional[str] = None
    egress_allowed: Optional[bool] = None

    # MultiPort class: when set, overrides port/protocol — the rule must
    # allow exactly this set of (protocol, port) pairs, nothing else.
    ports: Optional[list[PortSpec]] = None

    # MultiLabelSelector class: multi-key matchLabels (AND semantics) on the
    # policy's own pods and on the ingress peer, instead of podSelector: {}
    # or a single-label selector.
    target_labels: Optional[dict[str, str]] = None
    peer_labels: Optional[dict[str, str]] = None


class SemanticReport(BaseModel):
    """Deterministic verdict from semantic_verifier.py — own engine, no
    third-party policy-analysis tool involved."""
    matches_intent: bool
    reason: str = ""


class ExtractedIntent(BaseModel):
    """Structured fields pulled from a free-text intent by extractor.py —
    the production-realistic substitute for reading them off a CSV row.
    An LLM produced this, so it can be wrong; round_trip_sentence +
    round_trip_match are the cheap, deterministic self-check (see
    extractor.py:round_trip_check), not a guarantee of correctness."""
    policy_type: Optional[str] = None
    destination_cidr: Optional[str] = None
    allowed: Optional[bool] = None
    port: int = 80
    protocol: str = "TCP"
    # Mirror PolicyIntent's class-specific fields (see models.py:PolicyIntent)
    # — only populated when the extractor recognizes one of the 4 newer
    # classes. Anything the extractor can't map to a known policy_type/field
    # combination fails closed via parse_ok=False rather than guessing.
    peer_namespace: Optional[str] = None
    egress_destination_cidr: Optional[str] = None
    egress_allowed: Optional[bool] = None
    ports: Optional[list[PortSpec]] = None
    target_labels: Optional[dict[str, str]] = None
    peer_labels: Optional[dict[str, str]] = None
    parse_ok: bool = True
    raw_response: str = ""
    round_trip_sentence: str = ""
    # Embedding cosine similarity between the original intent and
    # round_trip_sentence — kept as a diagnostic only. Calibration
    # (calibrate_round_trip_threshold.py) showed generic sentence embeddings
    # can't reliably separate "same meaning, different words" from "a single
    # flipped word or CIDR digit" — a single differing detail barely moves
    # cosine similarity when the surrounding sentence is topically
    # near-identical. Not used to gate anything.
    round_trip_similarity: Optional[float] = None
    # The actual trust signal: an LLM asked to compare A (original) vs B
    # (round_trip_sentence) on the load-bearing details only — direction,
    # CIDR value, allow/deny — ignoring wording. See
    # extractor.py:round_trip_check and prompts/round_trip_judge.txt.
    round_trip_reason: str = ""
    round_trip_match: Optional[bool] = None


class CriticError(BaseModel):
    kind: str          # "syntax" | "semantic" | "security"
    message: str


class CriticReport(BaseModel):
    has_errors: bool
    errors: list[CriticError] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    raw_response: str = ""


class ConnectivityTest(BaseModel):
    description: str
    source: str
    destination: str
    port: int
    expected_allowed: bool
    actual_allowed: Optional[bool] = None

    @property
    def passed(self) -> bool:
        return self.actual_allowed == self.expected_allowed


class VerifierReport(BaseModel):
    dry_run_passed: bool
    dry_run_error: str = ""
    runtime_tests: list[ConnectivityTest] = Field(default_factory=list)
    # "intent" when the runtime test's expected outcome was derived from
    # structured intent (the hardened path); "yaml_shape" when it fell back
    # to inspecting the candidate YAML's own rule shapes (the flaw this was
    # built to fix — see verifier.py:_derive_test_cases_from_intent).
    test_source: str = "yaml_shape"

    @property
    def all_tests_passed(self) -> bool:
        return self.dry_run_passed and all(t.passed for t in self.runtime_tests)

    @property
    def failed_tests(self) -> list[ConnectivityTest]:
        return [t for t in self.runtime_tests if not t.passed]


class IterationResult(BaseModel):
    iteration: int
    yaml_policy: str
    critic_report: CriticReport
    verifier_report: VerifierReport
    semantic_report: Optional[SemanticReport] = None
    elapsed_seconds: float = 0.0
    generator_seconds: float = 0.0
    critic_seconds: float = 0.0
    verifier_seconds: float = 0.0
    refiner_seconds: float = 0.0


class PipelineResult(BaseModel):
    intent: PolicyIntent
    # Set only when pipeline.run(use_extraction=True) — the Extractor's own
    # output, which is what actually gated pass/fail in that mode (not
    # intent's oracle CSV fields, which stay untouched purely for scoring).
    extracted_intent: Optional[ExtractedIntent] = None
    final_yaml: str
    success: bool
    iterations: int
    iteration_history: list[IterationResult] = Field(default_factory=list)
    message: str = ""
    total_wall_clock_seconds: float = 0.0
    per_iteration_avg_seconds: float = 0.0
