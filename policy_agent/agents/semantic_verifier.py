"""
Agent — deterministic semantic verifier.

Own engine, no third-party policy-analysis tool. Computes the *effective
connectivity* a generated NetworkPolicy implies using exact CIDR/set
arithmetic (`ipaddress`), and compares it against the ground truth carried
on `PolicyIntent` (from the dataset row) — never against the candidate
YAML's own shape. That second part is what the old verifier got wrong: its
test derivation pattern-matched the generated YAML's structure, so a wrong
policy that happened to resemble a different valid pattern could grade
itself as correct (see IFIP_V2/README.md). It is also what makes an LLM
Critic unsafe as a gate: an LLM's opinion about a policy is not a source of
truth, and it can hallucinate objections to policies that are actually fine
(see the plan's root-cause #1).

This module understands six policy classes: the original two (Reachability,
Isolation) plus four added for the journal-extension's Checkpoint 4 dataset
(NamespaceSelector, CombinedIngressEgress, MultiPort, MultiLabelSelector).
Each is a declarative peer/port predicate in the same style — see the
per-class `_check_*` functions below.
"""

from __future__ import annotations

import ipaddress
from typing import Optional, Union

import yaml

from policy_agent.models import PolicyIntent, SemanticReport

IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]

_ALL_V4 = ipaddress.ip_network("0.0.0.0/0")


def _network(cidr: str) -> Optional[IPNetwork]:
    try:
        return ipaddress.ip_network(cidr, strict=False)
    except (ValueError, TypeError):
        return None


def _rule_peers(rule, peer_key: str) -> list[dict]:
    """A weak model can emit a malformed rule/peer (e.g. a bare string
    instead of a mapping) — treat anything that isn't the expected shape as
    "no peers here" rather than crashing the whole verification."""
    if not isinstance(rule, dict):
        return []
    peers = rule.get(peer_key)
    if not isinstance(peers, list):
        return []
    return [p for p in peers if isinstance(p, dict)]


def _ports_of(rule) -> Optional[list[tuple[str, int]]]:
    """None means 'no port restriction — all ports allowed'."""
    if not isinstance(rule, dict):
        return None
    ports = rule.get("ports")
    if not ports or not isinstance(ports, list):
        return None
    out = []
    for p in ports:
        if not isinstance(p, dict):
            continue
        proto = str(p.get("protocol") or "TCP").upper()
        port = p.get("port")
        if port is not None:
            try:
                out.append((proto, int(port)))
            except (TypeError, ValueError):
                continue
    return out


def _port_allowed(ports: Optional[list[tuple[str, int]]], want_proto: str, want_port: int) -> bool:
    if ports is None:
        return True
    return (want_proto.upper(), want_port) in ports


def _fmt_ports(ports: Optional[list[tuple[str, int]]]) -> str:
    if ports is None:
        return "all ports"
    return ",".join(f"{proto}:{port}" for proto, port in ports)


def _check_reachability(policy_types: list, egress: list, intent: PolicyIntent) -> SemanticReport:
    if "Egress" not in policy_types:
        return SemanticReport(matches_intent=False, reason="policyTypes does not include Egress")

    want_cidr = _network(intent.destination_cidr) if intent.destination_cidr else None
    if want_cidr is None:
        return SemanticReport(matches_intent=False, reason=f"invalid destination_cidr {intent.destination_cidr!r}")

    if intent.allowed:
        # Exactly one thing should be reachable: want_cidr on the required
        # port. Any other reachable destination violates "deny all other
        # egress by default".
        matched = False
        matched_ports: Optional[list[tuple[str, int]]] = None
        extra_destinations: list[str] = []

        for rule in egress:
            ports = _ports_of(rule)
            for peer in _rule_peers(rule, "to"):
                ipb = peer.get("ipBlock")
                if not isinstance(ipb, dict) or ipb.get("cidr") is None:
                    extra_destinations.append(str(peer))
                    continue
                net = _network(ipb["cidr"])
                if net is None:
                    extra_destinations.append(ipb["cidr"])
                    continue
                excepts = [_network(e) for e in (ipb.get("except") or [])]
                if net == want_cidr and not any(e == want_cidr for e in excepts):
                    matched = True
                    matched_ports = ports
                elif net == _ALL_V4 and any(e == want_cidr for e in excepts):
                    return SemanticReport(
                        matches_intent=False,
                        reason=(f"egress rule allows 0.0.0.0/0 except {intent.destination_cidr} — this BLOCKS the "
                                f"required CIDR instead of allowing only it (inverted semantics). "
                                f"FIX: delete this rule's ipBlock (cidr: 0.0.0.0/0, except: [...]) entirely and "
                                f"replace it with exactly one rule: to: [{{ipBlock: {{cidr: {intent.destination_cidr}}}}}], "
                                f"ports: [{{protocol: {intent.protocol}, port: {intent.port}}}] — nothing else."),
                    )
                else:
                    extra_destinations.append(str(net))

        if not matched:
            return SemanticReport(
                matches_intent=False,
                reason=(f"expected egress rule allowing only {intent.destination_cidr} not found. "
                        f"FIX: spec.egress must contain exactly one rule: "
                        f"to: [{{ipBlock: {{cidr: {intent.destination_cidr}}}}}], "
                        f"ports: [{{protocol: {intent.protocol}, port: {intent.port}}}] — remove any other egress rules."),
            )
        if not _port_allowed(matched_ports, intent.protocol, intent.port):
            return SemanticReport(
                matches_intent=False,
                reason=(f"egress to {intent.destination_cidr} exists but does not allow "
                        f"{intent.protocol}:{intent.port} (rule allows {_fmt_ports(matched_ports)}). "
                        f"FIX: change this rule's ports to exactly "
                        f"[{{protocol: {intent.protocol}, port: {intent.port}}}]."),
            )
        if extra_destinations:
            return SemanticReport(
                matches_intent=False,
                reason=(f"egress default-deny violated: additional destination(s) allowed: {extra_destinations}. "
                        f"FIX: remove every egress rule except the one allowing {intent.destination_cidr} — "
                        f"spec.egress must have exactly one rule, nothing else reachable."),
            )
        return SemanticReport(matches_intent=True, reason="egress allow-only-CIDR matches intent")

    else:
        # allowed=False: block want_cidr, allow everything else (ipBlock
        # 0.0.0.0/0 with `except: [want_cidr]`).
        for rule in egress:
            for peer in _rule_peers(rule, "to"):
                ipb = peer.get("ipBlock")
                if not isinstance(ipb, dict) or ipb.get("cidr") is None:
                    continue
                net = _network(ipb["cidr"])
                excepts = [_network(e) for e in (ipb.get("except") or [])]
                if net == _ALL_V4 and any(e == want_cidr for e in excepts):
                    return SemanticReport(matches_intent=True, reason="egress block-CIDR-except matches intent")
        return SemanticReport(
            matches_intent=False,
            reason=(f"expected egress rule blocking {intent.destination_cidr} not found. "
                    f"FIX: spec.egress must contain exactly one rule: "
                    f"to: [{{ipBlock: {{cidr: 0.0.0.0/0, except: [{intent.destination_cidr}]}}}}] — "
                    f"no ports restriction, remove any other egress rules."),
        )


def _check_isolation(policy_types: list, ingress: list, intent: PolicyIntent) -> SemanticReport:
    if "Ingress" not in policy_types:
        return SemanticReport(matches_intent=False, reason="policyTypes does not include Ingress")

    if not intent.allowed:
        # Deny-all: no ingress rules at all.
        if ingress:
            return SemanticReport(
                matches_intent=False,
                reason=(f"expected deny-all ingress (no rules) but found {len(ingress)} ingress rule(s). "
                        f"Any CIDR mentioned in this intent is CONTEXT ONLY and must not appear in the "
                        f"policy. FIX: delete the entire 'ingress:' key from spec — do not include any "
                        f"ingress rule, ipBlock, or except clause. Keep podSelector: {{}} and "
                        f"policyTypes: [Ingress] with nothing else."),
            )
        return SemanticReport(matches_intent=True, reason="deny-all ingress matches intent")

    want_cidr = _network(intent.destination_cidr) if intent.destination_cidr else None
    if want_cidr is None:
        return SemanticReport(matches_intent=False, reason=f"invalid destination_cidr {intent.destination_cidr!r}")

    for rule in ingress:
        ports = _ports_of(rule)
        for peer in _rule_peers(rule, "from"):
            ipb = peer.get("ipBlock")
            if not isinstance(ipb, dict) or ipb.get("cidr") is None:
                continue
            net = _network(ipb["cidr"])
            if net == want_cidr:
                if _port_allowed(ports, intent.protocol, intent.port):
                    return SemanticReport(matches_intent=True, reason="allow-ingress-from-CIDR matches intent")
                return SemanticReport(
                    matches_intent=False,
                    reason=(f"ingress from {intent.destination_cidr} exists but does not allow "
                            f"{intent.protocol}:{intent.port} (rule allows {_fmt_ports(ports)}). "
                            f"FIX: change this rule's ports to exactly "
                            f"[{{protocol: {intent.protocol}, port: {intent.port}}}]."),
                )
    return SemanticReport(
        matches_intent=False,
        reason=(f"expected ingress rule allowing {intent.destination_cidr} not found. "
                f"FIX: spec.ingress must contain exactly one rule: "
                f"from: [{{ipBlock: {{cidr: {intent.destination_cidr}}}}}], "
                f"ports: [{{protocol: {intent.protocol}, port: {intent.port}}}]."),
    )


def _extract_labels(selector) -> dict:
    """Return matchLabels dict from a podSelector, or {} for catch-all/
    malformed/missing (a weak model can emit a bare string or omit the
    selector entirely — treat as "no labels here" rather than crashing)."""
    if not isinstance(selector, dict):
        return {}
    match_labels = selector.get("matchLabels", {})
    return match_labels if isinstance(match_labels, dict) else {}


def _namespace_selector_info(selector) -> Optional[tuple[str, object]]:
    """Return ("only", ns) for a namespaceSelector matching exactly one
    namespace by identity, ("except", [ns, ...]) for a NotIn-based
    exclusion, or None if the selector isn't a recognized namespace-identity
    selector. Namespace identity is matched via the auto-populated
    kubernetes.io/metadata.name label — see generator.txt's rule for why
    podSelector/ipBlock must never be used for this instead."""
    if not isinstance(selector, dict):
        return None
    match_labels = selector.get("matchLabels")
    if isinstance(match_labels, dict):
        ns = match_labels.get("kubernetes.io/metadata.name")
        if ns:
            return ("only", ns)
    match_exprs = selector.get("matchExpressions")
    if isinstance(match_exprs, list):
        for expr in match_exprs:
            if not isinstance(expr, dict):
                continue
            if expr.get("key") == "kubernetes.io/metadata.name" and expr.get("operator") == "NotIn":
                values = expr.get("values")
                if isinstance(values, list) and values:
                    return ("except", values)
    return None


def _check_namespace_selector(policy_types: list, ingress: list, intent: PolicyIntent) -> SemanticReport:
    if "Ingress" not in policy_types:
        return SemanticReport(matches_intent=False, reason="policyTypes does not include Ingress")

    want_peer = intent.peer_namespace

    matched = False
    matched_ports: Optional[list[tuple[str, int]]] = None
    extra_peers: list[str] = []

    for rule in ingress:
        ports = _ports_of(rule)
        for peer in _rule_peers(rule, "from"):
            info = _namespace_selector_info(peer.get("namespaceSelector"))
            if info is None:
                extra_peers.append(str(peer))
                continue
            kind, val = info
            if intent.allowed and kind == "only" and val == want_peer:
                matched = True
                matched_ports = ports
            elif not intent.allowed and kind == "except" and want_peer in val:
                matched = True
                matched_ports = ports
            else:
                extra_peers.append(f"{kind}:{val}")

    if not matched:
        if intent.allowed:
            fix = f"from: [{{namespaceSelector: {{matchLabels: {{kubernetes.io/metadata.name: {want_peer}}}}}}}]"
        else:
            fix = (f"from: [{{namespaceSelector: {{matchExpressions: [{{key: kubernetes.io/metadata.name, "
                   f"operator: NotIn, values: [{want_peer}]}}]}}}}]")
        return SemanticReport(
            matches_intent=False,
            reason=(f"expected ingress rule using namespaceSelector "
                    f"({'only' if intent.allowed else 'except'} {want_peer}) not found. "
                    f"FIX: spec.ingress must contain exactly one rule: {fix}, "
                    f"ports: [{{protocol: {intent.protocol}, port: {intent.port}}}] — "
                    f"do not use podSelector or ipBlock for this."),
        )
    if not _port_allowed(matched_ports, intent.protocol, intent.port):
        return SemanticReport(
            matches_intent=False,
            reason=(f"namespaceSelector ingress rule exists but does not allow "
                    f"{intent.protocol}:{intent.port} (rule allows {_fmt_ports(matched_ports)}). "
                    f"FIX: change this rule's ports to exactly "
                    f"[{{protocol: {intent.protocol}, port: {intent.port}}}]."),
        )
    if extra_peers:
        return SemanticReport(
            matches_intent=False,
            reason=(f"ingress default-deny violated: additional peer(s) allowed: {extra_peers}. "
                    f"FIX: remove every ingress peer except the required namespaceSelector rule."),
        )
    return SemanticReport(matches_intent=True, reason="namespaceSelector ingress rule matches intent")


def _check_combined(policy_types: list, ingress: list, egress: list, intent: PolicyIntent) -> SemanticReport:
    missing = [t for t in ("Ingress", "Egress") if t not in policy_types]
    if missing:
        return SemanticReport(
            matches_intent=False,
            reason=f"policyTypes is missing {missing} — a combined ingress+egress policy needs both",
        )

    ingress_report = _check_isolation(policy_types, ingress, intent)
    if not ingress_report.matches_intent:
        return SemanticReport(matches_intent=False, reason=f"ingress side: {ingress_report.reason}")

    egress_intent = intent.model_copy(update={
        "destination_cidr": intent.egress_destination_cidr,
        "allowed": intent.egress_allowed,
    })
    egress_report = _check_reachability(policy_types, egress, egress_intent)
    if not egress_report.matches_intent:
        return SemanticReport(matches_intent=False, reason=f"egress side: {egress_report.reason}")

    return SemanticReport(matches_intent=True, reason="both ingress and egress sides match intent")


def _check_multi_port(policy_types: list, egress: list, intent: PolicyIntent) -> SemanticReport:
    if "Egress" not in policy_types:
        return SemanticReport(matches_intent=False, reason="policyTypes does not include Egress")

    want_cidr = _network(intent.destination_cidr) if intent.destination_cidr else None
    if want_cidr is None:
        return SemanticReport(matches_intent=False, reason=f"invalid destination_cidr {intent.destination_cidr!r}")

    want_ports = {(p.protocol.upper(), p.port) for p in (intent.ports or [])}

    matched = False
    matched_ports: Optional[list[tuple[str, int]]] = None
    extra_destinations: list[str] = []

    for rule in egress:
        ports = _ports_of(rule)
        for peer in _rule_peers(rule, "to"):
            ipb = peer.get("ipBlock")
            if not isinstance(ipb, dict) or ipb.get("cidr") is None:
                extra_destinations.append(str(peer))
                continue
            net = _network(ipb["cidr"])
            if net == want_cidr:
                matched = True
                matched_ports = ports
            else:
                extra_destinations.append(str(net) if net else ipb["cidr"])

    ports_fix = ", ".join(f"{{protocol: {p.protocol}, port: {p.port}}}" for p in (intent.ports or []))

    if not matched:
        return SemanticReport(
            matches_intent=False,
            reason=(f"expected egress rule allowing only {intent.destination_cidr} not found. "
                    f"FIX: spec.egress must contain exactly one rule: "
                    f"to: [{{ipBlock: {{cidr: {intent.destination_cidr}}}}}], "
                    f"ports: [{ports_fix}] — remove any other egress rules."),
        )
    if matched_ports is None:
        return SemanticReport(
            matches_intent=False,
            reason=(f"egress rule to {intent.destination_cidr} allows ALL ports (no ports: list) — "
                    f"must restrict to exactly ports: [{ports_fix}]."),
        )
    got_ports = set(matched_ports)
    missing_ports = want_ports - got_ports
    extra_ports = got_ports - want_ports
    if missing_ports:
        return SemanticReport(
            matches_intent=False,
            reason=(f"egress to {intent.destination_cidr} is missing required port(s): {missing_ports}. "
                    f"FIX: change this rule's ports to exactly [{ports_fix}]."),
        )
    if extra_ports:
        return SemanticReport(
            matches_intent=False,
            reason=(f"egress to {intent.destination_cidr} allows extra unrequested port(s): {extra_ports}. "
                    f"FIX: change this rule's ports to exactly [{ports_fix}]."),
        )
    if extra_destinations:
        return SemanticReport(
            matches_intent=False,
            reason=(f"egress default-deny violated: additional destination(s) allowed: {extra_destinations}. "
                    f"FIX: remove every egress rule except the one allowing {intent.destination_cidr}."),
        )
    return SemanticReport(matches_intent=True, reason="multi-port egress allow-only-CIDR matches intent")


def _check_multi_label(policy_types: list, ingress: list, pod_selector: dict, intent: PolicyIntent) -> SemanticReport:
    if "Ingress" not in policy_types:
        return SemanticReport(matches_intent=False, reason="policyTypes does not include Ingress")

    want_target = intent.target_labels or {}
    want_peer = intent.peer_labels or {}

    got_target = _extract_labels(pod_selector)
    if got_target != want_target:
        return SemanticReport(
            matches_intent=False,
            reason=(f"spec.podSelector.matchLabels {got_target} does not exactly match the required "
                    f"target labels {want_target}. FIX: set spec.podSelector.matchLabels to exactly "
                    f"{want_target} — every listed label together, nothing else."),
        )

    matched = False
    matched_ports: Optional[list[tuple[str, int]]] = None
    extra_peers: list[dict] = []

    for rule in ingress:
        ports = _ports_of(rule)
        for peer in _rule_peers(rule, "from"):
            got_peer = _extract_labels(peer.get("podSelector"))
            if got_peer == want_peer:
                matched = True
                matched_ports = ports
            elif got_peer:
                extra_peers.append(got_peer)

    if not matched:
        return SemanticReport(
            matches_intent=False,
            reason=(f"expected ingress from podSelector.matchLabels {want_peer} not found. "
                    f"FIX: spec.ingress must contain exactly one rule: "
                    f"from: [{{podSelector: {{matchLabels: {want_peer}}}}}], "
                    f"ports: [{{protocol: {intent.protocol}, port: {intent.port}}}]."),
        )
    if not _port_allowed(matched_ports, intent.protocol, intent.port):
        return SemanticReport(
            matches_intent=False,
            reason=(f"multi-label ingress rule exists but does not allow "
                    f"{intent.protocol}:{intent.port} (rule allows {_fmt_ports(matched_ports)}). "
                    f"FIX: change this rule's ports to exactly "
                    f"[{{protocol: {intent.protocol}, port: {intent.port}}}]."),
        )
    if extra_peers:
        return SemanticReport(
            matches_intent=False,
            reason=(f"ingress default-deny violated: additional peer selector(s) allowed: {extra_peers}. "
                    f"FIX: remove every ingress peer except the required podSelector.matchLabels rule."),
        )
    return SemanticReport(matches_intent=True, reason="multi-label podSelector ingress matches intent")


def _has_ground_truth(intent: PolicyIntent) -> bool:
    """Whether `intent` carries enough structured ground truth for this
    intent's own policy_type to be checked deterministically — every class
    has its own required-fields combination (see models.py:PolicyIntent)."""
    pt = intent.policy_type
    if pt in ("Reachability", "Isolation"):
        return intent.destination_cidr is not None and intent.allowed is not None
    if pt == "NamespaceSelector":
        return intent.peer_namespace is not None and intent.allowed is not None
    if pt == "CombinedIngressEgress":
        return (
            intent.destination_cidr is not None and intent.allowed is not None
            and intent.egress_destination_cidr is not None and intent.egress_allowed is not None
        )
    if pt == "MultiPort":
        return intent.destination_cidr is not None and bool(intent.ports)
    if pt == "MultiLabelSelector":
        return bool(intent.target_labels) and bool(intent.peer_labels)
    return False


def verify_semantics(yaml_text: str, intent: PolicyIntent) -> SemanticReport:
    """Deterministically check whether `yaml_text` implements what `intent`
    actually asked for (per its structured fields), not whether it merely
    looks like some recognizable pattern."""
    if intent.policy_type is None or not _has_ground_truth(intent):
        # No structured ground truth on this intent (e.g. a free-form/
        # real-world validation intent) — nothing to check deterministically.
        return SemanticReport(matches_intent=True, reason="no structured ground truth to check against")

    try:
        doc = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return SemanticReport(matches_intent=False, reason=f"YAML did not parse: {e}")

    if not isinstance(doc, dict):
        return SemanticReport(matches_intent=False, reason="YAML did not parse to a mapping")

    spec = doc.get("spec")
    spec = spec if isinstance(spec, dict) else {}
    policy_types = spec.get("policyTypes")
    policy_types = policy_types if isinstance(policy_types, list) else []
    egress = spec.get("egress")
    egress = egress if isinstance(egress, list) else []
    ingress = spec.get("ingress")
    ingress = ingress if isinstance(ingress, list) else []

    if intent.policy_type == "Reachability":
        return _check_reachability(policy_types, egress, intent)
    if intent.policy_type == "Isolation":
        return _check_isolation(policy_types, ingress, intent)
    if intent.policy_type == "NamespaceSelector":
        return _check_namespace_selector(policy_types, ingress, intent)
    if intent.policy_type == "CombinedIngressEgress":
        return _check_combined(policy_types, ingress, egress, intent)
    if intent.policy_type == "MultiPort":
        return _check_multi_port(policy_types, egress, intent)
    if intent.policy_type == "MultiLabelSelector":
        pod_selector = spec.get("podSelector")
        pod_selector = pod_selector if isinstance(pod_selector, dict) else {}
        return _check_multi_label(policy_types, ingress, pod_selector, intent)
    return SemanticReport(
        matches_intent=True,
        reason=f"unknown policy_type {intent.policy_type!r} — semantic verifier has no rule for this class yet",
    )
