"""Agent 3 — Verifier: kubectl dry-run + runtime connectivity tests."""

from __future__ import annotations
import subprocess
import tempfile
import time
import yaml

from policy_agent.config import (
    TEST_NAMESPACE,
    TEST_NAMESPACE_CLIENT,
    POD_READY_TIMEOUT,
    CONNECTIVITY_TIMEOUT,
)
from policy_agent.models import ConnectivityTest, VerifierReport


# ── helpers ──────────────────────────────────────────────────────────────────

# Which kind cluster to target. None = whatever kubectl's current-context is
# (single-cluster behavior, unchanged from before). Set via set_context() so
# a multi-cluster batch run can pin a specific cluster for its whole duration
# without relying on "whatever happens to be the current context" — the
# thing that would otherwise be one stray `kubectl config use-context` away
# from silently running against the wrong cluster.
_KUBE_CONTEXT: str | None = None


def set_context(context: str | None) -> None:
    """Pin all subsequent verifier.py kubectl calls to this context (e.g.
    "kind-calico"). Pass None to go back to using the current context."""
    global _KUBE_CONTEXT
    _KUBE_CONTEXT = context


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def _kubectl(*args: str, timeout: int = 30) -> tuple[int, str, str]:
    cmd = ["kubectl"]
    if _KUBE_CONTEXT:
        cmd += ["--context", _KUBE_CONTEXT]
    cmd += list(args)
    return _run(cmd, timeout=timeout)


# ── namespace & pod management ────────────────────────────────────────────────

def _ensure_namespace(ns: str) -> None:
    rc, _, _ = _kubectl("get", "namespace", ns)
    if rc != 0:
        _kubectl("create", "namespace", ns)


def _wait_pod_ready(ns: str, name: str, timeout: int = POD_READY_TIMEOUT) -> bool:
    rc, _, _ = _kubectl(
        "wait", "--for=condition=Ready",
        f"pod/{name}", f"-n={ns}",
        f"--timeout={timeout}s",
        timeout=timeout + 5,
    )
    return rc == 0


def _delete_pod(ns: str, name: str) -> None:
    _kubectl("delete", "pod", name, f"-n={ns}", "--ignore-not-found=true",
             "--grace-period=0", "--force")


def _get_pod_ip(ns: str, name: str) -> str:
    _, out, _ = _kubectl(
        "get", "pod", name, f"-n={ns}",
        "-o=jsonpath={.status.podIP}"
    )
    return out.strip()


# ── syntax / dry-run validation ───────────────────────────────────────────────

def _dry_run(yaml_text: str) -> tuple[bool, str]:
    """Apply the policy with --dry-run=server. Returns (passed, error_msg).
    Auto-creates the target namespace if missing so dry-run can proceed."""
    # Ensure namespace exists before server dry-run
    try:
        doc = yaml.safe_load(yaml_text)
        ns = doc.get("metadata", {}).get("namespace", "")
        if ns:
            _ensure_namespace(ns)
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_text)
        fname = f.name

    rc, out, err = _kubectl(
        "apply", "--dry-run=server", "-f", fname, timeout=20
    )
    if rc == 0:
        return True, ""
    return False, (err or out).strip()


# ── connectivity test helpers ─────────────────────────────────────────────────

def _can_connect(client_ns: str, client_pod: str,
                 target_ip: str, port: int) -> bool:
    """Return True if the client pod can TCP-connect to target_ip:port."""
    rc, _, _ = _kubectl(
        "exec", client_pod, f"-n={client_ns}", "--",
        "sh", "-c",
        f"nc -zw {CONNECTIVITY_TIMEOUT} {target_ip} {port}",
        timeout=CONNECTIVITY_TIMEOUT + 10,
    )
    return rc == 0


def _listen_cmd(ports: list[int]) -> str:
    """Shell command for a busybox pod to listen on every given port at
    once (the MultiPort class needs one server reachable on several ports
    simultaneously, not just one)."""
    loops = " & ".join(f"(while true; do nc -lp {p}; done)" for p in ports)
    return f"{loops} & wait"


# ── parse policy for test derivation ─────────────────────────────────────────

def _extract_policy_info(yaml_text: str) -> dict:
    """Extract key fields from the NetworkPolicy for test generation.

    A weak model chasing bad feedback across refiner iterations (e.g. the
    Cilium ipBlock quirk making it think a correct policy is broken — see
    the Checkpoint 2 CNI comparison writeup) can produce a malformed field
    shape, like `podSelector: "some string"` instead of a mapping. Coerce
    each field back to its expected type rather than crashing downstream
    `.get()` calls on it — a malformed shape should read as "no info here",
    not take down the whole batch row with an AttributeError.
    """
    try:
        doc = yaml.safe_load(yaml_text)
        spec = doc.get("spec", {}) if isinstance(doc.get("spec"), dict) else {}
        meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        pod_selector = spec.get("podSelector", {})
        ingress = spec.get("ingress", [])
        egress = spec.get("egress", [])
        policy_types = spec.get("policyTypes", [])
        return {
            "namespace":    meta.get("namespace", "default"),
            "name":         meta.get("name", "policy"),
            "pod_selector": pod_selector if isinstance(pod_selector, dict) else {},
            "policy_types": policy_types if isinstance(policy_types, list) else [],
            "ingress":      ingress if isinstance(ingress, list) else [],
            "egress":       egress if isinstance(egress, list) else [],
        }
    except Exception:
        return {}


def _extract_labels(selector) -> dict:
    """Return matchLabels dict from a selector, or {} for catch-all/malformed."""
    if not isinstance(selector, dict):
        return {}
    matchLabels = selector.get("matchLabels", {})
    return matchLabels if isinstance(matchLabels, dict) else {}


def _is_allow_all_except(egress_rules: list) -> bool:
    """Return True if the egress rules are 'allow 0.0.0.0/0 except [CIDRs]'."""
    for rule in egress_rules:
        if not isinstance(rule, dict):
            continue
        for to in rule.get("to", []) or []:
            if not isinstance(to, dict):
                continue
            ipb = to.get("ipBlock")
            if isinstance(ipb, dict):
                cidr    = ipb.get("cidr", "")
                excepts = ipb.get("except", [])
                if cidr == "0.0.0.0/0" and excepts:
                    return True
    return False


def _derive_test_cases_from_intent(
    intent, info: dict
) -> tuple[dict, dict, list[ConnectivityTest], str] | None:
    """
    Derive expected test outcomes from the STRUCTURED INTENT (policy_type,
    destination_cidr, allowed, port), not from the generated YAML's own
    rule shapes. This is the fix for the flaw documented in IFIP_V2/README.md
    and the journal-extension plan: the old `_derive_test_cases()` inspected
    the candidate's own ipBlock/except structure to decide what "should" be
    allowed, so a wrong-but-self-consistent policy (e.g. inverted
    allow/block direction) could pass by matching its own — wrong — shape
    instead of the actual intent. Topology (ingress vs egress test setup) is
    likewise chosen from `intent.policy_type`, not from the candidate's own
    `policyTypes` field — a candidate that emits the wrong direction
    entirely now fails the runtime test instead of silently being tested
    against its own mistake.

    Returns None when this intent's class isn't covered yet (no structured
    fields, or a combination this harness can't test — e.g. Isolation
    allowed=True needs source-CIDR-based ingress testing this harness
    doesn't implement), signalling the caller to fall back to
    `_derive_test_cases()`.

    For the two original classes (Reachability, Isolation) this returns the
    original 4-tuple (server_labels, client_labels, tests, topology), kept
    byte-for-byte identical to avoid touching the already-validated
    Checkpoint 1-3 live-cluster methodology. For the four newer classes
    (NamespaceSelector, CombinedIngressEgress, MultiPort, MultiLabelSelector)
    it returns a dict instead — see verify()'s topology dispatch for the
    shape each topology expects.
    """
    if intent is None or getattr(intent, "policy_type", None) not in (
        "Reachability", "Isolation", "NamespaceSelector",
        "CombinedIngressEgress", "MultiPort", "MultiLabelSelector",
    ):
        return None

    pod_selector_labels = _extract_labels(info.get("pod_selector", {}))
    test_port = getattr(intent, "port", None) or 80

    if intent.policy_type == "Isolation":
        if intent.allowed:
            # Ingress ALLOWED from a specific CIDR isn't testable with this
            # harness's label-based client pods (it would need a real
            # ipBlock-source-IP probe, not podSelector matching) — not yet
            # exercised by the dataset either. Fall back rather than guess.
            return None
        server_labels = pod_selector_labels or {"app": "server"}
        client_labels = {"app": "allowed-client"}
        tests = [
            ConnectivityTest(
                description=(
                    f"any-client → server:{test_port} "
                    f"(expect BLOCKED — deny-all ingress per intent)"
                ),
                source="any-client", destination="server",
                port=test_port, expected_allowed=False,
            )
        ]
        return server_labels, client_labels, tests, "ingress"

    if intent.policy_type == "NamespaceSelector":
        if intent.peer_namespace is None or intent.allowed is None:
            return None
        server_labels = pod_selector_labels or {"app": "server"}
        tests = [
            ConnectivityTest(
                description=(
                    f"client-in-{intent.peer_namespace} → server:{test_port} "
                    f"(expect {'ALLOWED' if intent.allowed else 'BLOCKED'} — peer namespace, per intent)"
                ),
                source="peer-ns-client", destination="server",
                port=test_port, expected_allowed=bool(intent.allowed),
            ),
            ConnectivityTest(
                description=(
                    f"client-in-other-ns → server:{test_port} "
                    f"(expect {'BLOCKED' if intent.allowed else 'ALLOWED'} — non-peer namespace, per intent)"
                ),
                source="other-ns-client", destination="server",
                port=test_port, expected_allowed=not bool(intent.allowed),
            ),
        ]
        return {
            "topology": "namespace_selector",
            "server_labels": server_labels,
            "tests": tests,
            "peer_namespace": intent.peer_namespace,
        }

    if intent.policy_type == "CombinedIngressEgress":
        if (intent.allowed is None or intent.egress_destination_cidr is None
                or intent.egress_allowed is None):
            return None
        if intent.allowed:
            # Ingress-allow-from-CIDR isn't live-testable with this harness's
            # label/namespace-based client pods — same documented gap as the
            # Isolation allowed=True case above. Fall back to legacy rather
            # than guess; the CombinedIngressEgress dataset rows are
            # generated with allowed=False (deny-all ingress) specifically
            # so this branch is the exception, not the common case.
            return None
        egress_expected = not bool(intent.egress_allowed)
        egress_reason = (
            "not in the allowed CIDR" if intent.egress_allowed else "blocked CIDR is external only"
        )
        tests = [
            ConnectivityTest(
                description=f"any-client → server:{test_port} (expect BLOCKED — deny-all ingress per intent)",
                source="any-client", destination="server",
                port=test_port, expected_allowed=False,
            ),
            ConnectivityTest(
                description=(
                    f"egress-client → external-server:{test_port} "
                    f"(expect {'ALLOWED' if egress_expected else 'BLOCKED'} — {egress_reason}, per intent)"
                ),
                source="egress-client", destination="external-server",
                port=test_port, expected_allowed=egress_expected,
            ),
        ]
        return {
            "topology": "combined",
            "server_labels": pod_selector_labels or {"app": "server"},
            "tests": tests,
        }

    if intent.policy_type == "MultiPort":
        if intent.destination_cidr is None or not intent.ports:
            return None
        # Every MultiPort dataset row is an allow-only-CIDR-on-N-ports rule
        # (Reachability's allowed=True shape, generalized to several
        # ports/protocols) — the live probe's destination is always a
        # cluster-internal IP outside the allowed CIDR, so every port must
        # come back BLOCKED regardless of which port is probed.
        tests = [
            ConnectivityTest(
                description=(
                    f"egress-client → external-server:{p.port} ({p.protocol}) "
                    f"(expect BLOCKED — non-matching destination, per intent)"
                ),
                source="egress-client", destination="external-server",
                port=p.port, expected_allowed=False,
            )
            for p in intent.ports
        ]
        return {
            "topology": "egress",
            "server_labels": {"app": "external-server"},
            "tests": tests,
        }

    if intent.policy_type == "MultiLabelSelector":
        if not intent.target_labels or not intent.peer_labels:
            return None
        # Only ONE of the required peer labels — this specifically tests
        # AND semantics (a pod matching some but not all required labels
        # must NOT match a multi-key matchLabels selector).
        partial_labels = dict(list(intent.peer_labels.items())[:1])
        tests = [
            ConnectivityTest(
                description=f"full-match-client → server:{test_port} (expect ALLOWED — all peer labels match)",
                source="full-match-client", destination="server",
                port=test_port, expected_allowed=True,
            ),
            ConnectivityTest(
                description=(
                    f"partial-match-client → server:{test_port} (expect BLOCKED — only "
                    f"{partial_labels} of {intent.peer_labels} present, AND semantics)"
                ),
                source="partial-match-client", destination="server",
                port=test_port, expected_allowed=False,
            ),
        ]
        return {
            "topology": "ingress",
            "server_labels": intent.target_labels,
            "client_labels": intent.peer_labels,
            "unauth_labels": partial_labels,
            "tests": tests,
        }

    # intent.policy_type == "Reachability"
    server_labels = {"app": "external-server"}
    client_labels = pod_selector_labels or {}
    # The runtime probe always targets a cluster-internal pod IP
    # (10.244.x.x), which is never inside this dataset's destination_cidr
    # (100.0.x.x/24) — so it's always a "non-matching destination" probe:
    #   allowed=True  (allow ONLY destination_cidr, deny the rest) → the
    #                 non-matching probe must be BLOCKED.
    #   allowed=False (block destination_cidr, allow everything else) → the
    #                 non-matching probe must be ALLOWED.
    # This is fixed by the intent's class, independent of whatever shape the
    # candidate YAML actually has.
    expected = not bool(intent.allowed)
    reason = "not in the allowed CIDR" if intent.allowed else "blocked CIDR is external only"
    tests = [
        ConnectivityTest(
            description=(
                f"egress-client → external-server:{test_port} "
                f"(expect {'ALLOWED' if expected else 'BLOCKED'} — {reason}, per intent)"
            ),
            source="egress-client", destination="external-server",
            port=test_port, expected_allowed=expected,
        )
    ]
    return server_labels, client_labels, tests, "egress"


def _derive_test_cases(info: dict) -> tuple[dict, dict, list[ConnectivityTest], str]:
    """
    FALLBACK ONLY — used when no structured intent is available, or the
    intent's class isn't covered by `_derive_test_cases_from_intent()` yet.
    This is the original YAML-shape-based derivation; it has the documented
    flaw of inferring "expected" behavior from the candidate's own rule
    shapes rather than from ground truth, so prefer the intent-driven path
    whenever structured intent is available.

    Derive server labels, client labels, expected test cases, and topology.

    Returns (server_labels, client_labels, tests, topology)
    topology: "ingress" → server in policy_ns, clients in policy_ns
              "egress"  → server in netpol-client ns, client in policy_ns
    """
    policy_types  = info.get("policy_types", [])
    ingress_rules = info.get("ingress", [])
    egress_rules  = info.get("egress", [])

    # ── INGRESS policies ──────────────────────────────────────────────────────
    if "Ingress" in policy_types:
        server_labels = _extract_labels(info.get("pod_selector", {})) or {"app": "server"}
        client_labels: dict = {"app": "allowed-client"}

        # Extract client labels from first ingress.from.podSelector if present
        if ingress_rules:
            for f in ingress_rules[0].get("from", []):
                if "podSelector" in f:
                    ml = f["podSelector"].get("matchLabels", {})
                    if ml:
                        client_labels = ml
                        break

        # Test port from ingress rule (default 80)
        test_port = 80
        if ingress_rules:
            ports = ingress_rules[0].get("ports", [])
            if ports:
                test_port = ports[0].get("port", 80)

        if ingress_rules:
            # Selective allow: allowed-client → ALLOWED, unauth → BLOCKED
            tests = [
                ConnectivityTest(
                    description=f"allowed-client → server:{test_port} (expect ALLOWED)",
                    source="allowed-client", destination="server",
                    port=test_port, expected_allowed=True,
                ),
                ConnectivityTest(
                    description=f"unauth-client → server:{test_port} (expect BLOCKED)",
                    source="unauth-client", destination="server",
                    port=test_port, expected_allowed=False,
                ),
            ]
        else:
            # Deny-all ingress: any client → BLOCKED
            tests = [
                ConnectivityTest(
                    description=f"any-client → server:{test_port} (expect BLOCKED — deny-all)",
                    source="any-client", destination="server",
                    port=test_port, expected_allowed=False,
                )
            ]
        return server_labels, client_labels, tests, "ingress"

    # ── EGRESS-ONLY policies ──────────────────────────────────────────────────
    if "Egress" in policy_types:
        server_labels: dict = {"app": "external-server"}
        client_labels = _extract_labels(info.get("pod_selector", {})) or {}

        if not egress_rules:
            # Deny-all egress: client (policy_ns) → external server → BLOCKED
            tests = [
                ConnectivityTest(
                    description="egress-client → external-server:80 (expect BLOCKED — deny-all egress)",
                    source="egress-client", destination="external-server",
                    port=80, expected_allowed=False,
                )
            ]
        elif _is_allow_all_except(egress_rules):
            # Allow-all-except-CIDR: test server is at 10.244.x.x (cluster IP)
            # which is NOT in the blocked CIDR (100.0.x.x) → should be ALLOWED
            tests = [
                ConnectivityTest(
                    description="egress-client → external-server:80 (expect ALLOWED — blocked CIDR is external only)",
                    source="egress-client", destination="external-server",
                    port=80, expected_allowed=True,
                )
            ]
        else:
            # Allow-only-specific-CIDR (e.g., 100.0.x.x):
            # test server at 10.244.x.x is NOT in allowed CIDR → BLOCKED
            tests = [
                ConnectivityTest(
                    description="egress-client → external-server:80 (expect BLOCKED — not in allowed CIDR)",
                    source="egress-client", destination="external-server",
                    port=80, expected_allowed=False,
                )
            ]
        return server_labels, client_labels, tests, "egress"

    # ── fallback ──────────────────────────────────────────────────────────────
    return {"app": "server"}, {"app": "allowed-client"}, [], "ingress"


# ── main entry point ──────────────────────────────────────────────────────────

def _apply_policy(yaml_text: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_text)
        fname = f.name
    _kubectl("apply", "-f", fname)
    time.sleep(2)


def _run_egress_topology(
    yaml_text: str, policy_ns: str, server_labels: dict,
    test_cases: list[ConnectivityTest], test_source: str,
) -> VerifierReport:
    """Server lives OUTSIDE the policy namespace (in netpol-client). Client
    lives INSIDE the policy namespace (egress rules apply to it). Test:
    client (policy_ns) → server (external) — governed by egress policy.
    Server listens on every port any test_case needs (MultiPort: several at
    once), and each test connects on its own tc.port rather than a single
    shared port."""
    test_ports = sorted({tc.port for tc in test_cases}) or [80]

    _ensure_namespace(policy_ns)
    _ensure_namespace(TEST_NAMESPACE_CLIENT)

    server_label_args = ",".join(f"{k}={v}" for k, v in server_labels.items())
    _delete_pod(TEST_NAMESPACE_CLIENT, "verify-ext-server")
    _kubectl(
        "run", "verify-ext-server",
        f"-n={TEST_NAMESPACE_CLIENT}",
        "--image=busybox:1.36",
        f"--labels={server_label_args}",
        "--restart=Never",
        "--command", "--",
        "sh", "-c", _listen_cmd(test_ports),
    )

    _delete_pod(policy_ns, "verify-egress-client")
    _kubectl(
        "run", "verify-egress-client",
        f"-n={policy_ns}",
        "--image=busybox:1.36",
        "--labels=app=egress-client",
        "--restart=Never",
        "--command", "--", "sleep", "300",
    )

    server_ready = _wait_pod_ready(TEST_NAMESPACE_CLIENT, "verify-ext-server")
    client_ready = _wait_pod_ready(policy_ns, "verify-egress-client")

    if not (server_ready and client_ready):
        _cleanup_egress(policy_ns)
        return VerifierReport(
            dry_run_passed=True,
            dry_run_error="Egress test pods did not become Ready in time",
        )

    _apply_policy(yaml_text)
    server_ip = _get_pod_ip(TEST_NAMESPACE_CLIENT, "verify-ext-server")

    results: list[ConnectivityTest] = []
    for tc in test_cases:
        actual = _can_connect(policy_ns, "verify-egress-client", server_ip, tc.port)
        results.append(tc.model_copy(update={"actual_allowed": actual}))

    _cleanup_egress(policy_ns)
    return VerifierReport(dry_run_passed=True, runtime_tests=results, test_source=test_source)


def _run_ingress_topology(
    yaml_text: str, policy_ns: str, server_labels: dict, client_labels: dict,
    unauth_labels: dict | None, test_cases: list[ConnectivityTest], test_source: str,
) -> VerifierReport:
    """Server and both clients are in the policy namespace. podSelector-based
    from-rules can match clients by label. `unauth_labels` overrides the
    second client's labels (default app=unauth-client) — MultiLabelSelector
    uses this to create a PARTIAL-match client (some but not all of the
    required peer labels) instead of a totally unrelated one, to test AND
    semantics specifically."""
    test_ports = sorted({tc.port for tc in test_cases}) or [80]
    unauth_labels = unauth_labels or {"app": "unauth-client"}

    _ensure_namespace(policy_ns)

    server_label_args = ",".join(f"{k}={v}" for k, v in server_labels.items())
    _delete_pod(policy_ns, "verify-server")
    _kubectl(
        "run", "verify-server",
        f"-n={policy_ns}",
        "--image=busybox:1.36",
        f"--labels={server_label_args}",
        "--restart=Never",
        "--command", "--",
        "sh", "-c", _listen_cmd(test_ports),
    )

    client_label_args = ",".join(f"{k}={v}" for k, v in client_labels.items())
    _delete_pod(policy_ns, "verify-allowed-client")
    _kubectl(
        "run", "verify-allowed-client",
        f"-n={policy_ns}",
        "--image=busybox:1.36",
        f"--labels={client_label_args}",
        "--restart=Never",
        "--command", "--", "sleep", "300",
    )

    unauth_label_args = ",".join(f"{k}={v}" for k, v in unauth_labels.items())
    _delete_pod(policy_ns, "verify-unauth-client")
    _kubectl(
        "run", "verify-unauth-client",
        f"-n={policy_ns}",
        "--image=busybox:1.36",
        f"--labels={unauth_label_args}",
        "--restart=Never",
        "--command", "--", "sleep", "300",
    )

    server_ready  = _wait_pod_ready(policy_ns, "verify-server")
    allowed_ready = _wait_pod_ready(policy_ns, "verify-allowed-client")
    unauth_ready  = _wait_pod_ready(policy_ns, "verify-unauth-client")

    if not (server_ready and allowed_ready and unauth_ready):
        _cleanup(policy_ns)
        return VerifierReport(
            dry_run_passed=True,
            dry_run_error="Ingress test pods did not become Ready in time",
        )

    _apply_policy(yaml_text)
    server_ip = _get_pod_ip(policy_ns, "verify-server")

    results = []
    for tc in test_cases:
        if tc.source in ("allowed-client", "client", "any-client", "full-match-client"):
            client_pod = "verify-allowed-client"
        else:
            client_pod = "verify-unauth-client"

        actual = _can_connect(policy_ns, client_pod, server_ip, tc.port)
        results.append(tc.model_copy(update={"actual_allowed": actual}))

    _cleanup(policy_ns)
    return VerifierReport(dry_run_passed=True, runtime_tests=results, test_source=test_source)


def _run_namespace_selector_topology(
    yaml_text: str, policy_ns: str, server_labels: dict,
    test_cases: list[ConnectivityTest], peer_namespace: str, test_source: str,
) -> VerifierReport:
    """Server lives in the policy namespace; the two clients live in TWO
    DIFFERENT namespaces (the peer namespace under test, and a fixed
    unrelated "other" namespace) rather than differing by label — this is
    what makes namespaceSelector matching (namespace identity) actually
    testable, as opposed to podSelector-based label matching."""
    test_ports = sorted({tc.port for tc in test_cases}) or [80]
    other_ns = TEST_NAMESPACE_CLIENT

    _ensure_namespace(policy_ns)
    _ensure_namespace(peer_namespace)
    _ensure_namespace(other_ns)

    server_label_args = ",".join(f"{k}={v}" for k, v in server_labels.items())
    _delete_pod(policy_ns, "verify-server")
    _kubectl(
        "run", "verify-server",
        f"-n={policy_ns}",
        "--image=busybox:1.36",
        f"--labels={server_label_args}",
        "--restart=Never",
        "--command", "--",
        "sh", "-c", _listen_cmd(test_ports),
    )

    _delete_pod(peer_namespace, "verify-peer-ns-client")
    _kubectl(
        "run", "verify-peer-ns-client",
        f"-n={peer_namespace}",
        "--image=busybox:1.36",
        "--labels=app=peer-ns-client",
        "--restart=Never",
        "--command", "--", "sleep", "300",
    )

    _delete_pod(other_ns, "verify-other-ns-client")
    _kubectl(
        "run", "verify-other-ns-client",
        f"-n={other_ns}",
        "--image=busybox:1.36",
        "--labels=app=other-ns-client",
        "--restart=Never",
        "--command", "--", "sleep", "300",
    )

    server_ready = _wait_pod_ready(policy_ns, "verify-server")
    peer_ready   = _wait_pod_ready(peer_namespace, "verify-peer-ns-client")
    other_ready  = _wait_pod_ready(other_ns, "verify-other-ns-client")

    if not (server_ready and peer_ready and other_ready):
        _cleanup_namespace_selector(policy_ns, peer_namespace, other_ns)
        return VerifierReport(
            dry_run_passed=True,
            dry_run_error="NamespaceSelector test pods did not become Ready in time",
        )

    _apply_policy(yaml_text)
    server_ip = _get_pod_ip(policy_ns, "verify-server")

    results = []
    for tc in test_cases:
        if tc.source == "peer-ns-client":
            client_ns, client_pod = peer_namespace, "verify-peer-ns-client"
        else:
            client_ns, client_pod = other_ns, "verify-other-ns-client"
        actual = _can_connect(client_ns, client_pod, server_ip, tc.port)
        results.append(tc.model_copy(update={"actual_allowed": actual}))

    _cleanup_namespace_selector(policy_ns, peer_namespace, other_ns)
    return VerifierReport(dry_run_passed=True, runtime_tests=results, test_source=test_source)


def _run_combined_topology(
    yaml_text: str, policy_ns: str, server_labels: dict,
    test_cases: list[ConnectivityTest], test_source: str,
) -> VerifierReport:
    """One applied policy, both directions tested: an ingress deny-all probe
    (any-client → server, in policy_ns) and an egress CIDR probe
    (egress-client in policy_ns → external server in netpol-client) —
    dispatched per test_case by `tc.destination` ("server" vs
    "external-server")."""
    test_ports = sorted({tc.port for tc in test_cases}) or [80]

    _ensure_namespace(policy_ns)
    _ensure_namespace(TEST_NAMESPACE_CLIENT)

    server_label_args = ",".join(f"{k}={v}" for k, v in server_labels.items())
    _delete_pod(policy_ns, "verify-server")
    _kubectl(
        "run", "verify-server",
        f"-n={policy_ns}",
        "--image=busybox:1.36",
        f"--labels={server_label_args}",
        "--restart=Never",
        "--command", "--",
        "sh", "-c", _listen_cmd(test_ports),
    )

    _delete_pod(policy_ns, "verify-any-client")
    _kubectl(
        "run", "verify-any-client",
        f"-n={policy_ns}",
        "--image=busybox:1.36",
        "--labels=app=any-client",
        "--restart=Never",
        "--command", "--", "sleep", "300",
    )

    _delete_pod(TEST_NAMESPACE_CLIENT, "verify-ext-server")
    _kubectl(
        "run", "verify-ext-server",
        f"-n={TEST_NAMESPACE_CLIENT}",
        "--image=busybox:1.36",
        f"--labels={server_label_args}",
        "--restart=Never",
        "--command", "--",
        "sh", "-c", _listen_cmd(test_ports),
    )

    _delete_pod(policy_ns, "verify-egress-client")
    _kubectl(
        "run", "verify-egress-client",
        f"-n={policy_ns}",
        "--image=busybox:1.36",
        "--labels=app=egress-client",
        "--restart=Never",
        "--command", "--", "sleep", "300",
    )

    ready = all([
        _wait_pod_ready(policy_ns, "verify-server"),
        _wait_pod_ready(policy_ns, "verify-any-client"),
        _wait_pod_ready(TEST_NAMESPACE_CLIENT, "verify-ext-server"),
        _wait_pod_ready(policy_ns, "verify-egress-client"),
    ])
    if not ready:
        _cleanup_combined(policy_ns)
        return VerifierReport(
            dry_run_passed=True,
            dry_run_error="Combined ingress+egress test pods did not become Ready in time",
        )

    _apply_policy(yaml_text)
    server_ip = _get_pod_ip(policy_ns, "verify-server")
    ext_server_ip = _get_pod_ip(TEST_NAMESPACE_CLIENT, "verify-ext-server")

    results = []
    for tc in test_cases:
        if tc.destination == "server":
            actual = _can_connect(policy_ns, "verify-any-client", server_ip, tc.port)
        else:
            actual = _can_connect(policy_ns, "verify-egress-client", ext_server_ip, tc.port)
        results.append(tc.model_copy(update={"actual_allowed": actual}))

    _cleanup_combined(policy_ns)
    return VerifierReport(dry_run_passed=True, runtime_tests=results, test_source=test_source)


def verify(yaml_text: str, intent=None) -> VerifierReport:
    """Full verification: dry-run + optional runtime tests.

    `intent` (a PolicyIntent/ExtractedIntent-like object exposing
    policy_type/destination_cidr/allowed/port/... ) drives what the runtime
    test expects, when available — see `_derive_test_cases_from_intent()`.
    Only falls back to inspecting the candidate YAML's own shape
    (`_derive_test_cases()`) when no usable intent is given.
    """

    # 1. Syntax / API validation
    dry_ok, dry_err = _dry_run(yaml_text)
    if not dry_ok:
        return VerifierReport(dry_run_passed=False, dry_run_error=dry_err)

    # 2. Parse policy
    info = _extract_policy_info(yaml_text)
    if not info:
        return VerifierReport(
            dry_run_passed=True,
            dry_run_error="Could not parse YAML for runtime tests",
        )

    derived = _derive_test_cases_from_intent(intent, info)
    test_source = "intent"
    extra: dict = {}
    if derived is None:
        server_labels, client_labels, test_cases, topology = _derive_test_cases(info)
        test_source = "yaml_shape"
    elif isinstance(derived, dict):
        extra = derived
        topology = derived["topology"]
        server_labels = derived.get("server_labels", {"app": "server"})
        client_labels = derived.get("client_labels", {})
        test_cases = derived["tests"]
    else:
        server_labels, client_labels, test_cases, topology = derived

    policy_ns = info["namespace"]

    if topology == "egress":
        return _run_egress_topology(yaml_text, policy_ns, server_labels, test_cases, test_source)
    if topology == "namespace_selector":
        return _run_namespace_selector_topology(
            yaml_text, policy_ns, server_labels, test_cases, extra["peer_namespace"], test_source
        )
    if topology == "combined":
        return _run_combined_topology(yaml_text, policy_ns, server_labels, test_cases, test_source)
    # "ingress"
    return _run_ingress_topology(
        yaml_text, policy_ns, server_labels, client_labels,
        extra.get("unauth_labels"), test_cases, test_source,
    )


def _cleanup(policy_ns: str) -> None:
    """Remove ingress test pods and the applied network policy."""
    _delete_pod(policy_ns, "verify-server")
    _delete_pod(policy_ns, "verify-allowed-client")
    _delete_pod(policy_ns, "verify-unauth-client")
    _kubectl("delete", "networkpolicies", "--all", f"-n={policy_ns}",
             "--ignore-not-found=true")


def _cleanup_egress(policy_ns: str) -> None:
    """Remove egress test pods and the applied network policy."""
    _delete_pod(policy_ns, "verify-egress-client")
    _delete_pod(TEST_NAMESPACE_CLIENT, "verify-ext-server")
    _kubectl("delete", "networkpolicies", "--all", f"-n={policy_ns}",
             "--ignore-not-found=true")


def _cleanup_namespace_selector(policy_ns: str, peer_namespace: str, other_ns: str) -> None:
    """Remove namespaceSelector test pods (across all 3 namespaces involved)
    and the applied network policy."""
    _delete_pod(policy_ns, "verify-server")
    _delete_pod(peer_namespace, "verify-peer-ns-client")
    _delete_pod(other_ns, "verify-other-ns-client")
    _kubectl("delete", "networkpolicies", "--all", f"-n={policy_ns}",
             "--ignore-not-found=true")


def _cleanup_combined(policy_ns: str) -> None:
    """Remove combined ingress+egress test pods and the applied network policy."""
    _delete_pod(policy_ns, "verify-server")
    _delete_pod(policy_ns, "verify-any-client")
    _delete_pod(policy_ns, "verify-egress-client")
    _delete_pod(TEST_NAMESPACE_CLIENT, "verify-ext-server")
    _kubectl("delete", "networkpolicies", "--all", f"-n={policy_ns}",
             "--ignore-not-found=true")
