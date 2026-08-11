"""
Batch runner: reads kubernetes_policies.csv, converts each row to a natural
language intent, runs the multi-agent pipeline, and writes results to a CSV.

Columns in output CSV:
  row_id, policy_type, namespace, destination_cidr, allowed,
  nl_intent, status, error_type, error_detail, iterations_used, yaml_file
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from policy_agent import pipeline
from policy_agent.models import PipelineResult, PolicyIntent, PortSpec

console = Console()

# ── NL intent generation ──────────────────────────────────────────────────────

def _parse_bool(val) -> bool:
    return str(val).strip().lower() == "true"


def _or_none(val):
    """Blank CSV cells mean "not applicable to this row's class" — treat as
    None rather than an empty string, so intent.destination_cidr is None
    (not "") for e.g. NamespaceSelector/MultiLabelSelector rows."""
    return val if val not in (None, "") else None


def _parse_bool_or_none(val):
    if val in (None, ""):
        return None
    return str(val).strip().lower() == "true"


def _ext_matches_oracle_row(ext, row: dict) -> bool:
    """Diagnostic-only comparison of the Extractor's structured output
    against this row's oracle CSV fields — never used to gate pass/fail
    (pipeline.run() already handles that via parse_ok/round_trip_match)."""
    if not ext.parse_ok or ext.policy_type != row["policy_type"]:
        return False
    ptype = row["policy_type"]
    if ptype in ("Reachability", "Isolation"):
        return (
            ext.destination_cidr == row["destination_cidr"]
            and ext.allowed == _parse_bool(row.get("allowed", ""))
        )
    if ptype == "NamespaceSelector":
        return (
            ext.peer_namespace == row.get("peer_namespace")
            and ext.allowed == _parse_bool(row.get("allowed", ""))
        )
    if ptype == "CombinedIngressEgress":
        return (
            ext.destination_cidr == row.get("destination_cidr")
            and ext.allowed == _parse_bool(row.get("allowed", ""))
            and ext.egress_destination_cidr == row.get("egress_destination_cidr")
            and ext.egress_allowed == _parse_bool(row.get("egress_allowed", ""))
        )
    if ptype == "MultiPort":
        want_ports = _parse_json_field(row, "ports_json") or []
        got_ports = [p.model_dump() for p in (ext.ports or [])]
        return ext.destination_cidr == row.get("destination_cidr") and got_ports == want_ports
    if ptype == "MultiLabelSelector":
        return (
            ext.target_labels == (_parse_json_field(row, "target_labels_json") or {})
            and ext.peer_labels == (_parse_json_field(row, "peer_labels_json") or {})
        )
    return False


def _class_fields_row(row: dict) -> dict:
    """The 8 class-specific OUTPUT_FIELDNAMES columns, passed straight
    through from the input row so the results CSV stays traceable."""
    return {
        "port":                     row.get("port", ""),
        "protocol":                 row.get("protocol", ""),
        "peer_namespace":           row.get("peer_namespace", ""),
        "egress_destination_cidr":  row.get("egress_destination_cidr", ""),
        "egress_allowed":           row.get("egress_allowed", ""),
        "ports_json":               row.get("ports_json", ""),
        "target_labels_json":       row.get("target_labels_json", ""),
        "peer_labels_json":         row.get("peer_labels_json", ""),
    }


def _parse_json_field(row: dict, key: str):
    """CSV cells for dict/list-valued fields (ports, labels) are JSON-encoded
    strings; blank/missing means the field doesn't apply to this row."""
    raw = row.get(key, "")
    if not raw or not str(raw).strip():
        return None
    return json.loads(raw)


def _fmt_labels(labels: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in labels.items())


def _fmt_ports(ports: list) -> str:
    return ", ".join(f"{p['protocol']} port {p['port']}" for p in ports)


def to_nl_intent(row: dict) -> str:
    """Convert a CSV row to a natural language policy intent string.

    Reachability=true  → pods in namespace CAN reach that CIDR
                         → allow egress to that CIDR only (deny all other egress)
    Reachability=false → pods in namespace CANNOT reach that CIDR
                         → allow all egress EXCEPT that CIDR (using ipBlock.except)
    Isolation=false    → namespace is isolated, deny all ingress
    Isolation=true     → allow ingress from that CIDR
    NamespaceSelector=true  → allow ingress only from a specific peer namespace
    NamespaceSelector=false → allow ingress from every namespace EXCEPT the peer
    CombinedIngressEgress   → independent ingress + egress rules in one policy
    MultiPort               → allow egress to a CIDR on several ports/protocols at once
    MultiLabelSelector      → multi-key (AND) podSelector on both target and peer
    """
    ptype   = row["policy_type"]      # Reachability | Isolation | ...
    ns      = row["namespace"]
    cidr    = row["destination_cidr"]
    allowed = _parse_bool(row.get("allowed", ""))
    port    = row.get("port") or "80"
    protocol = row.get("protocol") or "TCP"

    if ptype == "NamespaceSelector":
        peer_ns = row["peer_namespace"]
        if allowed:
            return (
                f"Allow ingress to ALL pods in namespace '{ns}' only from pods in "
                f"namespace '{peer_ns}' on {protocol} port {port}. Deny all other "
                f"ingress by default. Use podSelector: {{}} to apply to all pods in "
                f"'{ns}', and match the peer by namespaceSelector (namespace "
                f"identity) — not by podSelector or ipBlock."
            )
        else:
            return (
                f"Allow ingress to ALL pods in namespace '{ns}' from every "
                f"namespace EXCEPT '{peer_ns}' on {protocol} port {port}. Use "
                f"podSelector: {{}} to apply to all pods in '{ns}', and match "
                f"peers by namespaceSelector (namespace identity) excluding "
                f"'{peer_ns}' — not by podSelector or ipBlock."
            )

    elif ptype == "CombinedIngressEgress":
        egress_cidr = row["egress_destination_cidr"]
        egress_allowed = _parse_bool(row.get("egress_allowed", ""))

        if allowed:
            ingress_sentence = (
                f"Allow ingress to ALL pods in namespace '{ns}' from CIDR {cidr} "
                f"on {protocol} port {port}. Deny all other ingress by default."
            )
        else:
            ingress_sentence = (
                f"Deny ALL ingress to pods in namespace '{ns}'. Fully isolate "
                f"from inbound traffic — no ingress rules at all."
            )

        if egress_allowed:
            egress_sentence = (
                f"Allow egress from ALL pods in namespace '{ns}' to CIDR "
                f"{egress_cidr} on {protocol} port {port}. Deny all other "
                f"egress by default."
            )
        else:
            egress_sentence = (
                f"Block egress from ALL pods in namespace '{ns}' to CIDR "
                f"{egress_cidr}. Allow egress to all other destinations. "
                f"Implement with ipBlock cidr: 0.0.0.0/0 and "
                f"except: [{egress_cidr}]."
            )

        return (
            f"{ingress_sentence} {egress_sentence} Apply podSelector: {{}} to "
            f"all pods, with policyTypes: [Ingress, Egress] together in a "
            f"single NetworkPolicy."
        )

    elif ptype == "MultiPort":
        ports = _parse_json_field(row, "ports_json") or []
        ports_desc = _fmt_ports(ports)
        return (
            f"Allow egress from ALL pods in namespace '{ns}' to the external "
            f"CIDR {cidr} on the following ports: {ports_desc}. Apply to all "
            f"pods using podSelector: {{}}. Deny all other egress by default. "
            f"All listed ports must be allowed and nothing else — no port "
            f"outside this list."
        )

    elif ptype == "MultiLabelSelector":
        target_labels = _parse_json_field(row, "target_labels_json") or {}
        peer_labels = _parse_json_field(row, "peer_labels_json") or {}
        return (
            f"Allow ingress to pods in namespace '{ns}' with ALL of these "
            f"labels: {_fmt_labels(target_labels)}, only from pods with ALL "
            f"of these labels: {_fmt_labels(peer_labels)}, on {protocol} port "
            f"{port}. Deny all other ingress. Use podSelector.matchLabels "
            f"with every listed label together (AND semantics) — a pod "
            f"matching only some of the labels must NOT match."
        )

    elif ptype == "Reachability":
        if allowed:
            # Kubernetes: allow egress to that CIDR only, deny all other egress
            return (
                f"Allow egress from ALL pods in namespace '{ns}' to the external "
                f"CIDR {cidr} on TCP port 80. "
                f"Apply to all pods using podSelector: {{}}. "
                f"Deny all other egress by default."
            )
        else:
            # Kubernetes: allow all egress EXCEPT that CIDR (using ipBlock.except)
            # Kubernetes is whitelist-based — to block a specific CIDR,
            # allow 0.0.0.0/0 with that CIDR in the except list.
            return (
                f"Block egress from ALL pods in namespace '{ns}' to CIDR {cidr}. "
                f"Allow egress to all other destinations. "
                f"Use podSelector: {{}} to apply to all pods. "
                f"Implement with ipBlock cidr: 0.0.0.0/0 and except: [{cidr}]."
            )

    elif ptype == "Isolation":
        if not allowed:
            return (
                f"Deny ALL ingress to pods in namespace '{ns}' from CIDR {cidr}. "
                f"Fully isolate the namespace — block all inbound traffic. "
                f"Use podSelector: {{}} to apply to all pods. "
                f"policyTypes: [Ingress] with no ingress rules."
            )
        else:
            return (
                f"Allow ingress to ALL pods in namespace '{ns}' from CIDR {cidr} "
                f"on TCP port 80. Use podSelector: {{}}."
            )

    # Fallback
    action = "Allow" if allowed else "Deny"
    return f"{action} {ptype} traffic for namespace '{ns}' to/from CIDR {cidr}."


# ── result classification ─────────────────────────────────────────────────────

def _classify_failure(result: PipelineResult) -> tuple[str, str]:
    """Return (error_type, error_detail) from a failed pipeline result."""
    if not result.iteration_history:
        return "unknown", "No iteration history"

    # pipeline.py's best-of-N means final_yaml isn't necessarily the LAST
    # iteration's output — find the iteration that actually produced it so
    # the reported error matches what's actually being graded as the result.
    last = next(
        (it for it in result.iteration_history if it.yaml_policy == result.final_yaml),
        result.iteration_history[-1],
    )

    # Check dry-run failure
    if not last.verifier_report.dry_run_passed:
        err = last.verifier_report.dry_run_error
        # Classify API errors
        if "Invalid value" in err or "Unsupported value" in err or "required" in err.lower():
            return "syntax", err
        return "api_error", err

    # Live-cluster runtime result vs deterministic semantic_verifier result:
    # on any failing row, dry-run already passed above, so success requires
    # BOTH to agree the policy is correct. If exactly one says pass and the
    # other says fail, that's worth surfacing as its own category rather
    # than silently reporting only whichever check happens to be listed
    # first — a disagreement is exactly the case worth double-checking by
    # hand (see the plan's verifier-hardening design).
    runtime_ok = last.verifier_report.all_tests_passed
    semantic_ok = last.semantic_report.matches_intent if last.semantic_report is not None else None
    if semantic_ok is not None and runtime_ok != semantic_ok:
        if not runtime_ok:
            details = "; ".join(
                f"{t.description} expected={'ALLOW' if t.expected_allowed else 'BLOCK'} "
                f"got={'ALLOW' if t.actual_allowed else 'BLOCK'}"
                for t in last.verifier_report.failed_tests
            )
            return "verifier_disagreement", (
                f"live-cluster verifier FAILED ({details}) but semantic_verifier says MATCH"
            )
        return "verifier_disagreement", (
            f"live-cluster verifier PASSED but semantic_verifier says MISMATCH: {last.semantic_report.reason}"
        )

    # Check runtime test failures
    if last.verifier_report.failed_tests:
        details = "; ".join(
            f"{t.description} expected={'ALLOW' if t.expected_allowed else 'BLOCK'} "
            f"got={'ALLOW' if t.actual_allowed else 'BLOCK'}"
            for t in last.verifier_report.failed_tests
        )
        return "runtime", details

    # Deterministic semantic mismatch (own engine — see semantic_verifier.py)
    if last.semantic_report is not None and not last.semantic_report.matches_intent:
        return "semantic", last.semantic_report.reason

    # Critic-only failure (static analysis) — secondary/advisory, doesn't gate
    # acceptance anymore, but still worth surfacing if nothing else explains it.
    if last.critic_report.has_errors:
        kinds = list({e.kind for e in last.critic_report.errors})
        msgs  = "; ".join(e.message for e in last.critic_report.errors)
        return "/".join(kinds), msgs

    return "unknown", "Max iterations reached"


# ── YAML persistence ──────────────────────────────────────────────────────────

def _safe_name(text: str) -> str:
    """Convert text to a safe filename component."""
    return re.sub(r"[^a-z0-9_-]", "_", text.lower())[:40]


def _save_yaml(result: PipelineResult, row_id: int, out_dir: Path) -> str:
    """Save the verified YAML and return the file path."""
    ns    = _safe_name(result.intent.description.split("'")[1]
                       if "'" in result.intent.description else f"row{row_id}")
    fname = out_dir / f"policy_{row_id:04d}_{ns}.yaml"
    fname.write_text(result.final_yaml)
    return str(fname)


# ── CSV row iterator ──────────────────────────────────────────────────────────

def _read_csv(csv_path: str, limit: int | None = None) -> Iterator[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            yield row


# ── main batch entry ──────────────────────────────────────────────────────────

OUTPUT_FIELDNAMES = [
    "row_id", "policy_type", "namespace", "destination_cidr", "allowed",
    # Class-specific fields — blank ("") on rows whose policy_type doesn't
    # use them. Passed straight through from the input CSV for traceability.
    "port", "protocol", "peer_namespace",
    "egress_destination_cidr", "egress_allowed",
    "ports_json", "target_labels_json", "peer_labels_json",
    "nl_intent", "status", "error_type", "error_detail",
    "iterations_used", "yaml_file",
    "wall_clock_seconds", "per_iteration_avg_seconds",
    "generator_seconds", "critic_seconds", "verifier_seconds", "refiner_seconds",
    # Populated only when run_batch(use_extraction=True) — diagnostics
    # comparing what the Extractor pulled out of the free-text intent
    # against this row's own oracle CSV fields. Not used to gate pass/fail
    # (that already happens inside pipeline.run()); purely for measuring
    # how often extraction-based runs diverge from the oracle-intent runs.
    "extraction_parse_ok", "extraction_matches_oracle",
    "extraction_round_trip_match", "extraction_round_trip_similarity",
    # Whether pipeline.run() actually fed the extracted fields to the
    # verifier/semantic_verifier (parse_ok AND round_trip_match), or
    # gracefully degraded to "no structured ground truth" because the
    # round-trip judge couldn't confirm the extraction. Lets us measure how
    # often the gate fires and whether degraded rows still pass/fail
    # sensibly (they fall back to the pre-extraction legacy checks).
    "extraction_trusted",
]


def run_batch(
    csv_input:   str,
    csv_output:  str,
    yaml_dir:    str  = "verified_policies",
    limit:       int | None = None,
    max_iter:    int  = 3,            # reduced for batch speed
    mode:        str  = "full",       # full | generator_only | no_refine
    model:       str  = "",           # override all agent models (empty = use config defaults)
    generator_model: str = "",        # override just the Generator (heterogeneous-agent ablation)
    critic_model:    str = "",        # override just the Critic
    refiner_model:   str = "",        # override just the Refiner
    cluster:     str  = "kindnet",    # kindnet | calico | cilium
    use_extraction: bool = False,     # gate on Extractor output, not the CSV's oracle fields
    skip_critic:    bool = False,     # Checkpoint 5 ablation: stub out Critic feedback
    blind_refiner:  bool = False,     # Checkpoint 5 ablation: withhold specific Refiner feedback
) -> None:
    """
    Read csv_input, run the pipeline for each row, write results to csv_output.
    Passing policies have their YAML saved under yaml_dir/.

    `model` sets all three agent models at once; `generator_model` /
    `critic_model` / `refiner_model` override individual roles on top of
    that (or on top of config.py's defaults if `model` isn't given) — this
    is what a heterogeneous-agent ablation (e.g. a weak Generator paired
    with a stronger Refiner) actually needs.
    """
    import policy_agent.config as cfg
    from policy_agent.agents import verifier
    verifier.set_context(f"kind-{cluster}")
    if model:
        cfg.GENERATOR_MODEL = model
        cfg.CRITIC_MODEL    = model
        cfg.REFINER_MODEL   = model
    if generator_model:
        cfg.GENERATOR_MODEL = generator_model
    if critic_model:
        cfg.CRITIC_MODEL = critic_model
    if refiner_model:
        cfg.REFINER_MODEL = refiner_model
    out_path  = Path(csv_output)
    yaml_path = Path(yaml_dir)
    yaml_path.mkdir(parents=True, exist_ok=True)

    # Count total rows for progress bar
    rows = list(_read_csv(csv_input, limit=limit))
    total = len(rows)

    console.print(f"\n[bold cyan]Batch run:[/bold cyan] {total} policies from [bold]{csv_input}[/bold]")
    console.print(f"Output → [bold]{csv_output}[/bold]   YAMLs → [bold]{yaml_dir}/[/bold]\n")

    passed = failed = 0
    start_time = time.time()

    with open(out_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=4,
        ) as progress:
            task = progress.add_task("Processing policies…", total=total)

            for idx, row in enumerate(rows, start=1):
                nl = to_nl_intent(row)
                allowed_desc = (
                    "ALLOW" if row.get("allowed") == "true"
                    else "DENY " if row.get("allowed") == "false"
                    else "N/A  "
                )
                progress.update(
                    task,
                    description=f"[{idx:3d}/{total}] {row['namespace']:12s} {row['policy_type']:13s} "
                                f"{allowed_desc}",
                )

                ports_raw = _parse_json_field(row, "ports_json")
                intent = PolicyIntent(
                    description=nl,
                    target_namespace=row["namespace"],
                    policy_type=row["policy_type"],
                    destination_cidr=_or_none(row.get("destination_cidr")),
                    allowed=_parse_bool_or_none(row.get("allowed")),
                    port=int(row["port"]) if row.get("port") else 80,
                    protocol=row.get("protocol") or "TCP",
                    peer_namespace=_or_none(row.get("peer_namespace")),
                    egress_destination_cidr=_or_none(row.get("egress_destination_cidr")),
                    egress_allowed=_parse_bool_or_none(row.get("egress_allowed")),
                    ports=[PortSpec(**p) for p in ports_raw] if ports_raw else None,
                    target_labels=_parse_json_field(row, "target_labels_json"),
                    peer_labels=_parse_json_field(row, "peer_labels_json"),
                )

                # Override max iterations for batch speed
                import policy_agent.config as cfg
                original_max = cfg.MAX_ITERATIONS
                cfg.MAX_ITERATIONS = max_iter

                try:
                    result = pipeline.run(
                        intent, mode=mode, use_extraction=use_extraction,
                        skip_critic=skip_critic, blind_refiner=blind_refiner,
                    )
                except Exception as exc:
                    # A single transient failure (e.g. an Ollama request
                    # timeout) must not kill an entire multi-hour batch —
                    # record it as a FAIL row and move on to the next policy.
                    failed += 1
                    writer.writerow({
                        "row_id":          idx,
                        "policy_type":     row["policy_type"],
                        "namespace":       row["namespace"],
                        "destination_cidr": row["destination_cidr"],
                        "allowed":         row["allowed"],
                        **_class_fields_row(row),
                        "nl_intent":       nl,
                        "status":          "FAIL",
                        "error_type":      "exception",
                        "error_detail":    f"{type(exc).__name__}: {exc}",
                        "iterations_used": 0,
                        "yaml_file":       "",
                        "wall_clock_seconds": 0.0,
                        "per_iteration_avg_seconds": 0.0,
                        "generator_seconds": 0.0,
                        "critic_seconds": 0.0,
                        "verifier_seconds": 0.0,
                        "refiner_seconds": 0.0,
                        "extraction_parse_ok": "",
                        "extraction_matches_oracle": "",
                        "extraction_round_trip_match": "",
                        "extraction_round_trip_similarity": "",
                        "extraction_trusted": "",
                    })
                    out_f.flush()
                    progress.advance(task)
                    continue
                finally:
                    cfg.MAX_ITERATIONS = original_max

                # Classify result
                if result.success:
                    status     = "PASS"
                    error_type = ""
                    error_det  = ""
                    yaml_file  = _save_yaml(result, idx, yaml_path)
                    passed += 1
                else:
                    status              = "FAIL"
                    error_type, error_det = _classify_failure(result)
                    yaml_file           = ""
                    failed += 1

                # Calculate average agent times
                gen_avg = sum(it.generator_seconds for it in result.iteration_history) / result.iterations if result.iterations > 0 else 0.0
                crit_avg = sum(it.critic_seconds for it in result.iteration_history) / result.iterations if result.iterations > 0 else 0.0
                ver_avg = sum(it.verifier_seconds for it in result.iteration_history) / result.iterations if result.iterations > 0 else 0.0
                ref_avg = sum(it.refiner_seconds for it in result.iteration_history) / result.iterations if result.iterations > 0 else 0.0

                # Extraction diagnostics (only meaningful when use_extraction=True;
                # blank otherwise). Compares the Extractor's own output against
                # this row's oracle CSV fields — never fed back into the gate,
                # purely so an extraction-based run can be scored against the
                # oracle-intent run for the same rows.
                ext_parse_ok = ext_matches_oracle = ext_round_trip = ext_round_trip_sim = ext_trusted = ""
                if result.extracted_intent is not None:
                    ext = result.extracted_intent
                    ext_parse_ok = ext.parse_ok
                    ext_round_trip = ext.round_trip_match
                    ext_round_trip_sim = ext.round_trip_similarity
                    ext_trusted = bool(ext.parse_ok and ext.round_trip_match)
                    ext_matches_oracle = _ext_matches_oracle_row(ext, row)

                writer.writerow({
                    "row_id":          idx,
                    "policy_type":     row["policy_type"],
                    "namespace":       row["namespace"],
                    "destination_cidr": row["destination_cidr"],
                    "allowed":         row["allowed"],
                    **_class_fields_row(row),
                    "nl_intent":       nl,
                    "status":          status,
                    "error_type":      error_type,
                    "error_detail":    error_det,
                    "iterations_used": result.iterations,
                    "yaml_file":       yaml_file,
                    "wall_clock_seconds": round(result.total_wall_clock_seconds, 2),
                    "per_iteration_avg_seconds": round(result.per_iteration_avg_seconds, 2),
                    "generator_seconds": round(gen_avg, 2),
                    "critic_seconds": round(crit_avg, 2),
                    "verifier_seconds": round(ver_avg, 2),
                    "refiner_seconds": round(ref_avg, 2),
                    "extraction_parse_ok": ext_parse_ok,
                    "extraction_matches_oracle": ext_matches_oracle,
                    "extraction_round_trip_match": ext_round_trip,
                    "extraction_round_trip_similarity": ext_round_trip_sim,
                    "extraction_trusted": ext_trusted,
                })
                out_f.flush()  # write after each row so progress is saved

                progress.advance(task)

    elapsed = time.time() - start_time

    # Summary table
    console.print()
    summary = Table(title="Batch Summary", show_header=True, header_style="bold cyan")
    summary.add_column("Metric",   style="bold")
    summary.add_column("Value",    justify="right")
    summary.add_row("Total policies",  str(total))
    summary.add_row("[green]PASSED[/green]",       f"[green]{passed}[/green]")
    summary.add_row("[red]FAILED[/red]",           f"[red]{failed}[/red]")
    summary.add_row("Pass rate",       f"{passed/total*100:.1f}%")
    summary.add_row("Elapsed",         f"{elapsed:.0f}s  ({elapsed/total:.1f}s/policy)")
    summary.add_row("Results CSV",     csv_output)
    summary.add_row("YAML directory",  yaml_dir)
    console.print(summary)
