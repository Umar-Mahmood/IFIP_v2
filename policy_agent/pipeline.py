"""Main pipeline orchestrator: Generator → Critic → Verifier → Refiner loop."""

from __future__ import annotations
import time
from typing import Callable, Literal

from policy_agent.agents import critic, extractor, generator, refiner, verifier, semantic_verifier
import policy_agent.config as _cfg
from policy_agent.models import (
    CriticReport,
    IterationResult,
    PipelineResult,
    PolicyIntent,
    SemanticReport,
    VerifierReport,
)


def _badness(verifier_report: VerifierReport, semantic_report: SemanticReport) -> int:
    """Lower is better; 0 means fully passing. Used for best-of-N selection
    so the loop can't wander away from a good attempt and never come back."""
    score = 0 if verifier_report.dry_run_passed else 2
    score += len(verifier_report.failed_tests)
    score += 0 if semantic_report.matches_intent else 1
    return score

PipelineMode = Literal["full", "generator_only", "no_refine", "self_refine"]


def run(
    intent: PolicyIntent,
    on_iteration: Callable[[int, str, CriticReport, VerifierReport], None] | None = None,
    mode: PipelineMode = "full",
    use_extraction: bool = False,
    skip_critic: bool = False,
    blind_refiner: bool = False,
) -> PipelineResult:
    """
    Execute the multi-agent pipeline.

    mode="full"           – Generator → Critic → Verifier → Refiner loop (default)
    mode="generator_only" – Generator only, no validation (baseline: naive LLM)
    mode="no_refine"      – Generator → Critic → Verifier once, no refinement loop
    mode="self_refine"    – Checkpoint 5 baseline (Madaan et al.): same model in
                            every role (set via run_batch's --model). The loop
                            stops when the Critic's OWN self-review says
                            "no errors" — NOT when the deterministic
                            Verifier/Semantic Verifier agree — and the Refiner
                            sees only the Critic's feedback, never the
                            verifier/semantic detail. This can and does stop
                            early on a confidently-wrong attempt; `success` in
                            the returned result is still graded against the
                            real deterministic checks (never against the
                            model's own self-assessment), so a self-satisfied
                            but actually-wrong attempt correctly reports
                            success=False — that gap between "the model thinks
                            it's done" and "it's actually right" is the whole
                            point of this baseline.

    skip_critic / blind_refiner: Checkpoint 5 per-agent-removal ablations,
    orthogonal to `mode` (both only affect "full"/"no_refine" runs since
    generator_only never calls either agent). skip_critic replaces the
    Critic's analysis with a stub no-errors report every iteration — tests
    whether its advisory-only feedback still helps the Refiner even though
    it never gates pass/fail. blind_refiner tells the Refiner only that the
    previous attempt failed, withholding the specific semantic/verifier/
    critic detail — tests whether the loop's benefit comes from that
    specific feedback or just from additional inference-time attempts.

    use_extraction: when True, `intent.policy_type`/`destination_cidr`/
    `allowed`/`port` (the CSV's oracle ground truth) are NOT fed to the
    Verifier or Semantic Verifier — instead, those same fields are pulled
    from `intent.description` by the Extractor agent, exactly as a real
    deployment would have to (no CSV row exists in production). The oracle
    fields stay untouched on `intent` itself (still returned in
    `PipelineResult.intent`) purely so callers can score extraction accuracy
    against it — they're never read by anything that gates pass/fail when
    this flag is on. `PipelineResult.extracted_intent` carries the
    Extractor's raw output (including round_trip_match) for that scoring.
    """
    pipeline_start = time.time()
    history: list[IterationResult] = []

    extracted_intent = None
    check_intent = intent
    if use_extraction:
        extracted_intent = extractor.extract(intent.description)
        extracted_intent = extractor.round_trip_check(
            intent.description, extracted_intent, intent.target_namespace or ""
        )
        if extracted_intent.parse_ok and extracted_intent.round_trip_match:
            check_intent = intent.model_copy(update={
                "policy_type": extracted_intent.policy_type,
                "destination_cidr": extracted_intent.destination_cidr,
                "allowed": extracted_intent.allowed,
                "port": extracted_intent.port,
                "protocol": extracted_intent.protocol,
                "peer_namespace": extracted_intent.peer_namespace,
                "egress_destination_cidr": extracted_intent.egress_destination_cidr,
                "egress_allowed": extracted_intent.egress_allowed,
                "ports": extracted_intent.ports,
                "target_labels": extracted_intent.target_labels,
                "peer_labels": extracted_intent.peer_labels,
            })
        else:
            # Extraction failed to parse, OR parsed but the round-trip judge
            # (extractor.py:_llm_same_meaning) couldn't confirm it actually
            # matches what the user asked — degrade to "no structured ground
            # truth" rather than silently trusting an unconfirmed extraction
            # (or falling back to the oracle CSV fields, which would defeat
            # the point of this mode). semantic_verifier/verifier both
            # already handle None fields gracefully (skip the check they
            # can't ground) rather than failing outright.
            check_intent = intent.model_copy(update={
                "policy_type": None, "destination_cidr": None, "allowed": None,
                "peer_namespace": None, "egress_destination_cidr": None,
                "egress_allowed": None, "ports": None,
                "target_labels": None, "peer_labels": None,
            })

    # ── GENERATOR-ONLY baseline ──────────────────────────────────────────────
    if mode == "generator_only":
        iter_start = time.time()
        gen_start = time.time()
        yaml_out = generator.generate(intent)
        gen_time = time.time() - gen_start
        
        # Return a stub report — no validation performed
        stub_critic   = CriticReport(has_errors=False, errors=[], confidence=1.0)
        stub_verifier = VerifierReport(dry_run_passed=True, dry_run_error="skipped")
        iter_time = time.time() - iter_start
        
        history.append(IterationResult(
            iteration=1,
            yaml_policy=yaml_out,
            critic_report=stub_critic,
            verifier_report=stub_verifier,
            elapsed_seconds=iter_time,
            generator_seconds=gen_time,
        ))
        total_time = time.time() - pipeline_start
        return PipelineResult(
            intent=intent,
            final_yaml=yaml_out,
            success=True,          # assumed correct — no validation
            iterations=1,
            iteration_history=history,
            message="Generator-only mode: no validation performed.",
            total_wall_clock_seconds=total_time,
            per_iteration_avg_seconds=total_time,
        )

    # Read _cfg.MAX_ITERATIONS dynamically (not a plain `from ... import
    # MAX_ITERATIONS`) so that batch.py's temporary `cfg.MAX_ITERATIONS =
    # max_iter` override (used to reduce iterations for batch speed) actually
    # takes effect here — a bound-name import would have frozen this at
    # whatever config.py's module-level default was when pipeline.py was
    # first imported, silently ignoring every run's --max-iter flag. Real
    # impact found: 2/1410 rows in a completed run reached 5 iterations
    # despite an intended cap of 3 (stagnation-early-stop masked this in
    # most rows, since it independently stops most non-improving runs by
    # iteration 3 anyway — but not always).
    max_iters = _cfg.MAX_ITERATIONS if mode in ("full", "self_refine") else 1   # no_refine = 1 shot

    best_idx = -1
    best_score = None  # lower is better; None means "no iteration yet"
    stagnant_iterations = 0  # consecutive iterations with no improvement

    # Generate ONCE. Iterations 2+ test the Refiner's own output directly —
    # they do NOT re-invoke the Generator with the refined YAML as a mere
    # "hint". That was the actual reason the loop failed to converge for
    # weaker models: the Refiner's corrected YAML was fed back into another
    # full Generator call, which could (and did) simply ignore the hint and
    # regenerate its own preferred — wrong — structure again, discarding a
    # fix a stronger Refiner had already produced. Now what the Refiner
    # writes is what gets tested next iteration, full stop.
    gen_start = time.time()
    current_yaml = generator.generate(intent)
    initial_gen_time = time.time() - gen_start

    for iteration in range(1, max_iters + 1):
        iter_start = time.time()
        gen_time = initial_gen_time if iteration == 1 else 0.0

        # ── Step 2: Critic (static analysis) ────────────────────────────────
        crit_start = time.time()
        if skip_critic:
            critic_report = CriticReport(
                has_errors=False, errors=[], confidence=1.0,
                raw_response="skipped (Checkpoint 5 no-critic ablation)",
            )
        else:
            critic_report = critic.analyze(intent, current_yaml)
        crit_time = time.time() - crit_start

        # ── Step 3: Verifier (dry-run + runtime) ────────────────────────────
        ver_start = time.time()
        verifier_report = verifier.verify(current_yaml, check_intent)
        ver_time = time.time() - ver_start

        # ── Step 3b: Semantic verifier (deterministic, own engine) ──────────
        # Grades against ground truth, never against the candidate's own
        # shape or an LLM's opinion — see semantic_verifier.py. `check_intent`
        # is the oracle CSV fields normally, or the Extractor's own output
        # when use_extraction=True (see top of run()).
        semantic_report = semantic_verifier.verify_semantics(current_yaml, check_intent)

        iter_time = time.time() - iter_start

        # Record this iteration with timing (refiner_seconds filled in below
        # only if refinement actually runs — no point paying for a Refiner
        # call on an attempt that already passed).
        iter_result = IterationResult(
            iteration=iteration,
            yaml_policy=current_yaml,
            critic_report=critic_report,
            verifier_report=verifier_report,
            semantic_report=semantic_report,
            elapsed_seconds=iter_time,
            generator_seconds=gen_time,
            critic_seconds=crit_time,
            verifier_seconds=ver_time,
            refiner_seconds=0.0,
        )
        history.append(iter_result)

        score = _badness(verifier_report, semantic_report)
        if best_score is None or score < best_score:
            best_score, best_idx = score, iteration - 1  # 0-indexed into history
            stagnant_iterations = 0
        else:
            stagnant_iterations += 1

        if on_iteration:
            on_iteration(iteration, current_yaml, critic_report, verifier_report)

        # ── Decision: done? ──────────────────────────────────────────────────
        # Checked BEFORE refining, so a passing attempt's own (tested) YAML is
        # what gets saved as final_yaml — not an unverified rewrite of it.
        # Gated on the deterministic semantic verifier, NOT the free-form
        # Critic — an LLM's opinion is not a source of truth (it can and does
        # hallucinate objections to policies that are actually correct; see
        # the plan's root-cause #1). Critic still runs and is still logged
        # above as defense-in-depth commentary, it just doesn't block.
        #
        # self_refine is the one exception, by design: it stops when the
        # Critic (the model's OWN self-review) is satisfied, exactly like the
        # Madaan et al. baseline it's modeling — even though that can be
        # wrong. `real_success` below is always graded against the real
        # deterministic checks regardless of mode, so a self-satisfied but
        # actually-incorrect attempt still (correctly) reports success=False.
        real_success = verifier_report.all_tests_passed and semantic_report.matches_intent
        loop_satisfied = (not critic_report.has_errors) if mode == "self_refine" else real_success

        if loop_satisfied:
            total_time = time.time() - pipeline_start
            return PipelineResult(
                intent=intent,
                extracted_intent=extracted_intent,
                final_yaml=current_yaml,
                success=real_success,
                iterations=iteration,
                iteration_history=history,
                message=(
                    "All tests passed. Policy is valid and verified." if real_success else
                    "self_refine stopped: Critic self-review found no errors, but the "
                    "deterministic Verifier/Semantic Verifier disagree — reported as failure."
                ),
                total_wall_clock_seconds=total_time,
                per_iteration_avg_seconds=total_time / iteration,
            )

        # ── Step 4: Refine for next iteration (full / self_refine modes) ─────
        # Give up early if refinement has stopped helping — 2 consecutive
        # iterations with no improvement in score means the model is stuck
        # (repeating the same mistake despite specific feedback, as observed
        # with weaker models on structural YAML errors — see Checkpoint 3's
        # model-breadth writeup). Grinding through the remaining iterations
        # wastes a full LLM+live-cluster round-trip each with near-zero
        # chance of success; best-of-N below still returns whatever the best
        # attempt so far was, so this doesn't change correctness, only cost.
        if stagnant_iterations >= 2:
            break

        if mode in ("full", "self_refine") and iteration < _cfg.MAX_ITERATIONS:
            ref_start = time.time()
            current_yaml = refiner.refine(
                intent, current_yaml, critic_report, verifier_report, semantic_report,
                blind=blind_refiner, critic_only=(mode == "self_refine"),
            )
            iter_result.refiner_seconds = time.time() - ref_start

    # Max iterations reached (or no_refine failed its single shot) — return
    # the BEST-scoring attempt seen across the whole loop, not unconditionally
    # the last one. Without this, the loop can wander away from a good
    # attempt (e.g. chasing bad Refiner feedback) and never come back; the
    # final grade would then reflect wherever it happened to end up rather
    # than the best thing it actually produced.
    best = history[best_idx]
    all_tests_ok = best.verifier_report.all_tests_passed and best.semantic_report.matches_intent
    total_time = time.time() - pipeline_start
    return PipelineResult(
        intent=intent,
        extracted_intent=extracted_intent,
        final_yaml=best.yaml_policy,
        success=all_tests_ok,
        iterations=len(history),
        iteration_history=history,
        message=(
            f"Max iterations reached (best-of-{len(history)}: iteration {best.iteration}). "
            + ("Runtime and semantic checks passed on the best attempt."
               if all_tests_ok else "Some checks still failing on the best attempt.")
        ),
        total_wall_clock_seconds=total_time,
        per_iteration_avg_seconds=total_time / len(history) if history else 0.0,
    )
