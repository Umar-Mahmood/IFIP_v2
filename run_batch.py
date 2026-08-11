#!/usr/bin/env python3
"""
Batch runner for the multi-agent Kubernetes NetworkPolicy pipeline.

Usage:
  python3 run_batch.py kubernetes_policies.csv
  python3 run_batch.py kubernetes_policies.csv --limit 10
  python3 run_batch.py kubernetes_policies.csv --mode generator_only  # ablation: no validation
  python3 run_batch.py kubernetes_policies.csv --mode no_refine        # ablation: 1-shot verify
  python3 run_batch.py kubernetes_policies.csv --mode full             # full pipeline (default)
"""

import argparse
from policy_agent.batch import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-process kubernetes_policies.csv through the multi-agent pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv_input",
        nargs="?",
        default="kubernetes_policies.csv",
        help="Input CSV file (policy_type,namespace,destination_cidr,allowed)",
    )
    parser.add_argument(
        "--output", "-o",
        default="smollm2:latest_batch_results.csv",
        help="Output results CSV file path",
    )
    parser.add_argument(
        "--yaml-dir", "-y",
        default="smollm2:latest_verified_policies",
        help="Directory to save verified YAML files",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=0,
        help="Process only first N rows (0 = all rows)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Max pipeline iterations per policy",
    )
    parser.add_argument(
        "--model", "-m",
        default="",
        help="Override all agent models with this Ollama model name (e.g. phi3:mini)",
    )
    parser.add_argument(
        "--generator-model",
        default="",
        help="Override just the Generator's model (heterogeneous-agent ablation, "
             "Checkpoint 5) — applied on top of --model or config.py's defaults",
    )
    parser.add_argument(
        "--critic-model",
        default="",
        help="Override just the Critic's model (heterogeneous-agent ablation)",
    )
    parser.add_argument(
        "--refiner-model",
        default="",
        help="Override just the Refiner's model (heterogeneous-agent ablation) — "
             "e.g. a weak Generator paired with a stronger Refiner",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "generator_only", "no_refine", "self_refine"],
        default="full",
        help="Pipeline mode: full (default), generator_only (no validation), "
             "no_refine (1-shot verify), self_refine (Checkpoint 5 Madaan et "
             "al. baseline — loop stops on the model's own self-review, not "
             "the deterministic verifier; use --model to set one model for "
             "every role)",
    )
    parser.add_argument(
        "--cluster",
        choices=["kindnet", "calico", "cilium"],
        default="kindnet",
        help="Which kind cluster/CNI to verify against (default: kindnet). "
             "Pins the kubectl context for this whole run so it can't drift "
             "to whatever the current context happens to be.",
    )
    parser.add_argument(
        "--use-extraction",
        action="store_true",
        help="Gate pass/fail on the Extractor's own reading of the free-text "
             "intent instead of this CSV's oracle policy_type/destination_cidr/"
             "allowed fields — the production-realistic mode, since a real "
             "deployment has no CSV row to read ground truth from.",
    )
    parser.add_argument(
        "--skip-critic",
        action="store_true",
        help="Checkpoint 5 ablation: stub out the Critic every iteration (no "
             "advisory feedback reaches the Refiner) to test whether its "
             "non-gating notes still help.",
    )
    parser.add_argument(
        "--blind-refiner",
        action="store_true",
        help="Checkpoint 5 ablation: tell the Refiner only that the previous "
             "attempt failed, withholding specific semantic/verifier/critic "
             "detail — tests whether the loop's benefit is the specific "
             "feedback or just extra inference-time attempts.",
    )

    args = parser.parse_args()

    run_batch(
        csv_input  = args.csv_input,
        csv_output = args.output,
        yaml_dir   = args.yaml_dir,
        limit      = args.limit if args.limit > 0 else None,
        max_iter   = args.max_iter,
        mode       = args.mode,
        model      = args.model,
        generator_model = args.generator_model,
        critic_model    = args.critic_model,
        refiner_model   = args.refiner_model,
        cluster    = args.cluster,
        use_extraction = args.use_extraction,
        skip_critic    = args.skip_critic,
        blind_refiner  = args.blind_refiner,
    )


if __name__ == "__main__":
    main()
