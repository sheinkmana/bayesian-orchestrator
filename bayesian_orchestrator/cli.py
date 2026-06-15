from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_orchestrator.config import load_config
from bayesian_orchestrator.workflows.bayesian_llm_orchestrator import run_bayesian_llm_orchestrator
from bayesian_orchestrator.workflows.mmlu_bayesian_orchestrator import run_mmlu_bayesian_orchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bayesian-orchestrator",
        description="Run Bayesian LLM orchestration workflows.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run a workflow from a YAML config.")
    run.add_argument("--config", required=True, type=Path, help="Path to workflow config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    mode = config["mode"]

    try:
        if mode == "bayesian_llm_orchestrator":
            result = run_bayesian_llm_orchestrator(config)
        elif mode == "mmlu_bayesian_orchestrator":
            result = run_mmlu_bayesian_orchestrator(config)
        else:
            raise SystemExit(f"Unsupported workflow mode: {mode}")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Report written to: {result.report_path}")
    print(f"Machine-readable summary: {result.summary_path}")


if __name__ == "__main__":
    main()
