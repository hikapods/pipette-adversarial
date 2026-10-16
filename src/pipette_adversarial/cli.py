from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence

from .core.config import AttackCategory, BenchmarkConfig, QuantLevel
from .core.runner import run_benchmark
from .metrics.degradation import check_comparable, compute_degradation
from .reporters.console import print_report
from .reporters.json_report import load_report, write_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipette-adversarial",
        description="Quantization-aware adversarial robustness benchmark",
    )
    p.add_argument(
        "--model", required=True, help="Path to a GGUF model (ignored by the mock backend)"
    )
    p.add_argument("--backend", default="llama_cpp", choices=["llama_cpp", "mock"])
    p.add_argument(
        "--quant",
        required=True,
        choices=[q.value for q in QuantLevel],
        help="Quantization level label for this model file",
    )
    p.add_argument("--dataset-version", default="v1")
    p.add_argument(
        "--categories",
        nargs="+",
        default=["prompt_injection"],
        choices=[c.value for c in AttackCategory],
    )
    p.add_argument("--limit", type=int, default=None, help="Cap samples per category (smoke runs)")
    p.add_argument("--consistency-runs", type=int, default=5)
    p.add_argument(
        "--n-gpu-layers", type=int, default=-1, help="-1 = full GPU offload, 0 = CPU only"
    )
    p.add_argument("--n-threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--hardware",
        default=None,
        help="Hardware label for reports (default: auto-detected platform)",
    )
    p.add_argument(
        "--baseline-results",
        default=None,
        help="JSON from an earlier (e.g. F16) run on the same hardware",
    )
    p.add_argument("--output", default=None, help="Write JSON results here")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        model_path=args.model,
        quant_level=QuantLevel(args.quant),
        backend=args.backend,
        dataset_version=args.dataset_version,
        categories=[AttackCategory(c) for c in args.categories],
        num_consistency_runs=args.consistency_runs,
        seed=args.seed,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        limit=args.limit,
    )
    run = run_benchmark(config)

    degradation = None
    if args.baseline_results:
        baseline = load_report(args.baseline_results)
        try:
            for warning in check_comparable(baseline.environment, run.environment):
                print(f"WARNING: {warning}", file=sys.stderr)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        degradation = compute_degradation(
            baseline.breakdown,
            run.breakdown,
            baseline.results,
            run.results,
            baseline.config.quant_level.value,
            config.quant_level.value,
            args.hardware or platform.machine(),
        )

    print_report(run, degradation)
    if args.output:
        write_report(run, args.output, degradation)
        print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
