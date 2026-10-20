from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ..core.runner import RunResult
from ..metrics.degradation import DegradationReport

WIDTH = 48


def _fmt(v: float | None, digits: int = 1, suffix: str = " / 100") -> str:
    return "n/a" if v is None else f"{v:.{digits}f}{suffix}"


def format_report(run: RunResult, degradation: DegradationReport | None = None) -> str:
    b = run.breakdown
    env = run.environment
    model = Path(run.config.model_path).name
    lo, hi = b.pass_rate_ci

    def row(text: str = "") -> str:
        return f"│ {text:<{WIDTH}} │"

    bar = "─" * (WIDTH + 2)
    lines = [
        f"┌{bar}┐",
        row("PIPETTE ADVERSARIAL ROBUSTNESS BENCHMARK"),
        f"├{bar}┤",
        row(f"Model:    {model}"),
        row(f"Quant:    {run.config.quant_level.value}"),
        row(f"Backend:  {run.config.backend}"),
        row(f"Device:   {env.get('platform', 'unknown')[:36]}"),
        row(f"Dataset:  {run.config.dataset_version} ({b.total_samples} samples)"),
        f"├{bar}┤",
        row(f"COMPOSITE SCORE:  {_fmt(b.composite_score)}"),
        row(f"Pass rate 95% CI: [{lo * 100:.1f}, {hi * 100:.1f}] %"),
        f"├{bar}┤",
        row(f"Injection Resistance:  {_fmt(b.injection_resistance)}"),
        row(f"Boundary Respect:      {_fmt(b.boundary_respect)}"),
        row(f"Refusal Calibration:   {_fmt(b.refusal_calibration)}"),
        row(f"  Over-refusal rate:   {_fmt(b.over_refusal_rate, 3, '')}"),
        row(f"  Under-refusal rate:  {_fmt(b.under_refusal_rate, 3, '')}"),
        f"├{bar}┤",
        row("Diagnostics (not in composite score):"),
        row(f"  Repeat pass rate:    {_fmt(b.repeat_pass_rate)}"),
        row(f"  Stability (raw):     {_fmt(b.stability_score)}"),
    ]
    if b.excluded_no_decode:
        lines.append(row(f"Excluded (no decode capability): {b.excluded_no_decode}"))
    if degradation is not None:
        d = degradation
        lines += [
            f"├{bar}┤",
            row(f"vs {d.baseline_quant} baseline: {d.composite_delta:+.1f} pts (n={d.n_paired})"),
            row(f"McNemar p = {d.mcnemar_p:.3f}  significant: {'yes' if d.significant else 'no'}"),
        ]
    lines.append(f"└{bar}┘")
    return "\n".join(lines)


def print_report(run: RunResult, degradation: DegradationReport | None = None) -> None:
    console = Console()
    console.print(format_report(run, degradation), markup=False, highlight=False)
    if degradation is not None:
        console.print(degradation.interpretation, markup=False, highlight=False)
