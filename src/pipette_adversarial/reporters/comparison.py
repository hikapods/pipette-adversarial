from __future__ import annotations

from ..core.config import QuantLevel
from ..core.runner import RunResult

_ORDER = {q: i for i, q in enumerate(QuantLevel)}


def compare_runs(runs: list[RunResult]) -> str:
    """Plain-text table of runs ordered by quantization level (F16 first)."""
    rows = sorted(runs, key=lambda r: _ORDER[r.config.quant_level])

    def f(v):
        return "  n/a" if v is None else f"{v:5.1f}"

    header = f"{'quant':<8}{'composite':>10}{'inject':>8}{'bound':>8}{'calib':>8}{'consist':>9}{'over-ref':>10}{'under-ref':>11}"
    lines = [header, "-" * len(header)]
    for r in rows:
        b = r.breakdown
        over = "  n/a" if b.over_refusal_rate is None else f"{b.over_refusal_rate:5.3f}"
        under = "  n/a" if b.under_refusal_rate is None else f"{b.under_refusal_rate:5.3f}"
        lines.append(
            f"{r.config.quant_level.value:<8}{b.composite_score:>10.1f}{f(b.injection_resistance):>8}"
            f"{f(b.boundary_respect):>8}{f(b.refusal_calibration):>8}{f(b.consistency_score):>9}"
            f"{over:>10}{under:>11}"
        )
    return "\n".join(lines)
