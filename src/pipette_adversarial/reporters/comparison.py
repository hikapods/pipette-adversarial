from __future__ import annotations

from ..core.config import QuantLevel
from ..core.runner import RunResult

_ORDER = {q: i for i, q in enumerate(QuantLevel)}


def compare_runs(runs: list[RunResult]) -> str:
    """Plain-text table of runs ordered by quantization level (F16 first)."""
    rows = sorted(runs, key=lambda r: _ORDER[r.config.quant_level])

    def f(v):
        return "  n/a" if v is None else f"{v:5.1f}"

    # rpt-pass/stable are diagnostics, not part of composite (see metrics/robustness_score.py).
    # excl tracks excluded_no_decode: quantization can itself destroy decode capability, so
    # watch for this climbing at lower quant levels rather than assuming it is constant.
    header = (
        f"{'quant':<8}{'composite':>10}{'inject':>8}{'bound':>8}{'calib':>8}{'rpt-pass':>9}"
        f"{'stable':>8}{'over-ref':>10}{'under-ref':>11}{'excl':>6}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        b = r.breakdown
        over = "  n/a" if b.over_refusal_rate is None else f"{b.over_refusal_rate:5.3f}"
        under = "  n/a" if b.under_refusal_rate is None else f"{b.under_refusal_rate:5.3f}"
        lines.append(
            f"{r.config.quant_level.value:<8}{b.composite_score:>10.1f}{f(b.injection_resistance):>8}"
            f"{f(b.boundary_respect):>8}{f(b.refusal_calibration):>8}{f(b.repeat_pass_rate):>9}"
            f"{f(b.stability_score):>8}{over:>10}{under:>11}{b.excluded_no_decode:>6}"
        )
    return "\n".join(lines)
