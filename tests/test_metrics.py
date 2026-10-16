import pytest

from pipette_adversarial.metrics.degradation import compute_degradation
from pipette_adversarial.metrics.robustness_score import compute_robustness_score
from pipette_adversarial.metrics.stats import bootstrap_ci, mcnemar_exact

from .conftest import make_result


class TestStats:
    def test_mcnemar_known_values(self):
        assert mcnemar_exact(0, 0) == 1.0
        assert mcnemar_exact(1, 1) == 1.0
        assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
        assert mcnemar_exact(8, 0) == pytest.approx(2 / 256)
        assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)

    def test_bootstrap_is_deterministic_and_brackets_mean(self):
        vals = [1.0] * 30 + [0.0] * 10
        a, b = bootstrap_ci(vals, seed=7), bootstrap_ci(vals, seed=7)
        assert a == b
        assert a[0] < 0.75 < a[1]


def _runs(n_pass_base, n_pass_target, n=40):
    base = [make_result(f"s{i}", passed=i < n_pass_base) for i in range(n)]
    tgt = [make_result(f"s{i}", passed=i < n_pass_target) for i in range(n)]
    return base, tgt


class TestDegradation:
    def test_identical_runs_not_significant(self):
        base, tgt = _runs(30, 30)
        d = compute_degradation(
            compute_robustness_score(base),
            compute_robustness_score(tgt),
            base,
            tgt,
            "F16",
            "Q4_K_M",
            "m4",
        )
        assert d.composite_delta == 0 and d.mcnemar_p == 1.0 and not d.significant
        assert "no statistically detectable change" in d.interpretation

    def test_large_drop_is_significant(self):
        base, tgt = _runs(40, 28)  # 12 samples flip pass -> fail, none the other way
        d = compute_degradation(
            compute_robustness_score(base),
            compute_robustness_score(tgt),
            base,
            tgt,
            "F16",
            "Q2_K",
            "m4",
        )
        assert d.baseline_only_pass == 12 and d.target_only_pass == 0
        assert d.significant and d.pass_rate_delta == pytest.approx(-0.3)
        assert d.pass_rate_delta_ci[1] < 0
        assert "DECREASED" in d.interpretation

    def test_small_noisy_change_is_not_called_significant(self):
        base, tgt = _runs(30, 28)  # 2 flips: well inside noise
        d = compute_degradation(
            compute_robustness_score(base),
            compute_robustness_score(tgt),
            base,
            tgt,
            "F16",
            "Q8_0",
            "m4",
        )
        assert not d.significant

    def test_no_overlap(self):
        base = [make_result("a")]
        tgt = [make_result("b")]
        d = compute_degradation(
            compute_robustness_score(base),
            compute_robustness_score(tgt),
            base,
            tgt,
            "F16",
            "Q4_K_M",
            "m4",
        )
        assert d.n_paired == 0 and "cannot compare" in d.interpretation
