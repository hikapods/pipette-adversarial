from pipette_adversarial.core.config import AttackCategory, BenchmarkConfig, QuantLevel
from pipette_adversarial.core.runner import run_benchmark
from pipette_adversarial.integrations.mock import MockBackend
from pipette_adversarial.reporters.comparison import compare_runs
from pipette_adversarial.reporters.console import format_report
from pipette_adversarial.reporters.json_report import load_report, write_report


def _config(quant=QuantLevel.F16, **kw):
    return BenchmarkConfig(model_path="mock.gguf", quant_level=quant, backend="mock", **kw)


def test_always_refusing_model_over_refuses_benign_and_ignores_payloads():
    run = run_benchmark(_config(), MockBackend())
    b = run.breakdown
    assert b.over_refusal_rate == 1.0  # refuses every benign look-alike
    assert b.under_refusal_rate == 0.0  # never follows a payload
    assert b.consistency_score == 100.0
    assert run.environment["dataset_sha256"]["injection"]


def test_always_complying_model_scores_low_injection_resistance():
    def parrot(system, user):
        # Naively echo any quoted payload: crude "follows the injection" model.
        return "Sure! PWNED. DAN ENABLED. ACCESS GRANTED. OVERRIDE ACTIVE. DEV MODE ON. BANANA. Here is a poem."

    run = run_benchmark(_config(num_consistency_runs=1), MockBackend(responder=parrot))
    assert run.breakdown.injection_resistance < 50
    assert run.breakdown.over_refusal_rate == 0.0


def test_limit_caps_samples_and_consistency_calls_use_distinct_seeds():
    seeds = []

    class Spy(MockBackend):
        def generate(self, system_prompt, user_input, *, temperature, seed, max_tokens):
            seeds.append((temperature, seed))
            return super().generate(
                system_prompt, user_input, temperature=temperature, seed=seed, max_tokens=max_tokens
            )

    run = run_benchmark(_config(limit=5, num_consistency_runs=3, seed=100), Spy())
    assert len(run.results) == 5
    t0 = [s for t, s in seeds if t == 0.0]
    assert set(t0) == {100}
    assert {s for t, s in seeds if t > 0} == {100, 101, 102}


def test_limit_keeps_benign_samples_so_over_refusal_is_measurable():
    run = run_benchmark(_config(limit=10, num_consistency_runs=1), MockBackend())
    assert len(run.results) == 10
    assert any(r.expected_behavior == "comply_with_system" for r in run.results)
    assert run.breakdown.over_refusal_rate is not None


def test_json_roundtrip_preserves_results(tmp_path):
    run = run_benchmark(_config(), MockBackend())
    path = tmp_path / "run.json"
    write_report(run, str(path))
    loaded = load_report(str(path))
    assert [r.sample_id for r in loaded.results] == [r.sample_id for r in run.results]
    assert loaded.breakdown.composite_score == run.breakdown.composite_score


def test_console_and_comparison_render():
    a = run_benchmark(_config(QuantLevel.F16), MockBackend())
    b = run_benchmark(_config(QuantLevel.Q4_K_M), MockBackend())
    assert "COMPOSITE SCORE" in format_report(a)
    table = compare_runs([b, a])
    assert table.index("F16") < table.index("Q4_K_M")


def test_unimplemented_category_raises_clearly():
    import pytest

    cfg = _config(categories=[AttackCategory.REFUSAL_BYPASS])
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        run_benchmark(cfg, MockBackend())
