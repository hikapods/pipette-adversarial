from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field

from ..attacks.base import AdversarialSample, AttackResult
from ..attacks.classify import CLASSIFIER_VERSION
from ..attacks.registry import get_strategy
from ..datasets.loader import dataset_sha256
from ..integrations.base import Backend
from ..metrics.robustness_score import RobustnessBreakdown, compute_robustness_score
from .config import BenchmarkConfig

DATASET_NAMES = {"prompt_injection": "injection"}


@dataclass
class RunResult:
    config: BenchmarkConfig
    results: list[AttackResult]
    consistency_results: dict[str, list[AttackResult]]
    breakdown: RobustnessBreakdown
    environment: dict = field(default_factory=dict)


def environment_info(backend: Backend, config: BenchmarkConfig) -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dataset_version": config.dataset_version,
        "dataset_sha256": {},
        "classifier_version": CLASSIFIER_VERSION,
    }
    info.update(backend.metadata())
    for cat in config.categories:
        name = DATASET_NAMES.get(cat.value)
        if name:
            info["dataset_sha256"][name] = dataset_sha256(name, config.dataset_version)
    return info


def stratified_limit(samples: list[AdversarialSample], limit: int) -> list[AdversarialSample]:
    """Deterministic subset that keeps the benign look-alike share, so smoke runs
    still measure over-refusal. Order within each group follows the dataset."""
    if limit >= len(samples):
        return samples
    benign = [s for s in samples if s.expected_behavior == "comply_with_system"]
    adversarial = [s for s in samples if s.expected_behavior != "comply_with_system"]
    n_benign = min(len(benign), round(limit * len(benign) / len(samples))) if benign else 0
    if benign and n_benign == 0 and limit > 1:
        n_benign = 1
    chosen = adversarial[: limit - n_benign] + benign[:n_benign]
    keep = {s.id for s in chosen}
    return [s for s in samples if s.id in keep]


def create_backend(config: BenchmarkConfig) -> Backend:
    if config.backend == "llama_cpp":
        from ..integrations.llama_cpp import LlamaCppBackend

        return LlamaCppBackend(
            config.model_path,
            n_gpu_layers=config.n_gpu_layers,
            n_threads=config.n_threads,
            seed=config.seed,
        )
    if config.backend == "mock":
        from ..integrations.mock import MockBackend

        return MockBackend()
    raise ValueError(f"Unknown backend '{config.backend}'. Supported: llama_cpp, mock")


def _run_sample(
    strategy,
    backend: Backend,
    sample: AdversarialSample,
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
) -> AttackResult:
    gen = backend.generate(
        sample.system_prompt,
        sample.user_input,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )
    result = strategy.classify_response(sample, gen.text)
    result.latency_ms = gen.latency_ms
    if gen.error:
        result.actual_behavior = "error"
        result.passed = False
        result.confidence = 1.0
    return result


def run_benchmark(config: BenchmarkConfig, backend: Backend | None = None) -> RunResult:
    backend = backend or create_backend(config)
    results: list[AttackResult] = []
    consistency: dict[str, list[AttackResult]] = {}

    for category in config.categories:
        strategy = get_strategy(category)
        samples = strategy.load_samples(config.dataset_version)
        if config.limit:
            samples = stratified_limit(samples, config.limit)

        for sample in samples:
            results.append(
                _run_sample(
                    strategy,
                    backend,
                    sample,
                    temperature=config.temperature,
                    seed=config.seed,
                    max_tokens=config.max_tokens,
                )
            )

        # Consistency: repeat the adversarial subset at T>0 with distinct seeds.
        if config.num_consistency_runs > 1:
            subset = [s for s in samples if s.expected_behavior != "comply_with_system"][:10]
            for sample in subset:
                consistency[sample.id] = [
                    _run_sample(
                        strategy,
                        backend,
                        sample,
                        temperature=config.consistency_temperature,
                        seed=config.seed + i,
                        max_tokens=config.max_tokens,
                    )
                    for i in range(config.num_consistency_runs)
                ]

    breakdown = compute_robustness_score(results, consistency)
    return RunResult(
        config=config,
        results=results,
        consistency_results=consistency,
        breakdown=breakdown,
        environment=environment_info(backend, config),
    )
