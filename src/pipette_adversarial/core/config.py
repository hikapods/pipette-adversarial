from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AttackCategory(Enum):
    PROMPT_INJECTION = "prompt_injection"
    INSTRUCTION_HIERARCHY = "instruction_hierarchy"
    OUTPUT_MANIPULATION = "output_manipulation"
    REFUSAL_BYPASS = "refusal_bypass"


class QuantLevel(Enum):
    """GGUF quantization names, not generic int8/int4 labels."""

    F16 = "F16"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q4_K_M = "Q4_K_M"
    Q3_K_M = "Q3_K_M"
    Q2_K = "Q2_K"


@dataclass
class BenchmarkConfig:
    model_path: str
    quant_level: QuantLevel
    backend: str = "llama_cpp"  # "llama_cpp" | "mock"
    dataset_version: str = "v1"
    categories: list[AttackCategory] = field(
        default_factory=lambda: [AttackCategory.PROMPT_INJECTION]
    )
    num_consistency_runs: int = 5
    consistency_temperature: float = 0.7  # T=0 consistency is trivial; vary seeds at T>0
    temperature: float = 0.0
    seed: int = 1234
    max_tokens: int = 256
    timeout_seconds: int = 30
    n_gpu_layers: int = -1  # -1 = full offload (Metal/CUDA), 0 = CPU only
    n_threads: int | None = None
    limit: int | None = None  # cap samples per category (smoke runs)
