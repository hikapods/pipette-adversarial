from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# expected_behavior values:
#   "refuse"              adversarial request the model must decline
#   "ignore_injection"    legitimate task with an embedded injection; do the task, skip the payload
#   "comply_with_system"  benign look-alike; the model must answer normally (NOT refuse)
EXPECTED = ("refuse", "ignore_injection", "comply_with_system")

# actual_behavior values produced by classifiers:
#   "refused"   declined the request
#   "complied"  followed the injected / adversarial instruction
#   "partial"   both refusal and compliance signals
#   "answered"  produced a normal answer without refusing or following the payload
#   "error"     empty / unusable output
ACTUAL = ("refused", "complied", "partial", "answered", "error", "timeout")


@dataclass
class AdversarialSample:
    id: str
    category: str
    system_prompt: str
    user_input: str
    expected_behavior: str
    attack_technique: str
    severity: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AttackResult:
    sample_id: str
    category: str
    expected_behavior: str
    actual_behavior: str
    model_output: str
    passed: bool
    latency_ms: float
    confidence: float
    severity: str = "medium"
    attack_technique: str = "unknown"


def load_jsonl(path: Path) -> list[AdversarialSample]:
    samples: list[AdversarialSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(AdversarialSample(**json.loads(line)))
    return samples


class AttackStrategy(ABC):
    @abstractmethod
    def load_samples(self, dataset_version: str) -> list[AdversarialSample]: ...

    @abstractmethod
    def classify_response(self, sample: AdversarialSample, model_output: str) -> AttackResult:
        """Deterministic classification. No LLM-as-judge."""
