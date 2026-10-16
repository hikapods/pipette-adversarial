"""Prompt injection attack strategy (direct override, authority impersonation,
task appending, code-comment injection, base64 encoding, delimiter confusion,
few-shot poisoning, context overflow) plus benign look-alikes for over-refusal."""

from __future__ import annotations

from ..core.config import AttackCategory
from ..datasets.loader import load_dataset
from .base import AdversarialSample, AttackResult, AttackStrategy
from .classify import classify


class PromptInjectionAttack(AttackStrategy):
    CATEGORY = AttackCategory.PROMPT_INJECTION

    def load_samples(self, dataset_version: str) -> list[AdversarialSample]:
        return load_dataset("injection", dataset_version)

    def classify_response(self, sample: AdversarialSample, model_output: str) -> AttackResult:
        return classify(sample, model_output)
