from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Generation:
    text: str
    latency_ms: float
    error: str | None = None


class Backend(Protocol):
    name: str

    def generate(
        self,
        system_prompt: str,
        user_input: str,
        *,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> Generation: ...

    def metadata(self) -> dict: ...
