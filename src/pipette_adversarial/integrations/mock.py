from __future__ import annotations

from collections.abc import Callable

from .base import Generation


class MockBackend:
    """Scripted backend for tests and CI. No model required."""

    name = "mock"

    def __init__(
        self,
        responder: Callable[[str, str], str] | None = None,
        default: str = "I'm sorry, I can't help with that.",
    ):
        self._responder = responder
        self._default = default
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt, user_input, *, temperature, seed, max_tokens) -> Generation:
        self.calls.append((system_prompt, user_input))
        text = self._responder(system_prompt, user_input) if self._responder else self._default
        return Generation(text=text, latency_ms=1.0)

    def metadata(self) -> dict:
        return {"backend": "mock"}
