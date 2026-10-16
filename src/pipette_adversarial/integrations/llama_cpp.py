from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .base import Generation


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class LlamaCppBackend:
    """llama-cpp-python backend. Uses the chat template embedded in the GGUF.

    Verify per model that the template supports a true system role before trusting
    instruction-hierarchy results (some templates fold system into the user turn).
    """

    name = "llama_cpp"

    def __init__(
        self,
        model_path: str,
        *,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        n_ctx: int = 4096,
        seed: int = 1234,
    ):
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Install llama-cpp-python: pip install 'pipette-adversarial[llama]'"
            ) from e
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_ctx=n_ctx,
            seed=seed,
            verbose=False,
        )
        self._sha: str | None = None

    def generate(self, system_prompt, user_input, *, temperature, seed, max_tokens) -> Generation:
        start = time.perf_counter()
        try:
            out = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            text = out["choices"][0]["message"]["content"] or ""
            return Generation(text=text, latency_ms=(time.perf_counter() - start) * 1000)
        except Exception as e:  # noqa: BLE001
            return Generation(
                text="", latency_ms=(time.perf_counter() - start) * 1000, error=str(e)
            )

    def metadata(self) -> dict:
        if self._sha is None:
            self._sha = _sha256(self.model_path)
        try:
            import llama_cpp

            version = llama_cpp.__version__
        except Exception:  # noqa: BLE001
            version = "unknown"
        return {
            "backend": "llama_cpp",
            "llama_cpp_python": version,
            "model_file": Path(self.model_path).name,
            "model_sha256": self._sha,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
        }
