from __future__ import annotations

import hashlib
from pathlib import Path

from ..attacks.base import AdversarialSample, load_jsonl

BUILTIN_DIR = Path(__file__).parent / "builtin"


def dataset_path(name: str, version: str) -> Path:
    path = BUILTIN_DIR / f"{name}_{version}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.name} (available: {available()})")
    return path


def load_dataset(name: str, version: str) -> list[AdversarialSample]:
    return load_jsonl(dataset_path(name, version))


def dataset_sha256(name: str, version: str) -> str:
    """Hash of the pinned dataset file; recorded in every result for reproducibility."""
    return hashlib.sha256(dataset_path(name, version).read_bytes()).hexdigest()


def available() -> list[str]:
    return sorted(p.name for p in BUILTIN_DIR.glob("*.jsonl"))
