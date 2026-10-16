from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..attacks.base import AttackResult
from ..core.config import AttackCategory, BenchmarkConfig, QuantLevel
from ..core.runner import RunResult
from ..metrics.degradation import DegradationReport
from ..metrics.robustness_score import compute_robustness_score

SCHEMA_VERSION = 1


def to_dict(run: RunResult, degradation: DegradationReport | None = None) -> dict:
    cfg = asdict(run.config)
    cfg["quant_level"] = run.config.quant_level.value
    cfg["categories"] = [c.value for c in run.config.categories]
    out = {
        "schema_version": SCHEMA_VERSION,
        "config": cfg,
        "environment": run.environment,
        "breakdown": asdict(run.breakdown),
        "results": [asdict(r) for r in run.results],
        "consistency_results": {
            k: [asdict(r) for r in v] for k, v in run.consistency_results.items()
        },
    }
    if degradation is not None:
        out["degradation"] = asdict(degradation)
    return out


def write_report(run: RunResult, path: str, degradation: DegradationReport | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(to_dict(run, degradation), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_report(path: str) -> RunResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    c = data["config"]
    config = BenchmarkConfig(
        model_path=c["model_path"],
        quant_level=QuantLevel(c["quant_level"]),
        backend=c["backend"],
        dataset_version=c["dataset_version"],
        categories=[AttackCategory(x) for x in c["categories"]],
    )
    results = [AttackResult(**r) for r in data["results"]]
    consistency = {
        k: [AttackResult(**r) for r in v] for k, v in data["consistency_results"].items()
    }
    return RunResult(
        config=config,
        results=results,
        consistency_results=consistency,
        breakdown=compute_robustness_score(results, consistency),
        environment=data.get("environment", {}),
    )
