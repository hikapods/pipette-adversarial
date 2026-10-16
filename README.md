# pipette-adversarial

Quantization-aware adversarial robustness benchmark for on-device models.

**Question:** does quantizing a model (F16 to Q8_0 / Q4_K_M / Q3_K_M / Q2_K) change how well it resists prompt injection and instruction-boundary attacks, and does hardware or runtime change that?

## Status: early scaffold

This is a working skeleton, not a finished benchmark. Read this before citing any number from it.

- **Implemented:** prompt-injection category, llama.cpp and mock backends, deterministic classifier, composite score, paired fp16-vs-quant comparison (bootstrap CI + exact McNemar), JSON and console reporters, CLI, tests.
- **Not implemented yet:** instruction-hierarchy, output-manipulation and refusal-bypass categories (declared in `AttackCategory`, runner raises `NotImplementedError`). MLX and ONNX backends are intentionally out of scope until llama.cpp results exist.
- **Dataset is a 26-sample starter slice** (20 adversarial, 6 benign look-alikes). The target is 50+ per category. At this size most deltas will not be statistically distinguishable, and the report says so.
- **The classifier is a heuristic** (refusal phrases plus per-sample payload markers, negation-aware). It has not yet been audited against real model outputs. Do that before trusting scores.
- **Not yet checked against Pipette's real plugin/metric interface.** "Pipette-native" is an assumption until the Pipette repo has been read.

## Install

```bash
pip install -e ".[dev]"          # core + tests (mock backend only)
pip install -e ".[llama,dev]"    # adds llama-cpp-python for real models
pytest
```

## Usage

```bash
# Baseline at F16, then a quantized run paired against it (same hardware):
pipette-adversarial --model models/m-F16.gguf    --quant F16    --output results/m-F16-metal.json
pipette-adversarial --model models/m-Q4_K_M.gguf --quant Q4_K_M \
    --baseline-results results/m-F16-metal.json --output results/m-Q4_K_M-metal.json

# CPU-only second hardware axis on the same machine:
pipette-adversarial --model models/m-F16.gguf --quant F16 --n-gpu-layers 0 --output results/m-F16-cpu.json

# Mock backend (no model needed):
pipette-adversarial --backend mock --model none --quant F16
```

## Reproducibility

- Quantize every level yourself from the same F16 with `llama-quantize` and a fixed imatrix choice. Mixing third-party GGUFs confounds quant level with calibration data.
- Every result JSON records the dataset SHA-256, model SHA-256, llama-cpp-python version, platform, GPU layers and seed.
- Scoring is deterministic string matching. No LLM-as-judge.
- Consistency runs use temperature > 0 with distinct seeds. At temperature 0 on one backend the metric is trivially 100.

## Layout

```
src/pipette_adversarial/
  attacks/        sample schema, classifier, prompt-injection strategy
  core/           config, runner
  datasets/       loader + pinned builtin JSONL (injection_v1)
  integrations/   llama.cpp and mock backends
  metrics/        score, calibration, consistency, degradation, stats
  reporters/      console, JSON, comparison
benchmarks/       thin wrappers: quick_smoke.py, run_full_suite.py
tests/
```

License: Apache-2.0.
