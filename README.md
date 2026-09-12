# Specialized Small-Q Paged Attention

Research project for a specialized paged-KV attention kernel targeting the small query lengths produced by MTP/speculative decoding.

The project has two tracks:

- `small_q_attention`: the independent kernel, reference, benchmark, and profiling work.
- FlashInfer integration: a narrow patch or branch only after the standalone kernel has a correctness and performance result.

## Scope of v0

The first kernel contract is intentionally narrow: FP16, paged KV cache, page size 16, head dimension 128, GQA `Hq/Hkv=32/8`, and uniform `q_len` in `{2, 4, 8}`. `q_len=16` remains in the benchmark matrix as a crossover point, but is not required for the v0 kernel.

The initial benchmark matrix contains 36 cases:

```text
q_len:       2, 4, 8, 16
KV length:   1K, 8K, 32K
batch size:  1, 4, 16
dtype:       FP16
head_dim:    128
page_size:   16
GQA:         Hq/Hkv = 32/8
```

## Current machine

The development machine exposes an RTX 3050 (SM86) and a CUDA 11.3 toolkit. It is suitable for reference correctness, source development, compilation experiments, and non-Hopper paths. The XQA path discussed in FlashInfer issue #3420/#3859 requires SM90 or newer and must be evaluated on a Hopper-or-newer server.

## Layout

```text
configs/core_matrix.json       Fixed 36-case matrix
configs/experiments.json       Kernel variant registry
src/small_q_attention/         Reference and shared project code
tests/                         CPU-runnable contract tests
scripts/                       Environment and benchmark utilities
csrc/                           CUDA kernel implementation (added after baseline map)
```

## Development order

1. Run reference and contract tests locally.
2. Run the same benchmark matrix against available baselines before writing a complex kernel.
3. Implement the v0 kernel for `q_len=2/4/8` only.
4. Profile one variable at a time on the server.
5. Integrate only the strongest result into FlashInfer.

## 阶段报告

本地开发进展、硬件限制、验证状态和下一阶段入口见
[`docs/phase1_report.md`](docs/phase1_report.md)。

V0/V1 的逐阶段假设、数据和决策见 [`docs/stage_comparison.md`](docs/stage_comparison.md)。

## Local checks

The current Windows host can run the dependency-light checks:

```text
python scripts/env_report.py
python scripts/validate_matrix.py
python scripts/smoke_test.py
python scripts/cuda_smoke.py
python scripts/run_reference_benchmark.py
```

The reference benchmark reports CPU execution only. It must not be used as a GPU speedup claim.

`cuda_smoke.py` only verifies CUDA execution and tensor semantics; it is not a performance result.

The first CUDA baseline is `csrc/small_q_attention_ext.cu`. Build and check it with the project venv:

```bash
.venv/bin/python scripts/build_and_test_v0.py
```

This v0 is intentionally readable and redundant in its dot-product work. It is a correctness baseline for later mapping and KV-tile reuse experiments, not the final optimized kernel.

The v0 kernel contract and one-variable experiment list are documented in `docs/kernel_design.md`.

Run a memory-aware local subset and emit JSONL results:

```text
.venv/bin/python scripts/run_gpu_matrix.py --variants v0 v1 --output results/local_subset.jsonl
```

Use `--check-correctness --correctness-limit 3` for a small reference parity sample. The runner skips cases whose estimated KV allocation exceeds 70% of currently free GPU memory.

Raw local samples are kept as `results_local_batch1.jsonl`, `results_local_batch4.jsonl`, and `results_local_batch16.jsonl`; summarize them with `scripts/summarize_results.py`.

## References

- FlashInfer issue #3420: https://github.com/flashinfer-ai/flashinfer/issues/3420
- FlashInfer PR #3859: https://github.com/flashinfer-ai/flashinfer/pull/3859
