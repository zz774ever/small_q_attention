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

服务器侧（NVIDIA H20，SM90）的三份报告：

- [`docs/h20_baseline_report.md`](docs/h20_baseline_report.md) — FlashInfer prefill / XQA / trtllm-gen 的 36 用例基线，即 issue #3420 与 PR #3859 的 before/after。
- [`docs/h20_v2_report.md`](docs/h20_v2_report.md) — V2（split-KV + warp 分块）与差距归因。
- [`docs/h20_v3_report.md`](docs/h20_v3_report.md) — V3（GQA K/V 复用），在 `q_len=2, KV=8192` 上与 XQA 打平。

## 服务器复现（H20）

服务器上的 FlashInfer 是源码 checkout（`third_party/flashinfer` @ `a69ad808`）并对内核做 JIT 编译，所以有两件事必须先做，否则会分别遇到 `FileNotFoundError: 'ninja'` 和 `cutlass/arch/reg_reconfig.h: No such file`：

```bash
# 1) 激活 conda 环境：非登录 shell 不会自动激活，ninja 在 env 的 bin 目录里
source /usr/local/miniconda3/bin/activate py312

# 2) 初始化 FlashInfer 需要的子模块：XQA 路径不需要 CUTLASS，prefill 路径需要
git -C third_party/flashinfer submodule update --init --recursive \
  3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog
```

然后：

```bash
# 自研 kernel 正确性
python scripts/build_and_test_v1.py
python scripts/build_and_test_v2.py   # 覆盖多 chunk 的 merge 路径
python scripts/build_and_test_v3.py   # 覆盖自动测长与显式 max_seq_len 两条路径

# FlashInfer baseline（36 用例 x 3 后端）
python scripts/run_flashinfer_baseline.py --backends prefill xqa trtllm \
  --q-lens 2 4 8 16 --kv-lens 1024 8192 32768 --batch-sizes 1 4 16 \
  --output results/h20_flashinfer_matrix36.jsonl

# 自研 kernel 与 FlashInfer 的配对表
python scripts/run_gpu_matrix.py --variants v0 v1 v2 v3 \
  --q-lens 2 4 8 --kv-lens 1024 8192 --batch-sizes 1 4 \
  --output results/h20_v0_v1_v2_v3.jsonl
python scripts/compare_backends.py results/h20_v0_v1_v2_v3.jsonl \
  results/h20_flashinfer_matrix36.jsonl
```

三个坑记在这里：

- `run_flashinfer_baseline.py` 在 `q_len > 1` 时必须构造 spec-dec draft mask（脚本内部按上游测试的 `generate_spec_dec_mask` 实现），否则 XQA / trtllm-gen 会抛 `AssertionError: Mask is required for speculative decoding`。
- 本机 `ncu` 被驱动权限挡住（`ERR_NVGPUCTRPERM`），需要 kernel 级时间分解时用 `scripts/profile_v1_v2.py --torch-profiler`。
- 计时统一用 CUDA events；`run_gpu_matrix.py` 会在估算显存超过当前空闲的 70% 时跳过用例并记录 `skipped`，不要把跳过当成通过。

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
