# Progress Log

## Session: 2026-09-08

### Phase 0: Scope and evidence
- **Status:** complete
- **Started:** 2026-09-08
- Actions taken:
  - Read and extracted the supplied project plan DOCX.
  - Verified issue #3420 in the browser.
  - Inspected PR #3859 description, changed-file list, dispatch conditions, metadata setup, runtime fast path, tests, and hardware limitation.
  - Reframed the project from a new standalone kernel to an evaluation and extension of existing XQA MTP dispatch.
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)

### Plan revision: kernel-first dual-track scope
- **Status:** complete
- Actions taken:
  - Accepted the distinction between the resume project and the upstream contribution.
  - Reordered work to run the baseline performance map before complex kernel implementation.
  - Added a one-week kernel Go/No-Go gate and a standalone FP16 prototype phase.
  - Reduced the first benchmark matrix to 36 cases and made A100 validation optional.
  - Downgraded deep metadata/fallback coverage to integration-supporting work.
- Files created/modified:
  - `task_plan.md` (updated)
  - `findings.md` (updated)
  - `progress.md` (updated)

### Plan revision: v0 q_len boundary
- **Status:** complete
- Actions taken:
  - Kept `q_len=16` in the 36-case benchmark matrix as a crossover point.
  - Restricted kernel v0 correctness and optimization scope to `q_len=2/4/8`.

### Phase 1: Environment and harness skeleton
- **Status:** in_progress
- Actions taken:
  - Confirmed Windows `nvidia-smi` sees an RTX 3050; local toolkit reports CUDA 11.3.
  - Confirmed the available Python 3.8 environment has CPU-only `torch 1.12.0`; pytest is not installed.
  - Created the project README, fixed 36-case matrix, reference implementation, matrix validator, and tests.
  - Added a dependency-light smoke test so local validation does not require installing packages.
- Files created/modified:
  - `README.md`, `.gitignore`, `pyproject.toml`
  - `configs/core_matrix.json`
  - `src/small_q_attention/__init__.py`, `src/small_q_attention/reference.py`
  - `scripts/validate_matrix.py`, `scripts/smoke_test.py`
  - `tests/test_matrix.py`, `tests/test_reference.py`

### Phase 1: Environment assessment
- **Status:** in_progress
- Actions taken:
  - Confirmed the Windows host has RTX 3050 visibility through `nvidia-smi` and CUDA Toolkit 11.3.
  - Confirmed local Python 3.8 uses CPU-only torch 1.12.0 and no Visual Studio C++ compiler was found.
  - Documented WSL2/Linux requirements and added a cross-machine environment report.
- Files created/modified:
  - `docs/local_setup.md`
  - `scripts/env_report.py`

### WSL CUDA confirmation
- **Status:** in_progress
- Actions taken:
  - Created an isolated WSL project at `/home/ai/projectsvllm/small_q_attention`; the existing `/home/ai/projectsvllm/nano-vllm` was not modified.
  - WSL environment report confirms CUDA-enabled PyTorch 2.13.0+cu130 and RTX 3050 SM86.
  - Added a CUDA-only reference smoke test for the WSL GPU path.
- Files created/modified:
  - `scripts/cuda_smoke.py`
  - `README.md`

### Phase 3: CUDA v0 baseline
- **Status:** in_progress
- Actions taken:
  - Confirmed WSL has CUDA-enabled PyTorch 2.13.0+cu130 and installed `ninja` in the isolated project venv.
  - Added a standalone PyTorch CUDA extension with the fixed v0 contract: FP16, paged KV, page size 16, head_dim 128, GQA, q_len 2/4/8.
  - Added a lazy Python loader and a correctness runner comparing CUDA output with the CPU reference.
- Files created/modified:
  - `csrc/small_q_attention_ext.cu`
  - `src/small_q_attention/cuda.py`
  - `scripts/build_and_test_v0.py`
  - `README.md`

### Phase 1: Baseline runner plumbing
- **Status:** in_progress
- Actions taken:
  - Added deterministic JSONL case generation for the 36-case matrix.
  - Added a CPU reference benchmark with a result schema for future CUDA runners.
  - Documented that CPU timings are plumbing checks, not performance claims.
- Files created/modified:
  - `scripts/generate_cases.py`
  - `scripts/run_reference_benchmark.py`
  - `README.md`

### Phase 1: Protocol and edge coverage
- **Status:** complete
- Actions taken:
  - Defined a backend-neutral JSON case/result protocol for future GPU runners.
  - Documented causal `kv_len` semantics and CUDA-event timing requirements.
  - Added reference edge tests for non-contiguous physical page order and invalid metadata.
- Files created/modified:
  - `docs/benchmark_protocol.md`
  - `tests/test_reference_edges.py`

### Phase 1: Local validation checkpoint
- **Status:** complete
- Actions taken:
  - Ran `scripts/env_report.py`, `scripts/smoke_test.py`, `scripts/validate_matrix.py`, and Python `compileall`.
  - Confirmed 36 generated cases and v0 kernel q_len boundary `{2, 4, 8}`.
  - Confirmed current Windows CUDA toolkit/PyTorch mismatch; no GPU benchmark was attempted.
- Results:
  - Environment report: RTX 3050 compute capability 8.6 visible to `nvidia-smi`; nvcc 11.3; torch 1.12.0+cpu.
  - Reference smoke test: PASS.
  - Matrix validation: PASS.
  - Compileall: PASS.

### Phase 1: Kernel design contract
- **Status:** complete
- Actions taken:
  - Documented the v0 tensor contract, paged lookup semantics, causal boundary, numerical checks, profiling questions, and bounded mapping/tile experiments.
  - Added a machine-readable experiment registry.
- Files created/modified:
  - `docs/kernel_design.md`
  - `configs/experiments.json`
  - `README.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Source extraction | `项目.docx` | Full non-empty text | 198 paragraphs extracted | PASS |
| Issue verification | FlashInfer #3420 | Confirm title/body/status | Open issue; qo_len=2 routed to prefill; H20 gap reported | PASS |
| PR verification | FlashInfer #3859 | Confirm current scope | Open PR; XQA routing and tests, no new kernel | PASS |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-09-08 | GitHub CLI/curl network access blocked | 1 | Used browser-only read-only inspection. |
| 2026-09-08 | LibreOffice unavailable while attempting DOCX render | 1 | Continued with text/source review; no DOCX edit or delivery was requested. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 1 in progress: environment, harness, and runner plumbing. |
| Where am I going? | Baseline performance map, v0 kernel prototype, profiling, integration, and delivery. |
| What's the goal? | Build a specialized small-q paged-attention kernel for speculative decoding and optionally upstream the strongest result. |
| What have I learned? | #3859 fixes routing via existing XQA but does not eliminate the need for independent kernel research if a measurable gap exists. |
| What have I done? | Created the project skeleton, reference implementation, fixed benchmark matrix, environment report, and local smoke checks. |

### WSL availability check
- **Status:** agent automation blocked; user confirms WSL is available
- Result: The user's terminal is at `/home/ai/projectsvllm/nano-vllm` under Ubuntu 24.04. The agent's direct `wsl.exe`/UNC probes returned access denied.
- Decision: continue environment validation from the user's existing WSL terminal rather than treating WSL as unavailable.

### V0 build retry and loader hardening
- **Status:** complete
- The first extension build reached `torch.utils.cpp_extension.load` and failed because Ninja was not discoverable when invoking `.venv/bin/python` directly.
- Hardened `src/small_q_attention/cuda.py` to prepend the active interpreter's `bin` directory to `PATH` before loading the extension.
- Extended `scripts/build_and_test_v0.py` to validate all v0 query lengths (`2`, `4`, `8`) in one invocation.
- Host-side Python `compileall` passes.
- After using a clean Linux PATH and syncing the edited files into WSL, nvcc compiled the extension for SM86 and the V0 kernel passed q_len=2/4/8 parity.
- Max absolute errors were `2.44e-4`, `4.88e-4`, and `2.44e-4` respectively.
- CUDA-event baseline at batch=1/KV=1024: p50 `10.254 ms`, `20.246 ms`, `39.353 ms` for q_len=2/4/8; p99 `14.066 ms`, `21.994 ms`, `40.553 ms`.
- Added `docs/phase1_report.md` with the project explanation, current evidence, limitations, and next-stage commands.

### Phase 4: V1 cooperative-score experiment
- **Status:** in progress
- Hypothesis: V0 repeats each Q.K score once per output lane; warp-cooperative score computation should remove that factor-of-128 redundancy.
- V1 keeps the V0 contract and block mapping, but warp 0 computes each score with shuffle reduction and all lanes reuse the shared score for V accumulation.
- Correctness passed for q_len=2/4/8 with a reversed, non-contiguous page table. Max absolute error: `1.22e-4`, `4.88e-4`, `2.44e-4`.
- Paired RTX 3050 CUDA-event benchmark (batch=1, KV=1024, warmup=10, repeats=30):
  - q_len=2: V0 `10.268 ms` -> V1 `2.054 ms` (`5.00x` p50 speedup).
  - q_len=4: V0 `20.370 ms` -> V1 `2.131 ms` (`9.56x` p50 speedup).
  - q_len=8: V0 `39.598 ms` -> V1 `4.436 ms` (`8.93x` p50 speedup).
- Interpretation: strong evidence that repeated score work was the dominant V0 cost on this shape. This is not yet a Hopper/XQA or end-to-end claim.
- Next experiment: repeat V0/V1 over a bounded local subset (`kv_len` 1K/8K, batch 1/4), then profile the winning shape on server hardware before adding shared-memory KV staging.
- Long-context subset completed for batch=1, KV=8192 (warmup=5, repeats=15): V1 p50 speedups were `5.37x` (q_len=2), `11.72x` (q_len=4), and `11.11x` (q_len=8). q_len=4 V1 p99 was noisier (`18.347 ms`) than its p50 (`14.444 ms`), so this point needs repeated sampling.
- An earlier batch=4 measurement attempt was interrupted by approval timeout; it was later rerun successfully through the memory-aware matrix runner.
- Added `scripts/run_gpu_matrix.py` with bounded q/KV/batch controls, free-memory skip logic, CUDA-event p50/p99/min/max, optional reference checks, and JSONL output.
- Completed local paired subset: q_len 2/4/8, KV 1024/8192, batch 1/4/16 where selected. V1 won every completed pair; p50 speedups ranged from `6.37x` to `14.91x`.
- Added `scripts/summarize_results.py` and copied raw result files to `results_local_batch1.jsonl`, `results_local_batch4.jsonl`, and `results_local_batch16.jsonl` in the Windows project root.
- Added deterministic zero/high-magnitude reference fixtures and `docs/stage_comparison.md`.

## Session: 2026-09-13 (H20 FlashInfer baseline)

### Phase 2: FlashInfer baseline map on H20
- **Status:** complete for the prefill / XQA / trtllm-gen comparison
- Actions taken:
  - Added `scripts/run_flashinfer_baseline.py`, a protocol-compatible runner for the three FlashInfer paths that issue #3420 and PR #3859 touch, with optional reference checking.
  - Added `scripts/compare_backends.py` to join kernel results (`p50_us`) and FlashInfer results (`latency_us_p50`) on the same case key.
  - Found the root cause of the earlier `Mask is required for speculative decoding` failure: XQA and trtllm-gen need the packed draft mask `[batch, q_len, div_up(q_len, 32) * 2]` (uint16) whenever `q_len_per_req > 1`. Adapted `generate_spec_dec_mask` from `tests/attention/test_xqa_batch_decode.py`.
  - Fixed two environment blockers: `ninja` was missing from the non-login PATH (conda env `py312` must be activated), and the CUTLASS/CCCL/spdlog submodules were uninitialised so the prefill JIT could not compile.
  - Ran the 12-case paired subset and then the full 36-case matrix.
  - Re-ran the standalone kernel correctness check on H20 and saved it to `reports/h20/small_q_kernel_correctness.txt`.
- Results:
  - All 36 cases and 108 backend calls succeeded, with no failures or skips.
  - XQA is 1.12x-10.51x faster than the prefill routing; the gain grows with KV length (about 10x at KV=32768/batch=1).
  - `q_len=16` also benefits (1.7x-6.4x), so the #3859 gate is not limited to the small `q_len` values in the issue.
  - trtllm-gen is systematically 10-25% slower than XQA on this matrix.
  - XQA/trtllm-gen match the prefill reference within `2.44e-4` (FP16 level).
  - The standalone V1 kernel is 20-200x slower than XQA and the gap widens with KV: `q_len=2/KV=1024/batch=1` 1404.7 us vs 72.1 us, `q_len=8/KV=8192/batch=4` 24844.6 us vs 122.6 us.
  - Standalone kernel correctness on H20: max abs error `1.22e-4` / `2.44e-4` / `4.88e-4` for `q_len=2/4/8`.
- Files created/modified:
  - `scripts/run_flashinfer_baseline.py`, `scripts/compare_backends.py`
  - `results/h20_flashinfer_baseline.jsonl`, `results/h20_flashinfer_matrix36.jsonl`
  - `docs/h20_baseline_report.md`, `docs/stage_comparison.md`, `task_plan.md`, `progress.md`
- Interpretation:
  - The prefill path is still 4.7-10.5x behind XQA at KV>=8K, so a target gap exists;
  - but V1 cannot capture it. The measured bottleneck is structural: one block per `(batch, query_row, query_head)` leaves only 64 blocks at `batch=1/q_len=2`, the KV loop is fully serial per block, each key costs at least two `__syncthreads()`, and there is no tensor-core or tiled-KV reuse.
  - Decision: stop tuning V1; the next variant must be V2 with a different mapping (M1/M2) and shared-memory KV staging (T2).

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| FlashInfer probe | `q_len=2, KV=1024, batch=1` | prefill and XQA both run | prefill 82.0 us, XQA 45.8 us, both correct | PASS |
| Paired subset | 12 cases x 3 backends | no failures | 36 ok rows | PASS |
| Full matrix | 36 cases x 3 backends | no failures | 108 ok rows | PASS |
| XQA correctness | 3 sampled cases vs prefill | FP16-level agreement | max abs error <= 2.44e-4 | PASS |
| Standalone kernel on H20 | `q_len=2/4/8` | match CPU reference | 1.22e-4 / 2.44e-4 / 4.88e-4 | PASS |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-09-13 | `FileNotFoundError: 'ninja'` during prefill JIT | 1 | Activate the conda env (`source /usr/local/miniconda3/bin/activate py312`) before running |
| 2026-09-13 | `cutlass/arch/reg_reconfig.h` / `cute/tensor.hpp` missing | 1 | Initialise the `cutlass`, `cccl`, and `spdlog` submodules |
| 2026-09-13 | `AssertionError: Mask is required for speculative decoding` | 1 | Pass the packed spec-dec draft mask |
| 2026-09-13 | Intermittent SSH connection timeouts while the server was busy | 3 | Retried with a longer `ConnectTimeout` |

## Session: 2026-09-13 (V2 kernel: split-KV + warp tiling)

### Phase 4: mapping change instead of arithmetic tuning
- **Status:** V2 implemented, correct, and measured; M1 identified as the next experiment
- Hypothesis: V1's 20-203x gap to XQA is caused by the mapping, not the arithmetic. Disabling the per-key `__syncthreads()` and giving the KV range its own parallel dimension should recover most of it.
- Implementation:
  - `small_q_attention_v2_partial_kernel`: grid `(rows, num_chunks)`; four warps stride over the block's KV chunk with independent online-softmax state; no barrier inside the KV loop; one barrier per chunk to merge the warps.
  - `small_q_attention_v2_merge_kernel`: combines chunk partials with a log-sum-exp merge; partials are fp32, output stays fp16.
  - Added `forward_v2` to `src/small_q_attention/cuda.py` (with a `chunk_keys` knob), plus `scripts/build_and_test_v2.py`, `scripts/sweep_v2_chunk.py`, `scripts/profile_v1_v2.py`, and `v2` support in `scripts/run_gpu_matrix.py` and `scripts/compare_backends.py`.
- Results:
  - Correctness: `q_len` 2/4/8 x KV 33/1024/8192, reversed page table, chunk 128/512/single all pass with max abs error <= `4.88e-4`; V2 vs V1 differ by <= `2.44e-4`.
  - Paired 12-case benchmark at `chunk_keys=512`: V2 is 5.9x-51x faster than V1 and 1.38x-16.42x behind XQA, versus V1's 19-203x gap. Best cell: `q_len=2, KV=1024, batch=1` at 99.8 us vs XQA 72.1 us.
  - Chunk sweep at KV=8192: small chunks win when the row count is low (177.2 us at chunk 128 for `q_len=2/batch=1`), large chunks win when rows are many (1942.9 us at chunk 1024 for `q_len=8/batch=4`); 256-1024 is flat.
  - `ncu` is unusable on this host (`ERR_NVGPUCTRPERM`), so `torch.profiler` was used: V1 is a single 11.893 ms kernel, V2 is 128.16 us partial plus 8.08 us merge for the same shape.
  - Traffic model for `q_len=2, KV=8192, batch=1`: 268 MB read per call at ~2.1 TB/s (about half of H20's 4.0 TB/s), while reading each kv head once would be 33.5 MB. The remaining gap is GQA redundancy, not parallelism.
- Files created/modified:
  - `csrc/small_q_attention_ext.cu`, `src/small_q_attention/cuda.py`
  - `scripts/build_and_test_v2.py`, `scripts/sweep_v2_chunk.py`, `scripts/profile_v1_v2.py`, `scripts/run_gpu_matrix.py`, `scripts/compare_backends.py`
  - `results/h20_v0_v1_v2.jsonl`, `results/h20_v2_chunk_sweep.jsonl`
  - `reports/h20/torch_profile_v1.txt`, `reports/h20/torch_profile_v2.txt`
  - `docs/h20_v2_report.md`, `docs/stage_comparison.md`, `task_plan.md`, `progress.md`

## Test Results (V2 session)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| V2 correctness | `q_len` 2/4/8 x KV 33/1024/8192, chunk 128/512/single | match CPU reference | max abs error <= 4.88e-4 | PASS |
| V2 merge path | 8, 16, 64 chunks per row | match single-chunk result | agreement <= 3.05e-5 | PASS |
| Paired benchmark | 12 cases x v0/v1/v2 | no failures | 36 ok rows | PASS |
| Chunk sweep | 4 shapes x 5 chunk sizes | monotone trade-off | 256-1024 flat, extremes worse | PASS |
| Kernel decomposition | `q_len=2, KV=8192, batch=1` | identify bottleneck | V1 one 11.893 ms kernel; V2 128.16 us + 8.08 us | PASS |

## Error Log (V2 session)
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-09-13 | `ncu` returns `ERR_NVGPUCTRPERM` | 1 | Driver-level counter restriction; used `torch.profiler` kernel-time decomposition instead |
| 2026-09-13 | Extension rebuild takes three compilations because v0/v1/v2 share one `.cu` | 1 | Accepted; keep one translation unit for reviewability |

## Session: 2026-09-13 (V3 kernel: GQA K/V sharing reaches XQA parity)

### Phase 4: M1 experiment, kernel line frozen
- **Status:** complete; V3 is the delivered kernel
- Hypothesis: V2 is bandwidth-bound because each query head reads the KV again; sharing each K/V slice across the four query heads of one kv head should cut traffic 4x.
- Implementation:
  - `small_q_attention_v3_partial_kernel`: one block per `(batch, query_row, kv_head)`, all four query heads in registers, one K and one V slice loaded per lane per key and reused by all four heads; same barrier-free KV loop and one barrier per chunk as V2.
  - Partial layout is unchanged, so V3 reuses `small_q_attention_v2_merge_kernel`.
  - `small_q_attention_v3` gained a `max_seq_len` argument so callers can avoid a per-call device reduce plus DtoH copy.
  - Added `forward_v3`, `scripts/build_and_test_v3.py`, `v3` support in `scripts/run_gpu_matrix.py` and `scripts/compare_backends.py`, and a `--variant` flag on `scripts/sweep_v2_chunk.py`.
- Results:
  - Correctness: `q_len` 2/4/8 x KV 33/1024/8192, reversed page table, chunk 128/512, both explicit and auto `max_seq_len`: max abs error <= `4.88e-4`, identical to V1/V2.
  - Paired 12-case benchmark at chunk 512: V3 is 1.24x-2.40x faster than V2 and 17x-30x faster than V1; the gap to XQA is 1.12x-6.83x.
  - Chunk sweep at KV=8192: best 82.7 us at `q_len=2, batch=1, chunk=256`, which is 1.03x of XQA's 80.0 us, i.e. parity.
  - Traffic model for `q_len=2, KV=8192, batch=1`: about 268 MB (V2) to about 66.5 MB (V3) per call; estimated achieved bandwidth falls from about 2.1 TB/s to about 0.6 TB/s, so V3 is no longer memory-bound.
- Decision: freeze the kernel line at V3. The residual gap is concentrated in many-row, long-KV shapes and would require a tensor-core/TMA rewrite with uncertain payoff.
- Files created/modified:
  - `csrc/small_q_attention_ext.cu`, `src/small_q_attention/cuda.py`
  - `scripts/build_and_test_v3.py`, `scripts/sweep_v2_chunk.py`, `scripts/run_gpu_matrix.py`, `scripts/compare_backends.py`
  - `results/h20_v0_v1_v2_v3.jsonl`, `results/h20_v3_chunk_sweep.jsonl`, `reports/h20/backend_comparison_table.txt`
  - `docs/h20_v3_report.md`, `docs/stage_comparison.md`, `task_plan.md`, `progress.md`

## Test Results (V3 session)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| V3 correctness | `q_len` 2/4/8 x KV 33/1024/8192, chunk 128/512 | match CPU reference | max abs error <= 4.88e-4 | PASS |
| V3 auto length path | same cases with `max_seq_len=0` | match explicit path | identical errors | PASS |
| Paired benchmark | 12 cases x v0/v1/v2/v3 | no failures | 48 ok rows | PASS |
| V3 chunk sweep | 4 shapes x 5 chunk sizes | best near chunk 256 | 82.7-836.9 us, optimum flat 256-512 | PASS |

## Session: 2026-09-13 (V4: vectorised loads, negative result)

### Phase 4 follow-up: ruling out the memory-access hypothesis
- Hypothesis: V3 loads K and V with four strided two-byte accesses per lane, so giving each lane four consecutive dims and using 8-byte vector loads should improve throughput.
- Implementation: `small_q_attention_v4_partial_kernel` keeps V3's mapping and arithmetic and only changes the per-lane dim layout (`[4*lane, 4*lane+4)`) plus `__half2` loads; `forward_v4` mirrors `forward_v3`; `scripts/build_and_test_v3.py` now checks both kernels.
- Result: correctness identical to V3 (max abs error <= `4.88e-4`, V4 vs V3 <= `1.5e-4`). Performance over the 27-case matrix at chunk 512 is `0.95x`-`1.01x`, mean `0.98x` of V3.
- Interpretation: access efficiency is not the bottleneck. Combined with the V2 result, the residual cost sits in the per-key dependency chain (20 shuffles plus 8 `expf` per key per lane, serialised across keys). Closing the remaining gap would need tensor-core reductions, a much larger change with uncertain payoff.
- Decision: kernel line stays frozen at V3; V4 is kept as the experiment that ruled out the memory-access hypothesis.
- Files: `csrc/small_q_attention_ext.cu`, `src/small_q_attention/cuda.py`, `scripts/run_gpu_matrix.py`, `scripts/build_and_test_v3.py`, `results/h20_v3_v4.jsonl`, `docs/h20_v3_report.md`
