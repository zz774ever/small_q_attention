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
