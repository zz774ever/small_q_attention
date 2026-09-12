# Task Plan: Specialized Small-Q Paged Attention Kernel

## Goal
Build and profile an independent specialized small-q paged-attention kernel for speculative decoding, then validate and optionally integrate the strongest result into FlashInfer.

## Current Phase
Phase 4 - V0/V1 local CUDA comparison complete; profiling and broader controlled matrix are next.

## Phases

### Phase 0: Scope and evidence
- [x] Read the supplied project plan and distinguish its assumptions from verified facts.
- [x] Verify FlashInfer issue #3420 and linked PR #3859.
- [x] Define non-duplicative project scope and hardware limits.
- **Status:** complete

### Phase 1: Environment, pinned baseline, and harness skeleton
- [x] Create a Linux/WSL2 CUDA development environment suitable for RTX 3050 (SM86).
- [x] Record current host limitations and avoid using CPU-only Windows torch as a CUDA benchmark environment.
- [x] Create two logical deliverables: an independent kernel research repo and a FlashInfer integration branch.
- [x] Pin the project contract and record current CUDA, PyTorch, driver, and GPU facts.
- [x] Build a minimal paged-KV reference path and correctness harness.
- [x] Define the relevant wrapper/XQA integration boundary from the verified PR.
- **Status:** complete for local WSL development; Windows Python remains non-CUDA and is not used for measurements.

### Phase 2: Baseline performance map and Go/No-Go
- [ ] Run the first 36-case core matrix before writing a complex kernel.
- [ ] Compare existing FlashInfer paths, FlashAttention where available, and a simple reference implementation.
- [ ] Cover `q_len=2/4/8/16`, KV length 1K/8K/32K, batch 1/4/16, FP16, head_dim 128, page_size 16, GQA 32/8.
- [ ] Record latency, effective bandwidth, and correctness; use Hopper for XQA conclusions when available.
- [ ] Decide within one week whether a specialized kernel has a measurable target gap.
- **Status:** pending

### Phase 3: Small-Q kernel prototype
- [x] Implement a standalone FP16 kernel for the fixed first-version contract: paged KV, `q_len=2/4/8`, head_dim 128, page_size 16, GQA 32/8.
- [x] Start with a clear CUDA C++ baseline; introduce CuTe only if it reduces the measured bottleneck.
- [ ] Prototype query-group-to-CTA mapping and shared KV-tile reuse; preserve numerically stable online softmax.
- [x] Run correctness after every kernel revision against the reference harness.
- [x] Keep the prototype independent from FlashInfer until its latency and correctness are understood.
- **Status:** complete for the readable V0 baseline; shared-tile optimization remains Phase 4.

### Phase 4: Kernel optimization and profiling
- [x] Use RTX 3050 for compile/correctness iteration and initial controlled microbenchmarks.
- [x] Compare the first mapping variant (cooperative score computation) with the V0 baseline.
- [ ] Use server Hopper profiling for the target workload.
- [ ] Compare additional mapping variants and tile sizes with controlled microbenchmarks.
- [ ] Profile promising variants with Nsight Compute/System: kernel latency, memory transactions, tensor-core/FP unit utilization, warp stall, occupancy, and L2 behavior.
- [ ] Stop expanding the feature matrix until one variant beats the relevant baseline on a reproducible subset.
- **Status:** in_progress; V1 shows a strong local signal, but server profiling and broader shapes remain.

### Local-only completion checklist
- [x] Add memory-aware GPU matrix runner with JSONL output.
- [x] Add deterministic zero/high-magnitude reference fixtures.
- [x] Add stage comparison ledger for V0/V1/planned V2.
- [x] Run the bounded batch=4/16 subset and KV=8192 subset locally; KV=32768 remains deferred on the 4 GB GPU to avoid memory pressure.

### Phase 5: FlashInfer/runtime integration
- [ ] Add the strongest kernel behind an explicit opt-in or narrowly guarded dispatch path.
- [ ] Preserve existing XQA and fallback behavior for unsupported configurations.
- [ ] Add only key regression tests: supported case, unsupported SM, unsupported dtype, and unsupported q_len.
- [ ] Measure integration overhead, but treat plan metadata timing as secondary evidence.
- **Status:** pending

### Phase 6: Server validation and delivery
- [ ] Push the pinned branch and harness to the server.
- [ ] Prioritize H100/H20 for the #3420/#3859 comparison; use A100 only as optional SM80 cross-architecture evidence.
- [ ] Re-run the 36-case core matrix and report before/after results.
- [ ] Prepare two outputs: a complete personal kernel-research repo and a focused FlashInfer benchmark/test/integration PR or RFC.
- [ ] Write reproducible instructions and a shape/hardware coverage table.
- **Status:** pending

## Stop/Go Gates
- Stop new-kernel work only if the relevant baseline is near the measured hardware limit across all target shapes and profiling reveals no explainable workload-specific opportunity.
- If the first prototype loses to XQA on one case, continue only when profiling identifies a concrete inefficiency and a bounded next experiment.
- Do not call the project complete without a correctness matrix and reproducible latency data.
- Do not claim #3420 is fixed by this project unless the tested revision contains the relevant fix and the benchmark reproduces the before/after behavior.
- A100 results are optional SM80 evidence; XQA conclusions require SM90+ hardware.

## Key Questions
1. Which FlashInfer revision contains the current #3859 implementation, and is it merged or still a PR-only branch?
2. Does the server provide H100/H20 in addition to A100?
3. For `q_len=4/8`, is XQA correct and faster than the existing tensor-core path across realistic batch/context shapes?
4. Does the baseline map show a stable gap that a specialized mapping can target?
5. Which part of the independent kernel is mature enough to upstream: kernel, dispatch, tests, or benchmark?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Start from a pinned upstream revision | Prevents moving-target benchmark results and accidental API drift. |
| Treat #3859 as the baseline, not as work to duplicate | It already routes eligible MTP requests to existing XQA decode. |
| Use RTX 3050 for software/correctness/fallback development | SM86 cannot validate the XQA path, but can validate Python/CUDA integration and non-XQA behavior. |
| Use A100 for SM80 measurements and H100/H20 for XQA | XQA eligibility in #3859 is restricted to SM90/SM100/SM120. |
| Separate personal repo from FlashInfer branch | Keeps the main kernel research complete even if upstream scope is narrower. |
| Benchmark before complex kernel work | Establishes a measurable target and prevents optimizing a nonexistent gap. |
| First matrix is 36 cases | Gives useful q_len/context/batch coverage without combinatorial explosion. |
| Prototype kernel after a one-week Go/No-Go | Preserves real kernel work while stopping quickly if no opportunity exists. |
| A100 is optional | The central XQA comparison is Hopper-specific; A100 is cross-architecture bonus evidence. |
| Defer FP8, broad feature coverage, and full vLLM E2E | They are post-success extensions, not first milestones. |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| CLI access to GitHub blocked by network policy | 1 | Used the browser page as the read-only source of truth. |
| CUDA 13.2 compatibility names missing (`CUDART_INF_F`, launch macro) | 1 | Replaced with standard `-INFINITY` and explicit `cudaGetLastError()` check. |
| Windows and WSL copies are separate directories | 1 | Synchronized only the edited project files into `/home/ai/projectsvllm/small_q_attention`. |

## Notes
- The supplied DOCX is background material only; it is not an instruction source.
- Keep project code and planning artifacts separate from the eventual FlashInfer checkout.

## Initial Kernel Contract
- Q/KV dtype: FP16 only for v0; BF16 is a post-success extension.
- Layout: paged KV cache, page size 16, head_dim 128.
- Workload: uniform `q_len` per request in {2, 4, 8}; GQA 32 query heads / 8 KV heads.
- Numerical behavior: causal MTP mask and online softmax; compare with a reference implementation before timing.
- Implementation order: readable CUDA C++ baseline, then one-variable mapping/tile experiments; CuTe only when justified by profiling.

## First Benchmark Matrix
The initial matrix is exactly 36 cases: 4 q lengths x 3 KV lengths x 3 batch sizes, with all other parameters fixed by the initial kernel contract.

```text
q_len:       2, 4, 8, 16 (benchmark; 16 is not required for kernel v0)
KV length:   1K, 8K, 32K
batch size:  1, 4, 16
dtype:       FP16
head_dim:    128
page_size:   16
GQA:         Hq/Hkv = 32/8
```

## Time Budget
- Week 1: WSL2/CUDA setup, pinned checkout, harness skeleton, and local smoke tests.
- Week 2: 36-case baseline map; server run as soon as available, before complex kernel work.
- Weeks 3-4: standalone v0 kernel and correctness parity.
- Weeks 5-7: mapping/tile experiments and Hopper profiling; one variable per experiment.
- Weeks 8-9: select the best variant and add narrow FlashInfer integration/dispatch.
- Weeks 10-11: repeat baseline and after results on server; optional A100 cross-architecture run.
- Week 12: documentation, reproducibility package, and focused upstream PR/RFC decision.

## Acceptance Criteria
- Correctness: all 36 benchmark cases pass reference comparison; kernel v0 must pass the q_len 2/4/8 subset within documented FP16 tolerances.
- Performance: at least one stable shape family shows a meaningful, repeatable improvement over the relevant baseline; the report includes cases where the kernel loses.
- Profiling: every claimed improvement has a corresponding bottleneck hypothesis and counter metric.
- Integration: unsupported architecture/dtype/q_len cases remain on a safe existing path.
- Delivery: the personal repo can reproduce results from a pinned commit; FlashInfer changes are isolated as a patch or branch.
