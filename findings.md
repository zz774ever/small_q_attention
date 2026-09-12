# Findings: FlashInfer MTP Small-Q Dispatch

## Requirements
- Develop first on the user's RTX 3050.
- Push the work later to a server for GPU validation.
- Keep the project non-redundant with FlashInfer PR #3859.
- Produce a credible AI infrastructure project suitable for a resume and possible upstream contribution.

## Research Findings
- FlashInfer issue #3420 is an open performance issue for MTP/speculative decoding with `qo_len=2`.
- The issue reports dispatch to `PrefillWithKVCacheKernel` instead of a decode kernel and roughly 10x higher attention latency than FlashAttention on H20 for the reported shape.
- The issue explains the utilization problem: the prefill path uses a fixed `CTA_Q=128` arrangement; with two valid query rows, most tensor-core work is padding.
- PR #3859 is open and changes three files with approximately 407 additions and 20 deletions.
- PR #3859 does not introduce a new CUDA kernel. It routes eligible multi-token requests through the existing XQA batch decode path.
- New eligibility gating includes: `q_len_per_req > 1`; SM90/SM100/SM120; Q FP16/BF16; KV FP16/BF16/FP8 E4M3; head dimension 16..256 divisible by 16; page size 16/32/64/128; backend auto/fa2/fa3; no RoPE, soft cap, fixed split-kv, JIT module, LSE, or NVFP4 block-scale request.
- `plan()` now prepares causal masks, KV-length buffers, block tables, and max KV length for XQA MTP. `q_len_per_req` must be set during `plan()` to enable the fast path.
- `run()` invokes `xqa_batch_decode_with_kv_cache` when the planned metadata and runtime conditions match, otherwise preserving existing paths.
- Tests added/updated cover XQA support gating, causal-mask/block-table helpers, and CUDA correctness for MTP decode. The PR reports XQA CUDA tests skipped on RTX 3090/SM86 because XQA requires SM90+.
- PR #3859 therefore addresses the main #3420 routing issue, but does not establish broad q_len 2/4/8 performance coverage, cross-architecture behavior, or full vLLM end-to-end evidence.
- A kernel project remains justified only if the baseline map identifies a stable gap or an unserved workload shape; the first prototype is intentionally independent of FlashInfer integration.
- The current Windows host sees an RTX 3050 and CUDA Toolkit 11.3, but the available Python 3.8 torch is CPU-only and no Visual Studio C++ compiler was found. Local work must use WSL2/Linux or remain at reference/source level.
- The user has an Ubuntu 24.04 WSL terminal at `/home/ai/projectsvllm/nano-vllm`. The earlier `CreateInstance/E_ACCESSDENIED` came from this agent's restricted PowerShell automation, not proof that the user's WSL environment is unavailable.

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Baseline against #3859 before writing new kernels | Avoids duplicating an existing upstream solution. |
| Build a correctness/dispatch harness first | It reveals actual gaps before implementation and is testable on SM86. |
| Treat q_len=4/8 and fallback behavior as primary research targets | They are implied by the broader project idea but not demonstrated by the PR's main scope. |
| Separate A100 and Hopper conclusions | A100 is SM80; #3859's XQA gate is SM90/SM100/SM120. |
| Maintain two deliverables | The independent kernel repo is the resume project; FlashInfer changes are an integration/open-source contribution. |
| Move baseline performance mapping before kernel implementation | A one-week baseline establishes whether a kernel opportunity exists. |
| Use a 36-case first matrix | `q_len=2/4/8/16` x KV 1K/8K/32K x batch 1/4/16, with FP16/head_dim128/page16/GQA32/8. |
| Make metadata overhead and deep fallback coverage secondary | They support integration quality but are not the main kernel research contribution. |
| Keep q_len=16 in benchmark only | It identifies the XQA/prefill crossover without forcing v0 kernel templates and mappings to generalize prematurely. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| Direct PowerShell/curl requests to GitHub were blocked | Read the PR and changes through the in-app browser. |
| The current workspace is empty and not a Git checkout | Plan to clone/pin FlashInfer as an explicit Phase 1 action; do not assume the workspace itself is the repo. |

## Resources
- FlashInfer issue #3420: https://github.com/flashinfer-ai/flashinfer/issues/3420
- FlashInfer PR #3859: https://github.com/flashinfer-ai/flashinfer/pull/3859
- PR changes view: https://github.com/flashinfer-ai/flashinfer/pull/3859/changes
- User project document: C:\Users\14018\Desktop\infra_learn\项目.docx

## Visual/Browser Findings
- PR header shows status Open, 2 commits, 3 changed files, and approximately `+407 -20`.
- PR description explicitly says it closes #3420 and keeps unsupported variants on the existing tensor-core prefill behavior.
- PR validation notes report two local tests passed and one skipped on an RTX 3090/SM86 environment; full tests were not all run due to an unrelated missing template in that checkout.
