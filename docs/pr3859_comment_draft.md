# Draft comment for FlashInfer PR #3859

Post as a comment on https://github.com/flashinfer-ai/flashinfer/pull/3859
(and optionally link it from issue #3420).

---

Hi! I have been independently measuring this path on an H20 (SM90) and wanted
to share the numbers, since so far the discussion has no hardware data from the
Hopper target that issue #3420 is about.

**Setup.** `flashinfer` commit `a69ad808` (2026-09-12), H20 (SM90, 97 GB),
driver 595.71.05, CUDA 13.0, torch 2.14.0+cu130, FP16, NHD paged KV,
page_size 16, GQA 32/8, head_dim 128, batch=1/4. Every point is 3 rounds x 15
CUDA-event samples; the number below is the median of the three round medians.
The XQA column calls `xqa_batch_decode_with_kv_cache` directly with the packed
spec-dec draft mask (`[batch, q_len, div_up(q_len, 32) * 2]` uint16).

**First observation.** On `a69ad808`, MTP is still routed to the prefill kernel:
with `BatchDecodeWithPagedKVCacheWrapper(...).plan(..., q_len_per_req=q_len)` the
profiler shows
`BatchPrefillWithPagedKVCacheKernel` + `PersistentVariableLengthMergeStatesKernel`
for q_len = 1, 2, 4 and 8. So the routing this PR adds is indeed not in main.

**Second observation, which is the important one.** That existing route is not
slow. Comparing it against the XQA routing proposed here:

| q_len | kv_len | batch | decode wrapper (today) | raw XQA | XQA vs today |
|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 1 | 43.7 us | 45.7 us | 1.05x slower |
| 2 | 1024 | 4 | 47.2 us | 50.8 us | 1.08x slower |
| 2 | 8192 | 1 | 55.0 us | 63.9 us | 1.16x slower |
| 2 | 8192 | 4 | 83.1 us | 120.6 us | 1.45x slower |
| 4 | 1024 | 1 | 40.1 us | 43.0 us | 1.07x slower |
| 4 | 1024 | 4 | 42.0 us | 48.0 us | 1.14x slower |
| 4 | 8192 | 1 | 46.4 us | 58.9 us | 1.27x slower |
| 4 | 8192 | 4 | 79.4 us | 120.6 us | 1.52x slower |
| 8 | 1024 | 1 | 40.9 us | 42.7 us | 1.04x slower |
| 8 | 1024 | 4 | 53.2 us | 47.9 us | 1.11x faster |
| 8 | 8192 | 1 | 68.2 us | 58.9 us | 1.16x faster |
| 8 | 8192 | 4 | 169.4 us | 119.7 us | 1.41x faster |

Median across the 12 shapes: the XQA routing is **8% slower**. It only wins for
`q_len=8` with larger batch/context (up to 1.41x), and it loses for the
`q_len=2` shapes that the issue is nominally about — up to 1.45x slower.

**What this suggests.** The premise of #3420 ("qo_len=2 dispatches to prefill,
~10x slower than FA on H20") did not reproduce against the route that main
actually uses: that route runs at 40-169 us here, in the same range as XQA. The
10x gap may have come from a different plan configuration, a different
revision, or a comparison against FlashAttention rather than against the
current decode path. Either way, the benefit of this PR looks strongly
shape-dependent rather than a general win, so it may be worth measuring through
the PR's own wrapper path (`q_len_per_req` set during `plan()`) on H20, with
emphasis on `q_len >= 8` / larger batch, before merging. If the win only
materialises there, the gating could be narrowed accordingly, or the PR could
carry a note about the shapes where the tensor-core prefill route is already
faster.

**Caveats.** (1) This compares the current decode-wrapper path against a
*direct* `xqa_batch_decode_with_kv_cache` call, not against this PR's
wrapper-level routing, so it is evidence about which path is faster, not a
verification of the PR's dispatch logic — that still needs to be measured on
the branch itself. (2) Single GPU, single revision. (3) Hardware counters are
unavailable on this host (`ncu` returns `ERR_NVGPUCTRPERM`), so the numbers are
wall-clock CUDA events only. (4) Results are from my own harness; scripts and
raw JSONL are in a personal repo and I am happy to share the exact reproduction
commands or re-run anything you would like measured.

If it is useful, I can also run the same comparison on the PR branch, and/or
extend it to `q_len=16` and KV up to 32K.
