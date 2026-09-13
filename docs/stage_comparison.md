# Stage Comparison Ledger

This file keeps the project decisions and measurements comparable across
iterations. All numbers are measured with CUDA events and should be repeated
on the target server before making an upstream or end-to-end claim.

| Stage | Kernel change | Correctness evidence | Local p50 evidence | Decision |
|---|---|---|---|---|
| V0 | One block per `(batch, q_row, q_head)`; every output lane recomputes Q.K | q_len 2/4/8, FP16 error <= `4.88e-4`; non-contiguous pages | KV=1024, batch=1: 10.268/20.370/39.598 ms | Keep as readable baseline |
| V1 | Warp 0 computes each Q.K once with shuffle reduction; lanes reuse score | q_len 2/4/8, reversed page table; error <= `4.88e-4` | KV=1024, batch=1: 2.054/2.131/4.436 ms | Keep; profile and broaden matrix |
| V1-long | Same V1 mapping at KV=8192 | Same contract; correctness path unchanged | KV=8192, batch=1: 16.828/14.444/29.516 ms | Strong local signal; repeat p99 and batch scaling |
| V2 (planned) | Shared-memory KV tile or query-group reuse | Must pass the same harness | Not measured | Only start after profiling confirms global-load/reuse opportunity |

## Interpretation

V1's improvement is attributable to removing repeated score arithmetic, not to
changing the workload contract. It is therefore a meaningful kernel-stage
comparison, but it does not establish superiority over FlashInfer XQA,
FlashAttention, or a Hopper tensor-core path. Those comparisons require the
server matrix and architecture-specific profiling.

## Server baseline (H20)

The H20 comparison is now measured (`docs/h20_baseline_report.md`). It changes
the stage ledger in one important way: V1's local win over V0 is real, but it
says nothing about competitiveness against the production path.

| Stage | FlashInfer prefill | FlashInfer XQA | p50 speedup of XQA | V1 vs XQA |
|---|---|---|---|---|
| `q_len=2, KV=1024, batch=1` | 80.8 us | 72.1 us | 1.12x | 19.5x slower |
| `q_len=4, KV=8192, batch=1` | 344.7 us | 60.6 us | 5.69x | 196.7x slower |
| `q_len=8, KV=8192, batch=4` | 644.4 us | 122.6 us | 5.26x | 202.6x slower |

Over the full 36-case matrix XQA beats the prefill routing by 1.12x-10.51x,
so the gap that issue #3420 describes is real and is already addressed by the
XQA routing in this revision. The standalone prototype is 20-200x behind XQA,
and the gap grows with KV length, which rules out closing it by tuning the V1
layout. V2 must change parallelism (mapping) and KV reuse, not just arithmetic.

## Local matrix subset

The memory-aware runner completed 12 paired cases on the RTX 3050 (q_len
2/4/8, KV 1024/8192, batch 1/4/16 where memory permitted). The p50 speedup
range was `6.37x` to `14.91x`; every completed pair favored V1. The raw JSONL
files are `results_local_batch1.jsonl`, `results_local_batch4.jsonl`, and
`results_local_batch16.jsonl` in the project root. Re-run the summarizer to
avoid relying on rounded values in this document.

## Reproduction

```bash
.venv/bin/python scripts/build_and_test_v0.py
.venv/bin/python scripts/build_and_test_v1.py
.venv/bin/python scripts/run_gpu_matrix.py --variants v0 v1 --output results/local_subset.jsonl
```
