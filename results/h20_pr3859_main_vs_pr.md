# PR #3859 head-to-head: main vs PR branch (H20)

Same probe (`scripts/probe_decode_wrapper_mtp.py`), same shapes, only difference
is the PR diff. `main` = a69ad808, PR branch = db824b9b. H20 (SM90, 97 GB),
driver 595.71.05, CUDA 13.0, torch 2.14.0+cu130, FP16, NHD paged KV,
page_size 16, GQA 32/8, head_dim 128, batch=1, `plan(q_len_per_req=...)` on
`BatchDecodeWithPagedKVCacheWrapper`. 3 rounds x 15 CUDA-event samples per
point, p50 = median of the three round medians.

| q_len | kv_len | main | PR branch | PR vs main |
|---:|---:|---:|---:|---:|
| 1 | 1024 | 44.3 us | 47.0 us | 1.06x (control) |
| 1 | 8192 | 53.5 us | 55.8 us | 1.04x (control) |
| 2 | 1024 | 52.7 us | 72.3 us | 1.37x slower |
| 2 | 8192 | 52.4 us | 85.1 us | 1.62x slower |
| 4 | 1024 | 51.5 us | 75.2 us | 1.46x slower |
| 4 | 8192 | 53.2 us | 87.5 us | 1.64x slower |
| 8 | 1024 | 50.4 us | 70.9 us | 1.41x slower |
| 8 | 8192 | 73.0 us | 83.3 us | 1.14x slower |

Dispatched kernels: main runs `BatchPrefillWithPagedKVCacheKernel` +
`PersistentVariableLengthMergeStatesKernel`; the PR branch runs `kernel_mha`
(the XQA kernel). So the routing change does take effect; it is simply slower
on these shapes. `q_len=1` is unchanged on both, which is the expected control
because the PR only affects `q_len_per_req > 1`.

Caveat: batch=1 only. In a separate comparison XQA looked better at
`q_len=8` with larger batch, so that corner still needs its own measurement.
