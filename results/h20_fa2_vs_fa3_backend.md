# Root cause of issue #3420: prefill backend selection (fa3 vs fa2) on SM90

Measured on H20 (SM90, 97 GB), driver 595.71.05, CUDA 13.0,
torch 2.14.0+cu130, flashinfer `a69ad808`, FP16, NHD paged KV, page_size 16,
GQA 32/8, head_dim 128. `BatchPrefillWithPagedKVCacheWrapper(..., backend=X)`,
causal, qo_indptr built with `q_len` rows per request (the MTP verification
shape). 4 warmups + 12 CUDA-event samples, p50 reported.

`backend="auto"` resolves to fa3 on this hardware: at `q_len=2, KV=8192,
batch=1` auto measures 353.5 us and the profiler shows
`PrefillWithKVCacheKernel<SparseCollectiveMainloop<...>>` - the kernel named in
issue #3420. The fa2 backend uses `BatchPrefillWithPagedKVCacheKernel` with an
adaptive query tile (`q_len=2 -> CTA_TILE_Q=16`, `q_len=8 -> 64`), i.e. 12.5%
tensor-core utilisation instead of 1.6%.

| q_len | kv_len | batch | fa2 | fa3 | fa3/fa2 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 1 | 56.0 | 81.0 | 1.45 |
| 2 | 1024 | 16 | 62.5 | 321.0 | 5.14 |
| 2 | 8192 | 1 | 52.5 | 343.9 | 6.55 |
| 2 | 8192 | 16 | 212.0 | 2145.1 | 10.12 |
| 2 | 32768 | 1 | 90.0 | 1232.4 | 13.69 |
| 2 | 32768 | 16 | 720.3 | 8355.1 | 11.60 |
| 4 | 1024 | 1 | 48.5 | 75.3 | 1.55 |
| 4 | 1024 | 16 | 62.0 | 318.0 | 5.13 |
| 4 | 8192 | 1 | 53.5 | 341.5 | 6.38 |
| 4 | 8192 | 16 | 208.6 | 2142.7 | 10.27 |
| 4 | 32768 | 1 | 87.7 | 1231.5 | 14.05 |
| 4 | 32768 | 16 | 717.8 | 8337.3 | 11.62 |
| 8 | 1024 | 1 | 46.2 | 75.1 | 1.62 |
| 8 | 1024 | 16 | 105.5 | 319.7 | 3.03 |
| 8 | 8192 | 1 | 74.4 | 342.0 | 4.60 |
| 8 | 8192 | 16 | 570.3 | 2147.1 | 3.76 |
| 8 | 32768 | 1 | 159.4 | 1234.3 | 7.74 |
| 8 | 32768 | 16 | 2139.1 | 8342.9 | 3.90 |
| 16 | 1024 | 1 | 46.2 | 75.4 | 1.63 |
| 16 | 1024 | 16 | 103.4 | 315.4 | 3.05 |
| 16 | 8192 | 1 | 75.6 | 336.3 | 4.45 |
| 16 | 8192 | 16 | 566.3 | 2124.4 | 3.75 |
| 16 | 32768 | 1 | 160.3 | 1219.1 | 7.60 |
| 16 | 32768 | 16 | 2118.9 | 8287.5 | 3.91 |

fa3 is slower in all 24 shapes by 1.45x-14.05x, with the gap growing with KV
length and batch size. This reproduces the ~10x figure from #3420 at
`q_len=2` with longer context or larger batch.

Implication: the fix is a backend-selection change, not a new kernel. Force the
fa2 tensor-core backend for MTP-style calls (small query length per request)
instead of letting `auto` resolve to fa3 on SM90. Routing to XQA, as PR #3859
proposes, was measured slower than the fa2 path (63.9-85.1 us vs 51-53 us at
`q_len=2, KV=8192`).

Related: `backend="fa3"` raises `NotImplementedError: q_len_per_req > 1 is
currently only supported on the fa2 tensor-core backend` when used with
`q_len_per_req > 1`, which confirms fa3 is not intended for MTP at all.
