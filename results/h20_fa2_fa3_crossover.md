# fa2 vs fa3 crossover vs query length (H20, SM90, KV=8192)

Same setup as results/h20_fa2_vs_fa3_backend.md. `BatchPrefillWithPagedKVCacheWrapper(backend=X)`,
causal, `qo_indptr` built with `q_len` rows per request. 3 warmups + 10 samples, p50.

batch=16:

| q_len | fa2 | fa3 | fa3/fa2 |
|---:|---:|---:|---:|
| 2 | 213.0 | 2141.3 | 10.05 |
| 8 | 570.1 | 2142.9 | 3.76 |
| 16 | 570.7 | 2145.1 | 3.76 |
| 32 | 1055.8 | 2144.0 | 2.03 |
| 64 | 2033.7 | 2145.1 | 1.05 |
| 128 | 3483.5 | 2142.9 | 0.62 |
| 256 | 6803.1 | 4180.3 | 0.61 |

batch=1:

| q_len | fa2 | fa3 | fa3/fa2 |
|---:|---:|---:|---:|
| 2 | 52.4 | 337.2 | 6.43 |
| 8 | 73.4 | 341.5 | 4.65 |
| 16 | 75.2 | 338.2 | 4.50 |
| 32 | 100.1 | 340.9 | 3.40 |
| 64 | 169.7 | 339.7 | 2.00 |
| 128 | 314.5 | 338.8 | 1.08 |
| 256 | 583.1 | 342.7 | 0.59 |

Two things stand out:

1. fa3's cost is essentially independent of q_len (2141-2145 us at batch=16,
   337-343 us at batch=1), while fa2 scales linearly. That is the signature of a
   fixed CTA_Q=128 tile being paid for regardless of how many query rows are
   valid, and it confirms the mechanism described in issue #3420.
2. The crossover sits at q_len=128 (batch=1) / q_len=128 (batch=16). Below it
   fa2 never loses; at q_len=64 fa2 is still >= as fast in both batches, which
   is why a `max_q_per_req <= 64` guard (or 32 for margin) is the defensible
   threshold for the backend fix.
