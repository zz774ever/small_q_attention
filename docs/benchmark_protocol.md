# Benchmark Protocol

Every backend runner consumes one JSON case and emits one JSON result. This keeps CPU reference, FlashInfer, FlashAttention, and the project kernel comparable without coupling the measurement code to one framework.

## Case fields

```json
{
  "q_len": 2,
  "kv_len": 8192,
  "batch_size": 4,
  "dtype": "float16",
  "head_dim": 128,
  "page_size": 16,
  "num_query_heads": 32,
  "num_kv_heads": 8
}
```

`kv_len` is the total sequence length visible to each request after the MTP query tokens are appended. The causal reference therefore exposes `kv_len - q_len + 1`, ..., `kv_len` keys to query rows 0, ..., `q_len-1`.

## Result fields

```json
{
  "backend": "reference",
  "device": "cpu",
  "case": {"...": "..."},
  "latency_us_p50": 123.4,
  "latency_us_p99": 130.2,
  "correct": true,
  "max_abs_error": 0.0,
  "notes": []
}
```

GPU runners must use CUDA events after warmup. Host wall-clock time is acceptable only for the reference or for explicitly labeled setup/metadata measurements. Every result must record the exact backend and revision in the surrounding run manifest.

## Required comparisons

For each server run, compare the relevant existing FlashInfer path, FlashAttention where available, and the standalone kernel. Report both wins and losses; do not average away regressions in unsupported shapes.
