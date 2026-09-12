# Kernel V0 Design

## Target contract

The first standalone kernel targets one uniform request batch:

```text
Q:            [B, Q, Hq, D]
Paged K/V:    [num_pages, page_size, Hkv, D]
block_table:  [B, max_pages]
seq_lens:     [B]
Q:            {2, 4, 8}
Hq/Hkv:       32/8
D:            128
page_size:    16
dtype:        FP16
```

`seq_lens[b]` is the total visible sequence length after appending the query block. Query row `qi` can attend to `seq_lens[b] - Q + qi + 1` keys. Physical page lookup is:

```text
page_id = block_table[b, key_index // page_size]
offset  = key_index % page_size
```

## V0 algorithm

V0 is a correctness-first baseline, not the final optimized design:

1. Map one logical `(batch, query_row, query_head)` output to one CUDA block.
2. Derive `kv_head = query_head / (Hq/Hkv)` for GQA.
3. Walk the valid paged K/V rows and compute an online-softmax result.
4. Write one `[D]` output vector.

The implementation must not assume physical pages are contiguous or ordered. It must support a non-contiguous block table from the first test.

## Optimization experiments

Only one variable changes per experiment:

| ID | Variable | Question |
|----|----------|----------|
| M0 | one block per query head | Correctness and baseline cost |
| M1 | one block per `(batch, query_group)` | Can several query heads reuse a KV tile? |
| M2 | one block per `(batch, query_row, query_group)` | Can Q rows share KV loads without register spill? |
| T1 | KV tile 16 vs 32 tokens | Does a larger tile improve memory efficiency? |
| T2 | shared-memory KV staging | Does reuse offset staging/synchronization cost? |
| R1 | register-resident Q fragments | Does Q reuse reduce global traffic without hurting occupancy? |
| S1 | split-KV reduction | Is it useful only for long contexts or larger batches? |

No experiment is accepted from a single timing. Each candidate must pass the q_len 2/4/8 correctness subset and beat the relevant baseline on a repeatable shape family.

## Numerical checks

- Compare FP32 accumulation reference against the FP16 kernel.
- Record max absolute and relative error per case.
- Test causal rows individually, especially the first and last MTP query row.
- Test non-contiguous physical page IDs.
- Include an all-zero and a high-magnitude deterministic fixture before random cases.

## Profiling questions

For every optimization, write one hypothesis before profiling:

```text
Hypothesis: [specific source of waste]
Metric:     [counter or timing that should move]
Result:     [observed movement]
Decision:   [keep, revert, or investigate]
```

The key metrics are kernel latency, global load transactions, achieved occupancy, warp stall reasons, L2 hit rate, and active FP/tensor-core utilization. A speedup without a matching explanation is not considered a finished result.
