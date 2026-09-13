# Independent Assessment: Is V3 the End?

## Short answer

No. V3 is a strong research checkpoint, not a complete performance result.
It demonstrates that the original prototype's main waste was structural: split-KV
parallelism and GQA K/V sharing reduce the gap from tens or hundreds of times to
near parity on one shape. It does not yet demonstrate a production-quality win
over XQA across the target workload.

The right decision is a bounded continuation, not an open-ended rewrite:

1. repair the benchmark workload and make the timing comparable;
2. run one controlled token-group sharing experiment for the shapes where V3 is
   still 3-7x behind XQA;
3. freeze the kernel line if that experiment has no stable gain, then deliver
   the integration/reporting work.

## What is actually established

- V0 -> V1 removed repeated score computation and validated the first mapping
  hypothesis.
- V1 -> V2 added KV parallelism and removed a block-wide barrier per key.
- V2 -> V3 removed the 4-way GQA K/V reload for each query head.
- V4 changed only the per-lane load layout and did not improve V3, so a simple
  coalescing change is not the next answer.
- On the reported H20 subset, V3 reaches about 1.03x XQA at
  `q_len=2, KV=8192, batch=1`, but remains about 3.70x behind at
  `q_len=4, KV=8192, batch=4` and 6.83x behind at
  `q_len=8, KV=8192, batch=4`.

Therefore the claim supported by the current data is “parity on a narrow shape
and a large mapping improvement,” not “a generally competitive replacement for
XQA.”

## Evidence gaps that must be closed before freezing

### 1. Physical-page workload

The current runners use one page-id sequence repeated for every batch request.
That is a valid cache-reuse stress case, but it is not representative of a
normal batched KV cache where requests own different physical pages. It can
change L2 behavior and can make the parity point non-portable.

The benchmark must report both:

- `shared_pages`: the existing reproducible result;
- `unique_pages` or a deterministic random page permutation per request.

At minimum, repeat `q_len=2/4/8`, `KV=8192`, `batch=1/4` for V3 and XQA.

### 2. Launch and scratch behavior

V3 allocates FP32 partial buffers and launches a separate merge kernel on every
call. The FlashInfer runners keep their wrapper/workspace outside the timed
call. CUDA-event timings are still useful, but they do not establish that the
same workspace and allocator behavior would hold in an integrated runtime.

Measure partial and merge kernels separately, and add a persistent-scratch
path before making an end-to-end claim.

### 3. Bottleneck attribution

The traffic model and V4 negative result narrow the hypothesis, but they do not
prove that the remaining cost is exclusively the shuffle/`expf` dependency
chain. Register count, occupancy, and instruction mix are still missing because
Nsight Compute counters are unavailable on this H20.

The next low-cost evidence is compiler resource output (`ptxas -v` or equivalent)
and kernel-time decomposition. A tensor-core/TMA rewrite should not start until
this evidence shows that it is justified.

## Highest-value next kernel experiment

The V5 prototype now implements token-group sharing, beginning with
`token_group=2`:

```text
block = (batch, query-token group, kv head, KV chunk)
```

The current V3 block reuses K/V across the four query heads belonging to one KV
head, but it still scans the same KV chunk once for every query row. Grouping
two query rows can reuse the K/V load across rows and directly targets the
remaining `q_len=4/8`, long-KV, multi-request gap.

Keep the experiment narrow: FP16, head_dim=128, page_size=16, Hq/Hkv=32/8,
`KV=8192`, `q_len=4/8`, `batch=1/4`, and unique pages. Require correctness
against the existing reference before timing. Test `token_group=4` only if
the two-row version is correct and faster.

The H20 validation is still pending. This experiment can lose. More query rows per block increase register state,
softmax state, and occupancy pressure. A negative result is useful because it
would show that cross-token reuse is not worth the register cost for this
shape.

## Low-risk engineering opportunities

These are worthwhile polish, but they are not the main research contribution:

- fuse or bypass the merge path when there is only one KV chunk;
- choose `chunk_keys` from a small measured policy instead of one fixed value;
- reuse persistent partial/output workspace in an integration wrapper;
- specialize page size 16 so page-slot arithmetic is not a generic runtime
  division, and optionally stage page ids per chunk;
- add a narrow dispatch/fallback path and regression tests for unsupported
  architecture, dtype, and query length.

## Stop condition

Freeze the kernel line after the unique-page validation and one token-group
experiment if:

- V3's parity result is not stable under unique pages, or
- token grouping is not faster on the target long-KV multi-row cases, or
- the remaining gap requires tensor-core/TMA work without a measurable,
  explainable opportunity from resource data.

At that point the project is still resume-worthy: it contains a pinned baseline,
five kernel variants, correctness evidence, positive and negative experiments,
and an honest performance envelope. The remaining deliverable should then be
the reproducible report and a narrow FlashInfer benchmark/integration patch,
not an unsupported claim that #3420 was independently fixed.
