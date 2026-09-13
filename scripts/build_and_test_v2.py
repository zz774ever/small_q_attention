"""Build V2 and compare it with the CPU reference and V0/V1.

V2 keeps the V0 data contract but changes the mapping: each block owns a chunk
of the KV range for one `(batch, query_row, query_head)` output, and four warps
stride over that chunk with independent online-softmax state, so the KV loop has
no block-wide barrier. A second kernel merges the per-chunk partials with a
log-sum-exp combine.

Correctness is checked for `q_len` 2/4/8 over short and long contexts, with a
reversed, non-contiguous page table, and across several chunk sizes so both the
single-chunk and the multi-chunk merge paths are exercised.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention import paged_mtp_attention_reference  # noqa: E402
from small_q_attention.cuda import forward_v0, forward_v1, forward_v2  # noqa: E402


def run_case(torch, device, q_len, kv_len, page_size=16, hq=32, hkv=8, dim=128, batch=1):
    """Return {name: max_abs_error} for one case."""
    num_pages = (kv_len + page_size - 1) // page_size
    query = torch.randn(batch, q_len, hq, dim, device=device, dtype=torch.float16)
    key_cache = torch.randn(num_pages, page_size, hkv, dim, device=device, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    # Non-contiguous physical ordering exercises the paged lookup in every variant.
    block_tables = torch.arange(num_pages - 1, -1, -1, device=device, dtype=torch.int32).repeat(
        batch, 1
    )
    seq_lens = torch.full((batch,), kv_len, device=device, dtype=torch.int32)

    reference = paged_mtp_attention_reference(
        query.float().cpu(),
        key_cache.float().cpu(),
        value_cache.float().cpu(),
        block_tables.cpu(),
        seq_lens.cpu(),
        page_size,
        hkv,
    ).to(device=device, dtype=torch.float16)

    errors = {}
    v1_output = forward_v1(query, key_cache, value_cache, block_tables, seq_lens, page_size)
    for chunk_keys in (128, 512, 1 << 20):
        # 1 << 20 forces a single chunk, i.e. the merge kernel sees one partial.
        output = forward_v2(
            query, key_cache, value_cache, block_tables, seq_lens, page_size, chunk_keys
        )
        torch.cuda.synchronize()
        chunks = max(1, (kv_len + chunk_keys - 1) // chunk_keys)
        errors[f"v2_chunk{chunk_keys}_{chunks}c"] = (
            (output.float() - reference.float()).abs().max().item()
        )
        errors[f"v2_chunk{chunk_keys}_vs_v1"] = (
            (output.float() - v1_output.float()).abs().max().item()
        )

    for name, forward in (("v0", forward_v0), ("v1", forward_v1)):
        output = forward(query, key_cache, value_cache, block_tables, seq_lens, page_size)
        torch.cuda.synchronize()
        errors[name] = (output.float() - reference.float()).abs().max().item()
    return errors


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.manual_seed(2026)
    device = torch.device("cuda")

    failures = []
    for q_len in (2, 4, 8):
        for kv_len in (33, 1024, 8192):
            errors = run_case(torch, device, q_len, kv_len)
            summary = " ".join(f"{name}={value:.6g}" for name, value in errors.items())
            print(f"q_len={q_len} kv_len={kv_len} {summary}", flush=True)
            for name, value in errors.items():
                if value > 5e-2:
                    failures.append((q_len, kv_len, name, value))
    if failures:
        for q_len, kv_len, name, value in failures:
            print(f"FAIL q_len={q_len} kv_len={kv_len} {name}={value:.6g}")
        raise SystemExit(1)
    print("v2 correctness checks passed")


if __name__ == "__main__":
    main()
