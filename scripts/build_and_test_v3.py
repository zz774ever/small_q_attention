"""Build V3 and compare it with the CPU reference and V1/V2.

V3 keeps V2's split-KV grid and barrier-free KV loop, but one block computes
every query head that shares a kv head, so each K/V element is loaded once
instead of once per query head. The GQA contract is fixed at Hq/Hkv = 32/8, so
the group size is exactly four.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention import paged_mtp_attention_reference  # noqa: E402
from small_q_attention.cuda import forward_v1, forward_v2, forward_v3  # noqa: E402


def run_case(torch, device, q_len, kv_len, chunks=(128, 512), page_size=16, hq=32, hkv=8, dim=128, batch=1):
    num_pages = (kv_len + page_size - 1) // page_size
    torch.manual_seed(99 + q_len + kv_len)
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
    for name, forward in (("v1", forward_v1), ("v2", forward_v2)):
        output = forward(query, key_cache, value_cache, block_tables, seq_lens, page_size, 512)
        torch.cuda.synchronize()
        errors[name] = (output.float() - reference.float()).abs().max().item()

    for chunk_keys in chunks:
        output = forward_v3(
            query, key_cache, value_cache, block_tables, seq_lens, page_size, chunk_keys, kv_len
        )
        torch.cuda.synchronize()
        errors[f"v3_chunk{chunk_keys}"] = (output.float() - reference.float()).abs().max().item()
        # Also cover the path where the launcher measures the longest sequence.
        output_measured = forward_v3(
            query, key_cache, value_cache, block_tables, seq_lens, page_size, chunk_keys, 0
        )
        torch.cuda.synchronize()
        errors[f"v3_chunk{chunk_keys}_autolen"] = (
            (output_measured.float() - reference.float()).abs().max().item()
        )
    return errors


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
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
    print("v3 correctness checks passed")


if __name__ == "__main__":
    main()
