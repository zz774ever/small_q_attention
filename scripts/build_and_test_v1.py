"""Build V1 and compare it with the CPU reference and V0."""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention import paged_mtp_attention_reference  # noqa: E402
from small_q_attention.cuda import forward_v0, forward_v1  # noqa: E402


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.manual_seed(4321)
    device = torch.device("cuda")
    batch, hq, hkv, dim, page_size, kv_len = 1, 32, 8, 128, 16, 33
    for q_len in (2, 4, 8):
        num_pages = (kv_len + page_size - 1) // page_size
        query = torch.randn(batch, q_len, hq, dim, device=device, dtype=torch.float16)
        key_cache = torch.randn(num_pages, page_size, hkv, dim, device=device, dtype=torch.float16)
        value_cache = torch.randn_like(key_cache)
        # Non-contiguous physical ordering exercises paged lookup in both variants.
        block_tables = torch.arange(num_pages - 1, -1, -1, device=device, dtype=torch.int32).repeat(batch, 1)
        seq_lens = torch.full((batch,), kv_len, device=device, dtype=torch.int32)
        reference = paged_mtp_attention_reference(
            query.float().cpu(), key_cache.float().cpu(), value_cache.float().cpu(),
            block_tables.cpu(), seq_lens.cpu(), page_size, hkv,
        ).to(device=device, dtype=torch.float16)
        actual_v0 = forward_v0(query, key_cache, value_cache, block_tables, seq_lens, page_size)
        actual_v1 = forward_v1(query, key_cache, value_cache, block_tables, seq_lens, page_size)
        torch.cuda.synchronize()
        err_v0 = (actual_v0.float() - reference.float()).abs().max().item()
        err_v1 = (actual_v1.float() - reference.float()).abs().max().item()
        diff = (actual_v1.float() - actual_v0.float()).abs().max().item()
        print(f"v1 correctness q_len={q_len} v0_max_abs={err_v0:.6g} v1_max_abs={err_v1:.6g} v1_vs_v0={diff:.6g}")
        if err_v1 > 5e-2:
            raise SystemExit("v1 correctness check failed")


if __name__ == "__main__":
    main()
