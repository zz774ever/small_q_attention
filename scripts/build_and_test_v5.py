"""Build V5 token-group kernels and check them against the CPU reference."""

import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention import paged_mtp_attention_reference  # noqa: E402
from small_q_attention.cuda import forward_v5  # noqa: E402


def run_case(torch, device, q_len, kv_len, token_group, page_mode="unique", page_size=16,
             hq=32, hkv=8, dim=128, batch=2):
    num_pages = (kv_len + page_size - 1) // page_size
    physical_pages = num_pages * batch
    torch.manual_seed(5000 + q_len * 17 + kv_len + token_group)
    query = torch.randn(batch, q_len, hq, dim, device=device, dtype=torch.float16)
    key_cache = torch.randn(physical_pages, page_size, hkv, dim, device=device, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    if page_mode == "unique":
        page_ids = torch.arange(num_pages - 1, -1, -1, device=device, dtype=torch.int32)
        offsets = torch.arange(batch, device=device, dtype=torch.int32).unsqueeze(1) * num_pages
        block_tables = (page_ids.unsqueeze(0) + offsets).contiguous()
    else:
        raise ValueError(page_mode)
    seq_lens = torch.full((batch,), kv_len, device=device, dtype=torch.int32)
    reference = paged_mtp_attention_reference(
        query.float().cpu(), key_cache.float().cpu(), value_cache.float().cpu(),
        block_tables.cpu(), seq_lens.cpu(), page_size, hkv,
    ).to(device=device, dtype=torch.float16)
    output = forward_v5(
        query, key_cache, value_cache, block_tables, seq_lens, page_size,
        chunk_keys=512, token_group=token_group, max_seq_len=kv_len,
    )
    torch.cuda.synchronize()
    error = (output.float() - reference.float()).abs()
    return error.max().item(), (error / reference.float().abs().clamp_min(1e-5)).max().item()


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")
    failures = []
    for q_len in (2, 4, 8):
        groups = (2,) if q_len == 2 else (2, 4)
        for kv_len in (33, 1024, 8192):
            for token_group in groups:
                max_abs, max_rel = run_case(torch, device, q_len, kv_len, token_group)
                print(
                    f"v5 correctness q_len={q_len} kv_len={kv_len} "
                    f"token_group={token_group} max_abs={max_abs:.6g} max_rel={max_rel:.6g}",
                    flush=True,
                )
                if max_abs > 5e-2:
                    failures.append((q_len, kv_len, token_group, max_abs))
    if failures:
        raise SystemExit(f"V5 correctness failures: {failures}")
    print("v5 correctness checks passed")


if __name__ == "__main__":
    main()
