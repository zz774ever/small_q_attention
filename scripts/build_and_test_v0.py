"""Build the v0 CUDA extension and compare it with the reference implementation."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention import paged_mtp_attention_reference  # noqa: E402
from small_q_attention.cuda import forward_v0  # noqa: E402


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--q-lens", nargs="+", type=int, default=[2, 4, 8])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    batch, hq, hkv, dim, page_size, kv_len = 1, 32, 8, 128, 16, 33
    for q_len in args.q_lens:
        if q_len not in (2, 4, 8):
            raise SystemExit(f"unsupported v0 q_len={q_len}; choose from 2 4 8")
        num_pages = (kv_len + page_size - 1) // page_size
        query = torch.randn(batch, q_len, hq, dim, device=device, dtype=torch.float16)
        key_cache = torch.randn(num_pages, page_size, hkv, dim, device=device, dtype=torch.float16)
        value_cache = torch.randn_like(key_cache)
        block_tables = torch.arange(num_pages, device=device, dtype=torch.int32).repeat(batch, 1)
        seq_lens = torch.full((batch,), kv_len, device=device, dtype=torch.int32)

        actual = forward_v0(query, key_cache, value_cache, block_tables, seq_lens, page_size, args.verbose_build)
        reference = paged_mtp_attention_reference(
            query.float().cpu(), key_cache.float().cpu(), value_cache.float().cpu(),
            block_tables.cpu(), seq_lens.cpu(), page_size, hkv,
        ).to(device=device, dtype=torch.float16)
        torch.cuda.synchronize()
        max_abs = (actual.float() - reference.float()).abs().max().item()
        max_rel = ((actual.float() - reference.float()).abs() / reference.float().abs().clamp_min(1e-5)).max().item()
        print(f"v0 correctness: q_len={q_len} max_abs_error={max_abs:.6g} max_rel_error={max_rel:.6g}")
        if max_abs > 5e-2 and max_rel > 5e-2:
            raise SystemExit("v0 correctness check failed")
        print(f"device={torch.cuda.get_device_name(0)} shape={tuple(actual.shape)}")


if __name__ == "__main__":
    main()
