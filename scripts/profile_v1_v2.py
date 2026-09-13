"""Profile one V1 or V2 launch of the same case.

Two modes:

- default: warm up, then issue exactly one further launch, so Nsight Compute can
  select it with `--launch-skip`. V1 issues one kernel per call; V2 issues a
  partial kernel plus a merge kernel, so the skip count differs per variant.
- `--torch-profiler`: print per-kernel CUDA time with `torch.profiler`. This is
  the substitute on hosts where Nsight Compute cannot read performance counters
  (ERR_NVGPUCTRPERM), which is the case on this H20 server.
"""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention.cuda import forward_v1, forward_v2  # noqa: E402


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v1", "v2"), default="v2")
    parser.add_argument("--q-len", type=int, default=2)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-keys", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--torch-profiler", action="store_true")
    parser.add_argument("--profile-repeats", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")
    page_size, hq, hkv, dim = 16, 32, 8, 128
    num_pages = (args.kv_len + page_size - 1) // page_size
    torch.manual_seed(11)
    query = torch.randn(
        args.batch_size, args.q_len, hq, dim, dtype=torch.float16, device=device
    )
    key_cache = torch.randn(
        num_pages, page_size, hkv, dim, dtype=torch.float16, device=device
    )
    value_cache = torch.randn_like(key_cache)
    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    block_tables = page_ids.repeat(args.batch_size, 1).contiguous()
    seq_lens = torch.full((args.batch_size,), args.kv_len, dtype=torch.int32, device=device)
    tensors = (query, key_cache, value_cache, block_tables, seq_lens)

    if args.variant == "v1":
        forward = lambda: forward_v1(*tensors, page_size)  # noqa: E731
    else:
        forward = lambda: forward_v2(*tensors, page_size, args.chunk_keys)  # noqa: E731

    for _ in range(args.warmup):
        forward()
    torch.cuda.synchronize()
    print(f"# warmup done ({args.warmup} launches of {args.variant})", flush=True)

    if args.torch_profiler:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile_repeats):
                forward()
            torch.cuda.synchronize()
        print(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=8),
            flush=True,
        )
        return

    forward()
    torch.cuda.synchronize()
    print(f"# profiled launch of {args.variant} done", flush=True)


if __name__ == "__main__":
    main()
