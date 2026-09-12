"""Benchmark the standalone v0 kernel with CUDA events.

This runner intentionally measures only the local prototype. It is not a
FlashInfer/XQA comparison and must not be used to claim an end-to-end speedup.
"""

import argparse
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention.cuda import forward_v0, forward_v1  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", choices=(2, 4, 8), type=int, default=2)
    parser.add_argument("--variant", choices=("v0", "v1"), default="v0")
    parser.add_argument("--kv-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.kv_len <= 0 or args.batch_size <= 0:
        parser.error("--kv-len and --batch-size must be positive")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats positive")
    return args


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.manual_seed(1234)
    page_size, hq, hkv, dim = 16, 32, 8, 128
    num_pages = (args.kv_len + page_size - 1) // page_size
    device = torch.device("cuda")
    query = torch.randn(args.batch_size, args.q_len, hq, dim, device=device, dtype=torch.float16)
    key_cache = torch.randn(num_pages, page_size, hkv, dim, device=device, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.arange(num_pages, device=device, dtype=torch.int32).repeat(args.batch_size, 1)
    seq_lens = torch.full((args.batch_size,), args.kv_len, device=device, dtype=torch.int32)

    forward = forward_v0 if args.variant == "v0" else forward_v1
    for _ in range(args.warmup):
        forward(query, key_cache, value_cache, block_tables, seq_lens, page_size)
    torch.cuda.synchronize()

    samples_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.repeats):
        start.record()
        forward(query, key_cache, value_cache, block_tables, seq_lens, page_size)
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

    samples_us.sort()
    p50 = statistics.median(samples_us)
    p99 = samples_us[min(len(samples_us) - 1, int(0.99 * len(samples_us)))]
    print(
        " ".join(
            (
                f"{args.variant}_cuda_events",
                f"device={torch.cuda.get_device_name(0)!r}",
                f"q_len={args.q_len}",
                f"kv_len={args.kv_len}",
                f"batch_size={args.batch_size}",
                f"warmup={args.warmup}",
                f"repeats={args.repeats}",
                f"p50_us={p50:.3f}",
                f"p99_us={p99:.3f}",
            )
        )
    )


if __name__ == "__main__":
    main()
