"""Sweep the V2/V3 KV chunk size on representative shapes.

V2's parallelism is `rows * ceil(max_seq_len / chunk_keys)`, so the chunk size
trades launch parallelism against per-block work and the cost of the merge
kernel. This prints p50 latency per chunk size so the choice is measured rather
than assumed.
"""

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention.cuda import forward_v2, forward_v3  # noqa: E402

FIXED = {
    "dtype": "float16",
    "head_dim": 128,
    "page_size": 16,
    "num_query_heads": 32,
    "num_kv_heads": 8,
}


def build_case(torch, case, device):
    page_size = case["page_size"]
    num_pages = (case["kv_len"] + page_size - 1) // page_size
    batch, q_len = case["batch_size"], case["q_len"]
    query = torch.randn(
        batch * q_len, case["num_query_heads"], case["head_dim"],
        dtype=torch.float16, device=device,
    )
    query = query.view(batch, q_len, case["num_query_heads"], case["head_dim"])
    key_cache = torch.randn(
        num_pages, page_size, case["num_kv_heads"], case["head_dim"],
        dtype=torch.float16, device=device,
    )
    value_cache = torch.randn_like(key_cache)
    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    return (
        query,
        key_cache,
        value_cache,
        page_ids.repeat(batch, 1).contiguous(),
        torch.full((batch,), case["kv_len"], dtype=torch.int32, device=device),
    )


def measure(torch, call, warmup, repeats):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return {
        "p50_us": statistics.median(samples),
        "p99_us": samples[min(len(samples) - 1, int(0.99 * len(samples)))],
    }


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--q-lens", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--kv-lens", nargs="+", type=int, default=[8192])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--chunks", nargs="+", type=int, default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--variant", choices=("v2", "v3"), default="v2")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")

    rows = []
    for q_len, kv_len, batch in itertools.product(args.q_lens, args.kv_lens, args.batch_sizes):
        case = {"q_len": q_len, "kv_len": kv_len, "batch_size": batch, **FIXED}
        torch.manual_seed(4)
        tensors = build_case(torch, case, device)
        rows_for_case = []
        for chunk_keys in args.chunks:
            num_chunks = max(1, (kv_len + chunk_keys - 1) // chunk_keys)
            if args.variant == "v2":
                call = lambda ck=chunk_keys: forward_v2(*tensors, case["page_size"], ck)  # noqa: E731
            else:
                call = lambda ck=chunk_keys: forward_v3(  # noqa: E731
                    *tensors, case["page_size"], ck, kv_len
                )
            timing = measure(
                torch,
                call,
                args.warmup,
                args.repeats,
            )
            row = {
                "case": case,
                "variant": args.variant,
                "chunk_keys": chunk_keys,
                "num_chunks": num_chunks,
                "device": torch.cuda.get_device_name(0),
                **timing,
            }
            rows.append(row)
            rows_for_case.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        best = min(rows_for_case, key=lambda row: row["p50_us"])
        print(
            f"# best for q_len={q_len} kv_len={kv_len} batch={batch}: "
            f"chunk={best['chunk_keys']} p50={best['p50_us']:.1f}us",
            flush=True,
        )
        del tensors
        torch.cuda.empty_cache()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        print(f"# wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
