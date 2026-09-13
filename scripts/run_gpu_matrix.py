"""Run a bounded V0/V1 CUDA matrix and emit one JSON result per case.

The runner is intentionally memory-aware for a 4 GB development GPU. Use
``--check-correctness`` only for a small subset because the reference runs on
CPU and copies the case tensors back from CUDA.
"""

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention import paged_mtp_attention_reference  # noqa: E402
from small_q_attention.cuda import (  # noqa: E402
    forward_v0,
    forward_v1,
    forward_v2,
    forward_v3,
    forward_v4,
    forward_v5,
)


def _percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * len(values)))]


def _case_inputs(torch, case, device, page_mode):
    page_size, hq, hkv, dim = 16, 32, 8, 128
    num_pages = (case["kv_len"] + page_size - 1) // page_size
    batch = case["batch_size"]
    query = torch.randn(batch, case["q_len"], hq, dim, device=device, dtype=torch.float16)
    physical_pages = num_pages if page_mode == "shared" else batch * num_pages
    key_cache = torch.randn(physical_pages, page_size, hkv, dim, device=device, dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    if page_mode == "shared":
        page_ids = torch.arange(num_pages - 1, -1, -1, device=device, dtype=torch.int32)
        block_tables = page_ids.repeat(batch, 1).contiguous()
    elif page_mode == "unique":
        page_ids = torch.arange(num_pages - 1, -1, -1, device=device, dtype=torch.int32)
        offsets = torch.arange(batch, device=device, dtype=torch.int32).unsqueeze(1) * num_pages
        block_tables = (page_ids.unsqueeze(0) + offsets).contiguous()
    elif page_mode == "random":
        generator = torch.Generator(device=device)
        generator.manual_seed(7000 + batch * 31 + case["q_len"])
        rows = []
        for request in range(batch):
            perm = torch.randperm(num_pages, generator=generator, device=device, dtype=torch.int32)
            rows.append(perm + request * num_pages)
        block_tables = torch.stack(rows).contiguous()
    else:
        raise ValueError(f"unsupported page_mode: {page_mode}")
    seq_lens = torch.full((batch,), case["kv_len"], device=device, dtype=torch.int32)
    return query, key_cache, value_cache, block_tables, seq_lens


def _measure(torch, forward, tensors, page_size, warmup, repeats):
    for _ in range(warmup):
        forward(*tensors, page_size)
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        forward(*tensors, page_size)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return {
        "p50_us": statistics.median(samples),
        "p99_us": _percentile(samples, 0.99),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def _resolve_forward(variant, case, chunk_keys):
    """Bind the per-variant extras so every variant is callable as forward(*tensors, page_size)."""
    if variant == "v0":
        return forward_v0
    if variant == "v1":
        return forward_v1
    if variant == "v2":
        return lambda *call_args, **kwargs: forward_v2(
            *call_args, chunk_keys=chunk_keys, **kwargs
        )
    if variant == "v3":
        return lambda *call_args, **kwargs: forward_v3(
            *call_args, chunk_keys=chunk_keys, max_seq_len=case["kv_len"], **kwargs
        )
    if variant == "v4":
        return lambda *call_args, **kwargs: forward_v4(
            *call_args, chunk_keys=chunk_keys, max_seq_len=case["kv_len"], **kwargs
        )
    return lambda *call_args, **kwargs: forward_v5(
        *call_args, chunk_keys=chunk_keys, token_group=case["token_group"],
        max_seq_len=case["kv_len"], **kwargs
    )


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", nargs="+", choices=("v0", "v1", "v2", "v3", "v4", "v5"), default=["v0", "v1"]
    )
    parser.add_argument(
        "--chunk-keys", type=int, default=512, help="KV entries owned by one split-KV block"
    )
    parser.add_argument("--token-group", type=int, choices=(2, 4), default=2)
    parser.add_argument("--page-mode", choices=("shared", "unique", "random"), default="shared")
    parser.add_argument("--q-lens", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--kv-lens", nargs="+", type=int, default=[1024, 8192])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--limit", type=int, default=0, help="limit cases per variant; 0 means all")
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--correctness-limit", type=int, default=3)
    parser.add_argument("--output", type=Path, help="optional JSONL output path")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("warmup must be non-negative and repeats positive")

    cases = [
            {"q_len": q, "kv_len": kv, "batch_size": batch, "dtype": "float16", "head_dim": 128, "page_size": 16, "num_query_heads": 32, "num_kv_heads": 8, "token_group": args.token_group}
        for q, kv, batch in itertools.product(args.q_lens, args.kv_lens, args.batch_sizes)
    ]
    device = torch.device("cuda")
    results = []
    correctness_done = 0
    for variant in args.variants:
        for index, case in enumerate(cases):
            if args.limit and index >= args.limit:
                break
            physical_requests = 1 if args.page_mode == "shared" else case["batch_size"]
            estimated = physical_requests * case["kv_len"] * 8 * 128 * 2 * 2
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if estimated > int(free_bytes * 0.70):
                result = {"variant": variant, "case": case, "status": "skipped", "reason": "estimated_memory_limit"}
                print(json.dumps(result, sort_keys=True), flush=True)
                results.append(result)
                continue
            torch.manual_seed(1000 + index)
            tensors = _case_inputs(torch, case, device, args.page_mode)
            forward = _resolve_forward(variant, case, args.chunk_keys)
            timing = _measure(torch, forward, tensors, case["page_size"], args.warmup, args.repeats)
            result = {"variant": variant, "page_mode": args.page_mode, "case": case, "status": "ok", "device": torch.cuda.get_device_name(0), **timing}
            if args.check_correctness and correctness_done < args.correctness_limit:
                query, key_cache, value_cache, block_tables, seq_lens = tensors
                actual = forward(*tensors, case["page_size"])
                reference = paged_mtp_attention_reference(
                    query.float().cpu(), key_cache.float().cpu(), value_cache.float().cpu(),
                    block_tables.cpu(), seq_lens.cpu(), case["page_size"], case["num_kv_heads"],
                ).to(device=device, dtype=torch.float16)
                error = (actual.float() - reference.float()).abs()
                result["max_abs_error"] = error.max().item()
                result["max_rel_error"] = (error / reference.float().abs().clamp_min(1e-5)).max().item()
                correctness_done += 1
            print(json.dumps(result, sort_keys=True), flush=True)
            results.append(result)
            del tensors
            torch.cuda.empty_cache()
    if args.output:
        args.output.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in results), encoding="utf-8")


if __name__ == "__main__":
    main()
