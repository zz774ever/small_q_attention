"""Run a small CPU reference benchmark to validate runner plumbing.

This is not a GPU performance result. It exists to verify case generation,
result schema, and deterministic output before a CUDA runner is available.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from small_q_attention import paged_mtp_attention_reference  # noqa: E402


def run_case(case, repeats):
    import torch

    batch = case["batch_size"]
    q_len = case["q_len"]
    kv_len = case["kv_len"]
    head_dim = case["head_dim"]
    page_size = case["page_size"]
    hq = case["num_query_heads"]
    hkv = case["num_kv_heads"]
    num_pages = (kv_len + page_size - 1) // page_size
    query = torch.randn(batch, q_len, hq, head_dim)
    key_cache = torch.randn(num_pages, page_size, hkv, head_dim)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.arange(num_pages, dtype=torch.int32).repeat(batch, 1)
    seq_lens = torch.full((batch,), kv_len, dtype=torch.int32)
    output = paged_mtp_attention_reference(
        query, key_cache, value_cache, block_tables, seq_lens, page_size, hkv
    )
    start = time.perf_counter()
    for _ in range(repeats):
        output = paged_mtp_attention_reference(
            query, key_cache, value_cache, block_tables, seq_lens, page_size, hkv
        )
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return {"device": "cpu", "latency_ms": elapsed_ms, "output_shape": list(output.shape)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, help="JSON case file; defaults to a tiny smoke case")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    case = (
        json.loads(args.case.read_text(encoding="utf-8"))
        if args.case
        else {
            "q_len": 2,
            "kv_len": 32,
            "batch_size": 1,
            "head_dim": 8,
            "page_size": 16,
            "num_query_heads": 4,
            "num_kv_heads": 2,
        }
    )
    print(json.dumps({"case": case, "result": run_case(case, args.repeats)}, sort_keys=True))


if __name__ == "__main__":
    main()
