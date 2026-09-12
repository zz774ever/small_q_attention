"""Summarize paired V0/V1 JSONL results into a compact comparison table."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    grouped = defaultdict(dict)
    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("status") != "ok":
                continue
            case = item["case"]
            key = (case["q_len"], case["kv_len"], case["batch_size"])
            grouped[key][item["variant"]] = item
    print("q_len kv_len batch v0_p50_us v1_p50_us speedup")
    for (q_len, kv_len, batch), variants in sorted(grouped.items()):
        if "v0" not in variants or "v1" not in variants:
            continue
        v0 = variants["v0"]["p50_us"]
        v1 = variants["v1"]["p50_us"]
        print(f"{q_len:5d} {kv_len:7d} {batch:5d} {v0:11.3f} {v1:11.3f} {v0 / v1:7.2f}x")


if __name__ == "__main__":
    main()
