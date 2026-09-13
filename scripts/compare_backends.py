"""Pair standalone-kernel and FlashInfer results into one comparison table.

Reads any number of JSONL files produced by ``scripts/run_gpu_matrix.py``
(``variant`` rows) and ``scripts/run_flashinfer_baseline.py`` (``backend``
rows), joins them on ``(q_len, kv_len, batch_size)`` and prints one row per
case. Ratios are printed as ``x`` multiples of the FlashInfer XQA column,
which is the routing PR #3859 adds and therefore the relevant baseline.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


COLUMNS = ("v0", "v1", "flashinfer_prefill", "flashinfer_xqa", "flashinfer_trtllm")


def latency_p50(item):
    """Accept both result schemas: run_gpu_matrix.py and run_flashinfer_baseline.py."""
    return item.get("latency_us_p50", item.get("p50_us"))


def load(paths):
    grouped = defaultdict(dict)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("status") != "ok":
                continue
            p50 = latency_p50(item)
            if p50 is None:
                continue
            case = item["case"]
            key = (case["q_len"], case["kv_len"], case["batch_size"])
            name = item.get("variant") or item.get("backend")
            grouped[key][name] = p50
    return grouped


def ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return "-"
    return f"{numerator / denominator:.2f}x"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    grouped = load(args.inputs)
    header = ["q_len", "kv_len", "batch"] + list(COLUMNS) + ["xqa/prefill", "v1/xqa"]
    print(" ".join(f"{h:>9}" for h in header))
    for (q_len, kv_len, batch), variants in sorted(grouped.items()):
        values = [variants.get(name) for name in COLUMNS]
        row = [q_len, kv_len, batch] + [
            f"{v:.1f}" if v is not None else "-" for v in values
        ]
        row += [
            ratio(values[2], values[3]),
            ratio(values[1], values[3]),
        ]
        print(" ".join(f"{str(cell):>9}" for cell in row))

    missing = [name for name in COLUMNS if not any(name in v for v in grouped.values())]
    if missing:
        print(f"# columns with no data: {', '.join(missing)}")


if __name__ == "__main__":
    main()
