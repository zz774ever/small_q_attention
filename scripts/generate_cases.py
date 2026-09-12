"""Emit the fixed benchmark cases as JSONL for later GPU runners."""

import json
import itertools
from pathlib import Path


def main():
    config = json.loads((Path(__file__).parents[1] / "configs" / "core_matrix.json").read_text(encoding="utf-8"))
    fixed = {k: config[k] for k in ("dtype", "head_dim", "page_size", "num_query_heads", "num_kv_heads")}
    keys = ("q_len", "kv_len", "batch_size")
    for values in itertools.product(*(config[k] for k in keys)):
        case = dict(zip(keys, values))
        case.update(fixed)
        print(json.dumps(case, sort_keys=True))


if __name__ == "__main__":
    main()
