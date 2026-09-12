"""Validate the fixed benchmark matrix without requiring CUDA or PyTorch."""

import itertools
import json
from pathlib import Path


def load_cases(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    keys = ("q_len", "kv_len", "batch_size")
    cases = [dict(zip(keys, values)) for values in itertools.product(*(config[k] for k in keys))]
    for case in cases:
        case.update({k: config[k] for k in ("dtype", "head_dim", "page_size", "num_query_heads", "num_kv_heads")})
    return config, cases


def main():
    config, cases = load_cases(Path(__file__).parents[1] / "configs" / "core_matrix.json")
    assert len(cases) == 36, len(cases)
    assert set(config["kernel_v0_q_len"]) == {2, 4, 8}
    assert 16 in config["q_len"] and 16 not in config["kernel_v0_q_len"]
    print(f"validated {len(cases)} benchmark cases")
    print(f"kernel v0 q_len={config['kernel_v0_q_len']}")


if __name__ == "__main__":
    main()
