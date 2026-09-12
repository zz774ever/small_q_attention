from pathlib import Path

from scripts.validate_matrix import load_cases


def test_core_matrix_is_36_cases():
    config, cases = load_cases(Path(__file__).parents[1] / "configs" / "core_matrix.json")
    assert len(cases) == 36
    assert config["kernel_v0_q_len"] == [2, 4, 8]


def test_matrix_contract_is_fixed():
    config, _ = load_cases(Path(__file__).parents[1] / "configs" / "core_matrix.json")
    assert config["dtype"] == "float16"
    assert config["head_dim"] == 128
    assert config["page_size"] == 16
    assert (config["num_query_heads"], config["num_kv_heads"]) == (32, 8)
