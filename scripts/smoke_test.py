"""Dependency-light smoke tests for the local development environment."""

import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention import paged_mtp_attention_reference  # noqa: E402


def main():
    import torch

    torch.manual_seed(0)
    query = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    key_cache = torch.randn(2, 4, 2, 8, dtype=torch.float32)
    value_cache = torch.randn(2, 4, 2, 8, dtype=torch.float32)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([5], dtype=torch.int32)
    output = paged_mtp_attention_reference(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        page_size=4,
        num_kv_heads=2,
    )
    assert output.shape == query.shape
    assert torch.isfinite(output).all()

    # Exercise non-contiguous physical page order, which is central to paged KV.
    query = torch.randn(1, 2, 2, 4)
    key_cache = torch.randn(3, 2, 1, 4)
    value_cache = torch.randn(3, 2, 1, 4)
    block_tables = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    reordered = paged_mtp_attention_reference(
        query,
        key_cache,
        value_cache,
        block_tables,
        torch.tensor([5], dtype=torch.int32),
        page_size=2,
        num_kv_heads=1,
    )
    assert reordered.shape == query.shape
    assert torch.isfinite(reordered).all()
    print("reference smoke test passed")
    print(f"torch={torch.__version__} device={query.device} shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
