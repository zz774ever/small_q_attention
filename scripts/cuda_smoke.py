"""Run the reference implementation on CUDA to validate the WSL GPU path."""

import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_q_attention import paged_mtp_attention_reference  # noqa: E402


def main():
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this Python environment")
    device = torch.device("cuda")
    torch.manual_seed(0)
    batch, q_len, hq, hkv, dim, page_size, kv_len = 1, 2, 4, 2, 8, 4, 5
    query = torch.randn(batch, q_len, hq, dim, device=device)
    key_cache = torch.randn(2, page_size, hkv, dim, device=device)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.tensor([[1, 0]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([kv_len], dtype=torch.int32, device=device)
    output = paged_mtp_attention_reference(
        query, key_cache, value_cache, block_tables, seq_lens, page_size, hkv
    )
    torch.cuda.synchronize()
    assert output.shape == query.shape
    assert torch.isfinite(output).all().item()
    print(f"cuda reference smoke test passed: {torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda} shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
