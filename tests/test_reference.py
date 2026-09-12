import pytest


torch = pytest.importorskip("torch")

from small_q_attention import paged_mtp_attention_reference


def test_reference_matches_manual_single_head_case():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 2, 4, dtype=torch.float32)
    key_cache = torch.randn(2, 2, 1, 4, dtype=torch.float32)
    value_cache = torch.randn(2, 2, 1, 4, dtype=torch.float32)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([4], dtype=torch.int32)

    actual = paged_mtp_attention_reference(
        query, key_cache, value_cache, block_tables, seq_lens, page_size=2, num_kv_heads=1
    )

    keys = key_cache.reshape(4, 1, 4)[:, 0]
    values = value_cache.reshape(4, 1, 4)[:, 0]
    expected = []
    for qi in range(2):
        valid = 4 - 2 + qi + 1
        heads = []
        for qh in range(2):
            scores = torch.matmul(keys[:valid], query[0, qi, qh]) / (4**0.5)
            heads.append(torch.matmul(torch.softmax(scores, dim=0), values[:valid]))
        expected.append(torch.stack(heads))
    expected = torch.stack(expected).unsqueeze(0)
    torch.testing.assert_close(actual, expected)


def test_reference_supports_gqa_grouping():
    torch.manual_seed(1)
    query = torch.randn(1, 2, 4, 8)
    key_cache = torch.randn(2, 4, 2, 8)
    value_cache = torch.randn(2, 4, 2, 8)
    block_tables = torch.tensor([[0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([5], dtype=torch.int32)
    output = paged_mtp_attention_reference(
        query, key_cache, value_cache, block_tables, seq_lens, page_size=4, num_kv_heads=2
    )
    assert output.shape == query.shape
