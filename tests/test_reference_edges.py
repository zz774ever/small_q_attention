import pytest


torch = pytest.importorskip("torch")

from small_q_attention import paged_mtp_attention_reference


def _inputs():
    query = torch.randn(1, 2, 2, 4)
    key_cache = torch.randn(3, 2, 1, 4)
    value_cache = torch.randn(3, 2, 1, 4)
    block_tables = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    return query, key_cache, value_cache, block_tables


def test_reference_uses_physical_page_order():
    query, key_cache, value_cache, block_tables = _inputs()
    seq_lens = torch.tensor([5], dtype=torch.int32)
    actual = paged_mtp_attention_reference(
        query, key_cache, value_cache, block_tables, seq_lens, page_size=2, num_kv_heads=1
    )
    reordered_keys = key_cache.index_select(0, block_tables[0].to(torch.long)).reshape(-1, 1, 4)[:5]
    reordered_values = value_cache.index_select(0, block_tables[0].to(torch.long)).reshape(-1, 1, 4)[:5]
    expected = []
    for qi in range(2):
        valid = 5 - 2 + qi + 1
        scores = torch.matmul(reordered_keys[:valid, 0], query[0, qi, 0]) / (4**0.5)
        expected.append(torch.matmul(torch.softmax(scores, dim=0), reordered_values[:valid, 0]))
    expected = torch.stack(expected).unsqueeze(0).unsqueeze(2).expand(1, 2, 2, 4)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"page_size": 2, "num_kv_heads": 1, "seq_lens": [1]},
        {"page_size": 2, "num_kv_heads": 1, "seq_lens": [4, 4]},
    ],
)
def test_reference_rejects_invalid_metadata(kwargs):
    query, key_cache, value_cache, block_tables = _inputs()
    with pytest.raises(ValueError):
        paged_mtp_attention_reference(
            query,
            key_cache,
            value_cache,
            block_tables,
            torch.tensor(kwargs["seq_lens"], dtype=torch.int32),
            kwargs["page_size"],
            kwargs["num_kv_heads"],
        )


def test_reference_zero_fixture_is_finite_and_uniform():
    query = torch.zeros(1, 2, 2, 4)
    key_cache = torch.zeros(2, 2, 1, 4)
    value_cache = torch.arange(8, dtype=torch.float32).reshape(2, 2, 1, 4)
    output = paged_mtp_attention_reference(
        query, key_cache, value_cache, torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32), page_size=2, num_kv_heads=1,
    )
    expected_first = value_cache.reshape(4, 1, 4)[:1, 0].squeeze(0)
    expected_second = value_cache.reshape(4, 1, 4)[:2, 0].mean(dim=0)
    torch.testing.assert_close(output[0, 0], expected_first.expand(2, 4))
    torch.testing.assert_close(output[0, 1], expected_second.expand(2, 4))


def test_reference_high_magnitude_fixture_stays_finite():
    query = torch.full((1, 2, 2, 4), 100.0)
    key_cache = torch.full((2, 2, 1, 4), 100.0)
    value_cache = torch.ones_like(key_cache)
    output = paged_mtp_attention_reference(
        query, key_cache, value_cache, torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32), page_size=2, num_kv_heads=1,
    )
    assert torch.isfinite(output).all().item()
    torch.testing.assert_close(output, torch.ones_like(output))
