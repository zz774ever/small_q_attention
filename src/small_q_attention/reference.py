"""Small, readable reference implementation for paged MTP attention.

This is intentionally not a performance implementation. It defines the tensor
contract and causal semantics used by every later CUDA kernel revision.
"""

from typing import Tuple


def paged_mtp_attention_reference(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    page_size: int,
    num_kv_heads: int,
):
    """Compute causal paged attention for uniform multi-token requests.

    Args:
        query: Tensor [batch, q_len, num_query_heads, head_dim].
        key_cache/value_cache: Tensor [num_pages, page_size, num_kv_heads, head_dim].
        block_tables: Tensor [batch, max_pages] containing physical page ids.
        seq_lens: Tensor [batch] containing total KV length after appending Q.
        page_size: Number of tokens in each physical page.
        num_kv_heads: Number of KV heads; query heads use GQA grouping.

    Returns:
        Tensor [batch, q_len, num_query_heads, head_dim].
    """
    import torch

    if query.ndim != 4:
        raise ValueError("query must have shape [batch, q_len, heads, head_dim]")
    batch, q_len, num_query_heads, head_dim = query.shape
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    if key_cache.shape[1:] != (page_size, num_kv_heads, head_dim):
        raise ValueError("key_cache shape does not match page/head contract")
    if value_cache.shape != key_cache.shape:
        raise ValueError("key_cache and value_cache must have the same shape")
    if block_tables.shape[0] != batch or seq_lens.numel() != batch:
        raise ValueError("batch metadata shape mismatch")

    scale = head_dim ** -0.5
    group_size = num_query_heads // num_kv_heads
    outputs = []

    for b in range(batch):
        total_len = int(seq_lens[b].item())
        if total_len < q_len:
            raise ValueError("seq_lens must be at least q_len for MTP attention")
        num_pages = (total_len + page_size - 1) // page_size
        page_ids = block_tables[b, :num_pages].to(torch.long)
        keys = key_cache.index_select(0, page_ids).reshape(-1, num_kv_heads, head_dim)
        values = value_cache.index_select(0, page_ids).reshape(-1, num_kv_heads, head_dim)
        keys = keys[:total_len]
        values = values[:total_len]

        q_out = []
        for qi in range(q_len):
            valid_kv = total_len - q_len + qi + 1
            q_row = query[b, qi]
            q_heads = []
            for qh in range(num_query_heads):
                kvh = qh // group_size
                scores = torch.matmul(keys[:valid_kv, kvh], q_row[qh]) * scale
                probs = torch.softmax(scores.float(), dim=0).to(query.dtype)
                q_heads.append(torch.matmul(probs, values[:valid_kv, kvh]))
            q_out.append(torch.stack(q_heads, dim=0))
        outputs.append(torch.stack(q_out, dim=0))

    return torch.stack(outputs, dim=0)
