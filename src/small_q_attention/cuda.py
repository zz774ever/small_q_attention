"""Lazy loader and wrapper for the standalone CUDA v0 kernel."""

import os
import sys
from pathlib import Path


_MODULE = None
_MODULE_V1 = None
_MODULE_V2 = None


def load_v0(verbose: bool = False):
    """Compile/load the v0 extension for the current PyTorch/CUDA environment."""
    global _MODULE
    if _MODULE is None:
        import torch
        from torch.utils.cpp_extension import load

        # Direct invocation of ``.venv/bin/python`` does not activate the
        # virtualenv, so its Ninja executable may be absent from PATH.
        venv_bin = Path(sys.executable).resolve().parent
        os.environ["PATH"] = os.pathsep.join(
            part for part in (str(venv_bin), os.environ.get("PATH", "")) if part
        )

        source = Path(__file__).parents[2] / "csrc" / "small_q_attention_ext.cu"
        _MODULE = load(
            name="small_q_attention_v0",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=verbose,
        )
    return _MODULE


def load_v1(verbose: bool = False):
    """Compile/load the cooperative-score v1 extension."""
    global _MODULE_V1
    if _MODULE_V1 is None:
        import torch
        from torch.utils.cpp_extension import load

        venv_bin = Path(sys.executable).resolve().parent
        os.environ["PATH"] = os.pathsep.join(
            part for part in (str(venv_bin), os.environ.get("PATH", "")) if part
        )
        source = Path(__file__).parents[2] / "csrc" / "small_q_attention_ext.cu"
        _MODULE_V1 = load(
            name="small_q_attention_v1",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=verbose,
        )
    return _MODULE_V1


def forward_v0(query, key_cache, value_cache, block_tables, seq_lens, page_size: int, verbose: bool = False):
    """Run v0 after validating the fixed contract at the Python boundary."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v0 kernel")
    tensors = (query, key_cache, value_cache, block_tables, seq_lens)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all v0 inputs must be CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all v0 inputs must be contiguous")
    if query.dtype != torch.float16 or key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("v0 requires FP16 Q/K/V")
    if query.shape[1] not in (2, 4, 8):
        raise ValueError("v0 supports q_len 2, 4, or 8")
    if query.shape[-1] != 128:
        raise ValueError("v0 supports head_dim 128")
    return load_v0(verbose=verbose).forward(query, key_cache, value_cache, block_tables, seq_lens, page_size)


def forward_v1(query, key_cache, value_cache, block_tables, seq_lens, page_size: int, verbose: bool = False):
    """Run the v1 cooperative-score kernel under the same v0 contract."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v1 kernel")
    tensors = (query, key_cache, value_cache, block_tables, seq_lens)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all v1 inputs must be CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all v1 inputs must be contiguous")
    if query.dtype != torch.float16 or key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("v1 requires FP16 Q/K/V")
    if query.shape[1] not in (2, 4, 8) or query.shape[-1] != 128:
        raise ValueError("v1 supports q_len 2, 4, or 8 and head_dim 128")
    return load_v1(verbose=verbose).forward_v1(query, key_cache, value_cache, block_tables, seq_lens, page_size)


def load_v2(verbose: bool = False):
    """Compile/load the split-KV, warp-tiled v2 extension."""
    global _MODULE_V2
    if _MODULE_V2 is None:
        import torch
        from torch.utils.cpp_extension import load

        venv_bin = Path(sys.executable).resolve().parent
        os.environ["PATH"] = os.pathsep.join(
            part for part in (str(venv_bin), os.environ.get("PATH", "")) if part
        )
        source = Path(__file__).parents[2] / "csrc" / "small_q_attention_ext.cu"
        _MODULE_V2 = load(
            name="small_q_attention_v2",
            sources=[str(source)],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=verbose,
        )
    return _MODULE_V2


def forward_v2(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    page_size: int,
    chunk_keys: int = 512,
    verbose: bool = False,
):
    """Run the v2 split-KV kernel under the same v0 contract.

    ``chunk_keys`` is the number of KV entries one block owns; the grid gains a
    second dimension of ``ceil(max_seq_len / chunk_keys)`` so the kernel stops
    depending on ``batch * q_len * heads`` alone for parallelism.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v2 kernel")
    tensors = (query, key_cache, value_cache, block_tables, seq_lens)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all v2 inputs must be CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all v2 inputs must be contiguous")
    if query.dtype != torch.float16 or key_cache.dtype != torch.float16 or value_cache.dtype != torch.float16:
        raise ValueError("v2 requires FP16 Q/K/V")
    if query.shape[1] not in (2, 4, 8) or query.shape[-1] != 128:
        raise ValueError("v2 supports q_len 2, 4, or 8 and head_dim 128")
    if chunk_keys <= 0:
        raise ValueError("chunk_keys must be positive")
    return load_v2(verbose=verbose).forward_v2(
        query, key_cache, value_cache, block_tables, seq_lens, page_size, chunk_keys
    )
