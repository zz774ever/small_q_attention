"""FlashInfer baseline runners for the small-q MTP contract.

This compares the FlashInfer paths that issue #3420 / PR #3859 touch, on the
same case description that ``scripts/run_gpu_matrix.py`` uses for the
standalone V0/V1 kernel:

- ``prefill``: ``BatchPrefillWithPagedKVCacheWrapper``. This is the routing
  issue #3420 reports as slow for ``qo_len=2``, and it doubles as the
  correctness reference here.
- ``xqa``: ``xqa_batch_decode_with_kv_cache`` with the speculative-decoding
  draft mask, i.e. the XQA routing PR #3859 adds.
- ``trtllm``: ``trtllm_batch_decode_with_kv_cache`` with the same mask.

Every backend emits one JSON result line in the schema documented in
``docs/benchmark_protocol.md``, so results can be paired with
``scripts/run_gpu_matrix.py`` output.

The pinned FlashInfer checkout is added to ``sys.path``, so the script runs
from any working directory:

    /usr/local/miniconda3/envs/py312/bin/python scripts/run_flashinfer_baseline.py \
        --backends prefill xqa --q-lens 2 --kv-lens 1024 --batch-sizes 1 \
        --check-correctness --output results/h20_flashinfer_probe.jsonl

The first call into a given XQA / trtllm-gen configuration triggers a JIT
build that can take several minutes. It happens during warmup and is therefore
excluded from the reported timings.
"""

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parents[1]
FLASHINFER_ROOT = ROOT / "third_party" / "flashinfer"
if str(FLASHINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(FLASHINFER_ROOT))

FIXED = {
    "dtype": "float16",
    "head_dim": 128,
    "page_size": 16,
    "num_query_heads": 32,
    "num_kv_heads": 8,
}
WORKSPACE_BYTES = 256 * 1024 * 1024
XQA_MAX_ABS_ERROR = 2e-2


def build_spec_dec_mask(torch, batch_size, q_len, device, mode="causal"):
    """Packed draft-block attention mask for speculative decoding.

    Mirrors ``generate_spec_dec_mask`` in the upstream test
    ``tests/attention/test_xqa_batch_decode.py``. The KV prefix is always
    visible; the packed bits only describe draft-token visibility. Returns
    ``[batch_size, q_len, div_up(q_len, 32) * 2]`` uint16.
    """
    num_packed = (q_len + 31) // 32
    q_indices = torch.arange(q_len, device=device, dtype=torch.int32).unsqueeze(1)
    kv_indices = torch.arange(q_len, device=device, dtype=torch.int32).unsqueeze(0)

    if mode == "causal":
        visible = kv_indices <= q_indices
    elif mode == "full":
        visible = torch.ones(q_len, q_len, device=device, dtype=torch.bool)
    else:
        raise ValueError(f"unsupported spec-dec mask mode: {mode}")

    padded = num_packed * 32
    if padded > q_len:
        visible = torch.cat(
            [visible, torch.zeros(q_len, padded - q_len, device=device, dtype=torch.bool)],
            dim=1,
        )
    visible = visible.view(q_len, num_packed, 32)
    bit_positions = torch.tensor([1 << i for i in range(32)], device=device, dtype=torch.int64)
    packed = (visible.to(torch.int64) * bit_positions).sum(dim=-1).to(torch.uint32)
    packed = packed.unsqueeze(0).expand(batch_size, q_len, num_packed).contiguous()
    return packed.view(torch.uint16)


def build_case(torch, case, device):
    """Allocate one case on the GPU. Every request reuses the same physical pages."""
    page_size = case["page_size"]
    hq = case["num_query_heads"]
    hkv = case["num_kv_heads"]
    dim = case["head_dim"]
    batch = case["batch_size"]
    q_len = case["q_len"]
    kv_len = case["kv_len"]

    num_pages = (kv_len + page_size - 1) // page_size
    query = torch.randn(batch * q_len, hq, dim, dtype=torch.float16, device=device)
    # NHD interleaved 5-D cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_nhd = torch.randn(num_pages, 2, page_size, hkv, dim, dtype=torch.float16, device=device)
    # trtllm-gen wants HND: [num_pages, 2, num_kv_heads, page_size, head_dim]
    kv_hnd = kv_nhd.permute(0, 1, 3, 2, 4).contiguous()
    page_ids = torch.arange(num_pages, dtype=torch.int32, device=device)
    return {
        "query": query,
        "kv_nhd": kv_nhd,
        "kv_hnd": kv_hnd,
        "block_tables": page_ids.repeat(batch, 1).contiguous(),
        "seq_lens": torch.full((batch,), kv_len, dtype=torch.int32, device=device),
        "mask": build_spec_dec_mask(torch, batch, q_len, device) if q_len > 1 else None,
        "num_pages": num_pages,
        "last_page_len": kv_len % page_size or page_size,
    }


def fresh_workspace(torch, device):
    """XQA requires a zero-initialised workspace on first use."""
    return torch.zeros(WORKSPACE_BYTES, dtype=torch.int8, device=device)


def make_prefill_runner(torch, case, t, device):
    """Routing used before PR #3859: paged prefill with a causal mask."""
    import flashinfer

    batch = case["batch_size"]
    q_len = case["q_len"]
    num_pages = t["num_pages"]
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        fresh_workspace(torch, device), "NHD"
    )
    wrapper.plan(
        qo_indptr=torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=device),
        paged_kv_indptr=torch.arange(
            0, (batch + 1) * num_pages, num_pages, dtype=torch.int32, device=device
        ),
        paged_kv_indices=torch.arange(num_pages, dtype=torch.int32, device=device).repeat(batch),
        paged_kv_last_page_len=torch.full(
            (batch,), t["last_page_len"], dtype=torch.int32, device=device
        ),
        num_qo_heads=case["num_query_heads"],
        num_kv_heads=case["num_kv_heads"],
        head_dim_qk=case["head_dim"],
        page_size=case["page_size"],
        pos_encoding_mode="NONE",
        causal=True,
        logits_soft_cap=0.0,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    return lambda: wrapper.run(t["query"], t["kv_nhd"])


def make_xqa_runner(torch, case, t, device):
    """PR #3859 routing: XQA batch decode with the speculative-decoding mask."""
    from flashinfer.decode import xqa_batch_decode_with_kv_cache

    scale = 1.0 / math.sqrt(case["head_dim"])
    workspace = fresh_workspace(torch, device)
    out = torch.empty_like(t["query"])

    def call():
        return xqa_batch_decode_with_kv_cache(
            t["query"],
            t["kv_nhd"],
            workspace,
            t["block_tables"],
            t["seq_lens"],
            case["kv_len"],
            scale,
            1.0,
            -1,
            out=out,
            kv_layout="NHD",
            q_len_per_req=case["q_len"],
            mask=t["mask"],
        )

    return call


def make_trtllm_runner(torch, case, t, device):
    """trtllm-gen decode with the same mask; requires the HND layout."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    scale = 1.0 / math.sqrt(case["head_dim"])
    workspace = fresh_workspace(torch, device)

    def call():
        return trtllm_batch_decode_with_kv_cache(
            t["query"],
            t["kv_hnd"],
            workspace,
            t["block_tables"],
            t["seq_lens"],
            case["kv_len"],
            scale,
            1.0,
            -1,
            kv_layout="HND",
            q_len_per_req=case["q_len"],
            mask=t["mask"],
        )

    return call


RUNNERS = {
    "prefill": make_prefill_runner,
    "xqa": make_xqa_runner,
    "trtllm": make_trtllm_runner,
}


def measure(torch, call, warmup, repeats):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return {
        "latency_us_p50": statistics.median(samples),
        "latency_us_p99": samples[min(len(samples) - 1, int(0.99 * len(samples)))],
        "min_us": samples[0],
        "max_us": samples[-1],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", choices=sorted(RUNNERS), default=["prefill", "xqa"])
    parser.add_argument("--q-lens", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--kv-lens", nargs="+", type=int, default=[1024, 8192])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--correctness-cases", type=int, default=3)
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be non-negative and --repeats positive")
    return args


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")

    cases = [
        {"q_len": q, "kv_len": kv, "batch_size": batch, **FIXED}
        for q, kv, batch in itertools.product(args.q_lens, args.kv_lens, args.batch_sizes)
    ]
    results = []
    correctness_cases = 0
    for case in cases:
        page_size = case["page_size"]
        num_pages = (case["kv_len"] + page_size - 1) // page_size
        estimate = num_pages * 2 * page_size * case["num_kv_heads"] * case["head_dim"] * 2
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if estimate > int(free_bytes * args.memory_fraction):
            for backend in args.backends:
                results.append(
                    {
                        "backend": f"flashinfer_{backend}",
                        "case": case,
                        "status": "skipped",
                        "reason": "estimated_memory_limit",
                    }
                )
            continue

        torch.manual_seed(7)
        t = build_case(torch, case, device)

        reference = None
        if args.check_correctness and correctness_cases < args.correctness_cases:
            started = time.perf_counter()
            reference = make_prefill_runner(torch, case, t, device)().float()
            torch.cuda.synchronize()
            correctness_cases += 1
            print(
                f"# reference q_len={case['q_len']} kv_len={case['kv_len']} "
                f"batch={case['batch_size']} built in {time.perf_counter() - started:.1f}s",
                flush=True,
            )

        for backend in args.backends:
            result = {
                "backend": f"flashinfer_{backend}",
                "device": torch.cuda.get_device_name(0),
                "case": case,
            }
            try:
                runner = RUNNERS[backend](torch, case, t, device)
                result.update(measure(torch, runner, args.warmup, args.repeats))
                result["status"] = "ok"
                if reference is not None:
                    actual = runner()
                    torch.cuda.synchronize()
                    error = (actual.float() - reference).abs().max().item()
                    result["max_abs_error"] = error
                    result["correct"] = bool(error < XQA_MAX_ABS_ERROR)
            except Exception as exc:  # noqa: BLE001 - surface the backend failure verbatim
                result["status"] = "failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

        del t
        del reference
        torch.cuda.empty_cache()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in results), encoding="utf-8"
        )
        print(f"# wrote {len(results)} results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
