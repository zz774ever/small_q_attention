"""Strict apples-to-apples comparison of the FlashInfer paths for MTP small-q attention.

Four paths, same shapes, same harness, same paged-KV layout (NHD, page 16):

  decode_wrapper_tc  BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)
                     -- what production code actually runs for MTP today
  prefill_tc         BatchPrefillWithPagedKVCacheWrapper(use_tensor_cores=True)
  prefill_notc       BatchPrefillWithPagedKVCacheWrapper(use_tensor_cores=False)
                     -- the configuration the earlier baseline report measured
  xqa_raw            xqa_batch_decode_with_kv_cache with the spec-dec draft mask
                     -- the routing PR #3859 proposes

Each point is 3 rounds x 15 CUDA-event samples; the reported p50 is the median
of the three round medians.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "flashinfer"))

PS, HQ, HKV, DIM = 16, 32, 8, 128


def spec_dec_mask(torch, batch, q_len, dev):
    packed_words = (q_len + 31) // 32
    qi = torch.arange(q_len, device=dev, dtype=torch.int32).unsqueeze(1)
    ki = torch.arange(q_len, device=dev, dtype=torch.int32).unsqueeze(0)
    visible = ki <= qi
    padded = packed_words * 32
    if padded > q_len:
        visible = torch.cat(
            [visible, torch.zeros(q_len, padded - q_len, device=dev, dtype=torch.bool)], dim=1
        )
    visible = visible.view(q_len, packed_words, 32)
    bits = torch.tensor([1 << i for i in range(32)], device=dev, dtype=torch.int64)
    words = (visible.to(torch.int64) * bits).sum(dim=-1).to(torch.uint32)
    return words.unsqueeze(0).expand(batch, q_len, packed_words).contiguous().view(torch.uint16)


def build_case(torch, q_len, kv_len, batch, dev):
    pages = (kv_len + PS - 1) // PS
    return {
        "q": torch.randn(batch * q_len, HQ, DIM, dtype=torch.float16, device=dev),
        "kv": torch.randn(pages, 2, PS, HKV, DIM, dtype=torch.float16, device=dev),
        "indptr": torch.arange(0, (batch + 1) * pages, pages, dtype=torch.int32, device=dev),
        "indices": torch.arange(pages, dtype=torch.int32, device=dev).repeat(batch),
        "last": torch.full((batch,), kv_len % PS or PS, dtype=torch.int32, device=dev),
        "block_tables": torch.arange(pages, dtype=torch.int32, device=dev).repeat(batch, 1).contiguous(),
        "seq_lens": torch.full((batch,), kv_len, dtype=torch.int32, device=dev),
    }


def timeit(torch, call, warmup=5, repeats=15, rounds=3):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    round_medians = []
    for _ in range(rounds):
        samples = []
        for _ in range(repeats):
            start.record()
            call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)
        round_medians.append(statistics.median(samples))
    return statistics.median(round_medians), min(round_medians), max(round_medians)


def main():
    import torch
    import flashinfer
    from flashinfer.decode import xqa_batch_decode_with_kv_cache

    parser = argparse.ArgumentParser()
    parser.add_argument("--q-lens", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--kv-lens", nargs="+", type=int, default=[1024, 8192])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dev = "cuda"
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.int8, device=dev)
    rows = []

    for q_len in args.q_lens:
        for kv_len in args.kv_lens:
            for batch in args.batch_sizes:
                case = {"q_len": q_len, "kv_len": kv_len, "batch_size": batch}
                tensors = build_case(torch, q_len, kv_len, batch, dev)
                mask = spec_dec_mask(torch, batch, q_len, dev) if q_len > 1 else None
                paths = {}

                try:
                    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                        torch.zeros_like(workspace), "NHD", use_tensor_cores=True
                    )
                    wrapper.plan(
                        indptr=tensors["indptr"],
                        indices=tensors["indices"],
                        last_page_len=tensors["last"],
                        num_qo_heads=HQ,
                        num_kv_heads=HKV,
                        head_dim=DIM,
                        page_size=PS,
                        pos_encoding_mode="NONE",
                        q_data_type=torch.float16,
                        kv_data_type=torch.float16,
                        q_len_per_req=q_len,
                    )
                    paths["decode_wrapper_tc"] = lambda w=wrapper, t=tensors: w.run(t["q"], t["kv"])
                except Exception as exc:  # noqa: BLE001
                    print(json.dumps({**case, "path": "decode_wrapper_tc", "status": "plan_failed",
                                      "error": f"{type(exc).__name__}: {exc}"}), flush=True)

                for tag, use_tc in (("prefill_notc", False), ("prefill_tc", True)):
                    try:
                        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                            torch.zeros_like(workspace), "NHD", use_tensor_cores=use_tc
                        )
                        wrapper.plan(
                            qo_indptr=torch.arange(
                                0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=dev
                            ),
                            paged_kv_indptr=tensors["indptr"],
                            paged_kv_indices=tensors["indices"],
                            paged_kv_last_page_len=tensors["last"],
                            num_qo_heads=HQ,
                            num_kv_heads=HKV,
                            head_dim_qk=DIM,
                            page_size=PS,
                            pos_encoding_mode="NONE",
                            causal=True,
                            logits_soft_cap=0.0,
                            q_data_type=torch.float16,
                            kv_data_type=torch.float16,
                        )
                        paths[tag] = lambda w=wrapper, t=tensors: w.run(t["q"], t["kv"])
                    except Exception as exc:  # noqa: BLE001
                        print(json.dumps({**case, "path": tag, "status": "plan_failed",
                                          "error": f"{type(exc).__name__}: {exc}"}), flush=True)

                try:
                    scale = 1.0 / math.sqrt(DIM)
                    xqa_ws = torch.zeros_like(workspace)
                    out = torch.empty_like(tensors["q"])

                    def xqa_call(t=tensors, xqa_ws=xqa_ws, out=out, scale=scale, mask=mask, kv_len=kv_len, q_len=q_len):
                        return xqa_batch_decode_with_kv_cache(
                            t["q"], t["kv"], xqa_ws, t["block_tables"], t["seq_lens"], kv_len,
                            scale, 1.0, -1, out=out, kv_layout="NHD",
                            q_len_per_req=q_len, mask=mask,
                        )

                    paths["xqa_raw"] = xqa_call
                except Exception as exc:  # noqa: BLE001
                    print(json.dumps({**case, "path": "xqa_raw", "status": "setup_failed",
                                      "error": f"{type(exc).__name__}: {exc}"}), flush=True)

                for name, call in paths.items():
                    try:
                        call()
                        torch.cuda.synchronize()
                        p50, lo, hi = timeit(torch, call)
                        row = {"path": name, "case": case, "status": "ok",
                               "p50_us": p50, "round_min_us": lo, "round_max_us": hi}
                    except Exception as exc:  # noqa: BLE001
                        row = {"path": name, "case": case, "status": "failed",
                               "error": f"{type(exc).__name__}: {exc}"}
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)

                del tensors
                torch.cuda.empty_cache()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
        print(f"# wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
