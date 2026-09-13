"""Does BatchDecodeWithPagedKVCacheWrapper route MTP (q_len_per_req > 1) to XQA?

PR #3859 proposes exactly that routing. Our earlier measurement called
``xqa_batch_decode_with_kv_cache`` directly, so it proved XQA is fast for
q_len > 1 but said nothing about the wrapper path that the PR actually changes.

This probe calls the wrapper itself, reports which kernels CUDA actually runs,
and times it, so the question is answered by evidence instead of reading code.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "flashinfer"))

PS, HQ, HKV, DIM = 16, 32, 8, 128


def build(torch, q_len, kv_len, batch):
    num_pages = (kv_len + PS - 1) // PS
    dev = "cuda"
    q = torch.randn(batch * q_len, HQ, DIM, dtype=torch.float16, device=dev)
    kv = torch.randn(num_pages, 2, PS, HKV, DIM, dtype=torch.float16, device=dev)
    indptr = torch.arange(0, (batch + 1) * num_pages, num_pages, dtype=torch.int32, device=dev)
    indices = torch.arange(num_pages, dtype=torch.int32, device=dev).repeat(batch)
    last_page_len = torch.full((batch,), kv_len % PS or PS, dtype=torch.int32, device=dev)
    return q, kv, indptr, indices, last_page_len


def time_it(torch, call, warmup=5, repeats=15):
    import statistics

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def kernels_used(torch, call):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            call()
        torch.cuda.synchronize()
    names = []
    for evt in prof.key_averages():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.count:
            names.append(f"{evt.key.split('(')[0][:70]} x{evt.count}")
    return names


def main():
    import torch
    import flashinfer

    device = "cuda"
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.int8, device=device)
    results = []

    for q_len in (1, 2, 4, 8):
        for kv_len in (1024, 8192):
            q, kv, indptr, indices, last_page_len = build(torch, q_len, kv_len, 1)
            wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                workspace, "NHD", use_tensor_cores=True
            )
            try:
                wrapper.plan(
                    indptr=indptr,
                    indices=indices,
                    last_page_len=last_page_len,
                    num_qo_heads=HQ,
                    num_kv_heads=HKV,
                    head_dim=DIM,
                    page_size=PS,
                    pos_encoding_mode="NONE",
                    q_data_type=torch.float16,
                    kv_data_type=torch.float16,
                    q_len_per_req=q_len,
                )
                wrapper.run(q, kv)
                torch.cuda.synchronize()
                status = "OK"
            except Exception as exc:  # noqa: BLE001
                print(f"q_len={q_len} kv_len={kv_len} PLAN_FAIL {type(exc).__name__}: {exc}", flush=True)
                continue

            p50 = time_it(torch, lambda: wrapper.run(q, kv))
            used = kernels_used(torch, lambda: wrapper.run(q, kv))
            print(
                f"q_len={q_len} kv_len={kv_len} status={status} p50_us={p50:.1f} kernels={used}",
                flush=True,
            )
            results.append((q_len, kv_len, p50, used))

    print("--- dispatched kernel summary ---")
    for q_len, kv_len, p50, used in results:
        joined = " | ".join(used)
        tag = "XQA" if "xqa" in joined.lower() else ("PREFILL" if "prefill" in joined.lower() else "OTHER")
        print(f"q_len={q_len} kv_len={kv_len} p50={p50:.1f}us -> {tag}")


if __name__ == "__main__":
    main()
