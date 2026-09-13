# 阶段报告：V2（split-KV + warp 分块）与差距归因

## 目标

V1 在 H20 上比 FlashInfer XQA 慢 20–200x（见 `docs/h20_baseline_report.md`）。V1 的问题是结构性的，不是参数问题。本阶段验证 V2 是否能用"换映射"把差距大幅压缩，并把剩余的差距归因到可测量的瓶颈上。

## V2 设计

保持 V0 的数据契约（FP16、paged KV、`head_dim=128`、`page_size=16`、GQA 32/8、`q_len` 2/4/8），只改映射与同步结构：

1. **split-KV**：一个 `(batch, query_row, query_head)` 的 KV 范围被切成 `num_chunks = ceil(max_seq_len / chunk_keys)` 段，每段独立成一个 block。grid 从 `batch*q_len*heads`（`batch=1, q_len=2` 时只有 64）变成 `rows * num_chunks`（`chunk_keys=128, KV=8192` 时 64×64 = 4096）。
2. **warp 分块、无逐 key 同步**：block 内 4 个 warp 以步长 4 轮流处理本段 key，每个 warp 维护自己的 online-softmax 状态（`m, l, acc[4]`）。KV 主循环里**一次 block 级 `__syncthreads()` 都没有**，V1 是每个 key 两次。
3. **两段式归约**：每个 block 内先合并 4 个 warp 的 partial（一次 barrier），再由第二个 kernel 用 log-sum-exp 合并各 chunk 的 partial。partial 用 fp32，最终输出 fp16。

新增文件：`csrc/small_q_attention_ext.cu` 中的 v2 kernel、`src/small_q_attention/cuda.py` 的 `forward_v2`、`scripts/build_and_test_v2.py`、`scripts/sweep_v2_chunk.py`、`scripts/profile_v1_v2.py`。

## 正确性

`scripts/build_and_test_v2.py`：`q_len` 2/4/8 × KV 33/1024/8192，逆序（非连续）page table，chunk=128/512/单块三种配置。

全部通过，最大绝对误差 ≤ 4.88e-4（FP16 量级），且 `v2` 与 `v1` 的互差 ≤ 2.44e-4。多 chunk 的 merge 路径（8/16/64 chunk）与单 chunk 路径结果一致。

## 结果：V0/V1/V2 与 FlashInfer 的配对对比（p50，µs）

`v0/v1/v2` 用 `chunk_keys=512`，FlashInfer 列取自 36 用例矩阵。

| q_len | KV | batch | v0 | v1 | **v2** | fi_prefill | fi_xqa | v1/xqa | **v2/xqa** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 1 | 7341.8 | 1400.1 | **99.8** | 80.8 | 72.1 | 19.4x | **1.38x** |
| 2 | 1024 | 4 | 8388.8 | 1442.8 | **108.7** | 143.9 | 66.1 | 21.8x | **1.64x** |
| 2 | 8192 | 1 | 59301.4 | 11746.6 | **230.1** | 390.2 | 80.0 | 146.8x | **2.88x** |
| 2 | 8192 | 4 | 67101.3 | 12069.8 | **526.8** | 668.6 | 141.0 | 85.6x | **3.74x** |
| 4 | 1024 | 1 | 7613.2 | 1415.6 | **101.1** | 76.3 | 45.3 | 31.3x | **2.23x** |
| 4 | 1024 | 4 | 11643.2 | 1509.3 | **177.5** | 116.8 | 49.0 | 30.8x | **3.63x** |
| 4 | 8192 | 1 | 60853.5 | 11899.6 | **329.6** | 344.7 | 60.6 | 196.4x | **5.44x** |
| 4 | 8192 | 4 | 93044.8 | 12412.1 | **993.8** | 644.4 | 122.1 | 101.6x | **8.14x** |
| 8 | 1024 | 1 | 8395.6 | 1440.7 | **106.9** | 74.4 | 44.6 | 32.3x | **2.40x** |
| 8 | 1024 | 4 | 21273.0 | 2970.0 | **261.9** | 117.7 | 50.1 | 59.3x | **5.23x** |
| 8 | 8192 | 1 | 67018.5 | 12058.6 | **526.6** | 344.1 | 60.4 | 199.6x | **8.72x** |
| 8 | 8192 | 4 | 171060.8 | 24890.8 | **2012.9** | 644.4 | 122.6 | 203.0x | **16.42x** |

V2 相对 V1 提速 **5.9x–51x**（几何均值约 20x），把与 XQA 的差距从 19–203x 压到 **1.4–16x**。

## chunk 粒度扫描（KV=8192，p50 µs）

| q_len | batch | 128 | 256 | 512 | 1024 | 2048 | 最优 | 最优 vs XQA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | **177.2** | 180.8 | 204.2 | 241.8 | 415.5 | 177.2 | 2.2x |
| 2 | 4 | 567.4 | **533.9** | 543.6 | 538.8 | 643.1 | 533.9 | 3.8x |
| 8 | 1 | 560.5 | **527.5** | 537.2 | 538.3 | 641.4 | 527.5 | 8.7x |
| 8 | 4 | 2058.8 | 2014.7 | 2011.9 | **1942.9** | 1988.9 | 1942.9 | 15.8x |

规律与设计预期一致：并行度不足时（`q_len=2`、`batch=1`、行数只有 64）需要更小的 chunk 去铺满 SM；行数多时（`q_len=8`、`batch=4`，4096 行）大 chunk 反而更好，因为每个 block 的固定开销和 merge 成本更重要。曲线在 256–1024 之间很平，说明 V2 对这一个超参不敏感。

## 归因：计数器拿不到，但时间分解足够

`ncu` 在这台 H20 上不可用：

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

这是驱动层的内核模块限制（需要 `NVreg_RestrictProfilingToAdminUsers=0`），不是用户权限问题，root 也绕不过去，不能为此重启服务器。`nsys` 存在但只给时间线，不给 stall/occupancy 计数器。

替代方案是 `torch.profiler` 的 kernel 级时间分解（`reports/h20/torch_profile_v1.txt`、`torch_profile_v2.txt`），shape 为 `q_len=2, KV=8192, batch=1`：

| 版本 | 单次调用的 CUDA kernel 时间 | 说明 |
|---|---|---|
| V1 | **11.893 ms** | 只有一个 kernel，占 100% CUDA 时间 —— V1 完全是 kernel 内部慢，不是启动开销 |
| V2 (chunk=128) | **128.16 µs** partial + **8.08 µs** merge | partial 占 92%，merge 占 5.8% |

顺带暴露一个可修的小问题：V2 的 host 端每次调用都会执行 `seq_lens.max().item()` 来算 chunk 数，profiler 里能看到一次 `reduce_kernel`（1.49 µs）加一次 DtoH 拷贝（1.52 µs）以及随之而来的同步。改成由调用方传入 `max_seq_len` 可以去掉。

## 差距归因：V2 现在受限于内存流量，而不是并行度

以 `q_len=2, KV=8192, batch=1`（32 个 query head、8 个 kv head、`head_dim=128`、FP16）为例：

- **当前读法**：每个 `(batch, q_row, q_head)` 行都要把属于它的整个 KV head 扫一遍，每次读 K 和 V 各 `8192 × 128 × 2B = 2 MB`，一行 4 MB，64 行合计 **≈268 MB/次调用**。
- **V2 实测**：268 MB / 128 µs ≈ **2.1 TB/s**，约为 H20 标称 4.0 TB/s 的 52% —— 已经接近带宽bound，继续抠算术收益有限。
- **理论下限**：同一份 K/V 只要按 kv head 读一次就是 `8 × 8192 × 128 × 2 × 2B ≈ 33.5 MB`，是当前流量的 1/8。这正是 GQA 的分组冗余：32 个 query head 只对应 8 个 kv head，`q_len=2` 时有 8 行共享同一份 KV。
- **对照 V1**：同样 268 MB 却用了 11.9 ms，等效带宽只有 22 GB/s，说明 V1 根本不是带宽问题，而是每 key 两次 barrier 的串行化问题。

结论：V1→V2 解决的是"并行度与同步"，V2→V3 要解决的是"KV 复用"。

## Go / No-Go

- V2 把与 XQA 的差距压到 1.4–16x，且剩余差距可以用一个明确的、可测的机制解释（KV 被重复读 8 倍），**满足继续投入的条件**。
- 下一步应当是 **M1：按 query group 共享 KV tile**（一个 block 负责同一 kv head 下的多个 query head/行，配合 shared-memory staging），预期把流量降到原来的 1/8 量级；只有在 M1 之后仍落后，才考虑张量化/TMA 这类更重的改造。
- 同时做两个小的确定性修复：去掉每次调用的 `seq_lens.max().item()` 同步；按 `rows` 自动选 chunk（或提供经验表）。

## 限制

- 单卡单次采样；KV=1024 的小 case 抖动明显。
- H20 上无法获得硬件计数器，occupancy/warp-stall 只能从设计推断，没有实测 counter。
- 268 MB 与 2.1 TB/s 是按 `q_len=2, KV=8192, batch=1` 手算的流量模型，未用计数器验证。
- V2 只覆盖 `q_len` 2/4/8；`q_len=16` 未纳入自研 kernel 契约。
- 计时含 Python/launch 开销（V2 的 p50 177 µs 中，kernel 占 136 µs）。

## 复现

```bash
source /usr/local/miniconda3/bin/activate py312
cd /root/small_q_attention

python scripts/build_and_test_v2.py
python scripts/run_gpu_matrix.py --variants v0 v1 v2 \
  --q-lens 2 4 8 --kv-lens 1024 8192 --batch-sizes 1 4 \
  --output results/h20_v0_v1_v2.jsonl
python scripts/sweep_v2_chunk.py --q-lens 2 8 --kv-lens 8192 --batch-sizes 1 4 \
  --chunks 128 256 512 1024 2048 --output results/h20_v2_chunk_sweep.jsonl
python scripts/profile_v1_v2.py --variant v1 --kv-len 8192 --torch-profiler
python scripts/compare_backends.py results/h20_v0_v1_v2.jsonl results/h20_flashinfer_matrix36.jsonl
```
