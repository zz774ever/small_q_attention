# 阶段报告：H20 FlashInfer baseline 对照

## 目标

1. 在 SM90 上复现 issue #3420 的 before/after：被投诉的 prefill 路由 vs PR #3859 引入的 XQA 路由。
2. 把自研 V0/V1 kernel 放进同一坐标系，判断是否存在可争取的性能 gap，以及当前原型离目标有多远。

上一轮 `reports/h20/backend_probe_q2.txt` 只留下两条 `AssertionError: Mask is required for speculative decoding`，没有任何 baseline 数字；本报告补上了这部分。

## 环境与固定版本

| 项目 | 值 |
|---|---|
| GPU | NVIDIA H20，compute capability 9.0，97871 MiB |
| 驱动 / CUDA toolkit | 595.71.05 / 13.0（nvcc V13.0.88） |
| Python / PyTorch | 3.12.11（conda env `py312`）/ 2.14.0+cu130 |
| FlashInfer | 0.7.0，源码 checkout `third_party/flashinfer` @ `a69ad808f8ff4095df4460cbda86ebcf815aa31d` |
| 项目 | `0316fef`（本报告提交前） |
| 计时 | CUDA events，warmup=5，repeats=15，报告 p50/p99（微秒） |

## 方法

- 用例契约与本地一致：FP16、`head_dim=128`、`page_size=16`、GQA 32/8、paged KV、uniform `q_len`。
- 36 用例矩阵：`q_len` 2/4/8/16 × KV 1024/8192/32768 × batch 1/4/16。
- 三个后端：
  - `flashinfer_prefill` — `BatchPrefillWithPagedKVCacheWrapper`（`causal=True`），即 issue #3420 报告偏慢的路径，同时用作正确性参考。
  - `flashinfer_xqa` — `xqa_batch_decode_with_kv_cache` + speculative-decoding draft mask，即 PR #3859 的路由。
  - `flashinfer_trtllm` — `trtllm_batch_decode_with_kv_cache`，同一 mask，HND 布局。
- `q_len > 1` 时必须传 packed draft mask，形状 `[batch, q_len, div_up(q_len, 32) * 2]` uint16，构造方式取自上游测试 `tests/attention/test_xqa_batch_decode.py` 的 `generate_spec_dec_mask`（KV 前缀恒可见，packed bit 只描述 draft token 之间的可见性）。
- 自研 V0/V1 覆盖 12 个用例（`q_len` 2/4/8 × KV 1024/8192 × batch 1/4），数据来自 `results/h20_v0_v1.jsonl`。

## 过程中解决的阻塞

| 现象 | 根因 | 处理 |
|---|---|---|
| `FileNotFoundError: 'ninja'` | conda env `py312` 只在登录 shell 激活，JIT 子进程 PATH 里没有 `ninja` | 先 `source /usr/local/miniconda3/bin/activate py312` 再运行 |
| `cutlass/arch/reg_reconfig.h: No such file`、`cute/tensor.hpp: No such file` | `third_party/cutlass`、`3rdparty/cccl`、`3rdparty/spdlog` 子模块未初始化（XQA 不需要 CUTLASS，prefill 需要） | `git submodule update --init --recursive 3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog` |
| `AssertionError: Mask is required for speculative decoding`（`flashinfer/xqa.py:418`） | 之前的探针没有传 `mask` | 传入 packed draft mask 后 XQA 与 trtllm 均正常 |
| 上游 trace 测试在 H20 上 `2 skipped` | `tests/trace/*` 要求 `_skip_if_not_sm100()`，H20 是 SM90 | 不使用 trace 测试，直接走 API 并以 prefill 输出作为参考 |

## 正确性证据

- XQA / trtllm-gen 相对 prefill 参考：抽样 3 个 `q_len=2` 用例，`max_abs_error` 为 `0`（参考自身）、`1.22e-4`、`2.44e-4`、`6.10e-5`，全部 `correct=True` —— 属 FP16 量级误差。
- 自研 kernel 在 H20 上（`reports/h20/small_q_kernel_correctness.txt`）：`q_len=2/4/8` 的 max abs error 为 `1.22e-4` / `2.44e-4` / `4.88e-4`，与 CPU reference 一致。

## 结果：36 用例 p50 延迟（µs）

`-` 表示该后端在该用例上没有数据（自研 kernel 只覆盖 12 个用例）。

| q_len | kv_len | batch | v0 | v1 | flashinfer_prefill | flashinfer_xqa | flashinfer_trtllm | xqa/prefill | v1/xqa |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 1 | 7339.9 | 1404.7 | 80.8 | 72.1 | 87.1 | 1.12x | 19.5x |
| 2 | 1024 | 4 | 8395.5 | 1443.6 | 143.9 | 66.1 | 89.9 | 2.18x | 21.8x |
| 2 | 1024 | 16 | - | - | 370.2 | 105.2 | 127.6 | 3.52x | - |
| 2 | 8192 | 1 | 59335.2 | 11874.2 | 390.2 | 80.0 | 102.4 | 4.88x | 148.4x |
| 2 | 8192 | 4 | 67194.4 | 12045.5 | 668.6 | 141.0 | 161.4 | 4.74x | 85.4x |
| 2 | 8192 | 16 | - | - | 2134.0 | 366.8 | 376.5 | 5.82x | - |
| 2 | 32768 | 1 | - | - | 1230.1 | 117.0 | 128.6 | 10.51x | - |
| 2 | 32768 | 4 | - | - | 2410.0 | 367.6 | 381.2 | 6.56x | - |
| 2 | 32768 | 16 | - | - | 8338.8 | 1307.2 | 1323.4 | 6.38x | - |
| 4 | 1024 | 1 | 7615.9 | 1412.6 | 76.3 | 45.3 | 55.7 | 1.69x | 31.2x |
| 4 | 1024 | 4 | 11652.4 | 1506.8 | 116.8 | 49.0 | 60.9 | 2.39x | 30.8x |
| 4 | 1024 | 16 | - | - | 315.5 | 80.3 | 93.2 | 3.93x | - |
| 4 | 8192 | 1 | 60972.6 | 11915.4 | 344.7 | 60.6 | 72.5 | 5.69x | 196.7x |
| 4 | 8192 | 4 | 93151.4 | 12423.6 | 644.4 | 122.1 | 133.9 | 5.28x | 101.7x |
| 4 | 8192 | 16 | - | - | 2140.2 | 360.7 | 376.4 | 5.93x | - |
| 4 | 32768 | 1 | - | - | 1230.4 | 117.9 | 130.5 | 10.43x | - |
| 4 | 32768 | 4 | - | - | 2417.0 | 368.5 | 382.8 | 6.56x | - |
| 4 | 32768 | 16 | - | - | 8358.0 | 1309.2 | 1325.9 | 6.38x | - |
| 8 | 1024 | 1 | 8397.4 | 1449.0 | 74.4 | 44.6 | 55.5 | 1.67x | 32.5x |
| 8 | 1024 | 4 | 21298.3 | 2969.3 | 117.7 | 50.1 | 60.7 | 2.35x | 59.3x |
| 8 | 1024 | 16 | - | - | 319.0 | 83.7 | 93.6 | 3.81x | - |
| 8 | 8192 | 1 | 67153.6 | 12041.5 | 344.1 | 60.4 | 72.5 | 5.70x | 199.3x |
| 8 | 8192 | 4 | 171220.6 | 24844.6 | 644.4 | 122.6 | 132.4 | 5.26x | 202.6x |
| 8 | 8192 | 16 | - | - | 2155.3 | 373.9 | 384.0 | 5.76x | - |
| 8 | 32768 | 1 | - | - | 1246.8 | 122.4 | 134.1 | 10.19x | - |
| 8 | 32768 | 4 | - | - | 2437.6 | 370.3 | 383.4 | 6.58x | - |
| 8 | 32768 | 16 | - | - | 8359.7 | 1315.1 | 1334.3 | 6.36x | - |
| 16 | 1024 | 1 | - | - | 75.2 | 44.5 | 56.2 | 1.69x | - |
| 16 | 1024 | 4 | - | - | 115.9 | 66.6 | 78.4 | 1.74x | - |
| 16 | 1024 | 16 | - | - | 314.8 | 153.0 | 179.9 | 2.06x | - |
| 16 | 8192 | 1 | - | - | 340.7 | 84.4 | 96.1 | 4.04x | - |
| 16 | 8192 | 4 | - | - | 643.7 | 208.1 | 221.3 | 3.09x | - |
| 16 | 8192 | 16 | - | - | 2138.9 | 686.9 | 699.6 | 3.11x | - |
| 16 | 32768 | 1 | - | - | 1230.8 | 192.7 | 204.7 | 6.39x | - |
| 16 | 32768 | 4 | - | - | 2408.9 | 688.4 | 701.2 | 3.50x | - |
| 16 | 32768 | 16 | - | - | 8357.2 | 2579.9 | 2594.3 | 3.24x | - |

36 个用例、108 次后端调用全部 `status=ok`，没有失败和跳过。

## 结论

1. **#3420 的问题真实存在，且在该 revision 的 XQA 路由下已被显著改善。** 同一 shape 下 XQA 比 prefill 快 1.1x–10.5x；收益随 KV 增长而扩大：KV=1024 时约 1.1–1.7x，KV=8192 时约 3–6x，KV=32768/batch=1 时约 10x。
2. **收益不只限于小 `q_len`。** `q_len=16` 也在 XQA 路由下取得 1.7x–6.4x 提升，说明 #3859 的 `q_len_per_req > 1` 门控覆盖面比 issue 描述的 `qo_len=2` 更宽。
3. **trtllm-gen 在本矩阵上系统性慢于 XQA 约 10–25%**，不作为目标对照基线。
4. **自研 V0/V1 与 XQA 相差 20–200x，且差距随 KV 急剧扩大**（KV=1024 约 20–60x，KV=8192 约 85–203x）。当前原型在设计上不可能竞争，这不是调参问题。

## 为什么 V1 落后这么多（可解释的瓶颈）

- **并行度不足**：一个 `(batch, query_row, query_head)` 一个 block，`batch=1/q_len=2` 时只有 64 个 block，而 H20 有 148 个 SM。
- **KV 维没有并行**：每个 block 串行扫完整个 KV，没有 split-KV。
- **同步开销与 KV 成正比**：每个 key 至少两次 `__syncthreads()`，`kv_len=8192` 时单 block 就有上万次 barrier；V1 的两趟 score 计算还把这个数字翻倍。
- **算力利用**：Q·K 只用 1 个 warp 的 shuffles，没有 tensor core、没有 TMA、KV 也不是 swizzled 布局。
- **V 累加**是逐 key 标量 FMA，没有分块复用和寄存器缓存。

## Go / No-Go 判断

按 `task_plan.md` 的判据（"只有当基线接近硬件极限、且 profiling 找不到可解释的优化机会时才停止"）：

- prefill 相对 XQA 在 KV≥8K 上仍有 4.7–10.5x 空间，**目标 gap 存在**；
- 但用当前 V1 设计去吃这个 gap 没有希望，必须换成"提高并行度 + 复用 KV tile"的设计。

因此本阶段的结论是：**Phase 2 的 H20 基线地图完成**；Phase 4 的下一步不是继续微调 V1，而是 V2（M1/M2 mapping + T2 shared-memory KV staging，再考虑张量化）。

## 限制

- 单卡、单次采样，未做多轮重复；KV=1024 的小 case 抖动明显（`q_len=2/KV=1024/batch=1` 的 XQA 两次测量分别为 45.8 µs 与 72.1 µs）。
- 计时包含 Python/launch 开销，没有拆出纯 kernel 时间，也没有 Nsight 计数器。
- 所有请求复用同一物理页集合，真实 KV cache 的页分布更分散，可能改变 L2 行为。
- FlashAttention 对照尚未纳入（Phase 2 清单中仍为未完成项）。
- 自研 kernel 只覆盖 `q_len` 2/4/8，`q_len=16` 只有 FlashInfer 数据。

## 复现

```bash
source /usr/local/miniconda3/bin/activate py312
cd /root/small_q_attention

# 12 用例子集（含正确性抽样）
python scripts/run_flashinfer_baseline.py --backends prefill xqa trtllm \
  --q-lens 2 4 8 --kv-lens 1024 8192 --batch-sizes 1 4 \
  --check-correctness --correctness-cases 3 \
  --output results/h20_flashinfer_baseline.jsonl

# 完整 36 用例矩阵
python scripts/run_flashinfer_baseline.py --backends prefill xqa trtllm \
  --q-lens 2 4 8 16 --kv-lens 1024 8192 32768 --batch-sizes 1 4 16 \
  --output results/h20_flashinfer_matrix36.jsonl

# 自研 kernel 正确性与基线对照
python scripts/build_and_test_v1.py
python scripts/run_gpu_matrix.py --variants v0 v1 --q-lens 2 4 8 \
  --kv-lens 1024 8192 --batch-sizes 1 4 --output results/h20_v0_v1.jsonl

# 配对表
python scripts/compare_backends.py results/h20_v0_v1.jsonl results/h20_flashinfer_matrix36.jsonl
```
