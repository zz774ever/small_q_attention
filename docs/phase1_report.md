# 阶段报告：本地 CUDA V0/V1 原型

## 项目定位

本项目围绕 FlashInfer issue #3420 的 small-q paged attention 场景，开发一个独立的 CUDA kernel 研究原型。目标不是重复 PR #3859 的 XQA 路由，而是研究 q_len=2/4/8 在 speculative decoding / MTP 请求中的专用计算映射，并用可复现 benchmark 和 profiling 数据判断是否值得集成。

## 本阶段完成

- 建立独立仓库目录：`/home/ai/projectsvllm/small_q_attention`。
- 保持原有 `/home/ai/projectsvllm/nano-vllm` 不变。
- 固定 V0 contract：FP16、head_dim=128、page_size=16、GQA 32/8、paged KV、q_len=2/4/8。
- 保留 q_len=16 作为 benchmark crossover 点，不要求 V0 kernel 支持。
- 完成 CPU/reference 实现、矩阵生成、边界测试和 CUDA smoke test。
- 完成第一个可编译的 CUDA V0 kernel：每个 `(batch, q_row, query_head)` 使用一个 128-thread block，使用 FP32 online softmax 和 paged KV lookup。
- 完成 V1 kernel：保持 block 粒度和数据契约不变，让一个 warp 协作计算每个 Q·K score，消除 V0 对每个输出维度重复计算 score 的主要浪费。
- 修复直接调用 `.venv/bin/python` 时 Ninja 不在 PATH 的加载器问题。
- 正确性脚本现在一次覆盖 q_len=2、4、8。
- 增加 CUDA event benchmark 脚本，输出 p50/p99 微秒延迟。

## 当前硬件与限制

本地设备为 RTX 3050 Laptop GPU，compute capability 8.6。它适合编译、reference parity、SM86 kernel 调试和初步延迟测量；不能代表 Hopper XQA 的性能。FlashInfer issue/PR 中的 XQA 对比应在 H100/H20 等 SM90+ 设备完成。

## 当前验证状态

| 验证项 | 状态 |
|---|---|
| WSL Ubuntu 24.04 可访问 | PASS |
| RTX 3050 / SM86 可见 | PASS |
| Python reference smoke | PASS |
| benchmark matrix validation | PASS |
| Python compileall | PASS |
| CUDA V0 编译及 q_len=2/4/8 parity | PASS |

## 本地 V0 基线

配置：RTX 3050 Laptop（SM86）、FP16、batch=1、head_dim=128、GQA 32/8、page_size=16、KV length=1024、CUDA events，warmup=10，repeats=30。

| q_len | p50 | p99 | max abs error |
|---:|---:|---:|---:|
| 2 | 10.254 ms | 14.066 ms | 2.44e-4 |
| 4 | 20.246 ms | 21.994 ms | 4.88e-4 |
| 8 | 39.353 ms | 40.553 ms | 2.44e-4 |

该 V0 是 correctness-first 原型，dot-product 对每个输出 lane 有冗余计算，因此这些数字只作为优化前基线，不代表最终性能，也不与 Hopper XQA 直接比较。

## V0 -> V1 阶段对比

实验假设：V0 的主要浪费是同一个 `(query_head, key_index)` 的 Q·K 被 128 个输出 lane 重复计算；V1 仅将 score 计算改为 warp 协作，其他布局、paged lookup、online softmax 和 FP32 累加保持不变。

配置仍为 RTX 3050、batch=1、KV=1024、FP16、GQA 32/8、warmup=10、repeats=30。

| q_len | V0 p50 | V1 p50 | V0 p99 | V1 p99 | p50 speedup |
|---:|---:|---:|---:|---:|---:|
| 2 | 10.268 ms | 2.054 ms | 10.513 ms | 2.235 ms | 5.00x |
| 4 | 20.370 ms | 2.131 ms | 20.645 ms | 2.232 ms | 9.56x |
| 8 | 39.598 ms | 4.436 ms | 41.522 ms | 4.514 ms | 8.93x |

V1 在非连续 physical page table 下通过 q_len=2/4/8 正确性测试，最大绝对误差为 `1.22e-4`、`4.88e-4`、`2.44e-4`。这是本地 SM86 微基准结果，不是 Hopper/XQA 或端到端 speculative decoding 结论；V1 仍需要在更长 KV、更大 batch 和服务器 GPU 上重复测量，并用 Nsight 验证内存与 warp stall 假设。

补充长上下文样本（KV=8192、batch=1、warmup=5、repeats=15）：

| q_len | V0 p50 | V1 p50 | p50 speedup | 备注 |
|---:|---:|---:|---:|---|
| 2 | 90.313 ms | 16.828 ms | 5.37x | 稳定趋势 |
| 4 | 169.213 ms | 14.444 ms | 11.72x | V1 p99 抖动到 18.347 ms，需复测 |
| 8 | 327.794 ms | 29.516 ms | 11.11x | 稳定趋势 |

下一步必须在 batch=4/16、KV=32768 和 Hopper/A100 上复现，避免将单一 RTX 3050 样本外推为通用结论。

## 下一阶段

1. 重复 batch=1 的长上下文样本，确认 q_len=4 的 p99 抖动不是系统噪声。
2. 完成 batch=4/16、KV=32768 的受控子集，并统一输出 JSONL。
3. 在服务器 A100/H100/H20 上复现 V0/V1，使用 Nsight 验证 global-load、occupancy、warp-stall 假设。
4. 只有在 V1 优势稳定后，才进入 shared-memory KV staging、tile size 和 query-group mapping 实验。

## 结论

当前阶段已经具备一个真实 CUDA 算子项目的基础：有明确 workload contract、reference、可编译 kernel、正确性入口、计时协议和服务器迁移路径。性能收益尚未宣称，必须等待本地编译验证和后续 A100/H100 实测。
