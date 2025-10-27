## 运行方式

- 运行全部基线（含 baseline_3 内置计时与/或正确性检查）：

```bash
make s3
```

如需调整架构，请在 `Makefile` 中把 `-arch=sm_120a` 改为你的 GPU 架构（例如 H100 可用 `sm_90a`）。

## Baseline 3 消融实验（bug 修复与性能来源）

为便于消融，我们在 `baseline_3.cu` 中加入了几个可编译期开关（默认均为优化版设置）：

- `AB_Q_PRELOAD`（默认 1）：在 flash-decoding 计算前把 q 载入寄存器，既是正确性修复也是性能优化。
- `AB_FLASH_KV_SHMEM_SMALL`（默认 1）：flash-decoding block 仅分配 `KV_DIM_PER_BLOCK_BS=640` 的 `kv_indices`，显著降低共享内存占用、提高占用率。
- `AB_LOCK_INIT_ONLY_FLASH`（默认 1）：仅在 flash-decoding block 初始化 `lock=0`，更清晰的同步路径（性能影响极小）。
- `AB_DEBUG_PRINT`（默认未定义）：若定义，将启用 `DEBUG` 路径，打印关键中间信息并做三路输出对比（baseline_1/2/3）。

开关通过 nvcc 的 `-D` 传入，例如 `-DAB_Q_PRELOAD=0`。

### 复现命令

- 默认优化版（含计时）：

```bash
make s3
```

- 关闭 q 预载（验证 bug 会复现；打开 DEBUG 做对比）：

```bash
nvcc -O3 -std=c++17 -arch=sm_120a -DAB_Q_PRELOAD=0 -DAB_DEBUG_PRINT=1 -o baseline_3_ab_bug baseline_3.cu && ./baseline_3_ab_bug
```

- 将 flash-decoding 的 `kv_indices` 扩大到 2560（验证共享内存瘦身带来的性能收益）：

```bash
nvcc -O3 -std=c++17 -arch=sm_120a -DAB_FLASH_KV_SHMEM_SMALL=0 -o baseline_3_ab_shmem baseline_3.cu && ./baseline_3_ab_shmem
```

- 改变锁初始化策略（通常性能影响很小）：

```bash
nvcc -O3 -std=c++17 -arch=sm_120a -DAB_LOCK_INIT_ONLY_FLASH=0 -o baseline_3_ab_lock baseline_3.cu && ./baseline_3_ab_lock
```

### 实验结果（该机器的样例数值，仅供参考）

- 默认优化版（`AB_Q_PRELOAD=1, AB_FLASH_KV_SHMEM_SMALL=1, AB_LOCK_INIT_ONLY_FLASH=1`）
	- baseline 1 latency: ~47.1 us
	- baseline 2 latency: ~47.2 us
	- baseline 3 latency: ~133.2 us

- 关闭 q 预载（`AB_Q_PRELOAD=0, AB_DEBUG_PRINT=1`）
	- 正确性对比：baseline_1 与 baseline_3 出现明显数值差异（示例：max abs diff ≈ 0.149）
	- 结论：q 预载是关键正确性修复，亦有性能正效应。

- 扩大 kv_indices 共享内存（`AB_FLASH_KV_SHMEM_SMALL=0`）
	- baseline 3 latency: ~243.8 us（由 ~133.2 us 明显退化）
	- 结论：共享内存瘦身（kv_indices=640）是主要性能来源之一，提升占用率、降低延迟。

- 锁初始化策略（`AB_LOCK_INIT_ONLY_FLASH=0`）
	- baseline 3 latency: ~133.0 us（与默认 ~133.2 us 基本一致）
	- 结论：对性能影响很小，但同步路径更清晰。

### 结论

- Bug 修复：`AB_Q_PRELOAD`（q → 寄存器预载）是正确性关键；关闭会出现明显数值偏差。
- 性能提升：`AB_FLASH_KV_SHMEM_SMALL=1` 将 flash-decoding 的共享内存从 2560 → 640，显著提高占用率并降低延迟，是主要性能来源之一；q 预载也对性能有正向贡献。

提示：不同 GPU/驱动环境的绝对数值可能不同，但相对趋势应保持一致。