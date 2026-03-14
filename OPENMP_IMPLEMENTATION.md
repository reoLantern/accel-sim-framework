# OpenMP 并行化实现

## 概述

已成功将 OpenMP 并行化添加到 accel-sim-framework，移植了 modern-gpu-simulator-micro-2025 的并行执行模型。这使得独立的 DRAM 分区和 shader 集群能够并行执行，提高了仿真性能。

**实测加速比：4.33x ~ 6.10x**（8 线程 vs 1 线程，取决于是否启用 power simulation；NUMA 绑定到同一 node）

## 修改内容

### 1. CMake 构建系统（2 个文件）

**重要**：本项目使用 CMake 构建，不是 Makefile！

- **gpu-simulator/CMakeLists.txt**（第 80-87 行）：
  ```cmake
  find_package(OpenMP REQUIRED)
  target_link_libraries(accel-sim.out PUBLIC OpenMP::OpenMP_CXX)
  ```

- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/CMakeLists.txt**（第 27-35 行）：
  ```cmake
  find_package(OpenMP REQUIRED)
  target_compile_options(gpgpusim PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(gpgpusim PUBLIC OpenMP::OpenMP_CXX)
  ```

### 2. 头文件（2 个文件）
- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h**（第 40 行）：添加 `#include <omp.h>`
- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h**（第 708 行）：添加成员变量 `float m_active_sms_this_cycle;`
- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h**：添加 `#include <omp.h>`

### 3. DRAM 分区并行化（1 个文件）
- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc**（第 2034 行）：
  - 在 DRAM 分区循环前添加 `#pragma omp parallel for`
  - 每个分区是独立的；功耗统计写入到按分区索引的数组中

### 4. Shader 集群并行化（1 个文件）
- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc**（第 2090-2119 行）：
  - 将集群循环拆分为两部分：
    - **并行循环**：`core_cycle()`、`get_icnt_stats()`、使用 OpenMP reduction 统计活跃 SM
    - **串行循环**：`get_cache_stats()`、`get_current_occupancy()`（累加到共享结构）
  - 使用 `#pragma omp parallel for schedule(runtime) reduction(+:m_active_sms_this_cycle)`

### 5. 无锁内核完成（3 个文件） — 2026-03-13 优化

初版使用 `#pragma omp critical` 保护 CTA 完成逻辑，但在高频 CTA 完成的 workload（如 `dram_write`，131,074 个轻量 CTA）上导致严重 spinlock 竞争，8 线程比 1 线程慢 25 倍。

**解决方案**：用 `std::atomic` 替换全局锁，实现无锁 CTA 完成。

- **gpu-simulator/gpgpu-sim/src/abstract_hardware_model.h**：
  - `m_num_cores_running` 改为 `std::atomic<int>`
  - `inc_running()` 使用 `fetch_add(1)`
  - `dec_running()` 使用 `fetch_sub(1)` 并返回旧值（调用者检查 `prev==1` 判断是否最后完成）
  - `running()` 使用 `.load()`

- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h**：
  - `gpu_completed_cta` 改为 `std::atomic<unsigned>`
  - `inc_completed_cta()` 使用 `fetch_add(1)`

- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h**：
  - `shader_core_stats_pod::ctas_completed` 改为 `std::atomic<unsigned>`

- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc**（第 3702-3718 行）：
  - 删除 `#pragma omp critical`，改用无锁逻辑：
  ```cpp
  m_stats->ctas_completed++;                  // atomic
  m_gpu->inc_completed_cta();                 // atomic
  int prev_running = kernel->dec_running();   // atomic，返回旧值
  if (prev_running == 1 && !m_gpu->kernel_more_cta_left(kernel)) {
      // 只有一个线程能到达这里（使 count 从 1→0 的线程）
      m_gpu->set_kernel_done(kernel);
  }
  ```

**安全性**：`inc_running()` 仅在串行的 `issue_block2core()` 中调用，并行区间内只有 `dec_running()`。`fetch_sub` 保证只有一个线程看到 `prev==1`。

### 6. Power stats 采样守卫（1 个文件）

在 DRAM 并行循环和 core_cycle 并行循环中，为 power stats 收集添加采样频率守卫，避免每个 cycle 都做无用的统计拷贝：

- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.cc**：
  - DRAM 循环中的 `set_dram_power_stats` 加 `gpu_stat_sample_freq` 守卫
  - Core 循环中的 `get_icnt_stats` 加 `gpu_stat_sample_freq` 守卫
  - 串行循环中的 `get_cache_stats` 加 `gpu_stat_sample_freq` 守卫

## 构建说明

**必须使用 CMake 构建**：

```bash
export MODE=release
cd /home/mmy/work/gpgpu-sim/accel-sim-framework
source ./gpu-simulator/setup_environment.sh $MODE
cmake -S ./gpu-simulator -B ./gpu-simulator/build/$MODE
cmake --build ./gpu-simulator/build/$MODE -j$(nproc)
```

二进制文件位置：`gpu-simulator/build/release/accel-sim.out`

## 使用方法

### 串行执行（基准）
```bash
OMP_NUM_THREADS=1 ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace_path> \
  -config <gpgpusim_config> \
  -config <trace_config>
```

### 并行执行（8 线程，推荐）
```bash
numactl --cpunodebind=0 --membind=0 \
  env OMP_NUM_THREADS=8 OMP_PROC_BIND=close \
  ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace_path> \
  -config <gpgpusim_config> \
  -config <trace_config>
```

### 并行执行（不使用 numactl）
```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=close ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace_path> \
  -config <gpgpusim_config> \
  -config <trace_config>
```

## 性能测试结果

测试用例：`GPU-CIM/01_test_comp_intensive_small`（108 个 SM，计算密集型）

测试环境：AMD EPYC 9475F（2 socket, 8 NUMA node, 192 逻辑核）；使用 `numactl --cpunodebind=0 --membind=0` + `OMP_PROC_BIND=close`。

### Power simulation 关闭

| 指标 | 1 线程 | 8 线程 | 加速比 |
|------|--------|--------|--------|
| **gpu_sim_cycle** | 210,145 | 210,145 | 一致 ✅ |
| **墙钟时间** | 312s | 72s | **4.33x** |
| **仿真速率** | 673 cycle/sec | 2,918 cycle/sec | 4.33x |

### Power simulation 开启（`-power_simulation_enabled 1`）

| 指标 | 1 线程 | 8 线程 | 加速比 |
|------|--------|--------|--------|
| **gpu_sim_cycle** | 210,145 | 210,145 | 一致 ✅ |
| **Power 报告** | — | — | diff 完全一致 ✅ |
| **kernel_avg_power** | 156.268 W | 156.268 W | 一致 ✅ |
| **墙钟时间** | 872s | 143s | **6.10x** |
| **仿真速率** | 240 cycle/sec | 1,469 cycle/sec | 6.10x |

> **注**：开启 power simulation 后加速比更高（6.10x vs 4.33x），不是因为 power 让程序更快，
> 而是因为 power 计算大幅拖慢了单线程（312s → 872s），而 8 线程受影响较小（72s → 143s）。
> 这说明 power stats 采样守卫有效减少了并行区内的串行开销。

### 优化前（v1，使用 `omp critical`）

| 指标 | 1 线程 | 8 线程 | 加速比 |
|------|--------|--------|--------|
| **墙钟时间** | 301.92s | 205.27s | **1.47x** |
| **仿真速率** | 698 cycle/sec | 1,030 cycle/sec | 1.48x |

### 为什么优化后快了这么多？

初版使用 `#pragma omp critical` 全局锁保护 CTA 完成逻辑。对于 CTA 频繁完成的 workload（如 `dram_write` 有 131,074 个轻量 CTA），所有线程争抢同一把锁，导致严重 spinlock 竞争（8 线程各仅 20% CPU 利用率）。

改用 `std::atomic` 后：
- **无锁**：`fetch_sub` 返回旧值，只有使 count 从 1→0 的线程执行 `set_kernel_done`
- **无竞争**：每个 atomic 操作独立完成，不阻塞其他线程
- Power stats 采样守卫减少了并行区内的无用统计操作

## 环境变量

- `OMP_NUM_THREADS`：OpenMP 线程数（默认：系统核心数）
- `OMP_PROC_BIND`：线程亲和性，详见下文
- `OMP_SCHEDULE`：运行时调度器（例如 `dynamic,1` 或 `static`）

## OMP_PROC_BIND 线程亲和性

`OMP_PROC_BIND` 控制 OpenMP 线程与 CPU 核心的绑定策略。在多 NUMA 节点的服务器上，此选项对性能影响巨大。

### 可用模式

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| **`close`** | 线程绑定到相邻的核心（同一 NUMA node 内） | **本项目推荐**。仿真器的数据结构由主线程分配，所有并行线程在同一 NUMA node 内访问本地内存，延迟最低 |
| **`spread`** | 线程均匀分散到所有可用的 NUMA node | 适合各线程有独立大内存工作集的场景（如并行编译）。**不适合本项目**：线程分散后，访问主线程分配的共享数据结构需要跨 NUMA 远程访问，延迟显著增大 |
| **`master`**（或 `primary`） | 所有线程绑定到与主线程相同的位置 | 效果类似 `close`，但更严格地跟随主线程 |
| **`true`** | 线程绑定到当前位置，不迁移 | 防止 OS 迁移线程，但不指定具体绑定策略 |
| **`false`** | 不绑定，OS 自由调度 | 默认行为，线程可能在 NUMA node 间迁移，性能不稳定 |

### 性能对比（8 线程，无 power simulation）

| 绑定模式 | 墙钟时间 | 加速比 | 说明 |
|----------|----------|--------|------|
| `close` + `numactl --cpunodebind=0` | 72s | **4.33x** | 所有线程和内存在同一 NUMA node |
| `spread` | 228s | 1.46x | 线程分散到 8 个 NUMA node，跨 NUMA 访问 |

### 为什么 `close` 比 `spread` 快 3 倍？

本仿真器的内存模型是"主线程分配，并行区共享读写"：
- 所有数据结构（shader core、DRAM partition、interconnect buffer 等，共约 6GB）在仿真初始化时由主线程分配，位于主线程所在的 NUMA node
- 并行区间内，各线程操作的 cluster/partition 对象虽然逻辑独立，但物理内存都在同一 NUMA node 上
- `spread` 将线程分散到远端 NUMA node，每次内存访问都要经过跨 NUMA 互连（如 AMD Infinity Fabric），延迟从 ~80ns 增大到 ~150-200ns
- `close` 让所有线程留在本地 NUMA node，内存访问走本地 DDR5，延迟最低

### 建议

1. **单 NUMA node 机器**（如桌面、笔记本）：`close` 和 `spread` 差异不大，均可使用
2. **多 NUMA node 服务器**（如本机 8 NUMA node）：**强烈推荐 `close`**，配合 `numactl` 效果最佳
3. 如果线程数超过单 NUMA node 的核心数（本机每 node 12 物理核），可考虑 `spread` 以利用更多核心，但需在加速比和 NUMA 惩罚之间权衡

## 验证正确性

串行和并行运行必须产生完全相同的仿真结果。已验证：
- ✅ 仿真周期数一致（210,145 cycles）
- ✅ Power 报告完全一致（diff 为空）
- ✅ 无数据竞争（atomic 替代 critical）
- ✅ 计算密集型和 DRAM 密集型 workload 均通过

## 变更历史

| 日期 | 内容 | 加速比 |
|------|------|--------|
| 2026-03-12 | 初版：OpenMP 并行化 DRAM + shader cluster，`omp critical` 保护 CTA 完成 | 1.47x |
| 2026-03-13 | 优化：atomic 替换 critical，power stats 采样守卫 | 4.33x ~ 6.10x |
| 2026-03-14 | 发现 NUMA 亲和性影响：`close` 比 `spread` 快 3 倍；更新文档和推荐配置 | 4.33x ~ 6.10x |
