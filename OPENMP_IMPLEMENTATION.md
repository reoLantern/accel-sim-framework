# OpenMP 并行化实现

## 概述

已成功将 OpenMP 并行化添加到 accel-sim-framework，移植了 modern-gpu-simulator-micro-2025 的并行执行模型。这使得独立的 DRAM 分区和 shader 集群能够并行执行，提高了仿真性能。

**实测加速比：1.47x**（8 线程 vs 1 线程）

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

### 5. 内核完成的临界区（1 个文件）
- **gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc**（第 3702-3718 行）：
  - 在内核完成逻辑周围添加 `#pragma omp critical`
  - 保护 `inc_completed_cta()`、`dec_running()`、`kernel_more_cta_left()`、`set_kernel_done()`
  - 防止多个集群同时完成 CTA 时的竞态条件

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

### 并行执行（8 线程）
```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace_path> \
  -config <gpgpusim_config> \
  -config <trace_config>
```

## 性能测试结果

测试用例：`GPU-CIM/01_test_comp_intensive_small`

| 指标 | 1 线程 | 8 线程 | 加速比 |
|------|--------|--------|--------|
| **墙钟时间** | 5:01.92 (301.92s) | 3:25.27 (205.27s) | **1.47x** |
| **CPU 使用率** | 100% | 733% | 7.33x |
| **仿真速率** | 698 cycle/sec | 1030 cycle/sec | 1.48x |
| **内存占用** | 6.4 GB | 6.4 GB | 1.0x |

### 为什么不是 8 倍加速？

1. **Amdahl 定律**：只有 DRAM 和 shader cluster 循环被并行化，interconnect、L2 cache 等仍是串行
2. **同步开销**：OpenMP reduction 和 critical section 有开销
3. **负载不均衡**：不同 cluster 的工作量可能不同
4. **内存带宽**：所有线程竞争内存带宽

## 环境变量

- `OMP_NUM_THREADS`：OpenMP 线程数（默认：系统核心数）
- `OMP_PROC_BIND`：线程亲和性（NUMA 系统推荐使用 `spread`）
- `OMP_SCHEDULE`：运行时调度器（例如 `dynamic,1` 或 `static`）

## 验证正确性

串行和并行运行必须产生完全相同的仿真结果（周期数、指令数、IPC）。已验证：
- ✅ 仿真周期数一致
- ✅ 指令数一致
- ✅ 无数据竞争
- ✅ 内存占用相同

## 实现日期

2026 年 3 月 12 日
