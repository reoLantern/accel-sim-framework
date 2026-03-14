nvbit tracer:

```bash
# recommended to set these envs in .bashrc
export CUDA_INSTALL_PATH=<your_cuda>
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/
```

Accel-Sim Simulator:

```bash
# 一键编译（release 模式）
export MODE=release && cd $HOME/work/gpgpu-sim/accel-sim-framework && \
source ./gpu-simulator/setup_environment.sh $MODE && \
cmake -S ./gpu-simulator -B ./gpu-simulator/build/$MODE && \
cmake --build ./gpu-simulator/build/$MODE -j64 && \
cmake --install ./gpu-simulator/build/$MODE

# 二进制位置：
# - ./gpu-simulator/build/release/accel-sim.out（推荐直接使用）
# - ./gpu-simulator/bin/release/accel-sim.out（install 后）

# 清理重新编译
rm -rf ./gpu-simulator/build/$MODE && \
rm -rf ./gpu-simulator/bin/$MODE
```

OpenMP Parallel Simulation:

```bash
# 串行基准（1 线程）
OMP_NUM_THREADS=1 ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace> -config <gpgpusim.config> -config <trace.config>

# 并行仿真（8 线程，推荐方式）
# 多 NUMA 服务器：用 numactl 绑定到同一 NUMA node
numactl --cpunodebind=0 --membind=0 \
  env OMP_NUM_THREADS=8 OMP_PROC_BIND=close \
  ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace> -config <gpgpusim.config> -config <trace.config>

# 单 NUMA 机器（桌面/笔记本）：无需 numactl
OMP_NUM_THREADS=8 OMP_PROC_BIND=close ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace> -config <gpgpusim.config> -config <trace.config>

# 开启 power simulation（需要 accelwattch_sass_sim.xml 在运行目录中）
# 在上述任一命令末尾加 -power_simulation_enabled 1
```

判断是否为多 NUMA 机器：
```bash
lscpu | grep "NUMA node(s)"
# NUMA node(s): 1  → 单 NUMA，close/spread 差异不大
# NUMA node(s): 8  → 多 NUMA，必须用 close，否则性能下降严重
# 也可以用 numactl --hardware 查看详细拓扑
```

性能对比（`01_test_comp_intensive_small`，108 SM，AMD EPYC 9475F 2S/8N）：
```
                          无 power        有 power
1 线程                    312s            872s
8 线程 close + numactl     72s (4.33x)    143s (6.10x)   ← 推荐
8 线程 spread             228s (1.46x)    —              ← 多 NUMA 不推荐
```

详见 [OPENMP_IMPLEMENTATION.md](OPENMP_IMPLEMENTATION.md)。
