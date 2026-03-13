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
  -trace <trace> -config <config>

# 并行仿真（8 线程，~1.3x 加速）
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace> -config <config>

# 性能对比（01_test_comp_intensive_small）：
# - 1 线程：  4:16, 100% CPU, 820 cycle/sec
# - 8 线程：  3:23, 733% CPU, 1045 cycle/sec (1.26x 加速)
```
