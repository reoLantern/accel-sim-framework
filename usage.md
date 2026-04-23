nvbit tracer:

```bash
# recommended to set these envs in .bashrc
export CUDA_INSTALL_PATH=<your_cuda>
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

./util/tracer_nvbit/install_nvbit.sh
make -C ./util/tracer_nvbit/
```

Benchmark 构建（gpu-app-collection）：

```bash
# 依赖：setup_environment 已识别 CUDA_INSTALL_PATH
cd ./gpu-app-collection/src && source setup_environment

# 一次构建所有 suite（rodinia-2.0-ft/rodinia-3.1/ispass-2009/lonestargpu/pannotia/polybench/
# proxy-apps/GPU_Microbenchmark/custom_apps/...）
make all -j8
# binary 落在：./gpu-app-collection/bin/$CUDA_VERSION/release/

# 注意：make <single_suite> 可能会顺手清掉 bin/<cuda>/release 里其它 suite 的产物
#       （根因未定位，2026-04-23 观察到 make rodinia_2.0-ft 后 rodinia-3.1 等 binary 消失）
#       → 改动某个 suite 后若发现别的丢了，直接重新 make all

# 首次运行前下载 benchmark 输入数据（~2.5GB 压缩包）
cd ./gpu-app-collection && bash get_data.sh
# 覆盖 rodinia / ispass-2009 / lonestargpu / sdk；polybench 规模用源码宏，其它 suite 需另觅
```

HW trace（在真卡上跑）：

```bash
# -l 5 限制每 app 前 5 个 kernel；-t 开启 TERMINATE_UPON_LIMIT
./util/tracer_nvbit/run_hw_trace.py -B <suite> -D 0 -l 5 -t
# 输出：./hw_run/traces/device-0/$CUDA_VERSION/<exec>/<argdir>/traces/dynamic_trace.pb

# 大 kernel 应用可能在单 kernel 就 timeout（NVBit overhead 10-20×）
# 建议独立对每个 run.sh 套 timeout，先 120s 再按需 600s 重试
# 失败排查：per-app spinlock 检测
bash <run_dir>/run_spinlock_detection.sh   # spinlock_detection/spinlock_instructions.txt
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
# 单进程运行（推荐方式）
# 多 NUMA 服务器：绑定到同一 NUMA node
numactl --cpunodebind=0 --membind=0 \
  env OMP_NUM_THREADS=8 OMP_PROC_BIND=close \
  ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace> -config <gpgpusim.config> -config <trace.config>

# 单 NUMA 机器（桌面/笔记本）：无需 numactl
OMP_NUM_THREADS=8 OMP_PROC_BIND=close ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace> -config <gpgpusim.config> -config <trace.config>

# 串行基准（1 线程，用于正确性对比）
OMP_NUM_THREADS=1 ./gpu-simulator/build/release/accel-sim.out \
  -trace <trace> -config <gpgpusim.config> -config <trace.config>
```

多进程并发（批量仿真）：

```bash
# 重要：多进程共享 NUMA node 时，不要加 OMP_PROC_BIND=close
# 否则所有 OMP 线程会 pin 到同一个 CPU，利用率只有 ~100%
# 正确做法：只用 numactl 绑定 node，不设 OMP_PROC_BIND

NODE=0
for trace_dir in ...; do
  DIR=$trace_dir/sim_output && mkdir -p "$DIR" && cd "$DIR"
  nohup numactl --cpunodebind=$NODE --membind=$NODE \
    env OMP_NUM_THREADS=7 \
    $BIN -trace ../post_traces/kernelslist.g -config $CFG -config $TCFG \
    > accel-sim-output.log 2>&1 &
  NODE=$(( (NODE+1) % NUM_NUMA_NODES ))
done

# 参考并行度：假设服务器是 288 逻辑线程 / 6 NUMA nodes
# 32 runs × OMP=7 = 224 线程（~78% 利用率）
```

判断是否为多 NUMA 机器：
```bash
lscpu | grep "NUMA node(s)"
# NUMA node(s): 1  → 单 NUMA
# NUMA node(s): 8  → 多 NUMA，必须绑定 node
```
