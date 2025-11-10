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

export MODE=debug

cd /home/mmy/work/gpgpusim/accel-sim-framework

# 不加 debug 参数则默认是 release 模式
source ./gpu-simulator/setup_environment.sh $MODE

# 生成 Debug 构建目录
cmake -S ./gpu-simulator -B ./gpu-simulator/build/$MODE

# 编译
cmake --build ./gpu-simulator/build/$MODE -j64

# 安装到 bin/debug/
cmake --install ./gpu-simulator/build/$MODE

# 一条命令连起来：
export MODE=release && cd /home/mmy/work/gpgpusim/accel-sim-framework && source ./gpu-simulator/setup_environment.sh $MODE && cmake -S ./gpu-simulator -B ./gpu-simulator/build/$MODE && cmake --build ./gpu-simulator/build/$MODE -j64 && cmake --install ./gpu-simulator/build/$MODE

# 清理当前 MODE 的目标与中间产物
cmake --build ./gpu-simulator/build/$MODE --target clean

# 一键清理构建目录、可执行文件
MODE=debug \
&& cd /home/mmy/work/gpgpusim/accel-sim-framework \
&& rm -rf ./gpu-simulator/build/$MODE \
&& rm -f ./gpu-simulator/bin/$MODE/accel-sim.out

```
