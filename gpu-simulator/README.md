# Accel-Sim Trace-Driven Front-end

![Accel-Sim Class Overview](https://accel-sim.github.io/assets/img/accel-sim-class.png)

The Accel-Sim's performance model relies on GPGPU-Sim 4.0 performance model. We created new classes with "exec_" and "trace_" prefix that are inherited from GPGPU-Sim performance model classes, then we moved some function implementations from the performance model to the new inherited classes using virtual functions. These functions are the ones that differ from exec-driven mode versus trace-driven mode. For example, when GPGPU-Sim calls the function "get_next_inst()", the exec_shader_core_ctx implementation will get the next instruction from the functional model, whereas the trace_shader_core_ctx will get the next inst from the traces.

Accel-Sim 的 performance model 依赖于 GPGPU-Sim 4.0 的 performance model。我们创建了带有 `exec_` 和 `trace_` 前缀的新 classes，这些 classes 继承自 GPGPU-Sim performance model classes，然后使用 virtual functions 将原 performance model 中的一些 function 实现移动到这些新继承的 classes 中。这些函数正是 exec-driven 模式和 trace-driven 模式之间的差异所在。例如，当 GPGPU-Sim 调用 `get_next_inst()` 函数时，`exec_shader_core_ctx` 的实现会从 functional model 中获取下一条指令，而 `trace_shader_core_ctx` 则会从 traces 中获取下一条 inst。

The blue blocks in the image are maintained in GPGPU-Sim 4.0 repo [here](https://github.com/gpgpu-sim/gpgpu-sim_distribution), whereas the green blocks are maintained by Accel-Sim in this repo.

Our new frontend supports both vISA (PTX) execution-driven and mISA (SASS) trace-driven simulation. In traced-riven mode, mISA traces are converted into an ISA-independent intermediate representation, that has a 1:1 correspondence to the original SASS instructions. We generate the traces from NVIDIA GPUs using Accel-Sim’s tracer tool that is built on top of Nvbit. For further details about the tracer, see [this](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/tracer_nvbit/README.md). These compatible traces are parsed by our [trace-parser](https://github.com/accel-sim/accel-sim-framework/tree/dev/gpu-simulator/trace-parser) component and feed up the performance model with these traces. The trace parser is a standalone component and can be utilized in other simulation engines for different use cases.

我们的新 frontend 同时支持 vISA (PTX) execution-driven 和 mISA (SASS) trace-driven simulation。在 trace-driven 模式下，mISA traces 会被转换成与 ISA 无关的 intermediate representation，该表示与原始 SASS instructions 之间保持 1:1 对应。我们使用基于 Nvbit 的 Accel-Sim tracer tool 从 NVIDIA GPUs 生成这些 traces。关于 tracer 的更多细节，请参见 [this](https://github.com/accel-sim/accel-sim-framework/blob/dev/util/tracer_nvbit/README.md)。这些兼容的 traces 会被我们的 trace-parser 组件解析，并提供给 performance model。trace parser 是一个 standalone component，也可以被其他 simulation engine 重用，适用于不同的 use case。

For each new GPU generation, we have to crease ISA_def file that specifies the SASS instruction types and where each instruction should be executed. For now, we have created the ISA_def files for NVIDIA's Kepler, Pascal, Turing and Volta generations. Please see the directory [./ISA_Def](./ISA_Def).
We were able to generate these files using the NVIDIA's CUDA Binary Utilities documentation from [here](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref).

对于每一代新的 GPU，我们都需要创建一个 ISA\_def file，用来指定 SASS instruction types 以及每条 instruction 应该在哪执行。目前，我们已为 NVIDIA 的 Kepler、Pascal、Turing 和 Volta generations 创建了对应的 ISA\_def files，详见目录 `./ISA_Def`。我们基于 NVIDIA CUDA Binary Utilities documentation（见 [here](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref)）来生成这些文件。

# GPGPU-SIM 4.x

You do not need to clone the GPGPU-Sim 4.x performance model by yourself. The [./setup_environment.sh](./setup_environment.sh) will clone the recent GPGPU-Sim model and integrate it with Accel-Sim. For more info on the Accel-Sim front-end and how to compile, please see "Accel-Sim SASS Frontend" entry in the main read-me page [here](https://github.com/accel-sim/accel-sim-framework/blob/dev/README.md).

你无需手动 clone GPGPU-Sim 4.x performance model。运行 `./setup_environment.sh` 脚本即可自动 clone 最新的 GPGPU-Sim 模型并与 Accel-Sim 集成。关于 Accel-Sim front-end 以及如何编译，请参见主 read-me 页面中 “Accel-Sim SASS Frontend” 一节（见 [here](https://github.com/accel-sim/accel-sim-framework/blob/dev/README.md)）。

For more information about the major changes and new features in GPGPU-Sim 4.x, read [this](https://github.com/accel-sim/accel-sim-framework/blob/dev/gpu-simulator/gpgpu-sim4.md).

For more information about GPGPU-Sim, see the [original GPGPU-Sim manual](http://gpgpu-sim.org/manual/index.php/Main_Page).

The GPGPU-SIM 4.x integrated with Accel-Sim includes AccelWattch. For more information on AccelWattch, please see [AccelWattch Overview](https://github.com/VijayKandiah/accel-sim-framework#accelwattch-overview) entry in the main read-me page and the [AccelWattch MICRO'21 Artifact Manual](https://github.com/VijayKandiah/accel-sim-framework/blob/release/AccelWattch.md).
