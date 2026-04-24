# `modern-modeling-v2` branch

This branch ports the **MICRO 2025 "Dissecting and Modeling the Architecture of Modern
GPU Cores"** (Huerta et al.) simulator into the mainline Accel-Sim framework.

Upstream source: `modern-gpu-simulator-micro-2025/simulator-remodeled/` (a separate
fork-style release). This branch keeps Accel-Sim's tool ecosystem (`run_hw_trace.py`,
`run_hw.py`, `run_simulations.py`, AccelWattch, CI) while replacing the core simulator
with the MICRO 2025 tree.

**Baseline commit**: tagged `baseline-2026-04-24` on this branch, on
`gpu-simulator/gpgpu-sim` (inner), and on `gpu-app-collection` `modern-modeling-v2`.

## What changed vs `dev`

### Simulator core — wholesale-replaced
- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/` — core simulator (subcore, PRT, L0I/L1I/L1D pipelines, scoreboard, OOO path disabled)
- `gpu-simulator/gpgpu-sim/src/abstract_hardware_model.{h,cc}` — new fields (per-subcore state, `AccessCoalInfo`, kernel_info extensions)
- `gpu-simulator/gpgpu-sim/src/cuda-sim/` — ptx_ir extensions for MICRO 2025 trace format
- `gpu-simulator/trace-driven/` + `gpu-simulator/trace-parser/` — new protobuf trace loader (MICRO 2025 `.pb` format)
- `gpu-simulator/ISA_Def/` — opcode tables (Hopper preserved, others re-imported)
- `gpu-simulator/accel-sim.cc` — new entry seam
- `gpu-simulator/configs/tested-cfgs/SM75_RTX2070_S/` — updated trace.config for MICRO 2025 options
- `gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM75_RTX2070_S/gpgpusim.config` — 26 new MICRO 2025 options enabled

### Simulator — kept from vanilla
- AccelWattch power model (`src/accelwattch/`) — integrates cleanly with the ported core; verified byte-identical power report vs upstream MICRO 2025.
- Most of GPGPU-Sim's interconnect / DRAM / config parsing.

### Framework tools — patched
All in `util/` and documented in `usage.md`:

| File | Patch | Commit |
|---|---|---|
| `util/tracer_nvbit/run_hw_trace.py` | env var `DYNAMIC_KERNEL_LIMIT_END` (not `_RANGE`); skip missing binaries; drop post-traces-processing (tracer writes .pb directly); `rm -rf` to survive reruns | `1639078`, `a875f3b`, `fd2ce98` |
| `util/hw_stats/run_hw.py` | skip missing binaries | `59460ec` |
| `util/job_launching/run_simulations.py` | trace arg path `kernelslist.g` → `dynamic_trace.pb` | `ecd0a06` |
| `util/job_launching/configs/define-standard-cfgs.yml` | add `RTX2070_S` entry | `181031e` |
| `gpu-simulator/main.cc` | default `OMP_NUM_THREADS=1` if env unset | `5d5c92d` |

### Benchmark collection — `gpu-app-collection` patches
Fork: `github.com:reoLantern/gpu-app-collection` branch `modern-modeling-v2`.

| Area | Fix | Commit |
|---|---|---|
| `setup_environment` | CUDA 13 gencode block; drop pre-Turing SM | `00a8abf5` |
| polybench common.mk | drop obsolete `compute_10…70` gencodes | `5e30e377` |
| 18 suite Makefiles | sed-strip obsolete gencodes | `e02cc354` |
| 491 source files | `cudaThreadSynchronize` → `cudaDeviceSynchronize` | `279482c5` |
| huffman + GPU_Microbenchmark/gpuConfig.h | `cudaThreadExit`/`deviceOverlap`/`memoryClockRate` → CUDA 13 equivalents | `07e35b6d` |
| rodinia-3.1 gaussian | `deviceProp.clockRate`/`.deviceOverlap` → stubbed (printf-only) | `28bae3d6` |

## Validation status (baseline `2026-04-24`)

- **Port fidelity (Tier 0)**: 43 apps × 13–16 perf stats + `accelwattch_power_report.log` are byte-identical with upstream MICRO 2025 `simulator-remodeled`, when both are given enough wall-clock to finish. See `accel-hopper-paper/discussions/2026-04-24_micro2025_port_100pct_validated.md` and `scripts/stage2/ab_compare.sh`.
- **Cycle MAPE (Tier 2)**: 44 real apps on RTX 2070 Super, geomean 17.19% (under canonical `gpc__cycles_elapsed.avg`). See `accel-hopper-paper/docs/current_baseline.md`.
- **OpenMP determinism (Tier 1)**: deferred; last tested 2026-04-16 on 12-microbench (byte-identical OMP=1 vs =8 on that tier).
- **Hopper (Tier 3)**: not started.

## Known limitations

- 26 upstream Accel-Sim correlator stats (`Total_core_cache_stats_breakdown[...]` family) are not emitted by the MICRO 2025 core — `util/plotting/plot-correlation.py` fails with KeyError on the first missing stat. Use `scripts/stage2/compute_mape.py` (DIY MAPE) until stat-emission wrapper is ported.
- `OoO execution path` from MICRO 2025 is disabled (MICRO 2025 has it as opt-in; we did not port the opt-in machinery).
- `InterWarpCoalescingUnit` (MICRO 2025's optional inter-warp coalescer) is not ported; defaults to 1:1 `m_prts_requesting` vector.
- Thrust-heavy benchmarks (e.g. `lonestargpu-2.0/pta`) not yet buildable on CUDA 13 (`thrust::unary_function` removed).

## Relationship to upstream

- MICRO 2025 `simulator-remodeled` links `libcudart.so` dynamically; our build links statically → our binary is ~5% faster wall-clock in independent-A/B runs. Semantics identical.
- MICRO 2025 `run_hw_trace.py` has custom `TRACES_ROOT_DIR` + `--compressed` handling we did not port; we kept the simpler vanilla flow + our `-l` / skip-missing patches.
