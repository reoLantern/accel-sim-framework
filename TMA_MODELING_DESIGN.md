# TMA Modeling Design for Accel-Sim

## 1. Background & Motivation

NVIDIA Hopper (SM90) introduced three tightly-coupled features that form the core of modern GPU kernel pipelines (e.g., CUTLASS 3.x):

```
TMA Load → mbarrier Wait → wgmma Compute
```

- **TMA (Tensor Memory Accelerator)**: An independent hardware engine that asynchronously transfers tiles between global and shared memory, driven by descriptors (`CUtensorMap`).
- **mbarrier**: An asynchronous, transaction-count-based barrier in shared memory that synchronizes TMA completions with warp execution.
- **wgmma (GMMA)**: Warp-group-level matrix multiply-accumulate that operates on data in shared memory and registers.

As of March 2026, **no public GPU simulator models any of these features**. The official Accel-Sim / GPGPU-Sim codebase does not define the corresponding SASS opcodes (`UTMALDG`, `HGMMA`, `SYNCS`, `WARPGROUP`, etc.), and attempting to simulate traces containing these instructions results in an assertion failure:

```
ERROR: undefined instruction : WARPGROUP.ARRIVE  Opcode: WARPGROUP
Assertion `0 && "undefined instruction"' failed.
```

This document describes our design for extending Accel-Sim's trace-driven pipeline to model TMA, starting with the most critical missing piece: **capturing TMA descriptor information at trace time**.

## 2. The Core Problem: Descriptor-Driven Transfers

Unlike traditional load/store instructions where the address and size are visible in the instruction encoding and register operands, TMA transfers are **descriptor-driven**:

1. The host creates a `CUtensorMap` descriptor (128 bytes) via `cuTensorMapEncodeTiled()`.
2. The descriptor encodes: data type, tensor rank, global memory base address, tensor dimensions, strides, tile (box) dimensions, swizzle mode, etc.
3. The descriptor is passed to the kernel (typically as a `__grid_constant__` parameter).
4. Inside the kernel, a single SASS instruction like `UTMALDG.2D` references the descriptor to trigger an asynchronous multi-kilobyte transfer.

**The transfer size is not visible in the instruction trace.** It is determined entirely by the descriptor's `boxDim` and `dataType` fields. For example, `boxDim={32,32}` with `INT32` means each TMA operation transfers `32 × 32 × 4 = 4096` bytes. Without this information, the simulator cannot generate the correct memory access requests.

## 3. Solution: Extend the NVBit Tracer (Approach B)

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CUDA Application (e.g., CUTLASS GEMM)                                  │
│                                                                         │
│  1. cuTensorMapEncodeTiled(&desc_A, FP16, 2, d_A, ..., box={128,64})   │
│  2. cuTensorMapEncodeTiled(&desc_B, FP16, 2, d_B, ..., box={64,256})   │
│  3. kernel<<<grid, block>>>(desc_A, desc_B, ...)                        │
│       └→ UTMALDG.2D ... desc_A → load A tile (128×64×2 = 16384 bytes) │
│       └→ UTMALDG.2D ... desc_B → load B tile (64×256×2 = 32768 bytes) │
└────────────────────┬────────────────────────────────────────────────────┘
                     │ LD_PRELOAD=tracer_tool.so
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  NVBit Tracer (modified tracer_tool.cu)                                 │
│                                                                         │
│  nvbit_at_cuda_event(), cbid=697 (cuTensorMapEncodeTiled):             │
│    → Capture: data_type, rank, globalAddress, boxDim, transfer_bytes   │
│    → memcpy(raw_bytes, tensorMap, 128)   // fingerprint                │
│    → Store in g_tma_descriptors[]                                      │
│                                                                         │
│  enter_kernel_launch():                                                 │
│    → cuFuncGetParamInfo + memcmp: match raw_bytes to kernel params     │
│    → Record param_idx for each descriptor                              │
│    → Write descriptor info (with param_idx) to trace header            │
│    → Write per-instruction traces (including UTMALDG addresses)        │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Trace File (kernel-1-ctx_0x....trace.xz)                               │
│                                                                         │
│  Header:                                                                │
│    -tma_desc_count = 2                                                  │
│    -tma_desc 0 param_idx=0 ... transfer_bytes=16384 box_dim=128,64 ... │
│    -tma_desc 1 param_idx=1 ... transfer_bytes=32768 box_dim=64,256 ... │
│                                                                         │
│  Body:                                                                  │
│    ... 0 UTMALDG.2D 2 R0 R0 4 2 0x1000 0                              │
│    ... 0 UTMALDG.2D 2 R0 R0 4 2 0x22802C0 0   ← desc_A device addr   │
│    ... 0 UTMALDG.2D 2 R0 R0 4 2 0x5000 0                              │
│    ... 0 UTMALDG.2D 2 R0 R0 4 2 0x2280340 0   ← desc_B device addr   │
└────────────────────┬────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Accel-Sim Simulator (future modeling)                                  │
│                                                                         │
│  1. Parse header → get tma_desc list (sorted by param_idx)             │
│  2. Scan UTMALDG addrs → sort unique desc addrs (ascending)            │
│  3. 1:1 mapping: sorted_addrs[i] ↔ tma_desc[i]                        │
│  4. On UTMALDG: lookup desc_addr → get transfer_bytes                  │
│  5. Generate global memory read requests for transfer_bytes            │
│  6. On completion: write to shmem, signal mbarrier                     │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 NVBit Hook: How cuTensorMapEncodeTiled is Intercepted

NVBit v1.7.6 natively supports intercepting `cuTensorMapEncodeTiled`:

- **Callback ID**: 697 (defined in `tools_cuda_api_meta.h:731`)
- **Parameter struct**: `cuTensorMapEncodeTiled_params` (defined in `generated_cuda_meta.h:2444`)
- **Works with dynamic resolution**: Even when the application obtains the function pointer via `cudaGetDriverEntryPointByVersion()`, NVBit's CUPTI-based callback still fires.

The hook is added in `tracer_tool.cu`, inside `nvbit_at_cuda_event()`:

```cpp
case (nvbit_api_cuda_t)697: {  // cuTensorMapEncodeTiled
    if (is_exit && *pStatus == CUDA_SUCCESS) {
        cuTensorMapEncodeTiled_params *p = (cuTensorMapEncodeTiled_params *)params;
        TMADescriptorInfo info = {};
        info.host_ptr = p->tensorMap;
        info.data_type = p->tensorDataType;
        info.rank = p->tensorRank;
        info.global_address = p->globalAddress;
        uint32_t transfer = tma_data_type_size(p->tensorDataType);
        for (uint32_t r = 0; r < p->tensorRank && r < 5; r++) {
            info.box_dim[r] = p->boxDim[r];
            info.global_dim[r] = p->globalDim[r];
            transfer *= p->boxDim[r];
        }
        info.transfer_bytes = transfer;
        memcpy(info.raw_bytes, p->tensorMap, 128);  // <-- fingerprint
        g_tma_descriptors.push_back(info);
    }
} break;
```

Key points:
- We capture on **exit** (`is_exit == true`) so that the 128-byte `CUtensorMap` has already been filled by the driver.
- `raw_bytes` is a verbatim copy of the opaque hardware descriptor — the **fingerprint** used for matching later.
- `transfer_bytes` is computed as `product(boxDim[0..rank-1]) × sizeof(element)`.

### 3.3 Trace Header Output Format

At kernel launch time (`enter_kernel_launch()`), descriptors matched to this kernel's parameters are written to the trace header, **sorted by `param_idx`** (ascending = ascending device address):

```
-tma_desc_count = 3
-tma_desc 0 param_idx=0 type=FLOAT32 rank=2 global_addr=0x73e5cfc00000
  transfer_bytes=1024 box_dim=16,16 global_dim=64,64 raw=<256 hex chars>
-tma_desc 1 param_idx=1 type=FLOAT32 rank=2 global_addr=0x73e5cfc04000
  transfer_bytes=2048 box_dim=32,16 global_dim=128,64 raw=<256 hex chars>
-tma_desc 2 param_idx=2 type=INT32 rank=2 global_addr=0x73e5cfc0c000
  transfer_bytes=2048 box_dim=16,32 global_dim=64,128 raw=<256 hex chars>
```

Fields:
| Field | Description |
|---|---|
| `param_idx` | Absolute kernel parameter index (0-based); determines device address ordering |
| `type` | Element data type (INT32, FLOAT16, BFLOAT16, etc.) |
| `rank` | Tensor dimensionality (1-5) |
| `global_addr` | Device global memory base address of the tensor |
| `transfer_bytes` | Total bytes per TMA operation = product(box_dim) × sizeof(element) |
| `box_dim` | Tile dimensions transferred per TMA operation |
| `global_dim` | Full tensor dimensions |
| `raw` | Raw 128-byte CUtensorMap content in hex (for fingerprint matching) |

## 4. Multi-Descriptor Matching via `param_idx`

### 4.1 The Problem

A UTMALDG instruction in the trace references the descriptor by its **device-side address** in parameter/constant memory:

```
UTMALDG.2D 2 R0 R0 4 2 0x400 0         ← shmem destination
UTMALDG.2D 2 R0 R0 4 2 0x26280240 0    ← descriptor device addr
```

But the tracer captures descriptors on the **host side** during `cuTensorMapEncodeTiled`. When a kernel uses multiple descriptors, we need to know which trace-side descriptor address corresponds to which captured descriptor.

The challenge: **descriptor creation order may differ from kernel parameter order**. For example:

```cpp
CUtensorMap desc_C = create(...);   // tracer captures as g_tma_descriptors[0]
CUtensorMap desc_A = create(...);   // tracer captures as g_tma_descriptors[1]
kernel<<<...>>>(desc_A, desc_C);    // param 0 = desc_A, param 1 = desc_C
```

If the trace naively writes descriptors in creation order, the simulator would mismatch them.

### 4.2 The Solution: Raw Bytes Fingerprint + `cuFuncGetParamInfo`

The solution operates in two stages:

**Stage 1 — Capture (at `cuTensorMapEncodeTiled` time):**
Each descriptor's full 128-byte content (`raw_bytes`) is saved as a unique fingerprint.

**Stage 2 — Match (at kernel launch time):**
The tracer uses `cuFuncGetParamInfo()` to iterate all kernel parameters. For each 128-byte parameter, it `memcmp`s the content against all known `raw_bytes` fingerprints. A match tells us: "this captured descriptor is at kernel parameter index N."

```cpp
// match_tma_descriptors_to_params() — tracer_tool.cu
for (int pi = 0; ; pi++) {
    size_t paramOffset, paramSize;
    CUresult res = cuFuncGetParamInfo(func, pi, &paramOffset, &paramSize);
    if (res != CUDA_SUCCESS) break;       // no more params
    if (paramSize != 128) continue;       // CUtensorMap is exactly 128 bytes
    for (each captured descriptor di) {
        if (memcmp(kernelParams[pi], g_tma_descriptors[di].raw_bytes, 128) == 0) {
            mapping.push_back({desc_idx=di, param_idx=pi});  // matched!
            break;
        }
    }
}
```

The result is a mapping sorted by `param_idx` (ascending), which is written to the trace header:

```
-tma_desc 0 param_idx=0 type=FLOAT32 transfer_bytes=1024 ...  ← desc_A (kernel param 0)
-tma_desc 1 param_idx=2 type=INT32   transfer_bytes=2048 ...  ← desc_C (kernel param 2)
```

### 4.3 End-to-End Walkthrough (Multi-Descriptor, Reversed Creation Order)

Consider our test case where descriptors are created in **reverse order** (C, B, A) but passed to the kernel as (A, B, C):

#### Step 1: Host creates descriptors (reverse order)

```cpp
CUtensorMap desc_C = create_tma_desc_2d(INT32,   d_C, 64x128, tile 16x32);  // created 1st
CUtensorMap desc_B = create_tma_desc_2d(FLOAT32, d_B, 128x64, tile 32x16);  // created 2nd
CUtensorMap desc_A = create_tma_desc_2d(FLOAT32, d_A, 64x64,  tile 16x16);  // created 3rd
multi_desc_kernel<<<2, 32>>>(desc_A, desc_B, desc_C, 4);  // param order: A, B, C
```

#### Step 2: Tracer captures in creation order

```
g_tma_descriptors[0] = desc_C (INT32,   transfer=2048, raw=<raw_C>)
g_tma_descriptors[1] = desc_B (FLOAT32, transfer=2048, raw=<raw_B>)
g_tma_descriptors[2] = desc_A (FLOAT32, transfer=1024, raw=<raw_A>)
```

#### Step 3: At kernel launch, fingerprint matching

```
cuFuncGetParamInfo(func, 0) → size=128 → memcmp(kernelParams[0], ...) → matches raw_A → {desc_idx=2, param_idx=0}
cuFuncGetParamInfo(func, 1) → size=128 → memcmp(kernelParams[1], ...) → matches raw_B → {desc_idx=1, param_idx=1}
cuFuncGetParamInfo(func, 2) → size=128 → memcmp(kernelParams[2], ...) → matches raw_C → {desc_idx=0, param_idx=2}
cuFuncGetParamInfo(func, 3) → size=4   → skip (not 128 bytes)
```

#### Step 4: Trace header (sorted by param_idx)

```
-tma_desc_count = 3
-tma_desc 0 param_idx=0 type=FLOAT32 transfer_bytes=1024 box_dim=16,16 ...  ← desc_A
-tma_desc 1 param_idx=1 type=FLOAT32 transfer_bytes=2048 box_dim=32,16 ...  ← desc_B
-tma_desc 2 param_idx=2 type=INT32   transfer_bytes=2048 box_dim=16,32 ...  ← desc_C
```

Despite creation order being C→B→A, the trace output is correctly ordered A→B→C by parameter position.

#### Step 5: Simulator matches by address ordering

Device parameter memory layout (param_idx ascending = address ascending):

```
param 0: desc_A → device addr 0x26280240  (lowest)
param 1: desc_B → device addr 0x262802c0  (+128)
param 2: desc_C → device addr 0x26280340  (+128)
param 3: run_iters → ...
```

UTMALDG instructions in trace:
```
UTMALDG @ PC 0x03d0 → desc_addr 0x26280240 → sorted_addrs[0] → tma_desc[0] → 1024B ✓
UTMALDG @ PC 0x0470 → desc_addr 0x262802c0 → sorted_addrs[1] → tma_desc[1] → 2048B ✓
UTMALDG @ PC 0x0510 → desc_addr 0x26280340 → sorted_addrs[2] → tma_desc[2] → 2048B ✓
```

### 4.4 Why This Approach is Reliable

| Property | Guarantee |
|---|---|
| **Creation order independence** | `raw_bytes` fingerprint matches by content, not by capture order |
| **Handles non-TMA params** | `cuFuncGetParamInfo` gives exact size; only 128-byte params are checked |
| **Handles descriptor copies** | Content matching works even if the CUtensorMap was copied before passing to kernel |
| **param_idx ↔ device addr ordering** | CUDA parameter layout guarantees lower param_idx = lower device address |
| **Per-kernel matching** | Each kernel launch independently matches its own parameters |
| **Zero overhead for non-TMA kernels** | `g_tma_descriptors.empty()` check returns immediately |

## 5. Verified Results

### 5.1 Test: multi_tma_desc (3 descriptors, reversed creation order)

Test case: `hopper_feature_testcases/multi_tma_desc/multi_tma_desc.cu`

Creates 3 descriptors in **reverse order** (C, B, A) but passes them to the kernel as (A, B, C):

| Descriptor | Type | Matrix | Tile | transfer_bytes | Created |
|---|---|---|---|---|---|
| desc_A | FLOAT32 | 64×64 | 16×16 | 1024 | 3rd |
| desc_B | FLOAT32 | 128×64 | 32×16 | 2048 | 2nd |
| desc_C | INT32 | 64×128 | 16×32 | 2048 | 1st |

### 5.2 Tracer output (creation order)

```
[TMA] Captured descriptor #0: type=INT32 rank=2 global_addr=0x7f55adc0c000 transfer_bytes=2048 box_dim=16x32
[TMA] Captured descriptor #1: type=FLOAT32 rank=2 global_addr=0x7f55adc04000 transfer_bytes=2048 box_dim=32x16
[TMA] Captured descriptor #2: type=FLOAT32 rank=2 global_addr=0x7f55adc00000 transfer_bytes=1024 box_dim=16x16
```

### 5.3 Trace header (correctly reordered by param_idx)

```
-tma_desc_count = 3
-tma_desc 0 param_idx=0 type=FLOAT32 rank=2 global_addr=0x7f55adc00000 transfer_bytes=1024 box_dim=16,16 ...
-tma_desc 1 param_idx=1 type=FLOAT32 rank=2 global_addr=0x7f55adc04000 transfer_bytes=2048 box_dim=32,16 ...
-tma_desc 2 param_idx=2 type=INT32   rank=2 global_addr=0x7f55adc0c000 transfer_bytes=2048 box_dim=16,32 ...
```

Key observation: despite creation order C→B→A, the trace correctly outputs A→B→C (by kernel parameter position).

### 5.4 UTMALDG address verification

```
UTMALDG @ PC 0x03d0 → desc_addr 0x26280240  → tma_desc[0] (desc_A, 1024B) ✓
UTMALDG @ PC 0x0470 → desc_addr 0x262802c0  → tma_desc[1] (desc_B, 2048B) ✓
UTMALDG @ PC 0x0510 → desc_addr 0x26280340  → tma_desc[2] (desc_C, 2048B) ✓
```

Descriptor device addresses are 128 bytes apart (`0x80`), consistent with CUtensorMap size.
48 total UTMALDG lines = 3 descriptors × 2 blocks × 4 iterations × 2 memory operands.

### 5.5 Earlier test: tma_tensor benchmark (single descriptor)

```bash
./tma_tensor -w 64 -h 64 -o UTMALDG -i 1
```

- Single descriptor: `type=INT32, box_dim={32,32}, transfer_bytes=4096`
- Trace header correctly shows `param_idx=0`

## 6. SASS Opcode Reference (SM90 TMA-related)

From trace analysis, these are the SASS instructions involved in TMA operations:

| SASS Opcode | PTX Equivalent | Role |
|---|---|---|
| `UTMALDG.2D` | `cp.async.bulk.tensor.2d.global.shared` | TMA load: global → shared |
| `UTMASTG.2D` | `cp.async.bulk.tensor.2d.shared.global` | TMA store: shared → global |
| `UTMAREDG.2D` | `cp.reduce.async.bulk.tensor.2d` | TMA reduce: shared → global |
| `UTMAPF.2D` | `cp.async.bulk.prefetch.tensor.2d.L2` | TMA prefetch to L2 |
| `UBLKPF.L2` | `cp.async.bulk.prefetch.L2.global` | Bulk (1D) prefetch |
| `UBLKCP.S.G` | `cp.async.bulk.shared.global` | Bulk copy: global → shared |
| `SYNCS.EXCH.64` | `mbarrier.init` | Initialize mbarrier (64-bit, shmem) |
| `SYNCS.ARRIVE.TRANS64.ART0` | `mbarrier.arrive` | Arrive at mbarrier |
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | `mbarrier.try_wait.parity` | Try-wait on mbarrier |
| `FENCE.VIEW.ASYNC.S` | `fence.proxy.async.shared` | Async proxy fence |

None of these opcodes are currently defined in `hopper_opcode.h`.

## 7. Roadmap

### Phase 1: GMMA (Simplest)

The HGMMA instruction encodes the matrix shape directly in the opcode string:
```
HGMMA.64x256x16.F32   → M=64, N=256, K=16, acc=F32
```
- Shared memory read volume: A = M×K×sizeof(A_type) + B = K×N×sizeof(B_type)
- Register C/D: accumulator in registers
- Latency: ~128 cycles (from microbenchmark data)
- Key challenge: warp group scheduling (4 warps cooperate)

### Phase 2: mbarrier (Without TMA tx count)

Model as an enhanced barrier using the existing LDGSTS/DEPBAR async framework:
- `SYNCS.EXCH.64` → allocate barrier state at shmem address
- `SYNCS.ARRIVE.*` → decrement arrive count
- `SYNCS.PHASECHK.*` → stall warp if barrier incomplete

### Phase 3: TMA

With descriptor info now available in traces:
1. Parse `tma_desc` from trace header
2. On `UTMALDG`: generate `transfer_bytes` worth of global memory reads, bypassing L1
3. On completion: write to shmem, signal mbarrier (integrate with Phase 2)
4. Model TMA engine as independent from SM pipeline (does not occupy SM execution units)

### Phase 4: mbarrier + TMA Integration

- TMA completion decrements mbarrier transaction count by `transfer_bytes`
- Barrier phase flips when both arrive count and tx count reach zero
- Enables full TMA → mbarrier → wgmma pipeline modeling

## 8. Files Modified

### Tracer (NVBit)

**`util/tracer_nvbit/tracer_tool/tracer_tool.cu`**:
- Added `TMADescriptorInfo` struct and `g_tma_descriptors` global storage
- Added `tma_data_type_size()`, `tma_data_type_name()` helper functions
- Added `KernelTMAMapping` struct for per-kernel descriptor-to-param mapping
- Added `match_tma_descriptors_to_params()` — uses `cuFuncGetParamInfo` + `memcmp` to match raw_bytes fingerprints to kernel parameters at launch time
- Added `write_tma_descriptors_to_trace()` for trace header output (includes `param_idx`)
- Added `case (nvbit_api_cuda_t)697` in `nvbit_at_cuda_event()` to intercept `cuTensorMapEncodeTiled`
- Modified `enter_kernel_launch()` to extract `kernelParams`, call `match_tma_descriptors_to_params()`, and write matched descriptors to trace header

### Simulator (Future)

Files that will need modification:
- `ISA_Def/hopper_opcode.h` — add UTMALDG, UTMASTG, SYNCS, HGMMA, WARPGROUP, etc.
- `ISA_Def/trace_opcode.h` — add OP_UTMALDG, OP_HGMMA, OP_SYNCS, etc.
- `trace-driven/trace_driven.cc` — parse tma_desc from trace header; handle new opcodes
- `gpgpu-sim/shader.cc` — TMA engine state machine; mbarrier state; warp group scheduling
- `gpgpu-sim/shader.h` — new data structures for TMA/mbarrier/warp group state
