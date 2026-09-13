# AMD XDNA (Ryzen AI NPU) backend

`ggml-xdna` runs matrix multiplications on the NPU of AMD Ryzen APUs. It targets the first-generation
XDNA NPU (Phoenix / Hawk Point, e.g. the Ryzen 7 8700G), which no vendor LLM runtime supports.
It is developed and measured on a Ryzen 7 8700G under Windows 11; Linux with the amdxdna driver and
XDNA2 parts are untested (XDNA2 needs its kernels rebuilt, see
[kernels/README.md](../../ggml/src/ggml-xdna/kernels/README.md)).

## How it works

The backend lives on host memory, like `ggml-blas`: it claims `MUL_MAT` and `MUL_MAT_ID` (mixture of
experts) nodes from a CPU graph and leaves every other op to the CPU. Each claimed product is split
by output rows between the NPU, a private CPU backend and, in a Vulkan build, a private Vulkan backend
on the integrated GPU, all working at the same time. An auto split learns each worker's speed per
matrix shape and gives the NPU whole kernel blocks.

- The NPU runs precompiled bf16 (and int8-weight) block GEMM kernels built with the open IRON
  toolchain (mlir-aie). They ship in `ggml/src/ggml-xdna/kernels/`.
- The host converts the NPU's weight rows to bf16 (or int8) once and caches them; any ggml weight
  type works, including the CPU's repacked Q4_0 / Q4_K / Q2_K / IQ4_NL / MXFP4 layouts.
- By default only products with at least 32 tokens go to the NPU, i.e. prompt processing. Token
  generation stays on the CPU (or GPU) unless `GGML_XDNA_NPU_DECODE` is set.

## Requirements

- An XDNA NPU with its driver: on Windows the AMD NPU / Ryzen AI driver, which installs XRT
  (`C:\Windows\System32\xrt_coreutil.dll`); on Linux the amdxdna driver and XRT (`/opt/xilinx/xrt`).
- The XRT headers are vendored in `ggml/src/ggml-xdna/xrt-include`; on Windows the import library is
  generated from the driver's DLL at configure time.

## Build

From a Visual Studio developer shell on Windows (or any shell on Linux):

```bash
cmake -B build-xdna -G Ninja -DGGML_XDNA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-xdna --config Release
```

Add `-DGGML_VULKAN=ON` (and the Vulkan SDK paths) for the Vulkan worker on the integrated GPU; the
same build then also offloads layers to GPUs with `-ngl` as usual.

CMake cache variables:

| variable | default | meaning |
|---|---|---|
| `GGML_XDNA_XRT_INCLUDE_DIR` | `ggml/src/ggml-xdna/xrt-include` | XRT public headers |
| `GGML_XDNA_XRT_LIB` | generated (Windows) / found in `/opt/xilinx/xrt/lib` | `xrt_coreutil` library |
| `GGML_XDNA_KERNEL_DIR` | `ggml/src/ggml-xdna/kernels` | kernel directory compiled in as the default |

## Running

Keep the layers on the CPU and the backend takes the matmuls it can:

```bash
llama-server -m model.gguf -ngl 0
llama-bench  -m model.gguf -ngl 0 -p 512 -n 0
```

In a Vulkan build, use `-ngl 0 --device none` to keep the model off the GPUs (the backend's own
Vulkan worker still helps with the weights it can read), or `-ngl 999` to run the whole model on a
GPU, where the NPU plays no part.

Run one XDNA process at a time: the XDNA1 driver grants 5 NPU hardware contexts in total, one per
loaded kernel, and processes sharing them unload each other's kernels. Side processes such as
`--version` or `--list-devices` next to a running server should set `GGML_XDNA_DISABLE=1`.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `GGML_XDNA_NPU_SHARE` | `auto` | NPU share of each product: `auto` (planned per shape) or a fixed 0..1 |
| `GGML_XDNA_NPU_W8` | `0` | `1`: the NPU reads int8 weights (one scale per row and 1,024 weights): half the cache memory, slightly less accurate |
| `GGML_XDNA_NPU_CACHE_MB` | `4096` | memory for the NPU's converted weights (weights beyond it are converted per call) |
| `GGML_XDNA_NPU_WARM` | `1` | `0`: no load-time warm-up (converting weights and loading kernels while the model loads) |
| `GGML_XDNA_MIN_BATCH` | `32` | fewest tokens a product needs to go to the NPU (a huge value turns the NPU off) |
| `GGML_XDNA_NPU_DECODE` | `0` | experimental token generation on the NPU: `1` all rows, a fraction shares with the CPU |
| `GGML_XDNA_DECODE_CACHE_MB` | `8192` | memory for the decode path's weight store |
| `GGML_XDNA_KERNEL_DIR` | build-time kernel dir | where to load the NPU kernels from |
| `GGML_XDNA_MAX_KERNELS` | `5` | kernels kept loaded at once (one hardware context each) |
| `GGML_XDNA_REPACK` | `1` on x86 | `0`: leave matmuls on the CPU's repacked weights to the CPU |
| `GGML_XDNA_DISABLE` | `0` | `1`: report no XDNA device and never touch the NPU |
| `GGML_XDNA_DEBUG` | unset | log the backend's decisions (op splits, kernel choice; use `-v` in llama-bench) |
| `GGML_XDNA_VK_SHARE` | `0.8` | Vulkan build: the GPU worker's starting share |
| `GGML_XDNA_VK_DEVICE` | first AMD / Radeon device | Vulkan build: device index of the GPU worker |
| `GGML_XDNA_NPU_MIN_GFLOP` | `4` with a GPU worker, else `0` | smallest product the NPU takes part in next to the GPU worker |
| `GGML_XDNA_VK_ZERO_COPY` | `1` | Vulkan build: import activations and results instead of copying them |
| `GGML_XDNA_VK_IMPORT_ALIGN` | `4096` | Vulkan build: alignment of host-memory imports |
| `GGML_XDNA_VK_CACHE_MB` | `8192` | Vulkan build: memory for the GPU worker's weight copies |
| `GGML_XDNA_W8_EMULATE` | `0` | diagnostic: round bf16 weights through int8 with this group size |

Recommended: leave everything at its default. For models of 20B and more, `GGML_XDNA_NPU_W8=1` with
`GGML_XDNA_NPU_CACHE_MB=5120` covers as many weights as a 10 GB bf16 cache.

## Performance

Ryzen 7 8700G (8 Zen 4 cores, 4-column XDNA1 NPU), `-ngl 0`, warm prompt processing:

| model | CPU | CPU + NPU (auto) |
|---|---|---|
| Qwen3-0.6B Q4_0 (llama-bench pp512) | ~627 t/s | ~930 t/s |
| gemma-4-E2B Q4_K_M (llama-bench pp512) | ~231 t/s | 286-304 t/s |
| Qwen2.5-0.5B Q4_K_M (server, 2,134-token prompt) | 486 t/s | 774-793 t/s |
| llama3.2:3b Q4_K_M (server, 2,134-token prompt) | 144 t/s | 206 t/s |
| Mixtral 8x7B Q4_0 (llama-bench pp512) | 14.0 t/s | 18.0 t/s |
| gpt-oss-20b MXFP4 (ggml-org GGUF, llama-bench pp512) | 58.5 t/s | 72.3 t/s |
| 27B (llama-server) | 17.4 t/s | ~19.8 t/s |

Token generation is unchanged by default. With `GGML_XDNA_NPU_DECODE=1` the NPU generates tokens at
~0.22 ms per launch of a 1,024 x 1,024 weight block (Qwen3-0.6B: 7.8 t/s, against ~130 on the CPU),
so it is an experiment. Accuracy (KL divergence against the CPU) stays at the CPU's own kernel-noise
level: Qwen3-0.6B 0.002 (bf16) / 0.004 (int8).
