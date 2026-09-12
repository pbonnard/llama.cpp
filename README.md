# llama.cpp-xdna — CPU, iGPU and XDNA1 NPU on AMD Ryzen APUs

This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that adds an experimental ggml backend,
`ggml-xdna`, for the first-generation AMD XDNA NPU (Phoenix / Hawk Point: Ryzen 7040 / 8040 mobile, Ryzen 8600G /
8700G desktop). The backend works together with the CPU and the integrated Radeon GPU (through Vulkan).

For everything else about llama.cpp — installation, models, tools, the server, the other backends — see the
[original README](https://github.com/ggml-org/llama.cpp/blob/master/README.md) and the
[upstream repository](https://github.com/ggml-org/llama.cpp). This fork is not affiliated with the llama.cpp project.

## Objectives

- **Put the XDNA1 NPU to work for LLM inference.** The vendor LLM stacks (Ryzen AI Software, Lemonade, FastFlowLM)
  only support XDNA2 (Ryzen AI 300 and later), and llama.cpp has no NPU backend.
- **Do it as a regular ggml backend.** The CPU backend is unchanged; the scheduler hands the new backend only the
  operations it claims.
- **Combine the chip's compute units.** Each large matrix multiply is split across the NPU, the CPU and the Radeon 780M,
  all running at the same time.
- **Measure on real hardware** and keep as defaults only what actually helps.

## How it works

- Only prompt-processing matrix multiplies (`MUL_MAT` with 32 or more tokens) go to the backend. Token generation
  stays on the CPU.
- The NPU runs precompiled bf16 → f32 block matrix-multiply kernels, built with the open-source
  [IRON / mlir-aie](https://github.com/Xilinx/mlir-aie) toolchain and launched through XRT, which ships with the AMD
  NPU driver. Weights and activations are converted to bf16 on the host, and larger products are tiled into blocks.
- Each matrix multiply the backend receives is split by output rows between the NPU, a CPU backend and, when built
  with Vulkan, a Vulkan backend on the integrated GPU. On the APU's shared memory the Vulkan worker reads the
  activations and writes its results in place, and keeps weight slices cached on the GPU.
- For each block the backend picks the NPU kernel with the lowest measured launch time.

## Results

Ryzen 7 8700G, 8 threads, ~476-token prompt.

**Correctness:** `test-backend-ops -b XDNA -o MUL_MAT` passes 11/11 on both builds (CPU + NPU, and Vulkan + NPU).
One 512×512×128 NPU block runs in ~360 µs including dispatch (~185 GFLOPS), with a normalized mean squared error of
3e-14 against a bf16 reference.

**Prompt processing, CPU + NPU build** — median of 5 runs, tokens/s, run with `--no-repack` unless noted:

| configuration | Qwen3-0.6B Q4_0 | gemma-4-E2B Q4_K_M |
|---|---:|---:|
| CPU only, stock (repacked weights) | 624 | 218 |
| CPU only | 428 | 116 |
| NPU only | 311 | 77 |
| CPU + NPU, NPU share 0.4 (default) | 449 | 153 |
| CPU + NPU, NPU share 0.5 | 474 | 114 |

Token generation speed is the same in every configuration (~90 tokens/s for Qwen3-0.6B, ~30 for gemma-4-E2B).

**Prompt processing, Vulkan + NPU build** — best of 2 runs, tokens/s:

| configuration | Qwen3-0.6B Q4_0 | gemma-4-E2B Q4_K_M |
|---|---:|---:|
| XDNA backend: Vulkan 0.8 + CPU 0.2 (default with Vulkan) | ~715 | ~260 |
| XDNA backend: any split that includes the NPU | 400–540 | ~160 |
| Whole model on the 780M (llama.cpp's own Vulkan backend, `-ngl 99`) | 2000–3200 | ~530 |

**What this shows:**

- CPU + NPU is faster than the CPU alone in the same configuration: +5–11% on Qwen3-0.6B and +32% on gemma-4-E2B.
  The larger model gains more because its matrix multiplies cover the NPU's fixed per-launch cost better.
- The stock CPU path is still faster overall. Its repacked weight layout is invisible to other backends, so the NPU
  can only take part with `--no-repack`.
- The 780M is roughly 10× faster than the XDNA1 at batched matrix multiply. Next to it the NPU only makes each
  operation finish later, so it is off by default when a Vulkan worker is available.
- On an APU, keeping the whole model on the integrated GPU is by far the fastest option when it fits in memory.

## Building (Windows)

Requirements: the AMD NPU driver (it provides XRT, `xrt_coreutil.dll`), Visual Studio 2022 or later with C++, CMake
and Ninja. The Vulkan worker additionally needs `glslc`, the Vulkan headers and SPIRV-Headers.

```sh
# from a Visual Studio developer shell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA=ON                  # CPU + NPU
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA=ON -DGGML_VULKAN=ON # CPU + NPU + iGPU
cmake --build build --config Release
```

The import library for `xrt_coreutil.dll` is generated from the driver at configure time. Prebuilt XDNA1 kernels are
in [`ggml/src/ggml-xdna/kernels/`](ggml/src/ggml-xdna/kernels/README.md); rebuilding them needs mlir-aie (see that
folder's README). The CMake files also support Linux with XRT under `/opt/xilinx/xrt`, but only Windows has been
tested.

## Running

```sh
llama-cli -m model.gguf --no-repack   # --no-repack lets the NPU receive the prompt matrix multiplies
```

| variable | effect |
|---|---|
| `GGML_XDNA_NPU_SHARE` | NPU share of each matrix multiply, `0`–`1` or `auto` (default 0.4 without Vulkan, 0 with Vulkan) |
| `GGML_XDNA_VK_SHARE` | Vulkan share when the Vulkan worker is available (default 0.8); the CPU takes the rest |
| `GGML_XDNA_MIN_BATCH` | smallest token count sent to the backend (default 32) |
| `GGML_XDNA_NPU_MIN_GFLOP` | with a Vulkan worker, the NPU only takes operations at least this large (default 4) |
| `GGML_XDNA_VK_DEVICE` | Vulkan device used by the worker (default: the first AMD device) |
| `GGML_XDNA_KERNEL_DIR` | directory with the NPU kernels |
| `GGML_XDNA_DEBUG` | set to `1` to log routing decisions |

More detail is in the [XDNA section of docs/build.md](docs/build.md#xdna-amd-ryzen-ai-npu) and the
[kernels README](ggml/src/ggml-xdna/kernels/README.md).

## Limitations and next steps

- Weights are converted to bf16 on every call; caching the converted weights would remove most of the host-side work.
- The backend cannot read the CPU backend's repacked weight layouts yet, which is why `--no-repack` is needed.
- NPU launches are synchronous and don't overlap with other work in the graph.
- There are only bf16 kernels. Integer kernels that match the quantized weight formats would suit the NPU better.
- Only tested on Windows with a Ryzen 7 8700G.

## License

MIT, like llama.cpp (see [LICENSE](LICENSE)). The vendored XRT headers in `ggml/src/ggml-xdna/xrt-include/` come from
[Xilinx/XRT](https://github.com/Xilinx/XRT) under the Apache-2.0 license; four of them (`ert.h`, `xclbin.h`,
`xrt_error_code.h`, `xclerr.h`) are dual-licensed Apache-2.0 or GPL-2.0.
