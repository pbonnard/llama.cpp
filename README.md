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
  NPU driver. Model weights are converted to bf16 once and cached; activations are converted on each call. Larger
  products are tiled into blocks, and the host prepares the next block while the NPU computes the current one.
- Each matrix multiply the backend receives is split by output rows between the NPU, a CPU backend and, when built
  with Vulkan, a Vulkan backend on the integrated GPU. On the APU's shared memory the Vulkan worker reads the
  activations and writes its results in place, and keeps weight slices cached on the GPU.
- For each block the backend picks the NPU kernel with the lowest measured launch time.
- By default (`auto`) the backend learns how fast each worker is for every matrix shape, and gives the NPU the number
  of rows — in whole kernel blocks, possibly none — that makes the operation finish earliest.

## Results

Ryzen 7 8700G, 8 threads, ~476-token prompt.

**Correctness:** `test-backend-ops -b XDNA -o MUL_MAT` passes 11/11 on both builds (CPU + NPU, and Vulkan + NPU).
One 512×512×128 NPU block runs in ~360 µs including dispatch (~185 GFLOPS), with a normalized mean squared error of
3e-14 against a bf16 reference. Perplexity of Qwen3-0.6B on a short text is 110.8 on the CPU and 111.5–111.7 with the
NPU, well within the measurement's error.

**Prompt processing, CPU + NPU build** — median of 5 runs, tokens/s, run with `--no-repack` unless noted:

| configuration | Qwen3-0.6B Q4_0 | gemma-4-E2B Q4_K_M |
|---|---:|---:|
| CPU only, stock (repacked weights) | 582 | 217 |
| CPU only | 398 | 125 |
| NPU only | 380 | 105 |
| CPU + NPU, NPU share 0.4 | 474 | 164 |
| CPU + NPU, NPU share 0.5 | 509 | 153 |
| CPU + NPU, `auto` (default) | 518 | 162 |

Token generation speed is the same in every configuration (~90 tokens/s for Qwen3-0.6B, ~30 for gemma-4-E2B).

**Prompt processing, Vulkan + NPU build** — best of 2 runs, tokens/s, measured before the NPU pipelining change:

| configuration | Qwen3-0.6B Q4_0 | gemma-4-E2B Q4_K_M |
|---|---:|---:|
| XDNA backend: Vulkan 0.8 + CPU 0.2 (default with Vulkan) | ~715 | ~260 |
| XDNA backend: any split that includes the NPU | 400–540 | ~160 |
| Whole model on the 780M (llama.cpp's own Vulkan backend, `-ngl 99`) | 2000–3200 | ~530 |

**What this shows:**

- CPU + NPU is faster than the CPU alone in the same configuration: +19–30% on Qwen3-0.6B and +19–31% on
  gemma-4-E2B at NPU shares 0.3–0.5 or `auto`. `auto` is the fastest on Qwen3-0.6B and ties the best fixed share on
  gemma-4-E2B, without tuning per model. Pipelining the NPU blocks and caching bf16 weights made the NPU alone 17–26%
  faster in the same conditions (311 → 365 tokens/s on Qwen3-0.6B, 77 → 97 on gemma-4-E2B).
- The stock CPU path is still faster overall (~12% ahead on Qwen3-0.6B, ~34% on gemma-4-E2B). Its repacked weight
  layout is invisible to other backends, so the NPU can only take part with `--no-repack`.
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
| `GGML_XDNA_NPU_SHARE` | NPU share of each matrix multiply, `0`–`1` or `auto` (default: `auto` starting at 0.4 without Vulkan, 0 with Vulkan) |
| `GGML_XDNA_VK_SHARE` | Vulkan share when the Vulkan worker is available (default 0.8); the CPU takes the rest |
| `GGML_XDNA_MIN_BATCH` | smallest token count sent to the backend (default 32) |
| `GGML_XDNA_NPU_MIN_GFLOP` | with a Vulkan worker, the NPU only takes operations at least this large (default 4) |
| `GGML_XDNA_NPU_CACHE_MB` | memory for the NPU's cached bf16 weights (default 4096); `0` converts weights on every call |
| `GGML_XDNA_VK_DEVICE` | Vulkan device used by the worker (default: the first AMD device) |
| `GGML_XDNA_KERNEL_DIR` | directory with the NPU kernels |
| `GGML_XDNA_DEBUG` | set to `1` to log routing decisions |

More detail is in the [XDNA section of docs/build.md](docs/build.md#xdna-amd-ryzen-ai-npu) and the
[kernels README](ggml/src/ggml-xdna/kernels/README.md).

## Limitations and next steps

- The bf16 weight cache costs 2 bytes per cached weight (the NPU's share of the model's matrices); past
  `GGML_XDNA_NPU_CACHE_MB` the weights are converted on every call.
- The backend cannot read the CPU backend's repacked weight layouts yet, which is why `--no-repack` is needed.
- NPU blocks are pipelined within an operation, but each operation waits for all of its workers before the graph
  moves on.
- There are only bf16 kernels. Integer kernels that match the quantized weight formats would suit the NPU better.
- Only tested on Windows with a Ryzen 7 8700G.

## License

MIT, like llama.cpp (see [LICENSE](LICENSE)). The vendored XRT headers in `ggml/src/ggml-xdna/xrt-include/` come from
[Xilinx/XRT](https://github.com/Xilinx/XRT) under the Apache-2.0 license; four of them (`ert.h`, `xclbin.h`,
`xrt_error_code.h`, `xclerr.h`) are dual-licensed Apache-2.0 or GPL-2.0.
