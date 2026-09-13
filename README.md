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

- Prompt-processing matrix multiplies (`MUL_MAT`, and `MUL_MAT_ID` for mixture-of-experts layers, with 32 or more
  tokens) go to the backend. Token generation stays on the CPU unless the experimental `GGML_XDNA_NPU_DECODE` is set.
- The NPU runs precompiled block matrix-multiply kernels (bf16, or int8 weights with bf16 activations, f32 results),
  built with the open-source [IRON / mlir-aie](https://github.com/Xilinx/mlir-aie) toolchain and launched through
  XRT, which ships with the AMD NPU driver. A launch costs ~180 µs of dispatch whatever its size, so the kernels take
  a whole 512-token batch at once, in blocks of 512 or 256 weight rows and 1,024 to 2,048 inputs. The host prepares
  the next block while the NPU computes the current one.
- Model weights are converted to bf16 (or int8) once and cached. Every ggml weight type works, including the layouts
  the CPU backend re-arranges for its SIMD kernels (repacked Q4_0, Q4_K, Q2_K, IQ4_NL and MXFP4), which the backend
  converts back exactly.
- Each matrix multiply the backend receives is split by output rows between the NPU, a CPU backend and, when built
  with Vulkan, a Vulkan backend on the integrated GPU. On the APU's shared memory the Vulkan worker reads the
  activations and writes its results in place, and keeps a copy of each weight on the GPU.
- By default (`auto`) the backend learns how fast each worker is for every matrix shape, and gives the NPU the number
  of rows — in whole kernel blocks, possibly none — that makes the operation finish earliest. It predicts the NPU
  from each kernel's measured launch time, corrected per shape by measurement.

## Results

Ryzen 7 8700G, 8 threads, layers on the CPU (`-ngl 0`), warm prompt processing, tokens/s.

**Prompt processing** — the CPU alone (stock llama.cpp path, repacked weights) against CPU + NPU (defaults):

| model | CPU | CPU + NPU | |
|---|---:|---:|---:|
| Qwen3-0.6B Q4_0 (`llama-bench -p 512`) | ~627 | ~930 | +48% |
| gemma-4-E2B Q4_K_M (`llama-bench -p 512`) | ~231 | 315–333 | +40% |
| llama3.2:3b Q4_K_M (`llama-bench -p 512`) | ~167 | ~250 | +50% |
| llama3.2:3b Q4_K_M (llama-server, 2,134-token prompt, int8 weights) | 144 | 206 | +43% |
| Qwen2.5-0.5B Q4_K_M (llama-server, 2,134-token prompt) | 471–486 | 774–793 | +59–68% |
| gpt-oss-20b MXFP4 (`llama-bench -p 512`) | 58.5 | 72.3 | +24% |
| Mixtral 8x7B Q4_0 (`llama-bench -p 512`) | 14.0 | 18.0 | +29% |
| qwen3.8:27b Q4_K_M (llama-server, int8 weights, 5 GB cache) | ~17.4 | ~19.8 | +14% |

The repacked formats the backend converts back, on Qwen3-0.6B requantized to each (`llama-bench -p 512`): Q2_K
524 → 830, IQ4_NL 599 → 896, MXFP4 609 → 912.

**Vulkan + NPU build**, Qwen3-0.6B, `llama-bench -p 512`:

| configuration | Q4_0 | Q8_0 |
|---|---:|---:|
| CPU alone | 590–620 | 444–463 |
| XDNA backend, defaults, GPUs off (`-ngl 0 --device none`) | 849–876 | 983–1,032 |
| whole model on the 780M (llama.cpp's own Vulkan backend, `-ngl 99`) | 3,564–3,748 | |

The Vulkan worker cannot read the CPU's repacked weights (Q4_0), so there the NPU does the extra work; on plain weights
(Q8_0) the GPU worker does most of it.

**Token generation** is unchanged by the backend: 115.3 tokens/s on the CPU, 114.3 with the NPU on, 118.6 on the
CPU again (Qwen2.5-0.5B, llama-server, 256 tokens after a short prompt). The experimental NPU decode path
(`GGML_XDNA_NPU_DECODE=1`) generates on the NPU at 7.7 tokens/s (bf16) / 8.1 (int8) on Qwen3-0.6B, against ~130 on
the CPU.

**Correctness:** `test-backend-ops -b XDNA` passes `MUL_MAT` 11/11 (58/58 with NPU decode) and `MUL_MAT_ID` 308/308
on both builds. Every kernel matches a reference with a normalized mean squared error of ~1e-14. Against the CPU's
own results (`llama-perplexity --kl-divergence`), Qwen3-0.6B drifts by a KL divergence of 0.002 with bf16 weights
and 0.004 with int8 weights, the CPU's own kernel-to-kernel noise level. Converting repacked weights back is exact: the
NPU gives the same logits from repacked weights as from `--no-repack` (KL divergence 0) for all five formats.

**What this shows:**

- CPU + NPU is 14–68% faster than the CPU alone at prompt processing on every model tried (the least on the 27B
  model, whose NPU weights only partly fit the cache), with default settings and no tuning per model. `auto` matches
  or beats the best fixed NPU share.
- The main levers were large launches (a whole 512-token batch per launch roughly doubled the NPU's throughput: the
  NPU alone went from ~370 to ~815 tokens/s on Qwen3-0.6B), splitting finer (256-row blocks), and reading the CPU's
  repacked weights, so the CPU keeps its fastest kernels.
- Large models need the NPU's weights to fit its cache. On qwen3.8:27b, int8 weights (`GGML_XDNA_NPU_W8=1`) in a 5 GB
  cache give the same gain as bf16 weights in a 10 GB one; the default 4 GB bf16 cache adds nothing there.
- Mixture-of-experts models go through the NPU too: Mixtral 8x7B +29%, gpt-oss-20b +24%.
- The NPU does not help token generation: the XDNA1's smallest efficient block is 64 tokens, so generating one token
  at a time wastes almost all of it.
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
llama-server -m model.gguf -ngl 0                    # CPU + NPU (and the iGPU worker in a Vulkan build)
llama-server -m model.gguf -ngl 0 --device none      # Vulkan build: keep the model off the GPUs
llama-server -m model.gguf -ngl 999                  # Vulkan build: whole model on a GPU (the NPU plays no part)
```

The defaults are the recommended settings. The most useful variables:

| variable | effect |
|---|---|
| `GGML_XDNA_NPU_SHARE` | NPU share of each matrix multiply, `0`–`1` or `auto` (default `auto`) |
| `GGML_XDNA_NPU_W8` | `1` makes the NPU read model weights as int8 (one scale per row and 1,024 weights): half the cache memory, slightly less accurate. Use it on large models, with `GGML_XDNA_NPU_CACHE_MB` around 5120 |
| `GGML_XDNA_NPU_CACHE_MB` | memory for the NPU's converted weights (default 4096); weights beyond it are converted on every call |
| `GGML_XDNA_NPU_WARM` | `0` turns off the load-time warm-up (default: load the NPU's kernels and convert its weights in the background while the model loads) |
| `GGML_XDNA_NPU_DECODE` | experimental token generation on the NPU: `1` for all rows, or a fraction shared with the CPU (default off; slower than the CPU) |
| `GGML_XDNA_MIN_BATCH` | smallest token count sent to the backend (default 32); a huge value turns the NPU off |
| `GGML_XDNA_DISABLE` | `1` hides the backend entirely (no device, the NPU is not opened), e.g. for side processes |
| `GGML_XDNA_DEBUG` | `1` logs routing decisions |

Every variable, with its default, is listed in [docs/backend/XDNA.md](docs/backend/XDNA.md); the kernels are described
in the [kernels README](ggml/src/ggml-xdna/kernels/README.md).

## Limitations and next steps

- Run one XDNA process at a time. The XDNA1 driver grants 5 NPU hardware contexts in total, one per loaded kernel; a
  model uses 2–5 (gemma-4-E2B uses all 5), and processes sharing the NPU unload each other's kernels (~90 ms per
  reload).
- The NPU's weight cache costs 2 bytes per cached weight in bf16, 1 byte with `GGML_XDNA_NPU_W8=1`; past
  `GGML_XDNA_NPU_CACHE_MB` the weights are converted on every call, which on large models cancels the NPU's gain. Size
  the cache from the model and the free RAM (Windows without a page file fails large allocations outright; the cache
  stops growing 2 GB short of the commit limit).
- On gpt-oss-20b the NPU's results drift from the CPU's more than on the other models (KL divergence 0.015 with the
  default split, 0.028 with the NPU taking every row, against the CPU's own 0.0096); the weight conversion is exact, so
  the cause is elsewhere and still open.
- A prompt that arrives while the model is still loading (as with `llama-cli` or `llama-bench`) stops the load-time
  warm-up, and the weights it had not reached are converted on first use. Routing matrix multiplies through the
  backend costs up to ~3% on a small model (one extra graph split per operation).
- Token generation on the NPU would need a true matrix-vector kernel instead of a 64-token block; even then the XDNA1
  is unlikely to beat the CPU at it.
- Next: int8 × int8 kernels (the NPU's integer multiply rate is higher than its bf16 rate), and a smarter choice of
  kernels per model so that more kernel shapes fit the 5-context budget.
- Only tested on Windows with a Ryzen 7 8700G.

## License

MIT, like llama.cpp (see [LICENSE](LICENSE)). The vendored XRT headers in `ggml/src/ggml-xdna/xrt-include/` come from
[Xilinx/XRT](https://github.com/Xilinx/XRT) under the Apache-2.0 license; four of them (`ert.h`, `xclbin.h`,
`xrt_error_code.h`, `xclerr.h`) are dual-licensed Apache-2.0 or GPL-2.0.
