# ggml-xdna kernels

Precompiled NPU block-GEMM kernels loaded by the XDNA backend at runtime. Each kernel is a pair

```
mm_bf16_f32_M<M>_K<K>_N<N>.xclbin      # AIE array configuration + compiled tile programs
mm_bf16_f32_M<M>_K<K>_N<N>.insts.bin   # runtime-sequence (DMA) instruction stream
```

computing, for one fixed block shape,

```
A : [M x K] bf16 row-major   (weight rows       = ggml src0, ne01 x ne00)
B : [N x K] bf16 row-major   (activation rows   = ggml src1, ne11 x ne10)
C : [N x M] f32  row-major   C[t][r] = sum_k A[r][k] * B[t][k]   (= ggml dst layout)
```

They are the mlir-aie `programming_examples/basic/matrix_multiplication/whole_array` design built
with `--dtype_in bf16 --dtype_out f32 --b-col-maj 1 --c-col-maj 1`, so ggml's native layouts go
straight into the NPU with no transposes. The backend tiles larger matmuls over (M, K, N) blocks,
picks the kernel with the least padding, and accumulates partial K products on the host.

## Building

Device code is compiled with the open-source IRON toolchain (mlir-aie + Peano); no NPU is needed
at compile time, so this works from WSL:

```bash
# once: IRON environment (Python 3.12), see https://github.com/Xilinx/mlir-aie
git clone --depth 1 --branch v1.4.3 https://github.com/Xilinx/mlir-aie ~/mlir-aie
cd ~/mlir-aie && source utils/env_install.sh

# every time
source ~/mlir-aie/ironenv/bin/activate && source ~/mlir-aie/utils/env_setup.sh
bash build_kernels.sh            # writes *.xclbin / *.insts.bin next to this README
```

`build_kernels.sh` targets `--dev npu` (XDNA1: Phoenix / Hawk Point, e.g. Ryzen 8700G). Set
`XDNA_DEV=npu2 XDNA_COLS=8` for XDNA2 parts. Shapes are constrained by the design:
`M % (m*4) == 0`, `K % k == 0`, `N % (n*4) == 0`, `(M/m/4) % 2 == 0`.

The final packing step needs XRT's `xclbinutil`, which mlir-aie does not ship. If you have no XRT
install (e.g. building from WSL), `xrt-utils.Dockerfile` + `xclbinutil-docker.sh` provide it from
the Ubuntu 26.04 `libxrt-utils` package: build the image, then copy the shim onto the venv's PATH
as `xclbinutil` (see the comments in the script).

Shipped kernels (XDNA1, 4 columns, bf16 in / f32 out):

| kernel | block (M x K x N) | tiles (m,k,n) | launch | intended use |
|---|---|---|---|---|
| `mm_bf16_f32_M512_K1024_N512` | 512 x 1024 x 512 | 64,64,32 | 454 us | 512-token batches (llama.cpp's default ubatch) |
| `mm_bf16_f32_M256_K1024_N512` | 256 x 1024 x 512 | 32,64,64 | 303 us | the same, in 256-row steps of a matrix |
| `mm_bf16_f32_M512_K1024_N128` | 512 x 1024 x 128 | 64,64,32 | 239 us | smaller batches |
| `mm_bf16_f32_M256_K512_N64`   | 256 x 512 x 64   | 32,32,16 | 179 us | small batches / narrow layers |

Launch times: Ryzen 7 8700G, `pyxrt`, median of 50 launches, NMSE ~1e-14 against the bf16
reference. A launch costs ~180 us of dispatch whatever its size, plus ~1 TMAC/s of compute, so
large blocks win: one 512 x 1024 x 512 launch does the work of four 512 x 1024 x 128 launches
(956 us). The 256-row blocks let the backend's auto split give the NPU a quarter or three quarters
of a 1,024-row matrix instead of none, half or all of it. With these kernels the NPU alone
prefills Qwen3-0.6B at ~815 t/s (it managed ~370 with 128-token blocks).

### Weight formats

The NPU never computes on quantized weights: the host converts the rows it takes to bf16 (or int8
with `GGML_XDNA_NPU_W8=1`) once, with ggml's own dequantizers, and keeps them in the weight cache.
Every ggml weight type with a dequantizer works: F32, F16, BF16 (used in place), the legacy and
K-quants, the IQ types and MXFP4. On x86 the CPU backend re-lays out some types for its SIMD
kernels (`CPU_REPACK`: rows in groups of 8, quants interleaved in 8-byte chunks); the backend
converts those layouts back for Q4_0, Q4_K, Q2_K (repacked on AVX-512 CPUs), IQ4_NL and MXFP4,
which covers every type the x86 CPU backend repacks. The conversion is exact: with the NPU taking
all rows, a model's repacked weights give the same logits as `--no-repack` (KL divergence 0 on
Qwen3-0.6B requantized to Q2_K, IQ4_NL and MXFP4).

### Decode kernels (experimental, `GGML_XDNA_NPU_DECODE`)

```
mm_bf16_f32_M1024_K1024_N64 / mm_w8_i8bf16_f32_M1024_K1024_N64     SHAPES="1024x1024x64:64,64,16"
```

Token generation is off the NPU by default: decode-sized products (fewer than
`GGML_XDNA_MIN_BATCH` tokens) stay on the CPU. `GGML_XDNA_NPU_DECODE=1` sends all rows of each of
them to the NPU, a fraction (e.g. `0.5`) that share of its rows, with the CPU computing the rest at
the same time. The NPU then reads each model weight from a store laid out once in NPU buffers, one
per 1,024-row and 1,024-wide block (`GGML_XDNA_DECODE_CACHE_MB`, default 8192), so no weight is
copied per token; `GGML_XDNA_NPU_W8=1` stores them as int8 (half the memory). Weights that do not
fit stay on the CPU.

A decode kernel takes 1,024 weight rows against a block of 64 tokens per launch. 64 is the
smallest token block of the 4-column design: the bf16 compute kernel needs `n % 16 == 0`. At one
token per step almost all of that block is padding, so the cores, not memory, are the limit: a
launch takes ~285 us alone and ~220 us pipelined, bf16 or int8 alike. Kernels with more rows per
launch (2,048: 475 us, 4,096: 863 us) are no faster per row. Using more than one kernel is slower
still: every op that follows one on another kernel waits ~1.3 ms for the NPU to switch hardware
contexts, so the backend uses the kernel with the smallest row block for every weight.

Ryzen 7 8700G, llama-bench tg16/tg64, all decode rows on the NPU (KL divergence against the CPU
at the batched NPU path's level: Qwen 0.0021 bf16 / 0.0044 int8, gemma 0.0148 / 0.0204):

| model | CPU (8 threads) | NPU, bf16 | NPU, int8 | NPU 0.5 + CPU, int8 |
|---|---|---|---|---|
| Qwen3-0.6B Q4_0 | ~120-130 t/s | 7.7-7.8 t/s | 8.1-8.2 t/s | 9.7 t/s |
| gemma-4-E2B Q4_K_M | ~32 t/s | 1.7 t/s | | 3.1 t/s (measured with the older multi-kernel choice) |

The NPU is busy for ~97% of each token, so the lever left is the kernel: a true matrix-vector
design (one token, all 4 columns streaming weights) instead of a 64-token GEMM block. NPU power
modes (`xrt-smi configure --pmode performance|turbo`) made no difference.

### Hardware contexts

Each loaded kernel holds one NPU hardware context, and the XDNA1 driver (32.0.20102) grants at
most 5 at a time; the sixth fails with `Failed to create context`. The backend keeps at most
`GGML_XDNA_MAX_KERNELS` (default 5) kernels loaded and unloads the least recently used one to make
room, also when another process holds contexts and a creation fails earlier. Reloading a kernel
takes ~90 ms, so once all slots are taken the kernel picker charges that to any kernel that is not
loaded, and the warm-up never unloads one. Keep the set a model actually uses to 5 or fewer per
weight format (bf16 or int8); more files in the kernel directory are harmless.

## int8-weight kernels

```
mm_w8_i8bf16_f32_M<M>_K<K>_N<N>.xclbin / .insts.bin
A : [M x K] int8 row-major   (weight rows; B and C as above)
```

These are the same whole-array design with an int8 A: `whole_array_w8.py` is mlir-aie's
`whole_array.py` with int8 weight types and a different compute kernel, and `mm_w8.cc` is its
bf16 4x8x4 `mmul` kernel (`matmul_vectorized_4x4`) with the weight vectors widened from int8 to
bf16 in registers. Both are derived from mlir-aie (Apache-2.0 WITH LLVM-exception, see their
headers). `build_kernels.sh` builds them after the bf16 kernels, for the same shapes (`W8_SHAPES`
overrides). With `GGML_XDNA_NPU_W8=1` the backend keeps the NPU's weights as int8, with one
scale per row and 1,024 weights, and applies the scale when it accumulates each block's result.
Every kernel's K divides 1,024, so a block never spans two scales.

On a Ryzen 7 8700G they take 10-20% longer per launch than their bf16 twins (512 x 1024 x 512:
542 vs 454 us; 256 x 1024 x 512: 358 vs 303 us; 512 x 1024 x 128: 271 vs 239 us) with NMSE
~1e-14 against an int8 x bf16 reference. The gain is on the host: half the weight-cache memory
and half the bytes copied per block.

The `xclbinutil` shim only mounts your Windows home, so build into a directory under it when
building from WSL. To check a kernel on the NPU (`pyxrt` from the NPU driver is built for
Python 3.10; `XDNA_KERNEL_DIR` selects another directory):

```
uv run --no-project --python 3.10 --with numpy python npu_smoke.py mm_w8_i8bf16_f32_M512_K512_N128
```

At runtime the backend searches `$GGML_XDNA_KERNEL_DIR`, falling back to this directory's
build-time path. `GGML_XDNA_MIN_BATCH` (default 32) sets the smallest token count a MUL_MAT
needs before it is sent to the NPU; decode-sized products always stay on the CPU.
