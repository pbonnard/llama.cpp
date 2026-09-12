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

| kernel | block (M x K x N) | tiles (m,k,n) | intended use |
|---|---|---|---|
| `mm_bf16_f32_M512_K512_N128`  | 512 x 512 x 128  | 64,64,32 | general |
| `mm_bf16_f32_M512_K1024_N128` | 512 x 1024 x 128 | 64,64,32 | K >= 1024 layers, fewer launches |
| `mm_bf16_f32_M512_K512_N256`  | 512 x 512 x 256  | 64,64,32 | 256+ token batches |
| `mm_bf16_f32_M256_K512_N64`   | 256 x 512 x 64   | 32,32,16 | small batches / narrow layers |

Measured on a Ryzen 7 8700G (via `pyxrt`, bf16 random data): the 512 x 512 x 128 block runs in
~360 us including dispatch (~185 GFLOPS) with NMSE ~3e-14 against the bf16 reference.

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

On a Ryzen 7 8700G they launch in the same time as their bf16 twins (512 x 512 x 128: 200 vs
184 us; 512 x 1024 x 128: 240 vs 240 us; 512 x 512 x 256: 250 vs 245 us) with NMSE ~1e-14
against an int8 x bf16 reference. The NPU's compute time does not change; the gain is on the
host: half the weight-cache memory and half the bytes copied per block.

The `xclbinutil` shim only mounts your Windows home, so build into a directory under it when
building from WSL. To check a kernel on the NPU (`pyxrt` from the NPU driver is built for
Python 3.10; `XDNA_KERNEL_DIR` selects another directory):

```
uv run --no-project --python 3.10 --with numpy python npu_smoke.py mm_w8_i8bf16_f32_M512_K512_N128
```

At runtime the backend searches `$GGML_XDNA_KERNEL_DIR`, falling back to this directory's
build-time path. `GGML_XDNA_MIN_BATCH` (default 32) sets the smallest token count a MUL_MAT
needs before it is sent to the NPU; decode-sized products always stay on the CPU.
