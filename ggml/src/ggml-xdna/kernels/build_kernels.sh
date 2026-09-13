#!/usr/bin/env bash
# Build the precompiled block-GEMM kernels used by ggml-xdna, with mlir-aie (IRON).
#
# Run inside the IRON environment (Linux or WSL; the NPU is not needed to compile):
#   source ~/mlir-aie/ironenv/bin/activate && source ~/mlir-aie/utils/env_setup.sh
#   bash build_kernels.sh [output-dir]
#
# Each kernel is the mlir-aie "whole_array" matmul with
#   --b-col-maj 1 --c-col-maj 1 --dtype_in bf16 --dtype_out f32
# so that A = weight rows [M x K], B = activation rows [N x K], C = [N x M] (ggml dst layout).
# Shapes must satisfy: M % (m*4) == 0, K % k == 0, N % (n*4) == 0, (M/m/4) % 2 == 0.
# So a 256-row block needs m=32 (n=64 keeps the per-core tile as large as m=64,n=32 does), and N
# must be a multiple of 256 with n=64. Launches cost ~180 us of dispatch on XDNA1 whatever their
# size, so large N (whole 512-token batches) pays off.
set -euo pipefail

OUT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MLIR_AIE_DIR="${MLIR_AIE_DIR:-$HOME/mlir-aie}"
DESIGN="$MLIR_AIE_DIR/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py"
DEV="${XDNA_DEV:-npu}"     # npu = XDNA1 (Phoenix/Hawk Point), npu2 = XDNA2 (Strix)
COLS="${XDNA_COLS:-4}"     # 4 columns on npu1

# M x K x N  m k n   (the shipped mm_bf16_f32_M256_K512_N64 was built with SHAPES="256x512x64:32,32,16",
# and the decode kernels (GGML_XDNA_NPU_DECODE) with SHAPES="1024x1024x64:64,64,16")
SHAPES="${SHAPES:-512x1024x512:64,64,32 256x1024x512:32,64,64 512x1024x128:64,64,32 512x1536x512:64,64,32 512x2048x512:64,64,32}"

mkdir -p "$OUT"
for spec in $SHAPES; do
  dims="${spec%%:*}"; tile="${spec##*:}"
  M="${dims%%x*}"; rest="${dims#*x}"; K="${rest%%x*}"; N="${rest##*x}"
  m="${tile%%,*}"; rest="${tile#*,}"; k="${rest%%,*}"; n="${rest##*,}"
  name="mm_bf16_f32_M${M}_K${K}_N${N}"
  echo "==> $name  (m=$m k=$k n=$n cols=$COLS dev=$DEV)"
  python3 "$DESIGN" --dev "$DEV" \
      -M "$M" -K "$K" -N "$N" -m "$m" -k "$k" -n "$n" --n-aie-cols "$COLS" \
      --dtype_in bf16 --dtype_out f32 --b-col-maj 1 --c-col-maj 1 \
      --xclbin-path="$OUT/$name.xclbin" --insts-path="$OUT/$name.insts.bin"
  ls -la "$OUT/$name.xclbin" "$OUT/$name.insts.bin"
done

# int8-weight kernels (A = int8 weight rows, B = bf16 activation rows, C = f32): same layouts, half the
# weight bytes; the host applies the per-row, per-block weight scales. Design: whole_array_w8.py + mm_w8.cc.
W8_DESIGN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/whole_array_w8.py"
# int8 weights keep one scale per 1,024 inputs, so their K must divide 1,024
W8_SHAPES="${W8_SHAPES:-512x1024x512:64,64,32 256x1024x512:32,64,64 512x1024x128:64,64,32}"
for spec in $W8_SHAPES; do
  dims="${spec%%:*}"; tile="${spec##*:}"
  M="${dims%%x*}"; rest="${dims#*x}"; K="${rest%%x*}"; N="${rest##*x}"
  m="${tile%%,*}"; rest="${tile#*,}"; k="${rest%%,*}"; n="${rest##*,}"
  name="mm_w8_i8bf16_f32_M${M}_K${K}_N${N}"
  echo "==> $name  (m=$m k=$k n=$n cols=$COLS dev=$DEV)"
  python3 "$W8_DESIGN" --dev "$DEV" \
      -M "$M" -K "$K" -N "$N" -m "$m" -k "$k" -n "$n" --n-aie-cols "$COLS" \
      --xclbin-path="$OUT/$name.xclbin" --insts-path="$OUT/$name.insts.bin"
  ls -la "$OUT/$name.xclbin" "$OUT/$name.insts.bin"
done
echo "done: $OUT"
