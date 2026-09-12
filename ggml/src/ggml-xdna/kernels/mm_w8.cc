//===- mm_w8.cc - int8-weight x bf16-activation -> f32 block GEMM (AIE2) ---===//
//
// Derived from mlir-aie's aie_kernels/aie2/mm.cc (matmul_vectorized_4x4) and
// aie_kernels/aie2/zero.cc:
// Copyright (C) 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// The weight tile A arrives as int8 and is widened to bf16 in registers before
// the bf16 mmul, so the NPU reads half the weight bytes of the bf16 kernel. The
// weights' scales (one per row and K block) are applied by the host when it
// accumulates the block results. Specialized to ggml-xdna's layout: B holds
// activation rows (column-major), C is column-major.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

#if !defined(B_COL_MAJ) || !defined(C_COL_MAJ)
#error "mm_w8.cc supports -DB_COL_MAJ -DC_COL_MAJ only"
#endif

// C[n x m] (column-major, pre-tiled) += A[m x k] (int8, pre-tiled r x s) * B[k x n] (bf16, column-major),
// expanding the r x s x t mmul 4 x 4 times like mlir-aie's matmul_vectorized_4x4
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s, unsigned t>
static inline void matmul_w8_4x4(const int8_t *__restrict pA, const bfloat16 *__restrict pB,
                                 float *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;
  static_assert(rowA % 4 == 0 && colB % 4 == 0);

  for (unsigned z = 0; z < rowA; z += 4) {
    for (unsigned j = 0; j < colB; j += 4) {
      float *__restrict pC1 = pC + (j + 0) * rowA * MMUL::size_C + z * MMUL::size_C;
      float *__restrict pC2 = pC + (j + 1) * rowA * MMUL::size_C + z * MMUL::size_C;
      float *__restrict pC3 = pC + (j + 2) * rowA * MMUL::size_C + z * MMUL::size_C;
      float *__restrict pC4 = pC + (j + 3) * rowA * MMUL::size_C + z * MMUL::size_C;

      const int8_t *__restrict pA1 = pA + (z + 0) * colA * MMUL::size_A;
      const int8_t *__restrict pA2 = pA + (z + 1) * colA * MMUL::size_A;
      const int8_t *__restrict pA3 = pA + (z + 2) * colA * MMUL::size_A;
      const int8_t *__restrict pA4 = pA + (z + 3) * colA * MMUL::size_A;

      const bfloat16 *__restrict pB1 = pB + (j + 0) * colA * MMUL::size_B;
      const bfloat16 *__restrict pB2 = pB + (j + 1) * colA * MMUL::size_B;
      const bfloat16 *__restrict pB3 = pB + (j + 2) * colA * MMUL::size_B;
      const bfloat16 *__restrict pB4 = pB + (j + 3) * colA * MMUL::size_B;

      // Cxy: A row block z + x, B column block j + y
      MMUL C00(aie::transpose(aie::load_v<MMUL::size_C>(pC1), t, r));
      MMUL C01(aie::transpose(aie::load_v<MMUL::size_C>(pC2), t, r));
      MMUL C02(aie::transpose(aie::load_v<MMUL::size_C>(pC3), t, r));
      MMUL C03(aie::transpose(aie::load_v<MMUL::size_C>(pC4), t, r));
      MMUL C10(aie::transpose(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C), t, r));
      MMUL C11(aie::transpose(aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C), t, r));
      MMUL C12(aie::transpose(aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C), t, r));
      MMUL C13(aie::transpose(aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C), t, r));
      MMUL C20(aie::transpose(aie::load_v<MMUL::size_C>(pC1 + 2 * MMUL::size_C), t, r));
      MMUL C21(aie::transpose(aie::load_v<MMUL::size_C>(pC2 + 2 * MMUL::size_C), t, r));
      MMUL C22(aie::transpose(aie::load_v<MMUL::size_C>(pC3 + 2 * MMUL::size_C), t, r));
      MMUL C23(aie::transpose(aie::load_v<MMUL::size_C>(pC4 + 2 * MMUL::size_C), t, r));
      MMUL C30(aie::transpose(aie::load_v<MMUL::size_C>(pC1 + 3 * MMUL::size_C), t, r));
      MMUL C31(aie::transpose(aie::load_v<MMUL::size_C>(pC2 + 3 * MMUL::size_C), t, r));
      MMUL C32(aie::transpose(aie::load_v<MMUL::size_C>(pC3 + 3 * MMUL::size_C), t, r));
      MMUL C33(aie::transpose(aie::load_v<MMUL::size_C>(pC4 + 3 * MMUL::size_C), t, r));

      for (unsigned i = 0; i < colA; ++i) {
        const aie::vector<bfloat16, MMUL::size_A> A0 = aie::to_float<bfloat16>(aie::load_v<MMUL::size_A>(pA1), 0);
        pA1 += MMUL::size_A;
        const aie::vector<bfloat16, MMUL::size_A> A1 = aie::to_float<bfloat16>(aie::load_v<MMUL::size_A>(pA2), 0);
        pA2 += MMUL::size_A;
        const aie::vector<bfloat16, MMUL::size_A> A2 = aie::to_float<bfloat16>(aie::load_v<MMUL::size_A>(pA3), 0);
        pA3 += MMUL::size_A;
        const aie::vector<bfloat16, MMUL::size_A> A3 = aie::to_float<bfloat16>(aie::load_v<MMUL::size_A>(pA4), 0);
        pA4 += MMUL::size_A;

        const aie::vector<bfloat16, MMUL::size_B> B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), t, s);
        pB1 += MMUL::size_B;
        const aie::vector<bfloat16, MMUL::size_B> B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), t, s);
        pB2 += MMUL::size_B;
        const aie::vector<bfloat16, MMUL::size_B> B2 = aie::transpose(aie::load_v<MMUL::size_B>(pB3), t, s);
        pB3 += MMUL::size_B;
        const aie::vector<bfloat16, MMUL::size_B> B3 = aie::transpose(aie::load_v<MMUL::size_B>(pB4), t, s);
        pB4 += MMUL::size_B;

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);

        C02.mac(A0, B2);
        C03.mac(A0, B3);
        C12.mac(A1, B2);
        C13.mac(A1, B3);

        C20.mac(A2, B0);
        C21.mac(A2, B1);
        C30.mac(A3, B0);
        C31.mac(A3, B1);

        C22.mac(A2, B2);
        C23.mac(A2, B3);
        C32.mac(A3, B2);
        C33.mac(A3, B3);
      }

      aie::store_v(pC1, aie::transpose(C00.template to_vector<float>(), r, t));
      pC1 += MMUL::size_C;
      aie::store_v(pC2, aie::transpose(C01.template to_vector<float>(), r, t));
      pC2 += MMUL::size_C;
      aie::store_v(pC3, aie::transpose(C02.template to_vector<float>(), r, t));
      pC3 += MMUL::size_C;
      aie::store_v(pC4, aie::transpose(C03.template to_vector<float>(), r, t));
      pC4 += MMUL::size_C;
      aie::store_v(pC1, aie::transpose(C10.template to_vector<float>(), r, t));
      pC1 += MMUL::size_C;
      aie::store_v(pC2, aie::transpose(C11.template to_vector<float>(), r, t));
      pC2 += MMUL::size_C;
      aie::store_v(pC3, aie::transpose(C12.template to_vector<float>(), r, t));
      pC3 += MMUL::size_C;
      aie::store_v(pC4, aie::transpose(C13.template to_vector<float>(), r, t));
      pC4 += MMUL::size_C;
      aie::store_v(pC1, aie::transpose(C20.template to_vector<float>(), r, t));
      pC1 += MMUL::size_C;
      aie::store_v(pC2, aie::transpose(C21.template to_vector<float>(), r, t));
      pC2 += MMUL::size_C;
      aie::store_v(pC3, aie::transpose(C22.template to_vector<float>(), r, t));
      pC3 += MMUL::size_C;
      aie::store_v(pC4, aie::transpose(C23.template to_vector<float>(), r, t));
      pC4 += MMUL::size_C;
      aie::store_v(pC1, aie::transpose(C30.template to_vector<float>(), r, t));
      aie::store_v(pC2, aie::transpose(C31.template to_vector<float>(), r, t));
      aie::store_v(pC3, aie::transpose(C32.template to_vector<float>(), r, t));
      aie::store_v(pC4, aie::transpose(C33.template to_vector<float>(), r, t));
    }
  }
}

template <typename T, int M, int N>
static inline void zero_w8(T *__restrict c) {
  constexpr int r = 256 / (sizeof(T) * 8);   // one 256-bit store
  static_assert((M * N) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M * N;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

extern "C" {

void matmul_i8bf16_f32(int8_t *a_in, bfloat16 *b_in, float *c_out) {
  constexpr unsigned r = 4, s = 8, t = 4;
  static_assert(DIM_M % (4 * r) == 0 && DIM_K % s == 0 && DIM_N % (4 * t) == 0);
  matmul_w8_4x4<DIM_M / r, DIM_K / s, DIM_N / t, r, s, t>(a_in, b_in, c_out);
}

void zero_f32(float *c_out) { zero_w8<float, DIM_M, DIM_N>(c_out); }

} // extern "C"
