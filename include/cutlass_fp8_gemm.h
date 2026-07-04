#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

namespace cutlass_fp8
{

struct Fp8TnnGemmResult
{
  int cutlass_status = 0;
  cudaError_t cuda_error = cudaSuccess;
  float avg_ms = 0.0f;
  double tflops = 0.0;

  bool ok() const
  {
    return cutlass_status == 0 && cuda_error == cudaSuccess;
  }
};

// Hopper FP8 GEMM with E4M3 inputs, FP32 accumulation, and BF16 output.
//
// Logical GEMM:
//   D[M, N] = alpha * (op(A)[M, K] * op(B)[K, N]) + beta * C[M, N]
//
// TNN storage contract:
//   A is the transposed operand: physical KxM column-major, equivalent to MxK row-major.
//   B is non-transposed: physical KxN column-major.
//   C and D are non-transposed: physical MxN column-major.
Fp8TnnGemmResult cutlass_fp8_e4m3_bf16_tnn_gemm(
    int M, int N, int K,
    const __nv_fp8_e4m3 *d_A,
    const __nv_fp8_e4m3 *d_B,
    const __nv_bfloat16 *d_C,
    __nv_bfloat16 *d_D,
    float alpha = 1.0f,
    float beta = 0.0f,
    int warmup_iters = 1,
    int bench_iters = 10,
    cudaStream_t stream = nullptr);

} // namespace cutlass_fp8
