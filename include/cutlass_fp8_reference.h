#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace cutlass_ref
{

struct GemmResult
{
  float avg_ms = 0.0f;
  double tflops = 0.0;
  const char *kernel_name = nullptr;
};

// CUTLASS SM90 FP8 reference path.
//
// Layout contract:
//   A: row-major logical M x K, A[m, k] = d_A_e4m3[m * lda + k]
//   B: CUTLASS SM90 B operand view over logical N x K,
//      B[n, k] = d_B_e4m3[n * ldb + k]
//   D: column-major logical M x N, D[m, n] = d_D_bf16[m + n * ldd]
//
// This matches the initial CUTLASS profiler/heuristics layout "tnn" used for
// Hopper FP8 kernels. The public API stays free of CUTLASS-specific types.
void fp8_e4m3_gemm_bf16_reference(
    int M, int N, int K,
    const uint8_t *d_A_e4m3,
    const uint8_t *d_B_e4m3,
    __nv_bfloat16 *d_D_bf16,
    int lda, int ldb, int ldd,
    GemmResult *result = nullptr,
    cudaStream_t stream = nullptr,
    int warmup_iters = 1,
    int bench_iters = 10);

} // namespace cutlass_ref
