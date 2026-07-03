#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace utils
{

  // Reference SGEMM (Single Precision)
  void cublas_sgemm_reference(
      int M, int N, int K,
      const float *d_A, const float *d_B, float *d_C,
      float alpha = 1.0f, float beta = 0.0f,
      bool A_OP_T = false, bool B_OP_T = false,
      int warmup_iters = 1, int bench_iters = 10);

  // Reference GEMM for FP8 E4M3 inputs, FP32 accumulation, BF16 output.
  void cublaslt_fp8_e4m3_bf16_reference(
      int M, int N, int K,
      const __nv_fp8_e4m3 *d_A, const __nv_fp8_e4m3 *d_B, __nv_bfloat16 *d_C,
      float alpha = 1.0f, float beta = 0.0f,
      bool A_OP_T = false, bool B_OP_T = false,
      int warmup_iters = 1, int bench_iters = 10);

  // Compares a test tensor against a reference tensor and prints the results.
  // Both pointers must reside on the GPU (Device pointers).
  // Note that for HGEMM (using __half), you will typically need to
  //  pass much looser tolerances (like 1e-3 or 5e-3) than SGEMM (1e-5)
  //  due to the lower precision format.
  template <typename T>
  void compare_tensors(const T *d_test, const T *d_ref, int num_elements,
                       float abs_tol = 1e-5f, float rel_tol = 1e-5f);

  // Explicit template instantiation declarations
  extern template void compare_tensors<float>(const float *d_test, const float *d_ref, int num_elements, float abs_tol, float rel_tol);
  extern template void compare_tensors<__half>(const __half *d_test, const __half *d_ref, int num_elements, float abs_tol, float rel_tol);
  extern template void compare_tensors<__nv_bfloat16>(const __nv_bfloat16 *d_test, const __nv_bfloat16 *d_ref, int num_elements, float abs_tol, float rel_tol);

} // namespace utils
