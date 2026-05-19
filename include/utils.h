#pragma once
#include <cuda_fp16.h>

namespace utils
{

  // Reference SGEMM (Single Precision)
  void cublas_sgemm_reference(
      int M, int N, int K,
      const float *d_A, const float *d_B, float *d_C,
      float alpha = 1.0f, float beta = 0.0f,
      bool A_OP_T = false, bool B_OP_T = false,
      int warmup_iters = 1, int bench_iters = 10);

  // Reference HGEMM (Half Precision with Tensor Cores, float accumulation)
  void cublas_hgemm_reference(
      int M, int N, int K,
      const __half *d_A, const __half *d_B, __half *d_C,
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

} // namespace utils