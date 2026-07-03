#include "utils.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// Helper macro to catch cuBLAS errors
#define CHECK_CUBLAS(call)                                           \
  do                                                                 \
  {                                                                  \
    cublasStatus_t status = call;                                    \
    if (status != CUBLAS_STATUS_SUCCESS)                             \
    {                                                                \
      fprintf(stderr,                                                \
            "cuBLAS error: %s (%s) at %s:%d\n",                      \
            cublasGetStatusName(status),                             \
            cublasGetStatusString(status),                           \
            __FILE__,                                                \
            __LINE__);                                               \
      exit(EXIT_FAILURE);                                            \
    }                                                                \
  } while (0)

#define CHECK_CUBLASLT(call)                                         \
  do                                                                 \
  {                                                                  \
    cublasStatus_t status = call;                                    \
    if (status != CUBLAS_STATUS_SUCCESS)                             \
    {                                                                \
      fprintf(stderr,                                                \
            "cuBLAS error: %s (%s) at %s:%d\n",                      \
            cublasLtGetStatusName(status),                           \
            cublasLtGetStatusString(status),                         \
            __FILE__,                                                \
            __LINE__);                                               \
      exit(EXIT_FAILURE);                                            \
    }                                                                \
  } while (0)

namespace utils
{

  // use transpose trick to avoid C^T
  // also handle row-major data better

  void cublas_sgemm_reference(
      int M, int N, int K,
      const float *d_A, const float *d_B, float *d_C,
      float alpha, float beta,
      bool A_OP_T, bool B_OP_T,
      int warmup_iters, int bench_iters)
  {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cublasOperation_t OPA = A_OP_T ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t OPB = B_OP_T ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda = A_OP_T ? M : K;
    int ldb = B_OP_T ? K : N;
    int ldc = N;

    // 1. Warmup
    for (int i = 0; i < warmup_iters; ++i)
    {
      CHECK_CUBLAS(cublasSgemm(handle, OPB, OPA,
                               N, M, K,
                               &alpha,
                               d_B, ldb,
                               d_A, lda,
                               &beta,
                               d_C, ldc));
    }
    cudaDeviceSynchronize();

    // 2. Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
    {
      CHECK_CUBLAS(cublasSgemm(handle, OPB, OPA,
                               N, M, K,
                               &alpha,
                               d_B, ldb,
                               d_A, lda,
                               &beta,
                               d_C, ldc));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 3. Compute Metrics
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / bench_iters;

    double ops_per_gemm = 2.0 * static_cast<double>(M) * N * K;
    double tflops = (ops_per_gemm / (avg_ms / 1000.0)) / 1e12;

    std::cout << "[cuBLAS SGEMM] "
              << "M=" << M << ", N=" << N << ", K=" << K
              << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
              << " | Performance: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";

    // 4. Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
  }

  void cublaslt_fp8_e4m3_bf16_tn_reference(
      int M, int N, int K,
      const __nv_fp8_e4m3 *d_A, const __nv_fp8_e4m3 *d_B, __nv_bfloat16 *d_C,
      float alpha, float beta,
      int warmup_iters, int bench_iters)
  {
    cublasLtHandle_t handle;
    CHECK_CUBLASLT(cublasLtCreate(&handle));

    cublasLtMatmulDesc_t matmul_desc;
    cublasLtMatrixLayout_t a_desc;
    cublasLtMatrixLayout_t b_desc;
    cublasLtMatrixLayout_t c_desc;
    cublasLtMatmulPreference_t preference;

    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t lt_op_a = CUBLAS_OP_T;
    cublasOperation_t lt_op_b = CUBLAS_OP_N;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &lt_op_a, sizeof(lt_op_a)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &lt_op_b, sizeof(lt_op_b)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &a_desc, CUDA_R_8F_E4M3, K, M, K));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &b_desc, CUDA_R_8F_E4M3, K, N, K));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
        &c_desc, CUDA_R_16BF, M, N, M));

    constexpr uint64_t max_workspace_bytes = 32ull * 1024ull * 1024ull;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace_bytes,
        sizeof(max_workspace_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic_result = {};
    int returned_results = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmul_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        preference,
        1,
        &heuristic_result,
        &returned_results));

    if (returned_results == 0 || heuristic_result.state != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "cuBLASLt Error: no FP8 E4M3 -> BF16 matmul algorithm found" << std::endl;
      exit(EXIT_FAILURE);
    }

    void *workspace = nullptr;
    size_t workspace_size = heuristic_result.workspaceSize;
    if (workspace_size > 0)
    {
      cudaMalloc(&workspace, workspace_size);
    }

    auto run_matmul = [&]() {
      CHECK_CUBLASLT(cublasLtMatmul(
          handle,
          matmul_desc,
          &alpha,
          d_A,
          a_desc,
          d_B,
          b_desc,
          &beta,
          d_C,
          c_desc,
          d_C,
          c_desc,
          &heuristic_result.algo,
          workspace,
          workspace_size,
          0));
    };

    // 1. Warmup
    for (int i = 0; i < warmup_iters; ++i)
    {
      run_matmul();
    }
    cudaDeviceSynchronize();

    // 2. Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i)
    {
      run_matmul();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 3. Compute Metrics
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / bench_iters;

    double ops_per_gemm = 2.0 * static_cast<double>(M) * N * K;
    double tflops = (ops_per_gemm / (avg_ms / 1000.0)) / 1e12;

    std::cout << "[cuBLASLt FP8 E4M3->BF16] "
              << "M=" << M << ", N=" << N << ", K=" << K
              << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
              << " | Performance: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";

    // 4. Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (workspace != nullptr)
    {
      cudaFree(workspace);
    }
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtDestroy(handle);
  }


  template <typename T>
  void compare_tensors(const T *d_test, const T *d_ref, int num_elements, float abs_tol, float rel_tol)
  {
    // 1. Allocate host memory
    std::vector<T> h_test(num_elements);
    std::vector<T> h_ref(num_elements);

    // 2. Copy data from Device to Host
    cudaError_t err1 = cudaMemcpy(h_test.data(), d_test, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
    cudaError_t err2 = cudaMemcpy(h_ref.data(), d_ref, num_elements * sizeof(T), cudaMemcpyDeviceToHost);

    if (err1 != cudaSuccess || err2 != cudaSuccess)
    {
      std::cerr << "CUDA Memcpy failed in compare_tensors!" << std::endl;
      return;
    }

    // 3. Variables to track error metrics
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    double sum_abs_err = 0.0;
    double sum_rel_err = 0.0;

    int error_count = 0;
    const int MAX_PRINT_ERRORS = 5; // Cap the console spam

    // 4. Verification loop
    for (int i = 0; i < num_elements; ++i)
    {
      // Cast to float for math to handle both float and __half smoothly
      float val_test = static_cast<float>(h_test[i]);
      float val_ref = static_cast<float>(h_ref[i]);

      double abs_err = std::abs(val_test - val_ref);
      double rel_err = abs_err / (std::abs(val_ref) + 1e-7); // 1e-7 prevents div by zero

      max_abs_err = std::max(max_abs_err, abs_err);
      max_rel_err = std::max(max_rel_err, rel_err);

      sum_abs_err += abs_err;
      sum_rel_err += rel_err;

      // Check if the current element exceeds both tolerances
      if (abs_err > abs_tol && rel_err > rel_tol)
      {
        error_count++;
        if (error_count <= MAX_PRINT_ERRORS)
        {
          std::cout << "  [Mismatch] at index " << i
                    << ": Test=" << val_test
                    << ", Ref=" << val_ref
                    << " | AbsErr=" << abs_err
                    << ", RelErr=" << rel_err << "\n";
        }
      }
    }

    // 5. Output Summary
    double avg_abs_err = sum_abs_err / num_elements;
    double avg_rel_err = sum_rel_err / num_elements;

    std::cout << "--- Tensor Comparison Summary ---\n"
              << "Elements Checked : " << num_elements << "\n"
              << "Max Abs Error    : " << std::scientific << max_abs_err << "\n"
              << "Max Rel Error    : " << std::scientific << max_rel_err << "\n"
              << "Avg Abs Error    : " << std::scientific << avg_abs_err << "\n"
              << "Avg Rel Error    : " << std::scientific << avg_rel_err << "\n";

    if (error_count > 0)
    {
      std::cout << "Status           : FAILED (" << error_count << " elements exceeded tolerances)\n";
      if (error_count > MAX_PRINT_ERRORS)
      {
        std::cout << "                   (Omitted " << (error_count - MAX_PRINT_ERRORS) << " more errors)\n";
      }
    }
    else
    {
      std::cout << "Status           : PASSED\n";
    }
    std::cout << "---------------------------------\n";
  }

  // Explicit instantiations
  template void compare_tensors<float>(const float *d_test, const float *d_ref, int num_elements, float abs_tol, float rel_tol);
  template void compare_tensors<__half>(const __half *d_test, const __half *d_ref, int num_elements, float abs_tol, float rel_tol);
  template void compare_tensors<__nv_bfloat16>(const __nv_bfloat16 *d_test, const __nv_bfloat16 *d_ref, int num_elements, float abs_tol, float rel_tol);

} // namespace utils
