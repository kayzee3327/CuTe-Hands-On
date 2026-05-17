#include "utils.h"
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

// Helper macro to catch cuBLAS errors
#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = call;                                          \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace utils {

// use transpose trick to avoid C^T
// also handle row-major data better

void cublas_sgemm_reference(
    int M, int N, int K,
    const float* d_A, const float* d_B, float* d_C,
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
    for (int i = 0; i < warmup_iters; ++i) {
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
    for (int i = 0; i < bench_iters; ++i) {
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

void cublas_hgemm_reference(
    int M, int N, int K,
    const __half* d_A, const __half* d_B, __half* d_C,
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

    // 1. Warmup using cublasGemmEx for Tensor Cores + FP32 compute
    for (int i = 0; i < warmup_iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle, OPB, OPA,  
            N, M, K,
            &alpha,
            d_B, CUDA_R_16F, ldb,
            d_A, CUDA_R_16F, lda,
            &beta,
            d_C, CUDA_R_16F, ldc,
            CUBLAS_COMPUTE_32F,              // FP32 accumulation
            CUBLAS_GEMM_DEFAULT_TENSOR_OP    // Enable Tensor Cores
        ));
    }
    cudaDeviceSynchronize();

    // 2. Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle, OPB, OPA, 
            N, M, K,
            &alpha,
            d_B, CUDA_R_16F, ldb,
            d_A, CUDA_R_16F, lda,
            &beta,
            d_C, CUDA_R_16F, ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 3. Compute Metrics
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / bench_iters;

    double ops_per_gemm = 2.0 * static_cast<double>(M) * N * K;
    double tflops = (ops_per_gemm / (avg_ms / 1000.0)) / 1e12;

    std::cout << "[cuBLAS HGEMM] "
              << "M=" << M << ", N=" << N << ", K=" << K 
              << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
              << " | Performance: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";

    // 4. Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
}

template <typename T>
void compare_tensors(const T* d_test, const T* d_ref, int num_elements, float abs_tol, float rel_tol) {
    // 1. Allocate host memory
    std::vector<T> h_test(num_elements);
    std::vector<T> h_ref(num_elements);

    // 2. Copy data from Device to Host
    cudaError_t err1 = cudaMemcpy(h_test.data(), d_test, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
    cudaError_t err2 = cudaMemcpy(h_ref.data(), d_ref, num_elements * sizeof(T), cudaMemcpyDeviceToHost);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
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
    for (int i = 0; i < num_elements; ++i) {
        // Cast to float for math to handle both float and __half smoothly
        float val_test = static_cast<float>(h_test[i]);
        float val_ref  = static_cast<float>(h_ref[i]);

        double abs_err = std::abs(val_test - val_ref);
        double rel_err = abs_err / (std::abs(val_ref) + 1e-7); // 1e-7 prevents div by zero

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        
        sum_abs_err += abs_err;
        sum_rel_err += rel_err;

        // Check if the current element exceeds both tolerances
        if (abs_err > abs_tol && rel_err > rel_tol) {
            error_count++;
            if (error_count <= MAX_PRINT_ERRORS) {
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

    if (error_count > 0) {
        std::cout << "Status           : FAILED (" << error_count << " elements exceeded tolerances)\n";
        if (error_count > MAX_PRINT_ERRORS) {
            std::cout << "                   (Omitted " << (error_count - MAX_PRINT_ERRORS) << " more errors)\n";
        }
    } else {
        std::cout << "Status           : PASSED\n";
    }
    std::cout << "---------------------------------\n";
}

// Explicit instantiations
template void compare_tensors<float>(const float* d_test, const float* d_ref, int num_elements, float abs_tol, float rel_tol);
template void compare_tensors<__half>(const __half* d_test, const __half* d_ref, int num_elements, float abs_tol, float rel_tol);

} // namespace utils