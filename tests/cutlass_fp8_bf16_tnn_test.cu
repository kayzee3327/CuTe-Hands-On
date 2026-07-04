#include "cutlass_fp8_gemm.h"
#include "utils.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct TestCase
{
  int M;
  int N;
  int K;
  float alpha;
  float beta;
  std::string name;
};

float input_value(int index, int salt)
{
  int v = ((index * 37 + salt * 19) % 41) - 20;
  return static_cast<float>(v) / 32.0f;
}

float c_init_value(int index)
{
  int v = ((index * 13 + 7) % 23) - 11;
  return static_cast<float>(v) / 16.0f;
}

size_t a_tnn_offset(const TestCase &tc, int m, int k)
{
  return static_cast<size_t>(k) + static_cast<size_t>(m) * tc.K;
}

size_t b_tnn_offset(const TestCase &tc, int k, int n)
{
  return static_cast<size_t>(k) + static_cast<size_t>(n) * tc.K;
}

size_t c_tnn_offset(const TestCase &tc, int m, int n)
{
  return static_cast<size_t>(m) + static_cast<size_t>(n) * tc.M;
}

bool compare_bf16(
    const std::vector<__nv_bfloat16> &got,
    const std::vector<__nv_bfloat16> &ref,
    const TestCase &tc)
{
  constexpr float abs_tol = 0.25f;
  constexpr float rel_tol = 0.03f;

  int error_count = 0;
  int worst_index = 0;
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;

  for (size_t i = 0; i < got.size(); ++i)
  {
    float got_value = static_cast<float>(got[i]);
    float ref_value = static_cast<float>(ref[i]);
    double abs_err = std::abs(got_value - ref_value);
    double rel_err = abs_err / (std::abs(ref_value) + 1e-7f);

    if (abs_err > max_abs_err)
    {
      max_abs_err = abs_err;
      worst_index = static_cast<int>(i);
    }
    max_rel_err = std::max(max_rel_err, rel_err);

    if (abs_err > abs_tol && rel_err > rel_tol)
    {
      if (error_count < 5)
      {
        int m = static_cast<int>(i % tc.M);
        int n = static_cast<int>(i / tc.M);
        std::cout << "  [Mismatch] m=" << m
                  << " n=" << n
                  << " got=" << got_value
                  << " ref=" << ref_value
                  << " abs=" << abs_err
                  << " rel=" << rel_err << "\n";
      }
      ++error_count;
    }
  }

  int worst_m = worst_index % tc.M;
  int worst_n = worst_index / tc.M;
  std::cout << "[Compare] " << tc.name
            << " | max_abs=" << std::scientific << max_abs_err
            << " max_rel=" << max_rel_err
            << " worst=(" << worst_m << "," << worst_n << ")"
            << " | " << (error_count == 0 ? "PASSED" : "FAILED")
            << std::defaultfloat << "\n";

  if (error_count > 5)
  {
    std::cout << "  Omitted " << (error_count - 5) << " more mismatches\n";
  }

  return error_count == 0;
}

bool run_case(const TestCase &tc)
{
  size_t a_elements = static_cast<size_t>(tc.M) * tc.K;
  size_t b_elements = static_cast<size_t>(tc.K) * tc.N;
  size_t c_elements = static_cast<size_t>(tc.M) * tc.N;

  std::vector<__nv_fp8_e4m3> h_A(a_elements);
  std::vector<__nv_fp8_e4m3> h_B(b_elements);
  std::vector<__nv_bfloat16> h_C_init(c_elements);
  std::vector<__nv_bfloat16> h_cutlass(c_elements);
  std::vector<__nv_bfloat16> h_cublaslt(c_elements);

  for (int m = 0; m < tc.M; ++m)
  {
    for (int k = 0; k < tc.K; ++k)
    {
      h_A[a_tnn_offset(tc, m, k)] = __nv_fp8_e4m3(input_value(m * tc.K + k, 1));
    }
  }

  for (int k = 0; k < tc.K; ++k)
  {
    for (int n = 0; n < tc.N; ++n)
    {
      h_B[b_tnn_offset(tc, k, n)] = __nv_fp8_e4m3(input_value(k * tc.N + n, 2));
    }
  }

  for (int m = 0; m < tc.M; ++m)
  {
    for (int n = 0; n < tc.N; ++n)
    {
      h_C_init[c_tnn_offset(tc, m, n)] = __float2bfloat16(c_init_value(m * tc.N + n));
    }
  }

  __nv_fp8_e4m3 *d_A = nullptr;
  __nv_fp8_e4m3 *d_B = nullptr;
  __nv_bfloat16 *d_C_source = nullptr;
  __nv_bfloat16 *d_C_cutlass = nullptr;
  __nv_bfloat16 *d_C_cublaslt = nullptr;

  cudaMalloc(&d_A, a_elements * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&d_B, b_elements * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&d_C_source, c_elements * sizeof(__nv_bfloat16));
  cudaMalloc(&d_C_cutlass, c_elements * sizeof(__nv_bfloat16));
  cudaMalloc(&d_C_cublaslt, c_elements * sizeof(__nv_bfloat16));

  cudaMemcpy(d_A, h_A.data(), a_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), b_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C_source, h_C_init.data(), c_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C_cublaslt, h_C_init.data(), c_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

  std::cout << "[Test] running " << tc.name
            << " alpha=" << tc.alpha
            << " beta=" << tc.beta << "\n";

  cutlass_fp8::Fp8TnnGemmResult cutlass_result =
      cutlass_fp8::cutlass_fp8_e4m3_bf16_tnn_gemm(
          tc.M, tc.N, tc.K,
          d_A, d_B, d_C_source, d_C_cutlass,
          tc.alpha, tc.beta,
          1, 5);

  if (!cutlass_result.ok())
  {
    std::cout << "[CUTLASS FP8 TNN] FAILED"
              << " cutlass_status=" << cutlass_result.cutlass_status
              << " cuda_error=" << cudaGetErrorString(cutlass_result.cuda_error) << "\n";
    cudaFree(d_C_cublaslt);
    cudaFree(d_C_cutlass);
    cudaFree(d_C_source);
    cudaFree(d_B);
    cudaFree(d_A);
    return false;
  }

  std::cout << "[CUTLASS FP8 E4M3->BF16 TNN] "
            << "M=" << tc.M << ", N=" << tc.N << ", K=" << tc.K
            << " | Time: " << std::fixed << std::setprecision(3) << cutlass_result.avg_ms << " ms"
            << " | Performance: " << std::fixed << std::setprecision(2) << cutlass_result.tflops << " TFLOPS\n";

  utils::cublaslt_fp8_e4m3_bf16_tn_reference(
      tc.M, tc.N, tc.K,
      d_A, d_B, d_C_source, d_C_cublaslt,
      tc.alpha, tc.beta,
      1, 5);

  cudaMemcpy(h_cutlass.data(), d_C_cutlass, c_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_cublaslt.data(), d_C_cublaslt, c_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  cudaFree(d_C_cublaslt);
  cudaFree(d_C_cutlass);
  cudaFree(d_C_source);
  cudaFree(d_B);
  cudaFree(d_A);

  return compare_bf16(h_cutlass, h_cublaslt, tc);
}

int main()
{
  int device = 0;
  cudaError_t cuda_status = cudaGetDevice(&device);
  if (cuda_status != cudaSuccess)
  {
    std::cout << "SKIPPED: CUDA device query failed: "
              << cudaGetErrorString(cuda_status) << "\n";
    return EXIT_SUCCESS;
  }

  cudaDeviceProp prop = {};
  cuda_status = cudaGetDeviceProperties(&prop, device);
  if (cuda_status != cudaSuccess)
  {
    std::cout << "SKIPPED: CUDA device properties query failed: "
              << cudaGetErrorString(cuda_status) << "\n";
    return EXIT_SUCCESS;
  }

  std::cout << "Running on " << prop.name
            << " (sm_" << prop.major << prop.minor << ")\n";

  if (prop.major * 10 + prop.minor < 90)
  {
    std::cout << "SKIPPED: CUTLASS Hopper FP8 GEMM requires an sm_90+ GPU.\n";
    return EXIT_SUCCESS;
  }

  std::vector<TestCase> cases = {
      {64, 128, 128, 1.0f, 0.0f, "shape_64x128x128_beta0_TNN"},
      {128, 128, 128, 0.75f, -0.5f, "shape_128x128x128_beta_TNN"},
  };

  int failures = 0;
  for (const TestCase &tc : cases)
  {
    if (!run_case(tc))
    {
      ++failures;
    }
  }

  if (failures != 0)
  {
    std::cout << failures << " case(s) failed\n";
    return EXIT_FAILURE;
  }

  std::cout << "All CUTLASS FP8 E4M3 -> BF16 TNN tests passed against cuBLASLt\n";
  return EXIT_SUCCESS;
}
