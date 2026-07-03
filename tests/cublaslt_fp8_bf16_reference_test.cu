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

size_t a_tn_offset(const TestCase &tc, int m, int k)
{
  return static_cast<size_t>(k) + static_cast<size_t>(m) * tc.K;
}

size_t b_tn_offset(const TestCase &tc, int k, int n)
{
  return static_cast<size_t>(k) + static_cast<size_t>(n) * tc.K;
}

size_t c_tn_offset(const TestCase &tc, int m, int n)
{
  return static_cast<size_t>(m) + static_cast<size_t>(n) * tc.M;
}

std::vector<__nv_bfloat16> cpu_reference(
    const std::vector<__nv_fp8_e4m3> &A,
    const std::vector<__nv_fp8_e4m3> &B,
    const std::vector<__nv_bfloat16> &C_init,
    const TestCase &tc)
{
  std::vector<__nv_bfloat16> C_ref(static_cast<size_t>(tc.M) * tc.N);

  for (int m = 0; m < tc.M; ++m)
  {
    for (int n = 0; n < tc.N; ++n)
    {
      float acc = 0.0f;
      for (int k = 0; k < tc.K; ++k)
      {
        acc += static_cast<float>(A[a_tn_offset(tc, m, k)]) *
               static_cast<float>(B[b_tn_offset(tc, k, n)]);
      }

      size_t c_idx = c_tn_offset(tc, m, n);
      float c_old = static_cast<float>(C_init[c_idx]);
      float out = tc.alpha * acc + tc.beta * c_old;
      C_ref[c_idx] = __float2bfloat16(out);
    }
  }

  return C_ref;
}

bool run_case(const TestCase &tc)
{
  size_t a_elements = static_cast<size_t>(tc.M) * tc.K;
  size_t b_elements = static_cast<size_t>(tc.K) * tc.N;
  size_t c_elements = static_cast<size_t>(tc.M) * tc.N;

  std::vector<__nv_fp8_e4m3> h_A(a_elements);
  std::vector<__nv_fp8_e4m3> h_B(b_elements);
  std::vector<__nv_bfloat16> h_C_init(c_elements);

  for (int m = 0; m < tc.M; ++m)
  {
    for (int k = 0; k < tc.K; ++k)
    {
      h_A[a_tn_offset(tc, m, k)] = __nv_fp8_e4m3(input_value(m * tc.K + k, 1));
    }
  }
  for (int k = 0; k < tc.K; ++k)
  {
    for (int n = 0; n < tc.N; ++n)
    {
      h_B[b_tn_offset(tc, k, n)] = __nv_fp8_e4m3(input_value(k * tc.N + n, 2));
    }
  }
  for (int m = 0; m < tc.M; ++m)
  {
    for (int n = 0; n < tc.N; ++n)
    {
      h_C_init[c_tn_offset(tc, m, n)] = __float2bfloat16(c_init_value(m * tc.N + n));
    }
  }

  std::vector<__nv_bfloat16> h_C_ref = cpu_reference(h_A, h_B, h_C_init, tc);
  std::vector<__nv_bfloat16> h_C_got(c_elements);

  __nv_fp8_e4m3 *d_A = nullptr;
  __nv_fp8_e4m3 *d_B = nullptr;
  __nv_bfloat16 *d_C = nullptr;

  cudaMalloc(&d_A, a_elements * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&d_B, b_elements * sizeof(__nv_fp8_e4m3));
  cudaMalloc(&d_C, c_elements * sizeof(__nv_bfloat16));

  cudaMemcpy(d_A, h_A.data(), a_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), b_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C_init.data(), c_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

  std::cout << "[Test] running " << tc.name
            << " alpha=" << tc.alpha
            << " beta=" << tc.beta << "\n";

  utils::cublaslt_fp8_e4m3_bf16_tn_reference(
      tc.M, tc.N, tc.K,
      d_A, d_B, d_C,
      tc.alpha, tc.beta,
      0, 1);

  cudaMemcpy(h_C_got.data(), d_C, c_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);

  constexpr float abs_tol = 0.25f;
  constexpr float rel_tol = 0.03f;
  int error_count = 0;
  int worst_index = 0;
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;

  for (size_t i = 0; i < c_elements; ++i)
  {
    float got = static_cast<float>(h_C_got[i]);
    float ref = static_cast<float>(h_C_ref[i]);
    double abs_err = std::abs(got - ref);
    double rel_err = abs_err / (std::abs(ref) + 1e-7f);

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
                  << " got=" << got
                  << " ref=" << ref
                  << " abs=" << abs_err
                  << " rel=" << rel_err << "\n";
      }
      ++error_count;
    }
  }

  int worst_m = worst_index % tc.M;
  int worst_n = worst_index / tc.M;
  std::cout << "[Test] " << tc.name
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

int main()
{
  int device = 0;
  cudaGetDevice(&device);

  cudaDeviceProp prop = {};
  cudaGetDeviceProperties(&prop, device);
  std::cout << "Running on " << prop.name
            << " (sm_" << prop.major << prop.minor << ")\n";

  if (prop.major * 10 + prop.minor < 89)
  {
    std::cout << "SKIPPED: cuBLASLt FP8 GEMM requires an FP8-capable GPU.\n";
    return EXIT_SUCCESS;
  }

  std::vector<TestCase> cases = {
      {64, 80, 48, 1.0f, 0.0f, "shape_64x80x48_beta0_TN"},
      {80, 48, 64, 0.75f, -0.5f, "shape_80x48x64_beta_TN"},
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

  std::cout << "All cuBLASLt FP8 E4M3 -> BF16 TN reference tests passed\n";
  return EXIT_SUCCESS;
}
