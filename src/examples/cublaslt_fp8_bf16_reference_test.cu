#include "utils.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA_TEST(call)                                                   \
  do                                                                            \
  {                                                                             \
    cudaError_t status = call;                                                  \
    if (status != cudaSuccess)                                                  \
    {                                                                           \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "                   \
                << cudaGetErrorString(status) << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

struct TestCase
{
  int M;
  int N;
  int K;
  bool A_OP_T;
  bool B_OP_T;
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

float get_a(const std::vector<__nv_fp8_e4m3> &A, const TestCase &tc, int m, int k)
{
  return tc.A_OP_T ? static_cast<float>(A[k * tc.M + m])
                   : static_cast<float>(A[m * tc.K + k]);
}

float get_b(const std::vector<__nv_fp8_e4m3> &B, const TestCase &tc, int k, int n)
{
  return tc.B_OP_T ? static_cast<float>(B[n * tc.K + k])
                   : static_cast<float>(B[k * tc.N + n]);
}

std::vector<__nv_bfloat16> cpu_reference(
    const std::vector<__nv_fp8_e4m3> &A,
    const std::vector<__nv_fp8_e4m3> &B,
    const std::vector<__nv_bfloat16> &C_init,
    const TestCase &tc)
{
  std::vector<__nv_bfloat16> C_ref(tc.M * tc.N);

  for (int m = 0; m < tc.M; ++m)
  {
    for (int n = 0; n < tc.N; ++n)
    {
      float acc = 0.0f;
      for (int k = 0; k < tc.K; ++k)
      {
        acc += get_a(A, tc, m, k) * get_b(B, tc, k, n);
      }

      float c_old = static_cast<float>(C_init[m * tc.N + n]);
      float out = tc.alpha * acc + tc.beta * c_old;
      C_ref[m * tc.N + n] = __float2bfloat16(out);
    }
  }

  return C_ref;
}

bool run_case(const TestCase &tc)
{
  size_t a_elements = tc.A_OP_T ? static_cast<size_t>(tc.K) * tc.M
                                : static_cast<size_t>(tc.M) * tc.K;
  size_t b_elements = tc.B_OP_T ? static_cast<size_t>(tc.N) * tc.K
                                : static_cast<size_t>(tc.K) * tc.N;
  size_t c_elements = static_cast<size_t>(tc.M) * tc.N;

  std::vector<__nv_fp8_e4m3> h_A(a_elements);
  std::vector<__nv_fp8_e4m3> h_B(b_elements);
  std::vector<__nv_bfloat16> h_C_init(c_elements);

  for (size_t i = 0; i < a_elements; ++i)
  {
    h_A[i] = __nv_fp8_e4m3(input_value(static_cast<int>(i), 1));
  }
  for (size_t i = 0; i < b_elements; ++i)
  {
    h_B[i] = __nv_fp8_e4m3(input_value(static_cast<int>(i), 2));
  }
  for (size_t i = 0; i < c_elements; ++i)
  {
    h_C_init[i] = __float2bfloat16(c_init_value(static_cast<int>(i)));
  }

  std::vector<__nv_bfloat16> h_C_ref = cpu_reference(h_A, h_B, h_C_init, tc);
  std::vector<__nv_bfloat16> h_C_got(c_elements);

  __nv_fp8_e4m3 *d_A = nullptr;
  __nv_fp8_e4m3 *d_B = nullptr;
  __nv_bfloat16 *d_C = nullptr;

  CHECK_CUDA_TEST(cudaMalloc(&d_A, a_elements * sizeof(__nv_fp8_e4m3)));
  CHECK_CUDA_TEST(cudaMalloc(&d_B, b_elements * sizeof(__nv_fp8_e4m3)));
  CHECK_CUDA_TEST(cudaMalloc(&d_C, c_elements * sizeof(__nv_bfloat16)));

  CHECK_CUDA_TEST(cudaMemcpy(d_A, h_A.data(), a_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
  CHECK_CUDA_TEST(cudaMemcpy(d_B, h_B.data(), b_elements * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
  CHECK_CUDA_TEST(cudaMemcpy(d_C, h_C_init.data(), c_elements * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  utils::cublaslt_fp8_e4m3_bf16_reference(
      tc.M, tc.N, tc.K,
      d_A, d_B, d_C,
      tc.alpha, tc.beta,
      tc.A_OP_T, tc.B_OP_T,
      0, 1);

  CHECK_CUDA_TEST(cudaMemcpy(h_C_got.data(), d_C, c_elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

  CHECK_CUDA_TEST(cudaFree(d_C));
  CHECK_CUDA_TEST(cudaFree(d_B));
  CHECK_CUDA_TEST(cudaFree(d_A));

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
        std::cout << "  [Mismatch] index=" << i
                  << " got=" << got
                  << " ref=" << ref
                  << " abs=" << abs_err
                  << " rel=" << rel_err << "\n";
      }
      ++error_count;
    }
  }

  std::cout << "[Test] " << tc.name
            << " A_OP_T=" << tc.A_OP_T
            << " B_OP_T=" << tc.B_OP_T
            << " alpha=" << tc.alpha
            << " beta=" << tc.beta
            << " | max_abs=" << std::scientific << max_abs_err
            << " max_rel=" << max_rel_err
            << " worst_index=" << worst_index
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
  CHECK_CUDA_TEST(cudaGetDevice(&device));

  cudaDeviceProp prop = {};
  CHECK_CUDA_TEST(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running on " << prop.name
            << " (sm_" << prop.major << prop.minor << ")\n";

  if (prop.major * 10 + prop.minor < 89)
  {
    std::cout << "SKIPPED: cuBLASLt FP8 GEMM requires an FP8-capable GPU.\n";
    return EXIT_SUCCESS;
  }

  std::vector<TestCase> cases = {
      {64, 80, 48, false, false, 1.0f, 0.0f, "shape_64x80x48_beta0_NN"},
      {64, 80, 48, true, false, 1.0f, 0.0f, "shape_64x80x48_beta0_TN"},
      {64, 80, 48, false, true, 1.0f, 0.0f, "shape_64x80x48_beta0_NT"},
      {64, 80, 48, true, true, 1.0f, 0.0f, "shape_64x80x48_beta0_TT"},
      {80, 48, 64, false, false, 0.75f, -0.5f, "shape_80x48x64_beta_NN"},
      {80, 48, 64, true, false, 0.75f, -0.5f, "shape_80x48x64_beta_TN"},
      {80, 48, 64, false, true, 0.75f, -0.5f, "shape_80x48x64_beta_NT"},
      {80, 48, 64, true, true, 0.75f, -0.5f, "shape_80x48x64_beta_TT"},
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

  std::cout << "All cuBLASLt FP8 E4M3 -> BF16 reference tests passed\n";
  return EXIT_SUCCESS;
}
