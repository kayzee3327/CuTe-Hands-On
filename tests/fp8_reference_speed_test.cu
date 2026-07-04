#include "cutlass_fp8_gemm.h"
#include "utils.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace
{

struct BenchmarkCase
{
  int M;
  int N;
  int K;
  std::string name;
};

__global__ void fill_fp8_kernel(__nv_fp8_e4m3 *ptr, size_t count, int salt)
{
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
  {
    int v = static_cast<int>((index * 37 + static_cast<size_t>(salt) * 19) % 41) - 20;
    ptr[index] = __nv_fp8_e4m3(static_cast<float>(v) / 32.0f);
  }
}

__global__ void fill_bf16_kernel(__nv_bfloat16 *ptr, size_t count)
{
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
  {
    int v = static_cast<int>((index * 13 + 7) % 23) - 11;
    ptr[index] = __float2bfloat16(static_cast<float>(v) / 16.0f);
  }
}

bool check_cuda(cudaError_t status, const char *what)
{
  if (status != cudaSuccess)
  {
    std::cerr << "CUDA error during " << what << ": "
              << cudaGetErrorString(status) << "\n";
    return false;
  }
  return true;
}

bool fill_fp8(__nv_fp8_e4m3 *ptr, size_t count, int salt)
{
  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  fill_fp8_kernel<<<blocks, threads>>>(ptr, count, salt);
  return check_cuda(cudaGetLastError(), "fill_fp8 launch");
}

bool fill_bf16(__nv_bfloat16 *ptr, size_t count)
{
  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  fill_bf16_kernel<<<blocks, threads>>>(ptr, count);
  return check_cuda(cudaGetLastError(), "fill_bf16 launch");
}

int env_int(const char *name, int fallback)
{
  const char *value = std::getenv(name);
  if (value == nullptr)
  {
    return fallback;
  }

  int parsed = std::atoi(value);
  return parsed > 0 ? parsed : fallback;
}

double gib(size_t bytes)
{
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

size_t required_bytes(const BenchmarkCase &tc)
{
  size_t a_elements = static_cast<size_t>(tc.M) * tc.K;
  size_t b_elements = static_cast<size_t>(tc.K) * tc.N;
  size_t c_elements = static_cast<size_t>(tc.M) * tc.N;
  return (a_elements + b_elements) * sizeof(__nv_fp8_e4m3) +
         2 * c_elements * sizeof(__nv_bfloat16);
}

double problem_size(const BenchmarkCase &tc)
{
  return static_cast<double>(tc.M) * tc.N * tc.K;
}

bool run_case(const BenchmarkCase &tc, int warmup_iters, int bench_iters)
{
  size_t a_elements = static_cast<size_t>(tc.M) * tc.K;
  size_t b_elements = static_cast<size_t>(tc.K) * tc.N;
  size_t c_elements = static_cast<size_t>(tc.M) * tc.N;

  __nv_fp8_e4m3 *d_A = nullptr;
  __nv_fp8_e4m3 *d_B = nullptr;
  __nv_bfloat16 *d_C = nullptr;
  __nv_bfloat16 *d_D = nullptr;

  std::cout << "\n[Shape] " << tc.name
            << " M=" << tc.M << " N=" << tc.N << " K=" << tc.K
            << " | M*N*K=" << std::scientific << problem_size(tc)
            << " | device memory ~= " << std::fixed << std::setprecision(2)
            << gib(required_bytes(tc)) << " GiB"
            << std::defaultfloat << "\n";

  if (!check_cuda(cudaMalloc(&d_A, a_elements * sizeof(__nv_fp8_e4m3)), "cudaMalloc A") ||
      !check_cuda(cudaMalloc(&d_B, b_elements * sizeof(__nv_fp8_e4m3)), "cudaMalloc B") ||
      !check_cuda(cudaMalloc(&d_C, c_elements * sizeof(__nv_bfloat16)), "cudaMalloc C") ||
      !check_cuda(cudaMalloc(&d_D, c_elements * sizeof(__nv_bfloat16)), "cudaMalloc D"))
  {
    cudaFree(d_D);
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
    return false;
  }

  bool initialized = fill_fp8(d_A, a_elements, 1) &&
                     fill_fp8(d_B, b_elements, 2) &&
                     fill_bf16(d_C, c_elements) &&
                     check_cuda(cudaDeviceSynchronize(), "initialization synchronize");
  if (!initialized)
  {
    cudaFree(d_D);
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
    return false;
  }

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;

  std::cout << "[cuBLASLt reference API]\n";
  utils::cublaslt_fp8_e4m3_bf16_tn_reference(
      tc.M, tc.N, tc.K,
      d_A, d_B, d_C, d_D,
      alpha, beta,
      warmup_iters, bench_iters);

  std::cout << "[CUTLASS reference API]\n";
  cutlass_fp8::Fp8TnnGemmResult cutlass_result =
      cutlass_fp8::cutlass_fp8_e4m3_bf16_tnn_gemm(
          tc.M, tc.N, tc.K,
          d_A, d_B, d_C, d_D,
          alpha, beta,
          warmup_iters, bench_iters);

  if (!cutlass_result.ok())
  {
    std::cerr << "[CUTLASS FP8 TNN] failed for " << tc.name
              << " cutlass_status=" << cutlass_result.cutlass_status
              << " cuda_error=" << cudaGetErrorString(cutlass_result.cuda_error) << "\n";
    cudaFree(d_D);
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);
    return false;
  }

  std::cout << "[CUTLASS FP8 E4M3->BF16 TNN] "
            << "M=" << tc.M << ", N=" << tc.N << ", K=" << tc.K
            << " | Time: " << std::fixed << std::setprecision(3) << cutlass_result.avg_ms << " ms"
            << " | Performance: " << std::fixed << std::setprecision(2) << cutlass_result.tflops
            << " TFLOPS\n";

  cudaFree(d_D);
  cudaFree(d_C);
  cudaFree(d_B);
  cudaFree(d_A);
  return true;
}

} // namespace

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

  int warmup_iters = env_int("FP8_SPEED_WARMUP_ITERS", 5);
  int bench_iters = env_int("FP8_SPEED_BENCH_ITERS", 20);
  std::cout << "Warmup iterations: " << warmup_iters
            << " | benchmark iterations: " << bench_iters << "\n";

  constexpr double max_problem_size =
      static_cast<double>(16384) * 16384.0 * 16384.0;
  std::vector<BenchmarkCase> cases = {
      {10240, 10240, 10240, "cubic"},
      {15360, 8192, 8192, "m_dominant"},
      {8192, 15360, 8192, "n_dominant"},
      {8192, 8192, 15360, "k_dominant"},
  };

  int failures = 0;
  for (const BenchmarkCase &tc : cases)
  {
    if (problem_size(tc) >= max_problem_size)
    {
      std::cerr << "Shape " << tc.name
                << " violates the 16384*16384*16384 problem-size limit\n";
      ++failures;
      continue;
    }

    if (!run_case(tc, warmup_iters, bench_iters))
    {
      ++failures;
    }
  }

  if (failures != 0)
  {
    std::cout << failures << " speed case(s) failed\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
