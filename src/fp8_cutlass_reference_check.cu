#include "cutlass_fp8_reference.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                        \
  do                                                                           \
  {                                                                            \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess)                                                  \
    {                                                                          \
      std::cerr << "CUDA Error at line " << __LINE__ << ": "                  \
                << cudaGetErrorString(status) << std::endl;                    \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

__global__ void fill_e4m3_inputs(uint8_t *A, uint8_t *B, int M, int N, int K, int lda, int ldb)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_A = M * lda;
  int total_B = N * ldb;
  int total = total_A + total_B;
  if (idx >= total)
  {
    return;
  }

  // 0x38 is +1.0 in IEEE-like E4M3 encoding used by Hopper FP8.
  // Use small deterministic values so the reference is easy to sanity-check.
  uint8_t value = (idx & 1) ? uint8_t(0x30) : uint8_t(0x38);
  if (idx < total_A)
  {
    A[idx] = value;
  }
  else
  {
    B[idx - total_A] = value;
  }
}

int main(int argc, char **argv)
{
  int M = 4096;
  int N = 4096;
  int K = 4096;
  int warmup_iters = 1;
  int bench_iters = 10;

  if (argc == 4 || argc == 6)
  {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
    if (argc == 6)
    {
      warmup_iters = std::atoi(argv[4]);
      bench_iters = std::atoi(argv[5]);
    }
  }
  else if (argc != 1)
  {
    std::cerr << "Usage: " << argv[0] << " [M N K [warmup_iters bench_iters]]\n";
    return EXIT_FAILURE;
  }

  int lda = K;
  int ldb = K;
  int ldd = M;

  uint8_t *d_A = nullptr;
  uint8_t *d_B = nullptr;
  __nv_bfloat16 *d_D = nullptr;

  CHECK_CUDA(cudaMalloc(&d_A, static_cast<size_t>(M) * lda * sizeof(uint8_t)));
  CHECK_CUDA(cudaMalloc(&d_B, static_cast<size_t>(N) * ldb * sizeof(uint8_t)));
  CHECK_CUDA(cudaMalloc(&d_D, static_cast<size_t>(N) * ldd * sizeof(__nv_bfloat16)));

  int total = M * lda + K * ldb;
  int block = 256;
  int grid = (total + block - 1) / block;
  fill_e4m3_inputs<<<grid, block>>>(d_A, d_B, M, N, K, lda, ldb);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemset(d_D, 0, static_cast<size_t>(N) * ldd * sizeof(__nv_bfloat16)));

  cutlass_ref::GemmResult result;
  cutlass_ref::fp8_e4m3_gemm_bf16_reference(
      M, N, K,
      d_A, d_B, d_D,
      lda, ldb, ldd,
      &result,
      nullptr,
      warmup_iters,
      bench_iters);

  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_D));

  return EXIT_SUCCESS;
}
