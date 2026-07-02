#include "cutlass_fp8_reference.h"

#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#define CHECK_CUDA_FP8_REF(call)                                               \
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

#define CHECK_CUTLASS_FP8_REF(status)                                           \
  do                                                                           \
  {                                                                            \
    cutlass::Status status_ = (status);                                         \
    if (status_ != cutlass::Status::kSuccess)                                   \
    {                                                                          \
      std::cerr << "CUTLASS Error at line " << __LINE__ << ": "               \
                << cutlassGetStatusString(status_) << std::endl;               \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

namespace cutlass_ref
{
namespace
{

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = AlignmentC;

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _2, _1>;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

constexpr char kKernelName[] =
    "cutlass_sm90_fp8_e4m3_e4m3_f32_bf16_bf16_128x128x128_1x2x1_tnn";

template <class Stride>
Stride make_mk_row_major_stride(int lda)
{
  Stride stride{};
  cute::get<0>(stride) = static_cast<int64_t>(lda);
  cute::get<2>(stride) = int64_t(0);
  return stride;
}

template <class Stride>
Stride make_nk_operand_stride(int ldb)
{
  Stride stride{};
  cute::get<0>(stride) = static_cast<int64_t>(ldb);
  cute::get<2>(stride) = int64_t(0);
  return stride;
}

template <class Stride>
Stride make_mn_column_major_stride(int ldd)
{
  Stride stride{};
  cute::get<1>(stride) = static_cast<int64_t>(ldd);
  cute::get<2>(stride) = int64_t(0);
  return stride;
}

typename Gemm::Arguments make_arguments(
    int M, int N, int K,
    const uint8_t *d_A_e4m3,
    const uint8_t *d_B_e4m3,
    __nv_bfloat16 *d_D_bf16,
    int lda, int ldb, int ldd)
{
  auto ptr_A = reinterpret_cast<ElementA const *>(d_A_e4m3);
  auto ptr_B = reinterpret_cast<ElementB const *>(d_B_e4m3);
  auto ptr_D = reinterpret_cast<ElementD *>(d_D_bf16);

  StrideA stride_A = make_mk_row_major_stride<StrideA>(lda);
  StrideB stride_B = make_nk_operand_stride<StrideB>(ldb);
  StrideC stride_C = make_mn_column_major_stride<StrideC>(ldd);
  StrideD stride_D = make_mn_column_major_stride<StrideD>(ldd);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {ptr_A, stride_A, ptr_B, stride_B},
      {
          {1.0f, 0.0f},
          ptr_D, stride_C,
          ptr_D, stride_D}};

  arguments.scheduler.raster_order =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions::AlongN;
  arguments.scheduler.max_swizzle_size = 1;

  return arguments;
}

void run_reference(
    int M, int N, int K,
    const uint8_t *d_A_e4m3,
    const uint8_t *d_B_e4m3,
    __nv_bfloat16 *d_D_bf16,
    int lda, int ldb, int ldd,
    GemmResult *result,
    cudaStream_t stream,
    int warmup_iters,
    int bench_iters)
{
  if (warmup_iters < 0 || bench_iters <= 0)
  {
    std::cerr << "Invalid warmup/benchmark iteration count for CUTLASS FP8 reference\n";
    std::exit(EXIT_FAILURE);
  }

  Gemm gemm;
  auto arguments = make_arguments(M, N, K, d_A_e4m3, d_B_e4m3, d_D_bf16, lda, ldb, ldd);

  CHECK_CUTLASS_FP8_REF(gemm.can_implement(arguments));

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void *workspace = nullptr;
  if (workspace_size > 0)
  {
    CHECK_CUDA_FP8_REF(cudaMalloc(&workspace, workspace_size));
  }

  CHECK_CUTLASS_FP8_REF(gemm.initialize(arguments, workspace, stream));

  for (int i = 0; i < warmup_iters; ++i)
  {
    CHECK_CUTLASS_FP8_REF(gemm.run(stream));
  }
  CHECK_CUDA_FP8_REF(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CHECK_CUDA_FP8_REF(cudaEventCreate(&start));
  CHECK_CUDA_FP8_REF(cudaEventCreate(&stop));

  CHECK_CUDA_FP8_REF(cudaEventRecord(start, stream));
  for (int i = 0; i < bench_iters; ++i)
  {
    CHECK_CUTLASS_FP8_REF(gemm.run(stream));
  }
  CHECK_CUDA_FP8_REF(cudaEventRecord(stop, stream));
  CHECK_CUDA_FP8_REF(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA_FP8_REF(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / static_cast<float>(bench_iters);

  double ops_per_gemm = 2.0 * static_cast<double>(M) * N * K;
  double tflops = (ops_per_gemm / (avg_ms / 1000.0)) / 1e12;

  if (result != nullptr)
  {
    result->avg_ms = avg_ms;
    result->tflops = tflops;
    result->kernel_name = kKernelName;
  }

  std::cout << "[CUTLASS FP8 E4M3xE4M3->BF16] "
            << "M=" << M << ", N=" << N << ", K=" << K
            << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
            << " | Performance: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS"
            << " | Kernel: " << kKernelName << "\n";

  CHECK_CUDA_FP8_REF(cudaEventDestroy(start));
  CHECK_CUDA_FP8_REF(cudaEventDestroy(stop));
  if (workspace != nullptr)
  {
    CHECK_CUDA_FP8_REF(cudaFree(workspace));
  }
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

} // namespace

void fp8_e4m3_gemm_bf16_reference(
    int M, int N, int K,
    const uint8_t *d_A_e4m3,
    const uint8_t *d_B_e4m3,
    __nv_bfloat16 *d_D_bf16,
    int lda, int ldb, int ldd,
    GemmResult *result,
    cudaStream_t stream,
    int warmup_iters,
    int bench_iters)
{
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run_reference(M, N, K,
                d_A_e4m3, d_B_e4m3, d_D_bf16,
                lda, ldb, ldd,
                result, stream,
                warmup_iters, bench_iters);
#else
  (void)M;
  (void)N;
  (void)K;
  (void)d_A_e4m3;
  (void)d_B_e4m3;
  (void)d_D_bf16;
  (void)lda;
  (void)ldb;
  (void)ldd;
  (void)result;
  (void)stream;
  (void)warmup_iters;
  (void)bench_iters;
  std::cerr << "CUTLASS FP8 reference requires SM90 GMMA support. "
            << "Build this target with CUDA arch 90a.\n";
  std::exit(EXIT_FAILURE);
#endif
}

} // namespace cutlass_ref
