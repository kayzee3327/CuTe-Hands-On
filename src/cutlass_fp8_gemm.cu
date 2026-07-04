#include "cutlass_fp8_gemm.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

namespace cutlass_fp8
{
namespace
{

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

// CUTLASS canonical TNN: e4m3t/e4m3n/bf16n.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
using FusionOperation =
    cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    FusionOperation>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCount<6>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>, // M, N, K, L (batch dimension)
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using RasterOrderOptions = typename Gemm::GemmKernel::TileScheduler::RasterOrderOptions;

#endif

} // namespace

Fp8TnnGemmResult cutlass_fp8_e4m3_bf16_tnn_gemm(
    int M, int N, int K,
    const __nv_fp8_e4m3 *d_A,
    const __nv_fp8_e4m3 *d_B,
    const __nv_bfloat16 *d_C,
    __nv_bfloat16 *d_D,
    float alpha,
    float beta,
    int warmup_iters,
    int bench_iters,
    cudaStream_t stream)
{
  Fp8TnnGemmResult result;

  if (M <= 0 || N <= 0 || K <= 0 || d_A == nullptr || d_B == nullptr ||
      d_C == nullptr || d_D == nullptr)
  {
    result.cuda_error = cudaErrorInvalidValue;
    return result;
  }

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  // batch-mode `make_cute_packed_stride`
  // shape has a appended L batch mode
  // stride results will be like stride_A: (K, _1, _0) for mode M, K, L
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  int device_id = 0;
  cudaError_t cuda_status = cudaGetDevice(&device_id);
  if (cuda_status != cudaSuccess)
  {
    result.cuda_error = cuda_status;
    return result;
  }

  auto hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(device_id);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {
          reinterpret_cast<ElementA const *>(d_A), stride_A,
          reinterpret_cast<ElementB const *>(d_B), stride_B,
      },
      {
          {}, // epilogue.thread
          reinterpret_cast<ElementC const *>(d_C), stride_C,
          reinterpret_cast<ElementD *>(d_D), stride_D,
      },
      hw_info};

  arguments.epilogue.thread.alpha = alpha;
  arguments.epilogue.thread.beta = beta;
  arguments.scheduler.max_swizzle_size = 4;
  arguments.scheduler.raster_order = RasterOrderOptions::AlongN;

  Gemm gemm_op;
  cutlass::Status status = Gemm::can_implement(arguments);
  result.cutlass_status = static_cast<int>(status);
  if (status != cutlass::Status::kSuccess)
  {
    return result;
  }

  void *workspace = nullptr;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  if (workspace_size > 0)
  {
    cuda_status = cudaMalloc(&workspace, workspace_size);
    if (cuda_status != cudaSuccess)
    {
      result.cuda_error = cuda_status;
      return result;
    }
  }

  status = gemm_op.initialize(arguments, workspace, stream);
  result.cutlass_status = static_cast<int>(status);
  if (status != cutlass::Status::kSuccess)
  {
    if (workspace != nullptr)
    {
      cudaFree(workspace);
    }
    return result;
  }

  for (int i = 0; i < warmup_iters; ++i)
  {
    status = gemm_op.run(stream);
    result.cutlass_status = static_cast<int>(status);
    if (status != cutlass::Status::kSuccess)
    {
      if (workspace != nullptr)
      {
        cudaFree(workspace);
      }
      return result;
    }
  }

  cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess)
  {
    result.cuda_error = cuda_status;
    if (workspace != nullptr)
    {
      cudaFree(workspace);
    }
    return result;
  }

  int timed_iters = std::max(bench_iters, 1);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_status = cudaEventCreate(&start);
  if (cuda_status == cudaSuccess)
  {
    cuda_status = cudaEventCreate(&stop);
  }
  if (cuda_status != cudaSuccess)
  {
    result.cuda_error = cuda_status;
    if (start != nullptr)
    {
      cudaEventDestroy(start);
    }
    if (workspace != nullptr)
    {
      cudaFree(workspace);
    }
    return result;
  }

  cudaEventRecord(start, stream);
  for (int i = 0; i < timed_iters; ++i)
  {
    status = gemm_op.run(stream);
    result.cutlass_status = static_cast<int>(status);
    if (status != cutlass::Status::kSuccess)
    {
      cudaEventDestroy(stop);
      cudaEventDestroy(start);
      if (workspace != nullptr)
      {
        cudaFree(workspace);
      }
      return result;
    }
  }
  cudaEventRecord(stop, stream);
  cuda_status = cudaEventSynchronize(stop);
  if (cuda_status != cudaSuccess)
  {
    result.cuda_error = cuda_status;
  }
  else
  {
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    result.avg_ms = elapsed_ms / timed_iters;
    double ops_per_gemm = 2.0 * static_cast<double>(M) * N * K;
    result.tflops = (ops_per_gemm / (result.avg_ms / 1000.0)) / 1.0e12;
  }

  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  if (workspace != nullptr)
  {
    cudaFree(workspace);
  }

  return result;
#else
  result.cutlass_status = static_cast<int>(cutlass::Status::kInvalid);
  result.cuda_error = cudaErrorNoKernelImageForDevice;
  return result;
#endif
}

} // namespace cutlass_fp8
