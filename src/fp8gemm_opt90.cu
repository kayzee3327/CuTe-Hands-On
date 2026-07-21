#include <cute/tensor.hpp>
// #include <cuda_fp8.h>
// #include <cuda_bf16.h>
// cutlass provide all types in include/cute/numeric/numeric_types.hpp

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "cutlass_fp8_gemm.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"

#include "cutlass/device_kernel.h"
#include "utils.h"

namespace {

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[cute::size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[cute::size<2>(SmemLayoutA{})];
};

struct SM90_TNN_E4M3_F32_BF16
{
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementAccumulate = float;
  using ElementCompute = float;
  using ElementC = cutlass::bfloat16_t;

  using BM = cute::_128;
  using BN = cute::_128;
  using BK = cute::_128;
  using CtaTiler = cute::Shape<BM, BN, BK>;

  // One warpgroup issues TMA loads and one warpgroup performs WGMMA.
  static constexpr int kProducerThreads = 128;
  static constexpr int kConsumerThreads = 128;
  static constexpr int kThreads = kProducerThreads + kConsumerThreads;
  static constexpr int kStages = 6;
  static constexpr uint32_t kSwizzleGroupM = 4;
  static constexpr uint32_t kSwizzleGroupN = 4;

  static constexpr uint32_t cx = 2;
  static constexpr uint32_t cy = 1;
  static constexpr uint32_t cz = 1;
  using ClusterShape = cute::Shape<cute::Int<cx>,
                                   cute::Int<cy>,
                                   cute::Int<cz>>;

  // use factory for composed cute types for direct instantiation
  // aligned well with cute design
  CUTE_HOST_DEVICE
  static constexpr auto make_cta_tiler() { return CtaTiler{}; }

  CUTE_HOST_DEVICE
  static constexpr auto make_smem_layout_a()
  {
    using namespace cute;
    auto atom = GMMA::Layout_K_SW128_Atom<ElementA>{};
    return tile_to_shape(atom, make_shape(BM{}, BK{}, Int<kStages>{}));
  }

  CUTE_HOST_DEVICE
  static constexpr auto make_smem_layout_b()
  {
    using namespace cute;
    auto atom = GMMA::Layout_K_SW128_Atom<ElementB>{};
    return tile_to_shape(atom, make_shape(BN{}, BK{}, Int<kStages>{}));
  }

  CUTE_HOST_DEVICE
  static constexpr auto make_tiled_mma()
  {
    return cute::make_tiled_mma(
      cute::SM90_64x128x32_F32E4M3E4M3_SS_TN<>{});
  }

  struct Arguments
  {
    ElementA const* A;
    ElementB const* B;
    ElementC      * C;
    int M, N, K;
    ElementCompute alpha, beta;
  };

};

template <class Policy, class TmaA, class TmaB,
          class SwizzledTileLayout>
struct GemmParams
{
  using ElementA = typename Policy::ElementA;
  using ElementB = typename Policy::ElementB;
  using ElementC = typename Policy::ElementC;
  using ElementCompute = typename Policy::ElementCompute;

  ElementA const* A;
  ElementB const* B;
  ElementC      * C;
  int M, N, K;
  ElementCompute alpha, beta;

  TmaA tma_a;
  TmaB tma_b;

  SwizzledTileLayout tile_layout;
};

template<uint32_t GM, uint32_t GN, class ClusterShape>
auto make_swizzled_tile_layout(
  uint32_t tiles_m,
  uint32_t tiles_n,
  ClusterShape
) {
  using namespace cute;

  static_assert(rank_v<ClusterShape> == 3,
                "ClusterShape must have M, N, and K modes");

  constexpr uint32_t CM = size<0>(ClusterShape{});
  constexpr uint32_t CN = size<1>(ClusterShape{});
  constexpr uint32_t CK = size<2>(ClusterShape{});

  static_assert(CK == 1, "Grid swizzling only supports ClusterShape K == 1");
  static_assert(GM > 0 && GN > 0, "Swizzle-group dimensions must be positive");
  static_assert(GM % CM == 0,
                "The M swizzle group must contain complete clusters");
  static_assert(GN % CN == 0,
                "The N swizzle group must contain complete clusters");

  constexpr uint32_t cluster_group_m = GM / CM;
  constexpr uint32_t cluster_group_n = GN / CN;
  uint32_t clusters_m = tiles_m / CM;
  uint32_t clusters_n = tiles_n / CN;

  // Preconditions for this initial version:
  // tiles_m % GM == 0 && tiles_n % GN == 0

  // Physical cluster BID digits:
  //   (local_cluster_m, local_cluster_n, group_m, group_n)
  //
  // output:
  //   natural logical cluster index cluster_m + clusters_m * cluster_n
  //
  // Keep this traversal rank-1 so blocked_product treats each CM x CN CTA
  // cluster as one indivisible block.
  //
  // size(s) == clusters_m * clusters_n
  // cluster groups as rectangle tiles
  // This is a explicit result of division or product.
  auto cluster_bid_to_logical_linear = make_layout(
      make_tuple(make_shape(Int<cluster_group_m>{},
                            Int<cluster_group_n>{},
                            clusters_m / cluster_group_m,
                            clusters_n / cluster_group_n)),
      make_tuple(make_stride(_1{},
                             clusters_m,
                             Int<cluster_group_m>{},
                             clusters_m * cluster_group_n)));
  // inside a cluster
  auto cta_in_cluster_layout =
      Layout<Shape<Int<CM>, Int<CN>>>{};
  // get the grid view
  auto bid_to_logical_linear =
      blocked_product(cta_in_cluster_layout,
                      cluster_bid_to_logical_linear);

  // Natural logical linear index -> (logical_m, logical_n)
  auto logical_coord = make_identity_layout(make_shape(tiles_m, tiles_n));

  return composition(logical_coord, bid_to_logical_linear);
}

template <class Policy>
auto make_gemm_params(typename Policy::Arguments args)
{
  using namespace cute;

  auto sA = Policy::make_smem_layout_a();
  auto sB = Policy::make_smem_layout_b();

  auto dA = make_stride(args.K, _1{});
  auto dB = make_stride(args.K, _1{});

  auto mA = make_tensor(args.A, make_shape(args.M, args.K), dA);
  auto mB = make_tensor(args.B, make_shape(args.N, args.K), dB);

  using ClusterShape = typename Policy::ClusterShape;
  using TmaOpA = std::conditional_t<
    (size<1>(ClusterShape{}) > _1{}), 
    SM90_TMA_LOAD_MULTICAST, 
    SM90_TMA_LOAD>;
  using TmaOpB = std::conditional_t<
    (size<0>(ClusterShape{}) > _1{}),
    SM90_TMA_LOAD_MULTICAST,
    SM90_TMA_LOAD>;

  auto tma_a = make_tma_copy_A_sm90(TmaOpA{}, mA, sA(_,_,0),
                                    Policy::make_cta_tiler(),
                                    ClusterShape{});
  auto tma_b = make_tma_copy_B_sm90(TmaOpB{}, mB, sB(_,_,0),
                                    Policy::make_cta_tiler(),
                                    ClusterShape{});

  auto tiles_m = ceil_div(args.M, typename Policy::BM{});
  auto tiles_n = ceil_div(args.N, typename Policy::BN{});
  auto tile_layout =
      make_swizzled_tile_layout<Policy::kSwizzleGroupM,
                                Policy::kSwizzleGroupN>(
          tiles_m, tiles_n, typename Policy::ClusterShape{});

  return GemmParams<Policy, decltype(tma_a), decltype(tma_b),
                    decltype(tile_layout)> {
    args.A, args.B, args.C,
    args.M, args.N, args.K,
    args.alpha, args.beta,
    tma_a, tma_b,
    tile_layout
  };
}



template <class Policy, class Params>
__global__ __launch_bounds__(Policy::kThreads)
// v1: tma+wgmma (ss), use explicit producer-consumer synchronization
// v2: excessive DRAM access: grid swizzle + cluster multicast
// v3: dedicate one warpgroup to TMA and one warpgroup to WGMMA
void gemm(CUTLASS_GRID_CONSTANT Params const params)
{
  using namespace cute;
  using TA = typename Policy::ElementA;
  using TB = typename Policy::ElementB;

  auto smem_layout_a = Policy::make_smem_layout_a();
  auto smem_layout_b = Policy::make_smem_layout_b();
  using SmemLayoutA = decltype(smem_layout_a);
  using SmemLayoutB = decltype(smem_layout_b);

  auto cta_tiler = Policy::make_cta_tiler();

  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});
  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);
  CUTE_STATIC_ASSERT_V(size<0>(smem_layout_a) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(smem_layout_b) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(smem_layout_a) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(smem_layout_b) == size<2>(cta_tiler));  // BLK_K
  
  int M = params.M;
  int N = params.N;
  int K = params.K;

  auto const& tma_a = params.tma_a; // Multicast TiledCopy
  auto const& tma_b = params.tma_b; // Multicast TiledCopy
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K)); // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K)); // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(params.C), 
                          make_shape(M, N), 
                          make_stride(_1{}, M));          // (M,N)

  auto cta_coord = append(params.tile_layout(blockIdx.x + gridDim.x * blockIdx.y),_);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  extern __shared__ char raw_smem[];
  using SharedStorageT = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorageT& smem = *reinterpret_cast<SharedStorageT*>(raw_smem);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

#if 0
  if (thread0())
  {
    print("mA: "); print(mA); print("\n");
    print("mB: "); print(mB); print("\n");
    print("gA: "); print(gA); print("\n");
    print("gB: "); print(gB); print("\n");
    print("sA: "); print(sA); print("\n");
    print("sB: "); print(sB); print("\n");
  }
  
#endif

  dim3 cta_id = block_id_in_cluster();
  // CTA with same id y requires same A operand
  // Along the N-mode, CTA needs same A tile.
  auto cta_tma_copy_a = tma_a.get_slice(cta_id.y); 
  auto cta_tma_copy_b = tma_b.get_slice(cta_id.x); 
  Tensor tAgA = cta_tma_copy_a.partition_S(gA); // (TMA, TMA_M, TMA_K, k)
  Tensor tAsA = cta_tma_copy_a.partition_D(sA); // (TMA, TMA_M, TMA_K, PIPE)
  Tensor tBgB = cta_tma_copy_b.partition_S(gB); // (TMA, TMA_N, TMA_K, k)
  Tensor tBsB = cta_tma_copy_b.partition_D(sB); // (TMA, TMA_N, TMA_K, PIPE)

#if 0
  // requires shape check
  if (thread0())
  {
    print("cta_tma_copy_a = tma_a.get_slice(cta_id.y): "); print(cta_tma_copy_a); print("\n");
    print("cta_tma_copy_b = tma_b.get_slice(cta_id.x): "); print(cta_tma_copy_b); print("\n");
    print("tAgA: "); print(tAgA); print("\n");
    print("tAsA: "); print(tAsA); print("\n");
    print("tBgB: "); print(tBgB); print("\n");
    print("tBsB: "); print(tBsB); print("\n");
  }
  
#endif
  using ClusterShape = typename Policy::ClusterShape;
  auto cluster_layout = make_layout(ClusterShape{});
  [[maybe_unused]] uint16_t mc_mask_a = 0;
  [[maybe_unused]] uint16_t mc_mask_b = 0;

  for (int m = 0; m < size<0>(cluster_layout); m++)
    mc_mask_b |= uint16_t(1) << cluster_layout(m, cta_id.y, _0{});

  for (int n = 0; n < size<1>(cluster_layout); n++)
    mc_mask_a |= uint16_t(1) << cluster_layout(cta_id.x, n, _0{});

#if 1
  // requires shape check
  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  constexpr int K_PIPE_MAX = size<3>(tAsA);
  int k_tile_count = size<3>(tAgA);

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  int warp_group_thread_idx = int(threadIdx.x) % Policy::kConsumerThreads;
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
  constexpr int consumer_arrival_count =
      size<0>(ClusterShape{}) + size<1>(ClusterShape{}) - 1;

  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX; k_pipe++)
  {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[k_pipe], 1);
      // One completion signal is sent to every CTA in this CTA's cluster
      // row/column union. The local CTA occurs in both sets, so count it once.
      ConsumerBarType::init(&consumer_mbar[k_pipe], consumer_arrival_count);
    }
  }
  cutlass::arch::fence_barrier_init();
  cluster_sync();

  if (warp_group_idx == 0) {
    // The producer warpgroup owns only the TMA pipeline. Starting in the
    // opposite phase makes the first K_PIPE_MAX acquisitions immediately
    // available without a separate, potentially out-of-bounds prefill loop.
    cutlass::PipelineState<K_PIPE_MAX> write_state(0, 1, 0);

    if ((warp_idx == 0) && lane_predicate) {
      CUTE_NO_UNROLL
      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        int write_pipe = write_state.index();
        if (k_tile >= K_PIPE_MAX) {
          ConsumerBarType::wait(&consumer_mbar[write_pipe],
                                write_state.phase());
        }

        ProducerBarType::arrive_and_expect_tx(
            &producer_mbar[write_pipe], tma_transaction_bytes);

        if constexpr (size<1>(ClusterShape{}) > _1{})
          copy(tma_a.with(producer_mbar[write_pipe], mc_mask_a),
               tAgA(_,_,_,k_tile), tAsA(_,_,_,write_pipe));
        else
          copy(tma_a.with(producer_mbar[write_pipe]),
               tAgA(_,_,_,k_tile), tAsA(_,_,_,write_pipe));

        if constexpr (size<0>(ClusterShape{}) > _1{})
          copy(tma_b.with(producer_mbar[write_pipe], mc_mask_b),
               tBgB(_,_,_,k_tile), tBsB(_,_,_,write_pipe));
        else
          copy(tma_b.with(producer_mbar[write_pipe]),
               tBgB(_,_,_,k_tile), tBsB(_,_,_,write_pipe));

        ++write_state;
      }
    }
  }
  else {
    // A full warpgroup must execute every WGMMA operation in convergence.
    auto mma = Policy::make_tiled_mma();
    static_assert(decltype(size(mma))::value == Policy::kConsumerThreads,
                  "The consumer warpgroup must match the tiled MMA thread count");
    ThrMMA thr_mma = mma.get_slice(warp_group_thread_idx);
    Tensor tCsA = thr_mma.partition_A(sA);                   // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);                   // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCgC = thr_mma.partition_C(gC);                   // (MMA,MMA_M,MMA_N)

    Tensor tCrC = thr_mma.make_fragment_C(tCgC);             // (MMA,MMA_M,MMA_N)
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);             // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);             // (MMA,MMA_N,MMA_K,PIPE)
    clear(tCrC);

    cutlass::PipelineState<K_PIPE_MAX> read_state;
    cutlass::PipelineState<K_PIPE_MAX> release_state;
    constexpr int K_PIPE_MMAS = 1;

    auto release_stage = [&](int release_pipe) {
      // Signal every CTA in the cluster row/column union exactly once.
      bool signal = warp_group_thread_idx < consumer_arrival_count;
      uint32_t dst_cta = 0;
      if (warp_group_thread_idx < size<0>(ClusterShape{})) {
        dst_cta = cluster_layout(warp_group_thread_idx, cta_id.y, _0{});
      }
      else if (signal) {
        int n = warp_group_thread_idx - size<0>(ClusterShape{});
        n += (n >= int(cta_id.y)); // Skip the local CTA already in the M set.
        dst_cta = cluster_layout(cta_id.x, n, _0{});
      }
      ConsumerBarType::arrive(&consumer_mbar[release_pipe], dst_cta, signal);
    };

    // Keep one committed WGMMA batch in flight. Its stage is released one
    // iteration later, after warpgroup_wait<1>() proves that batch is done.
    int prologue_mma_count = k_tile_count > 0 ? K_PIPE_MMAS : 0;
    warpgroup_fence_operand(tCrC);
    if (prologue_mma_count > 0) {
      int read_pipe = read_state.index();
      ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
      warpgroup_commit_batch();
      ++read_state;
    }
    warpgroup_fence_operand(tCrC);

    CUTE_NO_UNROLL
    for (int k_tile = prologue_mma_count; k_tile < k_tile_count; ++k_tile) {
      int read_pipe = read_state.index();
      ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

      warpgroup_fence_operand(tCrC);
      warpgroup_arrive();
      gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
      warpgroup_commit_batch();

      // The newest batch may still be running, but the oldest one is complete.
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(tCrC);

      release_stage(release_state.index());
      ++read_state;
      ++release_state;
    }

    // Drain the one-batch WGMMA prologue and release its final stage.
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
    if (prologue_mma_count > 0) {
      release_stage(release_state.index());
    }

    // Epilogue (unpredicated; the launch checks guarantee full M/N tiles).
    using ElementCompute = typename Policy::ElementCompute;
    using ElementC = typename Policy::ElementC;
    cutlass::NumericConverter<ElementC, ElementCompute> convert_to_c;

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
      ElementCompute result = params.alpha * tCrC(i);
      if (params.beta != ElementCompute(0)) {
        result += params.beta * ElementCompute(tCgC(i));
      }
      tCgC(i) = convert_to_c(result);
    }
  }

  // Keep producer CTAs alive until all multicast consumers have released their
  // final stages and completed the epilogue.
  cluster_sync();
#endif
}

template<class Policy = SM90_TNN_E4M3_F32_BF16>
cudaError_t launch_gemm(typename Policy::ElementA const* A, 
                        typename Policy::ElementB const* B, 
                        typename Policy::ElementC      * C,
                        int M, int N, int K,
                        typename Policy::ElementCompute alpha,
                        typename Policy::ElementCompute beta,
                        int warmup_iters = 5,
                        int bench_iters = 100)
{
  using namespace cute;

  using TA = typename Policy::ElementA;
  using TB = typename Policy::ElementB;
  using Arguments = typename Policy::Arguments;
  using BM = typename Policy::BM;
  using BN = typename Policy::BN;

  int tiles_m = size(ceil_div(M, BM{}));
  int tiles_n = size(ceil_div(N, BN{}));
  if ((tiles_m % Policy::kSwizzleGroupM) != 0 ||
      (tiles_n % Policy::kSwizzleGroupN) != 0) {
    std::cerr << "Error: fp8gemm_opt90 requires tiles_m divisible by "
              << Policy::kSwizzleGroupM
              << " and tiles_n divisible by "
              << Policy::kSwizzleGroupN
              << " for the current swizzle." << std::endl;
    return cudaErrorInvalidValue;
  }
  
  Arguments args{A, B, C, M, N, K, alpha, beta};
  auto params = make_gemm_params<Policy>(args);

  dim3 dimBlock(Policy::kThreads);
  // test the diff between traditional launch and cluster launch
  dim3 dimCluster(Policy::cx, Policy::cy, Policy::cz);
  // round up to complete each cluster
  dim3 dimGrid(round_up(tiles_m, dimCluster.x),
               round_up(tiles_n, dimCluster.y));
  
  int smem_size = sizeof(SharedStorage<TA, TB, 
                                       decltype(Policy::make_smem_layout_a()),
                                       decltype(Policy::make_smem_layout_b())>);

  auto* kernel_ptr = &gemm<Policy, decltype(params)>;
  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
  
  cutlass::ClusterLaunchParams cparams = {dimGrid, dimBlock, dimCluster, smem_size};
  auto run_kernel = [&]() -> cudaError_t {
    cutlass::Status status =
        cutlass::launch_kernel_on_cluster(cparams, (const void*) kernel_ptr, params);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Error: Failed at kernel Launch" << std::endl;
      return cudaErrorLaunchFailure;
    }
    return cudaPeekAtLastError();
  };

  int timed_iters = std::max(bench_iters, 1);

  // Warm up the kernel before timing so one-time launch and cache effects do not
  // dominate the measurement.
  for (int i = 0; i < std::max(warmup_iters, 0); ++i) {
    cudaError_t status = run_kernel();
    if (status != cudaSuccess) {
      return status;
    }
  }
  cudaError_t cuda_status = cudaStreamSynchronize(cparams.cuda_stream);
  if (cuda_status != cudaSuccess) {
    return cuda_status;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_status = cudaEventCreate(&start);
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaEventCreate(&stop);
  }
  if (cuda_status != cudaSuccess) {
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    return cuda_status;
  }

  cuda_status = cudaEventRecord(start, cparams.cuda_stream);
  for (int i = 0; i < timed_iters && cuda_status == cudaSuccess; ++i) {
    cuda_status = run_kernel();
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaEventRecord(stop, cparams.cuda_stream);
  }
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaEventSynchronize(stop);
  }

  float elapsed_ms = 0.0f;
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaEventElapsedTime(&elapsed_ms, start, stop);
  }
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  if (cuda_status != cudaSuccess) {
    return cuda_status;
  }

  float avg_ms = elapsed_ms / timed_iters;
  double ops_per_gemm = 2.0 * static_cast<double>(M) * N * K;
  double tflops = (ops_per_gemm / (avg_ms / 1000.0)) / 1.0e12;

  std::cout << "[CuTe FP8 E4M3->BF16] "
            << "M=" << M << ", N=" << N << ", K=" << K
            << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
            << " | Performance: " << std::fixed << std::setprecision(2) << tflops
            << " TFLOPS\n";

  return cudaSuccess;
}

}; // namespace


int main(int argc, char *argv[])
{
  using Policy = SM90_TNN_E4M3_F32_BF16;
  using TA = typename Policy::ElementA;
  using TB = typename Policy::ElementB;
  using TC = typename Policy::ElementC;

  int m = 8192;
  int n = 8192;
  int k = 8192;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  thrust::host_vector<TC> h_C(m*n);

  for (int j = 0; j < m*k; ++j) h_A[j] = TA(int((rand() % 2) ? 1 : -1));
  for (int j = 0; j < n*k; ++j) h_B[j] = TB(int((rand() % 2) ? 1 : -1));
  for (int j = 0; j < m*n; ++j) h_C[j] = TC(0);

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;
  thrust::device_vector<TC> d_C_source = h_C;
  thrust::device_vector<TC> d_C_cublaslt = h_C;
  thrust::device_vector<TC> d_C_cutlass = h_C;

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  constexpr int warmup_iters = 5;
  constexpr int bench_iters = 100;

  cudaError_t status = launch_gemm<Policy>(
      d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k, alpha, beta,
      warmup_iters, bench_iters);
  if (status != cudaSuccess) {
    std::cerr << "[fp8gemm_opt90] launch failed: "
              << cudaGetErrorString(status) << "\n";
    return EXIT_FAILURE;
  }

  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "[fp8gemm_opt90] synchronize failed: "
              << cudaGetErrorString(status) << "\n";
    return EXIT_FAILURE;
  }

  auto* d_A_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(d_A.data().get());
  auto* d_B_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(d_B.data().get());
  auto* d_C_source_bf16 = reinterpret_cast<__nv_bfloat16*>(d_C_source.data().get());
  auto* d_C_bf16 = reinterpret_cast<__nv_bfloat16*>(d_C.data().get());
  auto* d_C_cublaslt_bf16 = reinterpret_cast<__nv_bfloat16*>(d_C_cublaslt.data().get());
  auto* d_C_cutlass_bf16 = reinterpret_cast<__nv_bfloat16*>(d_C_cutlass.data().get());

  utils::cublaslt_fp8_e4m3_bf16_tn_reference(
      m, n, k,
      d_A_fp8, d_B_fp8, d_C_source_bf16, d_C_cublaslt_bf16,
      alpha, beta,
      warmup_iters, bench_iters);

  cutlass_fp8::Fp8TnnGemmResult cutlass_result =
      cutlass_fp8::cutlass_fp8_e4m3_bf16_tnn_gemm(
          m, n, k,
          d_A_fp8, d_B_fp8, d_C_source_bf16, d_C_cutlass_bf16,
          alpha, beta,
          warmup_iters, bench_iters);
  if (!cutlass_result.ok()) {
    std::cerr << "[CUTLASS FP8 TNN] failed"
              << " cutlass_status=" << cutlass_result.cutlass_status
              << " cuda_error=" << cudaGetErrorString(cutlass_result.cuda_error) << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "[CUTLASS FP8 E4M3->BF16 TNN] "
            << "M=" << m << ", N=" << n << ", K=" << k
            << " | Time: " << std::fixed << std::setprecision(3) << cutlass_result.avg_ms << " ms"
            << " | Performance: " << std::fixed << std::setprecision(2) << cutlass_result.tflops
            << " TFLOPS\n";

  std::cout << "\n[Compare] fp8gemm_opt90 vs cuBLASLt\n";
  utils::compare_tensors(d_C_bf16, d_C_cublaslt_bf16, m * n, 0.25f, 0.03f);

  std::cout << "\n[Compare] fp8gemm_opt90 vs CUTLASS FP8 TNN\n";
  utils::compare_tensors(d_C_bf16, d_C_cutlass_bf16, m * n, 0.25f, 0.03f);

  std::cout << "\n[Compare] CUTLASS FP8 TNN vs cuBLASLt\n";
  utils::compare_tensors(d_C_cutlass_bf16, d_C_cublaslt_bf16, m * n, 0.25f, 0.03f);

  return 0;
}
