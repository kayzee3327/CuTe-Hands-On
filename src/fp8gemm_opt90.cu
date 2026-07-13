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
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/numeric_conversion.h"

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

  // static constexpr int kThreads = 384; // tiled mma already counts threads
  static constexpr int kStages = 6; // to be ...

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

template <class Policy, class TmaAtomA, class TmaAtomB, 
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

  TmaAtomA tma_atom_a;
  TmaAtomB tma_atom_b;

  SwizzledTileLayout tile_layout;
};

template<uint32_t G = 4>
auto make_swizzled_tile_layout(
  uint32_t tiles_m,
  uint32_t tiles_n
) {
  using namespace cute;

  // Precondition for this initial version:
  // tiles_m % G == 0 && tiles_n % G == 0

  // physical BID digits:
  //   (local_m, local_n, group_m, group_n)
  //
  // output:
  //   natural logical linear index m + tiles_m * n
  auto bid_to_logical_linear = make_layout(
      make_shape(Int<G>{}, Int<G>{}, tiles_m / G, tiles_n / G),
      make_stride( _1{}, tiles_m, Int<G>{}, tiles_m * G));

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

  auto tma_atom_a = make_tma_atom(
    SM90_TMA_LOAD{}, mA, sA(_,_,0), // SMEM layout
    make_shape(typename Policy::BM{}, typename Policy::BK{})); // GMEM tiler
  auto tma_atom_b = make_tma_atom(
    SM90_TMA_LOAD{}, mB, sB(_,_,0),
    make_shape(typename Policy::BN{}, typename Policy::BK{}));
  
  auto tiles_m = ceil_div(args.M, typename Policy::BM{});
  auto tiles_n = ceil_div(args.N, typename Policy::BN{});
  auto tile_layout = make_swizzled_tile_layout(tiles_m, tiles_m);

  return GemmParams<Policy, decltype(tma_atom_a), decltype(tma_atom_b),
                    decltype(tile_layout)> {
    args.A, args.B, args.C,
    args.M, args.N, args.K,
    args.alpha, args.beta,
    tma_atom_a, tma_atom_b,
    tile_layout
  };
}



template <class Policy, class Params>
__global__
// v1: tma+wgmma (ss), use explicit producer-consumer synchronization
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

  auto const& tma_atom_a = params.tma_atom_a;
  auto const& tma_atom_b = params.tma_atom_b;
  Tensor mA = tma_atom_a.get_tma_tensor(make_shape(M,K)); // (M,K) TMA Tensor
  Tensor mB = tma_atom_b.get_tma_tensor(make_shape(N,K)); // (N,K) TMA Tensor
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

  auto [tAgA, tAsA] = tma_partition(tma_atom_a, _0{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA)); // (TMA,k) and (TMA,k_pipe)
  auto [tBgB, tBsB] = tma_partition(tma_atom_b, _0{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB)); // (TMA,k) and (TMA,k_pipe)

#if 0
  if (thread0())
  {
    print("tAgA: "); print(tAgA); print("\n");
    print("tAsA: "); print(tAsA); print("\n");
    print("tBgB: "); print(tBgB); print("\n");
    print("tBsB: "); print(tBsB); print("\n");
  }
  
#endif

  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  auto K_PIPE_MAX = size<1>(tAsA); // static
  int k_tile_count = size<1>(tAgA); // dynamic
  int k_tile_next = 0;

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA

  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX; k_pipe++)
  {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[k_pipe],   1); // 1 thread per TMA
      ConsumerBarType::init(&consumer_mbar[k_pipe], 128); // 128 thread per wgmma
    }
  }
  cluster_sync();

  // Start async loads for all pipes
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX; k_pipe++)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[k_pipe], tma_transaction_bytes);
      // check TMA_LOAD_Unpack for execution after `with` 
      copy(tma_atom_a.with(producer_mbar[k_pipe]), tAgA(_,k_tile_next), tAsA(_,k_pipe));
      copy(tma_atom_b.with(producer_mbar[k_pipe]), tBgB(_,k_tile_next), tBsB(_,k_pipe));
    }
    --k_tile_count;
    ++k_tile_next;
  }
  
  auto mma = Policy::make_tiled_mma();
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

#if 0
  if (thread0())
  {
    print("tCrA: "); print(tCrA); print("\n");
    print("tCsA: "); print(tCsA); print("\n");
    print("tCrB: "); print(tCrB); print("\n");
    print("tCsB: "); print(tCsB); print("\n");
  }
#endif

  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads
  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX)
  {
    // Wait for Producer to complete
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // MMAs to cover 1 K_TILE
    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
    warpgroup_commit_batch();

    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>();

    // Notify that consumption is done
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    // Only issue new TMA copies if there are more tiles to fetch
    if ((warp_idx == 0) && lane_predicate && (k_tile_count > 0))
    {
      int pipe = write_state.index();
      // Wait for Consumer to complete consumption
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_atom_a.with(producer_mbar[pipe]), tAgA(_,k_tile_next), tAsA(_,pipe));
      copy(tma_atom_b.with(producer_mbar[pipe]), tBgB(_,k_tile_next), tBsB(_,pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile_next;
  }

  //
  // Epilogue (unpredicated)
  //

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
  
  Arguments args{A, B, C, M, N, K, alpha, beta};
  auto params = make_gemm_params<Policy>(args);

  dim3 dimBlock(size(Policy::make_tiled_mma()));
  // test the diff between traditional launch and cluster launch
  dim3 dimCluster(1,1,1); 
  // round up to complete each cluster
  dim3 dimGrid(round_up(size(ceil_div(M, BM{})), dimCluster.x),
               round_up(size(ceil_div(N, BN{})), dimCluster.y));
  
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
