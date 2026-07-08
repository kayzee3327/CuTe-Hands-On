#include <cute/tensor.hpp>
// #include <cuda_fp8.h>
// #include <cuda_bf16.h>
// cutlass provide all types in include/cute/numeric/numeric_types.hpp

#include "cutlass/cluster_launch.hpp"

#include "cutlass/device_kernel.h"

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

template <class Policy, class TmaAtomA, class TmaAtomB>
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
};

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

  return GemmParams<Policy, decltype(tma_atom_a), decltype(tma_atom_b)>{
    args.A, args.B, args.C,
    args.M, args.N, args.K,
    args.alpha, args.beta,
    tma_atom_a, tma_atom_b
  };
}


// Annotating a __global__ function parameter with __grid_constant__ 
//  prevents the compiler from creating a per-thread copy of the parameter. 
// Instead, all threads in the grid will access the parameter 
//  through a single address, which can improve performance.
// For TMA desc, NVIDIA explicitly recommends passing the tensor map 
//  as a const __grid_constant__ kernel parameter 
//  rather than placing it behind a global-memory pointer.
template <class Policy, class Params>
__global__
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

  auto tma_atom_a = params.tma_atom_a;
  auto tma_atom_b = params.tma_atom_b;
  Tensor mA = tma_atom_a.get_tma_tensor(make_shape(M,K)); // (M,K) TMA Tensor
  Tensor mB = tma_atom_b.get_tma_tensor(make_shape(N,K)); // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(params.C), 
                          make_shape(M, N), 
                          make_stride(_1{}, M));          // (M,N)
  
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  extern __shared__ char raw_smem[];
  using SharedStorageT = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorageT& smem = *reinterpret_cast<SharedStorageT*>(raw_smem);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

#if 1
  if (thread0())
  {
    print("mA: "); print(mA); print("\n");
    print("mB: "); print(mB); print("\n");
    print("sA: "); print(sA); print("\n");
    print("sB: "); print(sB); print("\n");
  }
  
#endif

}

template<class Policy = SM90_TNN_E4M3_F32_BF16>
cudaError_t launch_gemm(typename Policy::ElementA const* A, 
                        typename Policy::ElementB const* B, 
                        typename Policy::ElementC      * C,
                        int M, int N, int K,
                        typename Policy::ElementCompute alpha,
                        typename Policy::ElementCompute beta)
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
  cutlass::Status status = cutlass::launch_kernel_on_cluster(cparams, (const void*) kernel_ptr, params);
  CUTE_CHECK_LAST();
  
  if (status != cutlass::Status::kSuccess)
  {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
    return cudaErrorLaunchFailure;
  }

  return cudaSuccess;
}

}; // namespace


int main(int argc, char *argv[])
{
  using Policy = SM90_TNN_E4M3_F32_BF16;
  Policy::ElementA *dummyA = nullptr;
  Policy::ElementB *dummyB = nullptr;
  Policy::ElementC *dummyC = nullptr;
  launch_gemm<Policy>(dummyA, dummyB, dummyC, 1, 1, 1, 1.0f, 0.0f);
  return 0;
}

