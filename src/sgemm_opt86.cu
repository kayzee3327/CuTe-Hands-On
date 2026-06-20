#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

// #include "ref.h"
#include "utils.h"

#define OPT_CALLER call_sgemm_opt86_nt_v4

// In this v1 impl, we expand the implicit optimization of
//  `gemm(mma, tCsA, tCsB, tCrC);`
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ void sgemm_opt86_nt_v1(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                  TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                                  TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                                  TC *C, CStride dC, CSmemLayout, TiledMma mma,
                                  Alpha alpha, Beta beta)
{
  using namespace cute;
  // step1.3 Preconditions: dynamic shapes
  // rank-3
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == _3{});
  // shape MN/KM/KN congruent with strides
  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));

  // step1.4 Represent the full tensors
  // M x K, N x K, M x N
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  // step2.2 Get the appropriate blocks for this threadblock (tiling)
  auto coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, select<0, 2>(cta_tiler), select<0, 2>(coord));
  Tensor gB = local_tile(mB, select<1, 2>(cta_tiler), select<1, 2>(coord));
  Tensor gC = local_tile(mC, select<0, 1>(cta_tiler), select<0, 1>(coord));

  // step2.4 Preconditions: smem layouts
  // static smem layout shape
  static_assert(is_static_v<ASmemLayout>);
  static_assert(is_static_v<BSmemLayout>);
  static_assert(is_static_v<CSmemLayout>);
  // smem shape corresponds to cta_tiler shape
  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));

  // step2.5 Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  // step3.2 partitioning AB (g/s/r) via a TiledCopy
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);
  // Allocate registers same shape/layout as partitioned data
  Tensor tArA = make_fragment_like(tAsA);
  Tensor tBrB = make_fragment_like(tBsB);

  // if (thread0())
  // {
  //     print(tAgA.layout());
  //     print("\n");
  //     print(tAsA.layout());
  //     print("\n");
  //     print(size<0>(tAgA));
  //     print("\n");
  // }

  // step3.3 Conditions: AB thread layouts (g/s/r) partitioning shape
  // require consistency with g, no need to check CPY mode
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));

  // if (thread0())
  // {
  //     print(tAgA(_,_,_,0).layout()); print("\n");
  //     print(tArA.layout()); print("\n");
  // }

#if 1
  // step3.4 Copy gmem to rmem for k_tile=0
  // pre-heatup the rmem
  copy(copy_a, tAgA(_, _, _, 0), tArA);
  copy(copy_b, tBgB(_, _, _, 0), tBrB);

  // Step4.2 Define A/B partitioning and C accumulators
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = make_fragment_like(tCgC);

  // if (thread0())
  // {
  //     print(tCsA.layout());
  //     print("\n");
  //     print(tCsB.layout());
  //     print("\n");
  //     print(tCgC.layout());
  //     print("\n");
  // }

  // Step4.3 Conditions: accum shape consistent and MMA_M, MMA_N, MMA_K
  CUTE_STATIC_ASSERT_V(shape(tCgC) == shape(tCrC));
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));

  // main loop
  auto K_TILE_MAX = size<3>(tAgA); // same as size<3>(gA), but we work on tAgA
  // uint K_TILE_MAX = 1;

  // ---- expand gemm(mma, tCsA, tCsB, tCrC); ----
  auto K_BLOCK_MAX = size<2>(tCsA);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(tCrC));
  // CUTE_STATIC_ASSERT_V(size(tCrC) == size(tCrC)) omitted

  // ---------------------------------------------

  CUTE_UNROLL
  for (uint k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
  {
    __syncthreads();
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    int k_tile_next = k_tile == K_TILE_MAX - 1 ? k_tile : k_tile + 1;
    copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);

    // ---- expand gemm(mma, tCsA, tCsB, tCrC); ----
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++)
    {
      copy(tCsA(_, _, k_block), tCrA(_, _, k_block));
      copy(tCsB(_, _, k_block), tCrB(_, _, k_block));

      auto M = size<1>(tCrA);
      auto N = size<1>(tCrB);

      // REGISTER .reuse OPTIMIZATIONS
      // 32-bit traversal specialization -- kinked serpentine path
      // Here I use the implementation for 4-byte data type in include/cute/algorithm/gemm.hpp for simplicity
      // if constexpr (decltype(size<0>(tCrA))::value * sizeof(typename TA::value_type) == 4 &&
      //               decltype(size<0>(tCrB))::value * sizeof(typename TB::value_type) == 4)
      // {
      // NOTE: Row- vs Col- major could depend on the C-matrix order... (which we can test)
      // Row-major kinked serpentine iteration
      for (int m = 0; m < M; m += 2)
      {
        for (int n = 0; n < N; ++n)
        {
          int ns = (m & 2) ? N-1-n : n;
          gemm(mma, tCrA(_,m+0,k_block), tCrB(_,ns,k_block), tCrC(_,m+0,ns));
          if (m+1 < M) 
          {
            gemm(mma, tCrA(_,m+1,k_block), tCrB(_,ns,k_block), tCrC(_,m+1,ns));
          }
        }
      
      }
    }
    // ---------------------------------------------
  }

  axpby(alpha, tCrC, beta, tCgC);
#endif
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void call_sgemm_opt86_nt_v1(TA *A, TB *B, TC *C,
                            int M, int N, int K,
                            Alpha alpha, Beta beta)
{
  using namespace cute;
  // step1.1 Define shapes (dynamic)
  auto prob_shape = make_shape(M, N, K);

  // step1.2 Define NT strides (mixed)
  // make sure vectorized copy dim is contiguous
  auto dA = make_stride(_1{}, M);
  auto dB = make_stride(_1{}, N);
  auto dC = make_stride(N, _1{});

  // step2.1 Define CTA tile sizes (static)
  auto bM = _128{};
  auto bN = _128{};
  auto bK = _8{};
  auto cta_tiler = make_shape(bM, bN, bK);

  // step2.3 Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));

  // step3.1 Define AB thread layouts (static) using TiledCopy
  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});

  // step4.1 Define C thread layouts (static) using TiledMMA
  TiledMMA mmaC = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{});

  // call
  dim3 dimBlock(32 * 8);
  dim3 dimGrid(ceil_div(M, bM),
               ceil_div(N, bN));
  sgemm_opt86_nt_v1<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler,
                                           A, dA, sA, copyA,
                                           B, dB, sB, copyB,
                                           C, dC, sC, mmaC,
                                           alpha, beta);
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ void sgemm_opt86_nt_v2(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                  TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                                  TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                                  TC *C, CStride dC, CSmemLayout, TiledMma mma,
                                  Alpha alpha, Beta beta)
{
  using namespace cute;
  
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == _3{});
  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  auto coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, select<0, 2>(cta_tiler), select<0, 2>(coord));
  Tensor gB = local_tile(mB, select<1, 2>(cta_tiler), select<1, 2>(coord));
  Tensor gC = local_tile(mC, select<0, 1>(cta_tiler), select<0, 1>(coord));

  static_assert(is_static_v<ASmemLayout>);
  static_assert(is_static_v<BSmemLayout>);
  static_assert(is_static_v<CSmemLayout>);
  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);
  Tensor tArA = make_fragment_like(tAsA);
  Tensor tBrB = make_fragment_like(tBsB);

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));

  copy(copy_a, tAgA(_, _, _, 0), tArA);
  copy(copy_b, tBgB(_, _, _, 0), tBrB);

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = make_fragment_like(tCgC);

  CUTE_STATIC_ASSERT_V(shape(tCgC) == shape(tCrC));
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));

  auto K_TILE_MAX = size<3>(tAgA); // same as size<3>(gA), but we work on tAgA

  auto K_BLOCK_MAX = size<2>(tCsA);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(tCrC));

  // SMEM->RMEM TiledCopy: 128-bit (LDS.128) loads, retiled to match MMA's A/B partitioning.
  // Requires the MMA's PermutationMNK to make each thread's per-M / per-N elements contiguous
  // (see call_sgemm_opt86_nt_v2). retile_D rebinds the same registers under the s2r thread/value layout.
  auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<UniversalCopy<uint128_t>, TA>{}, mma);
  auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<UniversalCopy<uint128_t>, TB>{}, mma);
  auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

  CUTE_UNROLL
  for (uint k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
  {
    __syncthreads();
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    int k_tile_next = k_tile == K_TILE_MAX - 1 ? k_tile : k_tile + 1;
    copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);

    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++)
    {
      copy(s2r_tiled_copy_a, tXsA(_, _, k_block), tXrA(_, _, k_block));
      copy(s2r_tiled_copy_b, tXsB(_, _, k_block), tXrB(_, _, k_block));

      auto M = size<1>(tCrA);
      auto N = size<1>(tCrB);

      for (int m = 0; m < M; m += 2)
      {
        for (int n = 0; n < N; ++n)
        {
          int ns = (m & 2) ? N-1-n : n;
          gemm(mma, tCrA(_,m+0,k_block), tCrB(_,ns,k_block), tCrC(_,m+0,ns));
          if (m+1 < M) 
          {
            gemm(mma, tCrA(_,m+1,k_block), tCrB(_,ns,k_block), tCrC(_,m+1,ns));
          }
        }
      }
    }
  }

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void call_sgemm_opt86_nt_v2(TA *A, TB *B, TC *C,
                            int M, int N, int K,
                            Alpha alpha, Beta beta)
{
  using namespace cute;

  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(_1{}, M);
  auto dB = make_stride(_1{}, N);
  auto dC = make_stride(N, _1{});

  auto bM = _128{};
  auto bN = _128{};
  auto bK = _8{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});

  // PermutationMNK reshuffles M and N so each thread's 8 per-M (per-N) values
  // are contiguous in SMEM, enabling LDS.128 in the s2r TiledCopy.
  // M dim (size 128): new coord (i=thread_m in [0,16), j=m_rep in [0,8))
  //   -> old M = i*8 + j   (each thread owns 8 contiguous M floats)
  // K is untouched.
  TiledMMA mmaC = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, _8>, Stride<_8, _1>>,
                                      Layout<Shape<_16, _8>, Stride<_8, _1>>,
                                      _1>{});

  dim3 dimBlock(32 * 8);
  dim3 dimGrid(ceil_div(M, bM),
               ceil_div(N, bN));
  sgemm_opt86_nt_v2<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler,
                                           A, dA, sA, copyA,
                                           B, dB, sB, copyB,
                                           C, dC, sC, mmaC,
                                           alpha, beta);
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ void sgemm_opt86_nt_v3(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                  TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                                  TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                                  TC *C, CStride dC, CSmemLayout, TiledMma mma,
                                  Alpha alpha, Beta beta)
{
  using namespace cute;
  
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == _3{});
  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  auto coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, select<0, 2>(cta_tiler), select<0, 2>(coord));
  Tensor gB = local_tile(mB, select<1, 2>(cta_tiler), select<1, 2>(coord));
  Tensor gC = local_tile(mC, select<0, 1>(cta_tiler), select<0, 1>(coord));

  static_assert(is_static_v<ASmemLayout>);
  static_assert(is_static_v<BSmemLayout>);
  static_assert(is_static_v<CSmemLayout>);
  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);
  Tensor tArA = make_fragment_like(tAsA);
  Tensor tBrB = make_fragment_like(tBsB);

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));

  copy(copy_a, tAgA(_, _, _, 0), tArA);
  copy(copy_b, tBgB(_, _, _, 0), tBrB);

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = make_fragment_like(tCgC);

  CUTE_STATIC_ASSERT_V(shape(tCgC) == shape(tCrC));
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));

  auto K_TILE_MAX = size<3>(tAgA); // same as size<3>(gA), but we work on tAgA

  auto K_BLOCK_MAX = size<2>(tCsA);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(tCrC));

  auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<UniversalCopy<uint128_t>, TA>{}, mma);
  auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<UniversalCopy<uint128_t>, TB>{}, mma);
  auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

  CUTE_UNROLL
  for (uint k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
  {
    __syncthreads();
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    int k_tile_next = k_tile == K_TILE_MAX - 1 ? k_tile : k_tile + 1;
    copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);

    // apply register multi-stage/prefetch to increase ILP
    // see if we can mitigate the wait&dispatch problem
    copy(s2r_tiled_copy_a, tXsA(_, _, _0{}), tXrA(_, _, _0{}));
    copy(s2r_tiled_copy_b, tXsB(_, _, _0{}), tXrB(_, _, _0{}));
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++)
    {
      auto k_block_next = (k_block + _1{}) % K_BLOCK_MAX;
      copy(s2r_tiled_copy_a, tXsA(_, _, k_block_next), tXrA(_, _, k_block_next));
      copy(s2r_tiled_copy_b, tXsB(_, _, k_block_next), tXrB(_, _, k_block_next));

      auto M = size<1>(tCrA);
      auto N = size<1>(tCrB);

      for (int m = 0; m < M; m += 2)
      {
        for (int n = 0; n < N; ++n)
        {
          int ns = (m & 2) ? N-1-n : n;
          gemm(mma, tCrA(_,m+0,k_block), tCrB(_,ns,k_block), tCrC(_,m+0,ns));
          if (m+1 < M) 
          {
            gemm(mma, tCrA(_,m+1,k_block), tCrB(_,ns,k_block), tCrC(_,m+1,ns));
          }
        }
      }
    }
  }

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void call_sgemm_opt86_nt_v3(TA *A, TB *B, TC *C,
                            int M, int N, int K,
                            Alpha alpha, Beta beta)
{
  using namespace cute;

  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(_1{}, M);
  auto dB = make_stride(_1{}, N);
  auto dC = make_stride(N, _1{});

  auto bM = _128{};
  auto bN = _128{};
  auto bK = _8{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});

  // Control the whole thread tile:
  // TiledMMA mmaC = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
  //                                Layout<Shape<_16, _16, _1>>{},
  //                                Tile<Layout<Shape<_16, Shape<_4, _2>>, Stride<_4, Stride<_1, _64>>>,
  //                                     Layout<Shape<_16, Shape<_4, _2>>, Stride<_4, Stride<_1, _64>>>,
  //                                     _1>{});
  // Same effects but less compile time:
  TiledMMA mmaC = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, _4>, Stride<_4, _1>>,
                                      Layout<Shape<_16, _4>, Stride<_4, _1>>,
                                      _1>{});

  dim3 dimBlock(32 * 8);
  dim3 dimGrid(ceil_div(M, bM),
               ceil_div(N, bN));
  sgemm_opt86_nt_v3<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler,
                                           A, dA, sA, copyA,
                                           B, dB, sB, copyB,
                                           C, dC, sC, mmaC,
                                           alpha, beta);
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ void sgemm_opt86_nt_v3a(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                  TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                                  TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                                  TC *C, CStride dC, CSmemLayout, TiledMma mma,
                                  Alpha alpha, Beta beta)
{
  using namespace cute;
  
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == _3{});
  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  auto coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, select<0, 2>(cta_tiler), select<0, 2>(coord));
  Tensor gB = local_tile(mB, select<1, 2>(cta_tiler), select<1, 2>(coord));
  Tensor gC = local_tile(mC, select<0, 1>(cta_tiler), select<0, 1>(coord));

  static_assert(is_static_v<ASmemLayout>);
  static_assert(is_static_v<BSmemLayout>);
  static_assert(is_static_v<CSmemLayout>);
  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));

  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);
  Tensor tArA = make_fragment_like(tAsA);
  Tensor tBrB = make_fragment_like(tBsB);

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));

  copy(copy_a, tAgA(_, _, _, 0), tArA);
  copy(copy_b, tBgB(_, _, _, 0), tBrB);

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = make_fragment_like(tCgC);

  CUTE_STATIC_ASSERT_V(shape(tCgC) == shape(tCrC));
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));

  auto K_TILE_MAX = size<3>(tAgA); // same as size<3>(gA), but we work on tAgA

  auto K_BLOCK_MAX = size<2>(tCsA);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(tCrC));

  auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<UniversalCopy<uint128_t>, TA>{}, mma);
  auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<UniversalCopy<uint128_t>, TB>{}, mma);
  auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

  CUTE_UNROLL
  for (uint k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
  {
    __syncthreads();
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    int k_tile_next = k_tile == K_TILE_MAX - 1 ? k_tile : k_tile + 1;
    copy(copy_a, tAgA(_, _, _, k_tile_next), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBrB);

    // remove register prefetch in v3
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++)
    {
      
      copy(s2r_tiled_copy_a, tXsA(_, _, k_block), tXrA(_, _, k_block));
      copy(s2r_tiled_copy_b, tXsB(_, _, k_block), tXrB(_, _, k_block));

      auto M = size<1>(tCrA);
      auto N = size<1>(tCrB);

      for (int m = 0; m < M; m += 2)
      {
        for (int n = 0; n < N; ++n)
        {
          int ns = (m & 2) ? N-1-n : n;
          gemm(mma, tCrA(_,m+0,k_block), tCrB(_,ns,k_block), tCrC(_,m+0,ns));
          if (m+1 < M) 
          {
            gemm(mma, tCrA(_,m+1,k_block), tCrB(_,ns,k_block), tCrC(_,m+1,ns));
          }
        }
      }
    }
  }

  axpby(alpha, tCrC, beta, tCgC);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void call_sgemm_opt86_nt_v3a(TA *A, TB *B, TC *C,
                            int M, int N, int K,
                            Alpha alpha, Beta beta)
{
  using namespace cute;

  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(_1{}, M);
  auto dB = make_stride(_1{}, N);
  auto dC = make_stride(N, _1{});

  auto bM = _128{};
  auto bN = _128{};
  auto bK = _8{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = make_layout(make_shape(bM, bK));
  auto sB = make_layout(make_shape(bN, bK));
  auto sC = make_layout(make_shape(bM, bN));

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});

  TiledMMA mmaC = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, _4>, Stride<_4, _1>>,
                                      Layout<Shape<_16, _4>, Stride<_4, _1>>,
                                      _1>{});

  dim3 dimBlock(32 * 8);
  dim3 dimGrid(ceil_div(M, bM),
               ceil_div(N, bN));
  sgemm_opt86_nt_v3a<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler,
                                           A, dA, sA, copyA,
                                           B, dB, sB, copyB,
                                           C, dC, sC, mmaC,
                                           alpha, beta);
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ void sgemm_opt86_nt_v4(ProblemShape shape_MNK, CtaTiler cta_tiler,
                                  TA const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                                  TB const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                                  TC *C, CStride dC, CSmemLayout, TiledMma mma,
                                  Alpha alpha, Beta beta)
{
  using namespace cute;
  
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == _3{});
  CUTE_STATIC_ASSERT_V(congruent(select<0, 2>(shape_MNK), dA));
  CUTE_STATIC_ASSERT_V(congruent(select<1, 2>(shape_MNK), dB));
  CUTE_STATIC_ASSERT_V(congruent(select<0, 1>(shape_MNK), dC));

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  auto coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, select<0, 2>(cta_tiler), select<0, 2>(coord));
  Tensor gB = local_tile(mB, select<1, 2>(cta_tiler), select<1, 2>(coord));
  Tensor gC = local_tile(mC, select<0, 1>(cta_tiler), select<0, 1>(coord));

  static_assert(is_static_v<ASmemLayout>);
  static_assert(is_static_v<BSmemLayout>);
  static_assert(is_static_v<CSmemLayout>);
  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));

  // consider using dynamic allocation for multi-stage due to 48KB limit per CTA
  // but in our case 1 stage only uses 8192 bytes
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);         // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);         // (CPY,CPY_M,CPY_K,PIPE)
  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);         // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);         // (CPY,CPY_N,CPY_K,PIPE)
  // Single-stage register buffer that relays each k-tile from GMEM into SMEM.
  // (CPY,CPY_M,CPY_K)
  Tensor tArA = make_tensor<float>(shape(tAsA(_,_,_,0)));
  Tensor tBrB = make_tensor<float>(shape(tBsB(_,_,_,0)));

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA(_,_,0));
  Tensor tCsB = thr_mma.partition_B(sB(_,_,0));
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = make_fragment_like(tCgC);

  CUTE_STATIC_ASSERT_V(shape(tCgC) == shape(tCrC));
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));
  
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(tCrC));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(tCrC));

  auto s2r_tiled_copy_a = make_tiled_copy_A(Copy_Atom<UniversalCopy<uint128_t>, TA>{}, mma);
  auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(Copy_Atom<UniversalCopy<uint128_t>, TB>{}, mma);
  auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

#if 0
  if(thread0())
  {
    print("tArA         : "); print(tArA.layout());          print("\n");
    print("tAsA         : "); print(tAsA.layout());          print("\n");
    print("tXsA(_,_,_,0): "); print(tXsA(_,_,_,0).layout()); print("\n");
    print("tXrA         : "); print(tXrA.layout());          print("\n");
    print("tCrA         : "); print(tCrA.layout());          print("\n");
    print("tCrB         : "); print(tCrB.layout());          print("\n");
    print("tCrC         : "); print(tCrC.layout());          print("\n");

  }
#endif

  // LDG/STS prologue
  //
  // With a single register buffer, each k-tile is relayed GMEM -> reg -> SMEM.
  // Fill (K_PIPE_SMEM_MAX - 1) SMEM stages, then stage one more k-tile in the
  // register buffer so the mainloop's first STS has data ready.
  int k_tile_count = size<3>(tAgA); // same as size<3>(gA), but we work on tAgA, not a constant in pipeline design
  int k_tile_next = 0;
  auto K_PIPE_SMEM_MAX = size<3>(tAsA);
  int smem_pipe_write = 0;
  for (int k_pipe = 0; k_pipe < K_PIPE_SMEM_MAX - 1; k_pipe++)
  {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
    k_tile_count --;
    if (k_tile_count > 0) k_tile_next ++;
    copy(tArA, tAsA(_,_,_,smem_pipe_write));
    copy(tBrB, tBsB(_,_,_,smem_pipe_write));
    smem_pipe_write = (smem_pipe_write + 1) % K_PIPE_SMEM_MAX;
  }
  // Stage the next k-tile in registers (drained to SMEM by the mainloop).
  copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
  copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
  k_tile_count --;
  if (k_tile_count > 0) k_tile_next ++;


  // S2R prologue
  //
  int smem_pipe_read = 0;
  auto K_BLOCK_MAX = size<2>(tCsA);
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);
  if (K_BLOCK_MAX > 1)
  {
    __syncthreads();
    copy(s2r_tiled_copy_a, tXsA_p(_,_,_0{}), tXrA(_,_,_0{}));
    copy(s2r_tiled_copy_b, tXsB_p(_,_,_0{}), tXrB(_,_,_0{}));
  }
  
  CUTE_NO_UNROLL
  while (k_tile_count + K_PIPE_SMEM_MAX > 0)
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_SMEM_MAX;
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);
        __syncthreads();
      }
      
      auto k_block_next = (k_block + _1{}) % K_BLOCK_MAX;
      copy(s2r_tiled_copy_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_tiled_copy_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));

      if (k_block == 0)
      {
        // Drain the staged k-tile from the register buffer into the next SMEM
        // stage, then load the following k-tile into the same register buffer.
        // STS reads tArA before LDG overwrites it, so the GMEM load latency is
        // hidden across the mainloop iteration.
        copy(tArA, tAsA(_,_,_,smem_pipe_write));
        copy(tBrB, tBsB(_,_,_,smem_pipe_write));
        smem_pipe_write = (smem_pipe_write + 1) % K_PIPE_SMEM_MAX;
        copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
        k_tile_count--;
        if (k_tile_count > 0) k_tile_next++;
      }
      
      auto M = size<1>(tCrA);
      auto N = size<1>(tCrB);

      for (int m = 0; m < M; m += 2)
      {
        for (int n = 0; n < N; ++n)
        {
          int ns = (m & 2) ? N-1-n : n;
          gemm(mma, tCrA(_,m+0,k_block), tCrB(_,ns,k_block), tCrC(_,m+0,ns));
          if (m+1 < M) 
          {
            gemm(mma, tCrA(_,m+1,k_block), tCrB(_,ns,k_block), tCrC(_,m+1,ns));
          }
        }
      }

    }
  }

  axpby(alpha, tCrC, beta, tCgC);

}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void call_sgemm_opt86_nt_v4(TA *A, TB *B, TC *C,
                            int M, int N, int K,
                            Alpha alpha, Beta beta)
{
  using namespace cute;

  auto prob_shape = make_shape(M, N, K);

  auto dA = make_stride(_1{}, M);
  auto dB = make_stride(_1{}, N);
  auto dC = make_stride(N, _1{});

  auto bM = _128{};
  auto bN = _128{};
  auto bK = _8{};
  auto cta_tiler = make_shape(bM, bN, bK);
  auto bP = _2{};

  auto sA_atom = make_layout(make_shape(bM, bK));
  auto sB_atom = make_layout(make_shape(bN, bK));
  auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP)); // similar to Tile in TiledMMA
  auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP)); // similar to Tile in TiledMMA
  auto sC = make_layout(make_shape(bM, bN));

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                    Layout<Shape<_32, _8>>{},
                                    Layout<Shape<_4, _1>>{});

  TiledMMA mmaC = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                 Layout<Shape<_16, _16, _1>>{},
                                 Tile<Layout<Shape<_16, _4>, Stride<_4, _1>>,
                                      Layout<Shape<_16, _4>, Stride<_4, _1>>,
                                      _1>{});

  dim3 dimBlock(32 * 8);
  dim3 dimGrid(ceil_div(M, bM),
               ceil_div(N, bN));
  sgemm_opt86_nt_v4<<<dimGrid, dimBlock>>>(prob_shape, cta_tiler,
                                           A, dA, sA, copyA,
                                           B, dB, sB, copyB,
                                           C, dC, sC, mmaC,
                                           alpha, beta);
}


int main(int argc, char *argv[])
{
  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  int M = 5120, N = 5120, K = 4096;
  TI alpha = 1.0, beta = 0.0;
  int warmup_iters = 1, bench_iters = 5;
  // Check if an argument was provided
  if (argc > 1)
  {
    // Convert to string_view for safe, easy, and efficient comparison
    std::string_view arg = argv[1];

    if (arg == "-p")
    {
      std::cout << "[INFO] Profile mode detected (-p). Adjusting iterations.\n";
      warmup_iters = 1;
      bench_iters = 0;
    }
    else
    {
      std::cerr << "Unknown argument: " << arg << "\n";
      std::cerr << "Usage: " << argv[0] << " [-p]\n";
      return 1; // Return an error code
    }
  }
  else
  {
    std::cout << "[INFO] Running with default configuration.\n";
  }

  // NT: A is K×M, B is K×N (column-major M×K and N×K stored row-major)
  // Use non-uniform random inputs (fixed seed for reproducibility). All-ones
  // inputs cannot catch k-tile ordering/duplication bugs in the pipeline, since
  // every k-tile contributes the same value regardless of order.
  thrust::host_vector<TA> h_A(K * M);
  thrust::host_vector<TB> h_B(K * N);
  thrust::host_vector<TC> h_C(M * N, TC(0.0));
  thrust::host_vector<TC> h_RefC(M * N, TC(0.0));
  {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &a : h_A) a = TA(dist(gen));
    for (auto &b : h_B) b = TB(dist(gen));
  }

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;
  thrust::device_vector<TC> d_RefC = h_RefC;

  // NT: A is K×M, B is K×N → C = A^T * B
  // ref_gemm(
  //     thrust::raw_pointer_cast(d_A.data()),
  //     thrust::raw_pointer_cast(d_B.data()),
  //     thrust::raw_pointer_cast(d_RefC.data()),
  //     alpha, beta, M, N, K,
  //     CUBLAS_OP_T, CUBLAS_OP_N);
  utils::cublas_sgemm_reference(
      M, N, K,
      d_A.data().get(),
      d_B.data().get(),
      d_RefC.data().get(),
      1.0, 0.0,
      true, false,
      warmup_iters, bench_iters);

  for (int i = 0; i < warmup_iters; i++)
  {
    OPT_CALLER(
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()),
        M, N, K, alpha, beta);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < bench_iters; i++)
  {
    OPT_CALLER(
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()),
        M, N, K, alpha, beta);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / bench_iters;

  double ops_per_gemm = 2.0 * static_cast<double>(M) * N * K;
  double tflops = (ops_per_gemm / (avg_ms / 1000.0)) / 1e12;

  std::cout << "[opt_86 SGEMM] "
            << "M=" << M << ", N=" << N << ", K=" << K
            << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
            << " | Performance: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaDeviceSynchronize();

  h_C = d_C;
  h_RefC = d_RefC;

  // tensor_cmp(
  //     thrust::raw_pointer_cast(h_C.data()),
  //     thrust::raw_pointer_cast(h_RefC.data()),
  //     M, N);
  utils::compare_tensors(d_C.data().get(), d_RefC.data().get(), M * N);

  return 0;
}
