#include "cutlass/util/reference/host/tensor_fill.h"
#include <cute/tensor.hpp>

#include "ref.h"


template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__
void sgemm2_nt(ProblemShape shape_MNK, CtaTiler cta_tiler,
               TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
               TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
               TC* C,       CStride dC, CSmemLayout          , TiledMma mma,
               Alpha alpha, Beta beta) 
{
    using namespace cute;
    // step1.3 Preconditions: dynamic shapes
    // rank-3
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == _3{});
    // shape MN/KM/KN congruent with strides
    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

    // step1.4 Represent the full tensors
    // M x K, N x K, M x N
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

    // step2.2 Get the appropriate blocks for this threadblock (tiling)
    auto coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(coord));
    Tensor gB = local_tile(mB, select<1,2>(cta_tiler), select<1,2>(coord));
    Tensor gC = local_tile(mC, select<0,1>(cta_tiler), select<0,1>(coord));

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
    copy(copy_a, tAgA(_,_,_,0), tArA);
    copy(copy_b, tBgB(_,_,_,0), tBrB);

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
    for (uint k_tile = 0; k_tile < K_TILE_MAX; k_tile++)
    {
        __syncthreads();
        copy(tArA, tAsA);
        copy(tBrB, tBsB);
        __syncthreads();

        int k_tile_next = k_tile == K_TILE_MAX - 1 ? k_tile : k_tile + 1;
        copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
        
        gemm(mma, tCsA, tCsB, tCrC);
    }
    
    axpby(alpha, tCrC, beta, tCgC);
#endif
}

template <class TA, class TB, class TC, 
          class Alpha, class Beta>
void call_sgemm2_nt(TA *A, TB *B, TC *C, 
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
    auto dC = make_stride(_1{}, M);

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
                                      Layout<Shape<_32,_8>>{},
                                      Layout<Shape<_4, _1>>{});
    TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                      Layout<Shape<_32,_8>>{},
                                      Layout<Shape<_4, _1>>{});                                  

    // step4.1 Define C thread layouts (static) using TiledMMA
    TiledMMA mmaC = make_tiled_mma(MMA_Atom<UniversalFMA<float>>{},
                                   Layout<Shape<_16, _16, _1>>{});
    
    // call
    dim3 dimBlock(32 * 8);
    dim3 dimGrid(ceil_div(M, bM),
                 ceil_div(N, bN));
    sgemm2_nt<<<dimGrid, dimBlock>>>
        (prob_shape, cta_tiler, 
         A, dA, sA, copyA, 
         B, dB, sB, copyB, 
         C, dC, sC, mmaC,
         alpha, beta);
}


int main() {
    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    int M = 5120, N = 5120, K = 4096;
    TI alpha = 1.0, beta = 0.0;

    // nt
    cutlass::HostTensor<TA, cutlass::layout::RowMajor> A({M, K});
    cutlass::HostTensor<TB, cutlass::layout::RowMajor> B({N, K});
    cutlass::HostTensor<TC, cutlass::layout::RowMajor> C({M, N});
    cutlass::HostTensor<TC, cutlass::layout::RowMajor> Reference_C({M, N});

    cutlass::reference::host::TensorFill(A.host_view(), TA(1.0));
    cutlass::reference::host::TensorFill(B.host_view(), TB(1.0));
    cutlass::reference::host::TensorFill(C.host_view(), TC(0.0));
    cutlass::reference::host::TensorFill(Reference_C.host_view(), TC(0));
    
    // Push the initialized host data to the GPU
    A.sync_device();
    B.sync_device();
    C.sync_device();
    Reference_C.sync_device();

    call_sgemm2_nt(A.device_data(), B.device_data(), C.device_data(), M, N, K, alpha, beta);
    ref_gemm(A, B, Reference_C, alpha, beta, M, N, K);

    // Wait for the GPU reference kernel to finish
    cudaDeviceSynchronize();

    C.sync_host();
    Reference_C.sync_host();

    tensor_cmp(C, Reference_C);
    
    return 0;
}
