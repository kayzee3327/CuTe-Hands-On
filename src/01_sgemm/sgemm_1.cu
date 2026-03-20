#include "cutlass/util/reference/host/tensor_fill.h"
#include <cute/tensor.hpp>

#include "ref.h"


template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__
void sgemm1_tn(ProblemShape shape_MNK, CtaTiler cta_tiler,
               TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
               TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
               TC* C,       CStride dC, CSmemLayout          , CThreadLayout tC,
               Alpha alpha, Beta beta) 
{
    using namespace cute;
    // step1.3 Preconditions: dynamic shapes
    // use `CUTE_STATIC_ASSERT_V` for CuTe related compile-time comparing
    // rank-3
    CUTE_STATIC_ASSERT_V(rank(shape_MNK) == _3{});
    // shape MN/KM/KN congruent with strides
    CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));
    CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));
    CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));

    // step1.4 Represent the full tensors
    // M x K, N x K, M x N, transposition should be handled by strides
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K):()
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K):()
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N):()

    // if (thread0())
    // {
    //     print(mA.layout());
    //     print("\n");
    //     print(mB.layout());
    //     print("\n");
    //     print(mC.layout());
    //     print("\n");
    // }

    // step2.2 Get the appropriate blocks for this threadblock (tiling)
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, select<0,2>(cta_tiler), select<0,2>(cta_coord));
    Tensor gB = local_tile(mB, select<1,2>(cta_tiler), select<1,2>(cta_coord));
    Tensor gC = local_tile(mC, select<0,1>(cta_tiler), select<0,1>(cta_coord));

    // if (thread0())
    // {
    //     print("hello from thread0\n");
    // }

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
    // allocate smem space
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    // mark the space with Tensor (name "sA" is used twice with diff meaning)
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // if (thread0())
    // {
    //     print("hello from thread0\n");
    // }

    // step3.2 Preconditions: AB thread layouts
    // static thread layout shape
    static_assert(is_static_v<AThreadLayout>);
    static_assert(is_static_v<BThreadLayout>);
    // equal num of threads
    CUTE_STATIC_ASSERT_V(size(tA) == size(tB)); //static, size() returns a type
    // check AB thread layout shape
    // BLK_M / THR_M
    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tA) == _0{});
    // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tA) == _0{});
    // BLK_N / THR_N
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<0>(tB) == _0{});
    // BLK_K / THR_K
    CUTE_STATIC_ASSERT_V(size<2>(cta_tiler) % size<1>(tB) == _0{});

    // if (thread0())
    // {
    //     print("hello from thread0\n");
    // }
    

    // step3.3 Partiton gA gB for copy
    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    // step3.4 Preconditions: copy partitioning
    // gmem partition compatible with smem partition
    CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA));
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
    CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB));
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));

    // step4.2 Preconditions: C thread layouts
    // static thread layout shape
    static_assert(is_static_v<CThreadLayout>);
    // check C layout shape
    CUTE_STATIC_ASSERT_V(size<0>(cta_tiler) % size<0>(tC) == _0{});
    CUTE_STATIC_ASSERT_V(size<1>(cta_tiler) % size<1>(tC) == _0{});

    // step4.3 Partition sA sB gC for computing 
    Tensor tCsA = local_partition(sA, select<0>(tC), threadIdx.x); // (BLK_M, BLK_K) -> (THR_M, BLK_K)
    Tensor tCsB = local_partition(sB, select<1>(tC), threadIdx.x); // (BLK_N, BLK_K) -> (THR_N, BLK_K)
    Tensor tCgC = local_partition(gC, tC, threadIdx.x);
    // accumulator
    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    // step4.4 Preconditions: math partitioning
    // check register matmul shape
    // CUTE_STATIC_ASSERT_V(rank(tCsA) == _3{});

    CUTE_STATIC_ASSERT_V(size<0>(tCsA) == size<0>(tCrC));
    CUTE_STATIC_ASSERT_V(size<0>(tCsB) == size<1>(tCrC));
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));
    CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));
    CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));

    // if (thread0())
    // {
    //     /* code */
    //     print(tCsA.layout());
    //     print("\n");
    //     print(tCsB.layout());
    //     print("\n");
    //     print(tCgC.layout());
    //     print("\n");
    //     // print_tensor(tAsA);
    //     // print(gA.layout());
    //     // print("\n");
    //     // print(tAgA(_,_,0).layout());
    //     // print("\n");
        
    // }
    
#if 1
    // step5 main loop
    auto K_TILES = size<2>(tAgA);
    for (uint k_tile = 0; k_tile < K_TILES; k_tile++)
    {
        copy(tAgA(_, _, k_tile), tAsA);
        copy(tBgB(_, _, k_tile), tBsB);
        cp_async_fence();        // Label the end of (potential) cp.async instructions
        cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
        __syncthreads();         // Wait for all threads to write to smem
        gemm(tCsA, tCsB, tCrC);
        __syncthreads();         // Wait for all threads to read from smem
    }
    axpby(alpha, tCrC, beta, tCgC);
#endif   
    // if (thread0()) 
    // {
    //     print_tensor(tCgC);
    // }
}

template <class TA, class TB, class TC, class Alpha, class Beta>
void call_sgemm1_tn(TA *A, TB *B, TC *C, int M, int N, int K, Alpha alpha, Beta beta) 
{
    using namespace cute;
    // step1.1 Define shapes (dynamic)
    auto prob_shape = make_shape(M, N, K);

    // step1.2 Define TN strides (mixed)
    // strides here are used with shapes of gmem ref Tensor mA, mB, mC
    // You should decide what shapes you are going to work on, 
    //      and then map data to shape using strides
    auto dA = make_stride(K, _1{});
    auto dB = make_stride(K, _1{});
    auto dC = make_stride(N, _1{});

    // step2.1 Define CTA tile sizes (static)
    // cta_tiler is used by local_tile()
    auto bM = _128{};
    auto bN = _128{};
    auto bK = _8{};
    auto cta_tiler = make_shape(bM, bN, bK);

    // step2.3 Define the smem layouts (static)
    // smem layout is for physical storage of cta tile
    auto sA = make_layout(make_shape(bM, bK), LayoutRight{});
    auto sB = make_layout(make_shape(bN, bK), LayoutRight{});
    // auto sB = make_layout(make_shape(bK, bN));
    auto sC = make_layout(make_shape(bM, bN));
    
    // step3.1 Define AB thread layouts (static)
    // use (_32,_8) since bK == 8
    auto tA = make_layout(make_shape(_32{}, _8{}), LayoutRight{});
    auto tB = make_layout(make_shape(_32{}, _8{}), LayoutRight{});

    // step4.1 Define C thread layouts (static)
    auto tC = make_layout(make_shape(_16{}, _16{}));

    // call
    // the design of block should be considered at the very beginning
    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
    // print("hello from host\n");
    
    sgemm1_tn<<<dimGrid, dimBlock>>>
        (prob_shape, cta_tiler,
         A, dA, sA, tA,
         B, dB, sB, tB,
         C, dC, sC, tC, 
         alpha, beta);
    
}


int main() {
    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    int M = 5120, N = 5120, K = 4096;
    TI alpha = 1.0, beta = 0.0;

    // tn
    cutlass::HostTensor<TA, cutlass::layout::RowMajor> A({K, M});
    cutlass::HostTensor<TB, cutlass::layout::RowMajor> B({K, N});
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

    call_sgemm1_tn(A.device_data(), B.device_data(), C.device_data(), M, N, K, alpha, beta);
    ref_gemm(A, B, Reference_C, alpha, beta, M, N, K);

    // Wait for the GPU reference kernel to finish
    cudaDeviceSynchronize();

    C.sync_host();
    Reference_C.sync_host();

    tensor_cmp(C, Reference_C);
    
    return 0;
}
