#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// #include "ref.h"
#include "utils.h"


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


int main(int argc, char* argv[]) {
    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    int M = 5120, N = 5120, K = 4096;
    TI alpha = 1.0, beta = 0.0;
    int warmup_iters = 1, bench_iters = 5;
    // Check if an argument was provided
    if (argc > 1) {
        // Convert to string_view for safe, easy, and efficient comparison
        std::string_view arg = argv[1];

        if (arg == "-p") {
            std::cout << "[INFO] Profile mode detected (-p). Adjusting iterations.\n";
            warmup_iters = 1;
            bench_iters = 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Usage: " << argv[0] << " [-p]\n";
            return 1; // Return an error code
        }
    } else {
        std::cout << "[INFO] Running with default configuration.\n";
    }

    // NT: A is K×M, B is K×N (column-major M×K and N×K stored row-major)
    thrust::host_vector<TA> h_A(K * M, TA(1.0));
    thrust::host_vector<TB> h_B(K * N, TB(1.0));
    thrust::host_vector<TC> h_C(M * N, TC(0.0));
    thrust::host_vector<TC> h_RefC(M * N, TC(0.0));

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
        false, true,
        warmup_iters, bench_iters
    );

    for (int i = 0; i < warmup_iters; i++)
    {
        call_sgemm2_nt(
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
        call_sgemm2_nt(
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

    std::cout << "[sgemm2 SGEMM] "
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
    utils::compare_tensors(d_C.data().get(), d_RefC.data().get(), M*N);

    return 0;
}
