#include <cute/tensor.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// #include "ref.h"
#include "utils.h"

template<class TA,
         class TB,
         class ASmemLayout,
         class BSmemLayout>
struct SharedStorage
{
    cute::ArrayEngine<TA, cute::cosize_v<ASmemLayout>> A;
    cute::ArrayEngine<TB, cute::cosize_v<BSmemLayout>> B;
};


template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMMA,
          class Alpha, class Beta>
__global__
void sgemm_sm80_tn(ProblemShape shape_MNK, CtaTiler cta_tiler,
                   TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
                   TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
                   TC* C,       CStride dC, CSmemLayout          , TiledMMA mma,
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
    Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(select<0,2>(shape_MNK), dA)); // make_layout() can be omitted
    Tensor mB = make_tensor(make_gmem_ptr(B), make_layout(select<1,2>(shape_MNK), dB));
    Tensor mC = make_tensor(make_gmem_ptr(C), make_layout(select<0,1>(shape_MNK), dC));

    // step2.2 Get the appropriate blocks for this threadblock (tiling)
    auto coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, coord, Step<_1, _1, X>{});

    // step2.4 Preconditions: smem layouts
    // static smem layout shape
    static_assert(is_static_v<ASmemLayout>);
    static_assert(is_static_v<BSmemLayout>);
    static_assert(is_static_v<CSmemLayout>);
    // smem shape corresponds to cta_tiler shape
    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));

    // step2.5 Shared memory buffers
    // dynamic allocated smem
    extern __shared__ char raw_smem[];
    using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(raw_smem);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

    // step3.2 partitioning AB (g/s) via a TiledCopy
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);   // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA);   // (CPY,CPY_M,CPY_K,PIPE)
    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

#if 0
    if (thread0())
    {
        print(tAgA.layout()); print("\n");
        print(tAsA.layout()); print("\n");
        print(tBgB.layout()); print("\n");
        print(tBsB.layout()); print("\n");
    }
#endif

    // step3.3 Conditions: AB thread layouts (g/s) partitioning shape
    // check CPY_K, CPY_M, CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tBgB));
    CUTE_STATIC_ASSERT_V(size<2>(tAsA) == size<2>(tBsB));
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));

    // step3.4 Pipeline: prefetch
    // pipeline stage count
    auto K_PIPE_MAX = size<3>(tAsA);
    // Total count of tiles
    int k_tile_count = size<3>(tAgA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;

    // Start async loads for all pipes but the last
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; k_pipe++)
    {
        copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe));
        copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe));
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0)
        {
            k_tile_next++;
        }
    }

    // Step4.2 Define A/B partitioning and C accumulators
    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    // global parition
    Tensor tCgC = thr_mma.partition_C(gC);
    // Allocate registers for pipelining
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0)); // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0)); // (MMA,MMA_N,MMA_K)
    // Allocate the accumulators -- same size as the projected data
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);              // (MMA,MMA_M,MMA_N)

#if 0
    if (thread0())
    {
        print("gC:   "); print(  gC); print("\n");
        print("tCgC: "); print(tCgC); print("\n");
        print("sA(_, _, 0):"); print(sA(_, _, 0)); print("\n");
        print("sB(_, _, 0):"); print(sB(_, _, 0)); print("\n");
        print("tCrA: "); print(tCrA); print("\n");
        print("tCrB: "); print(tCrB); print("\n");
        print("tCrC: "); print(tCrC); print("\n");
    }
#endif
    

    // Step4.3 Conditions: accum shape consistent and MMA_M, MMA_N, MMA_K
    CUTE_STATIC_ASSERT_V(shape(tCrC) == shape(tCgC));
    CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCrA));
    CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCrB));
    // Clear the accumulators
    clear(tCrC);

    // Step4.4 Copy Atom retiling
    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

#if 0
    if(thread0()) {
        print("  mC : "); print(  mC); print("\n");
        print("  gC : "); print(  gC); print("\n");
        print("tCgC : "); print(tCgC); print("\n");
        print("sA : "); print(sA); print("\n");
        print("sB : "); print(sB); print("\n");
        print("tCrA : "); print(tCrA); print("\n");
        print("tCrB : "); print(tCrB); print("\n");
        print("tCrC : "); print(tCrC); print("\n");

        print("tXsA : "); print(tXsA); print("\n");
        print("tXrA : "); print(tXrA); print("\n");
        print("tXsB : "); print(tXsB); print("\n");
        print("tXrB : "); print(tXrB); print("\n");
    }
#endif


    // Step5.1 Pipeline: start
    // Current pipe index in smem to read from
    int smem_pipe_read = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX - 1;
    // Pipe slice
    Tensor tXsA_p = tXsA(_,_,_, smem_pipe_read);
    Tensor tXsB_p = tXsB(_,_,_, smem_pipe_read);
    // Size of the register pipeline
    auto K_BLOCK_MAX = size<2>(tCrA);
    CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

    // PREFETCH register pipeline
    // Wait until our first prefetched tile is loaded in
    // Prefetch the first rmem from the first k-tile
    if (K_BLOCK_MAX > 1)
    {
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
        copy(s2r_atom_a, tXsA_p(_,_,_0{}), tXrA(_,_,_0{}));
        copy(s2r_atom_b, tXsB_p(_,_,_0{}), tXrB(_,_,_0{}));
    }

    // Step5.2 Pipeline: Main Loop
    while (k_tile_count + K_PIPE_MAX - 1 > 0) // 可以优化成k_tile_count > -(K_PIPE_MAX-1)
    {
        for (int k_block = 0; k_block < K_BLOCK_MAX; k_block++)
        {
            if (k_block == K_BLOCK_MAX - 1)
            {
                // Slice the smem_pipe_read smem
                tXsA_p = tXsA(_,_,_, smem_pipe_read);
                tXsB_p = tXsB(_,_,_, smem_pipe_read);
                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX-2>();
                __syncthreads();
            }
            
            auto k_block_next = (k_block + _1{}) % K_BLOCK_MAX; // static
            copy(s2r_atom_a, tXsA_p(_,_, k_block_next), tXrA(_,_, k_block_next));
            copy(s2r_atom_b, tXsB_p(_,_, k_block_next), tXrB(_,_, k_block_next));
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0)
            {
                copy(copy_a, tAgA(_,_,_, k_tile_next), tAsA(_,_,_, smem_pipe_write));
                copy(copy_b, tBgB(_,_,_, k_tile_next), tBsB(_,_,_, smem_pipe_write));
                cp_async_fence();
                // Advance the gmem tile
                k_tile_count--;
                if (k_tile_count > 0)
                {
                    k_tile_next++;
                }
                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                // smem_pipe_read = (smem_pipe_read + 1) % K_PIPE_MAX;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
            }

            gemm(mma, tCrA(_,_, k_block), tCrB(_,_, k_block), tCrC);
            
        }
        
    }

    // epilogue
    axpby(alpha, tCrC, beta, tCgC);
    
}


template <class TA, class TB, class TC,
          class Alpha, class Beta>
void call_sgemm_sm80_tn(TA *A, TB *B, TC *C, 
                        int M, int N, int K,
                        Alpha alpha, Beta beta)
{
    using namespace cute;
    // step1.1 Define shapes (dynamic)
    auto prob_shape = make_shape(M, N, K);

    // step1.2 Define TN strides (mixed)
    // make sure vectorized copy dim is contiguous
    auto dA = make_stride(K, _1{});
    auto dB = make_stride(K, _1{});
    auto dC = make_stride(_1{}, M);

    // step2.1 Define CTA tile sizes (static) and pipeline stages
    auto bM = _128{};
    auto bN = _128{};
    auto bK = _8{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = _3{};

    // step2.3 Define the smem layouts (static)
    auto sA_atom = make_layout(make_shape(bM, bK), make_stride(_1{}, bM));
    auto sB_atom = make_layout(make_shape(bN, bK), make_stride(_1{}, bN));
    auto sA = tile_to_shape(sA_atom, make_shape(bM, bK, bP));
    auto sB = tile_to_shape(sB_atom, make_shape(bN, bK, bP));
    auto sC = make_layout(make_shape(bM, bN));

    // step3.1 Define AB thread layouts (static) using TiledCopy
    TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TA>, TA>{},
                                      Layout<Shape<_32, _8>, Stride<_8, _1>>{},
                                      Layout<Shape<_1, _1>>{});

    TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<TB>, TB>{},
                                      Layout<Shape<_32, _8>, Stride<_8,_1>>{},
                                      Layout<Shape<_1, _1>>{});            
    
    
    // step4.1 Define C thread layouts (static) using TiledMMA
    TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                   Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA

    // call
    int smem_sz = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
    dim3 dimBlock(size(mmaC));
    dim3 dimGrid(ceil_div(M, bM), ceil_div(N, bN));
    sgemm_sm80_tn<<<dimGrid, dimBlock, smem_sz, 0>>>
        (prob_shape, cta_tiler, 
         A, dA, sA, copyA, Copy_Atom<AutoVectorizingCopy, TA>{},
         B, dB, sB, copyB, Copy_Atom<AutoVectorizingCopy, TB>{},
         C, dC, sC, mmaC,
         alpha, beta);
}

int main(int argc, char* argv[])
{
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

    // TN: A is M×K, B is N×K (row-major)
    thrust::host_vector<TA> h_A(M * K, TA(1.0));
    thrust::host_vector<TB> h_B(N * K, TB(1.0));
    thrust::host_vector<TC> h_C(M * N, TC(0.0));
    thrust::host_vector<TC> h_RefC(M * N, TC(0.0));

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    thrust::device_vector<TC> d_RefC = h_RefC;

    // TN: A is M×K, B is N×K → C = A * B^T (default transA=N, transB=T)
    // ref_gemm(
    //     thrust::raw_pointer_cast(d_A.data()),
    //     thrust::raw_pointer_cast(d_B.data()),
    //     thrust::raw_pointer_cast(d_RefC.data()),
    //     alpha, beta, M, N, K);
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
        call_sgemm_sm80_tn(
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
        call_sgemm_sm80_tn(
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

    std::cout << "[sgem80 SGEMM] "
              << "M=" << M << ", N=" << N << ", K=" << K 
              << " | Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
              << " | Performance: " << std::fixed << std::setprecision(2) << tflops << " TFLOPS\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
    CUTE_CHECK_LAST();

    h_C = d_C;
    h_RefC = d_RefC;

    // tensor_cmp(
    //     thrust::raw_pointer_cast(h_C.data()),
    //     thrust::raw_pointer_cast(h_RefC.data()),
    //     M, N);
    utils::compare_tensors(d_C.data().get(), d_RefC.data().get(), M*N);

    return 0;
}