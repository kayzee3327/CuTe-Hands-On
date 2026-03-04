#include "cutlass/util/host_tensor.h"
#include "cutlass/layout/matrix.h"

#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/util/reference/device/gemm.h"

#include "01_sgemm1.cuh"

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

    // Define the problem size
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Launch the CUTLASS reference GEMM on the GPU
    cutlass::reference::device::Gemm<
        TA, cutlass::layout::RowMajor,
        TB, cutlass::layout::RowMajor,
        TC, cutlass::layout::RowMajor,
        TA,
        TI
    > gemm_operator;
    gemm_operator(
        problem_size,
        alpha,
        A.device_ref(),
        B.device_ref(),
        beta,
        Reference_C.device_ref(), // Output D
        Reference_C.device_ref()  // Input C (for D = alpha*A*B + beta*C)
    );

    // Wait for the GPU reference kernel to finish
    cudaDeviceSynchronize();

    C.sync_host();
    Reference_C.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(
        Reference_C.host_view(), 
        C.host_view()
    );
    double err = cutlass::reference::host::TensorGreatestError(
        Reference_C.host_view(), 
        C.host_view()
    );

    std::cout << "Greatest Error: "<< err << std::endl;
    
    if (passed) {
        std::cout << "Success! CuTe GEMM matches CUTLASS reference." << std::endl;
    } else {
        std::cout << "Mismatch detected." << std::endl;
        // Optional: use cutlass::reference::host::TensorRelativeAndAbsoluteError to debug
    }
    
        return 0;
    }
