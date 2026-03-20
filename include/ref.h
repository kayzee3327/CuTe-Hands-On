#pragma once

#include "cutlass/util/host_tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

template <class TA, class LayoutA,
          class TB, class LayoutB,
          class TC, class LayoutC,
          class TI>
void ref_gemm(cutlass::HostTensor<TA, LayoutA>& A,
              cutlass::HostTensor<TB, LayoutB>& B,
              cutlass::HostTensor<TC, LayoutC>& C,
              TI alpha, TI beta, 
              int M, int N, int K) 
{
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    cutlass::reference::device::Gemm<
        TA, LayoutA,
        TB, LayoutB,
        TC, LayoutC,
        TA,
        TI
    > gemm_operator;

    gemm_operator(
        problem_size,
        alpha,
        A.device_ref(),
        B.device_ref(),
        beta,
        C.device_ref(), // Output D
        C.device_ref()  // Input C (for D = alpha*A*B + beta*C)
    );

}

template <class TA, class LayoutA,
          class TB, class LayoutB>
void tensor_cmp(cutlass::HostTensor<TA, LayoutA>& result,
                cutlass::HostTensor<TB, LayoutB>& target) 
{
    
    bool passed = cutlass::reference::host::TensorEquals(
        result.host_view(), 
        target.host_view()
    );
    double err = cutlass::reference::host::TensorGreatestError(
        result.host_view(), 
        target.host_view()
    );

    std::cout << "Greatest Error: "<< err << std::endl;
    
    if (passed) {
        std::cout << "Success! CuTe GEMM matches CUTLASS reference." << std::endl;
    } else {
        std::cout << "Mismatch detected." << std::endl;
        // Optional: use cutlass::reference::host::TensorRelativeAndAbsoluteError to debug
    }
}