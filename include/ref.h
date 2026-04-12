#pragma once

#include "cutlass/util/host_tensor.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/mma.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include <cublas_v2.h>

#include <iostream>

// Raw-pointer ref_gemm using cuBLAS (row-major, all matrices contiguous).
// transA/transB describe the math operation on A and B:
//   TN (A is M×K, B is N×K): transA=CUBLAS_OP_N, transB=CUBLAS_OP_T  (default)
//   NT (A is K×M, B is K×N): transA=CUBLAS_OP_T, transB=CUBLAS_OP_N
// Leading dims are derived automatically from the op flags and M,N,K.
template <class TA, class TB, class TC, class TI>
void ref_gemm(TA* d_A, TB* d_B, TC* d_C,
              TI alpha, TI beta, int M, int N, int K,
              cublasOperation_t transA = CUBLAS_OP_N,
              cublasOperation_t transB = CUBLAS_OP_T)
{
    static_assert(std::is_same_v<TA, float> && std::is_same_v<TB, float> && std::is_same_v<TC, float>,
                  "ref_gemm cuBLAS path only supports float");
    // Row-major leading dims:
    //   transA=N → A is M×K → lda=K;  transA=T → A is K×M → lda=M
    //   transB=T → B is N×K → ldb=K;  transB=N → B is K×N → ldb=N
    int lda = (transA == CUBLAS_OP_N) ? K : M;
    int ldb = (transB == CUBLAS_OP_T) ? K : N;
    float alpha_f = static_cast<float>(alpha);
    float beta_f  = static_cast<float>(beta);

    cublasHandle_t handle;
    cublasCreate(&handle);
    // cuBLAS is column-major: compute C^T = op(B) * op(A) with swapped M↔N
    cublasSgemm(handle, transB, transA, N, M, K,
                &alpha_f, d_B, ldb, d_A, lda, &beta_f, d_C, N);
    cublasDestroy(handle);
}

template <class TA, class TB>
void tensor_cmp(TA* h_result, TB* h_target, int M, int N)
{
    using Layout = cutlass::layout::RowMajor;
    cutlass::MatrixCoord extent(M, N);

    cutlass::TensorView<TA, Layout> view_result(h_result, Layout::packed(extent), extent);
    cutlass::TensorView<TB, Layout> view_target(h_target, Layout::packed(extent), extent);

    bool passed = cutlass::reference::host::TensorEquals(view_result, view_target);
    double err = cutlass::reference::host::TensorGreatestError(view_result, view_target);

    std::cout << "Greatest Error: " << err << std::endl;
    if (passed) {
        std::cout << "Success! CuTe GEMM matches CUTLASS reference." << std::endl;
    } else {
        std::cout << "Mismatch detected." << std::endl;
    }
}

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
