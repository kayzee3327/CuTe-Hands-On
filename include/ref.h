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

#include <iostream>
#include <utility>
#include <concepts> // For std::invocable
#include <type_traits> // For std::is_same_v and std::invoke_result_t

// check cuda apis
#define CUDA_CHECK(status)                                              \
    {                                                                   \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " in "\
                  << __func__ << " at "                                 \
                  << __FILE__ << ":" << __LINE__ << std::endl;          \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
    }
// check kernel launch
#define CUDA_CHECK_LAST_ERROR()                                                 \
    do {                                                                        \
        cudaError_t error = cudaGetLastError();                                 \
        if (error != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " in "  \
                      << __func__ << " at "                                     \
                      << __FILE__ << ":" << __LINE__ << std::endl;              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define LAMBDA_WRAPPER(func) [](auto&&... fw_args){return func(std::forward<decltype(fw_args)>(fw_args)...);}

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


template<typename Callable, typename... Args>
requires std::invocable<Callable, Args...>
bool is_cuda_kernel(Callable f)
{
    if constexpr (std::is_pointer_v<Callable>) {
        cudaFuncAttribute attr;
        cudaError_t err = cudaFuncGetAttributes(&attr, f);
        if (err == cudaSuccess)
        {
            return true;
        }
        cudaGetLastError();
    }
    return false;
}


template<typename Callable, typename... Args>
requires std::invocable<Callable, Args...>
auto profile(int iters, double flop, Callable func, Args&&... args) 
{
    using RetType = std::invoke_result_t<Callable, Args...>;
    
    GPU_Clock timer;
    double tflops = flop * 1e-12;
    if constexpr (std::is_same_v<RetType, void>)
    {
        // warm up caches
        func(std::forward<Args>(args)...);

        timer.start();
        for (int i = 0; i < iters; i++)
        {
            func(std::forward<Args>(args)...);
        }
        double time = timer.seconds() / iters;
        CUDA_CHECK_LAST_ERROR();
        printf("Profiling results:    [%f]TFlop/s  (%f)ms\n", 
                tflops / time, time*1000);
        return;
    }
    else
    {
        // warm up caches
        func(std::forward<Args>(args)...);
        
        std::vector<RetType> results;
        results.reserve(iters);
        
        timer.start();
        for (int i = 0; i < iters; i++)
        {
            results.push_back(func(std::forward<Args>(args)...));
        }
        double time = timer.seconds() / iters;
        CUDA_CHECK_LAST_ERROR();
        printf("Profiling results:    [%f]TFlop/s  (%f)ms\n", 
                tflops / time, time*1000);
        return results;
            
    }

    
    
}