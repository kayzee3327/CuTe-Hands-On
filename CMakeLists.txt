cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(CuTe-Hands-On LANGUAGES CXX CUDA)

# --- 1. Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 90a) # Build for Ampere (80) and Hopper (90a)

# --- 2. Dependencies ---
# We point directly to the submodule. 
# "include" has the core CuTe/CUTLASS headers.
# "tools/util/include" has helper classes like HostTensor and reference GEMMs.
set(CUTLASS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/extern/cutlass)
set(CUTLASS_INCLUDE_DIRS 
    ${CUTLASS_PATH}/include 
    ${CUTLASS_PATH}/tools/util/include
)

# --- 3. Helper Function to Add Examples ---
# This macro saves you from writing 5 lines of cmake for every new file you create
function(add_cute_example TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    
    # Include paths for this target
    target_include_directories(${TARGET_NAME} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include # Your local headers
        ${CUTLASS_INCLUDE_DIRS}             # CUTLASS & CuTe headers
    )

    # Standard flags needed for CUTLASS/CuTe
    target_compile_options(${TARGET_NAME} PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
    )
endfunction()

# --- 4. Define Your Targets ---

# Example 1: Basic SGEMM
add_cute_example(sgemm1 src/01_sgemm/sgemm_1.cu)
add_cute_example(sgemm2 src/01_sgemm/sgemm_2.cu)

# Example 2: Hopper Specific (Ping Pong)
# You can override properties for specific targets if needed
# add_cute_example(hopper_pingpong src/03_hopper/pingpong_gemm.cu)
# set_target_properties(hopper_pingpong PROPERTIES CUDA_ARCHITECTURES 90a)