cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR) # cuda c++ 20
project(CuTe-Hands-On LANGUAGES CXX CUDA)

# --- 1. Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# old versions of cmake does not know how to compile cuda >=c++20
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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

find_package(CUDAToolkit REQUIRED)

# We make it STATIC so the heavy compilation happens once and is archived.
add_library(utils STATIC 
    src/utils.cu
)
# PUBLIC means any target that links against 'utils' will automatically 
# be able to #include "utils.h"
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(utils PUBLIC CUDA::cublas)

# --- 3. Helper Function to Add Examples ---
# This macro saves you from writing 5 lines of cmake for every new file you create
function(add_cute_example TARGET_NAME SOURCE_FILE)
  add_executable(${TARGET_NAME} ${SOURCE_FILE})
    
  # Include paths for this target
  target_include_directories(${TARGET_NAME} PRIVATE 
      ${CMAKE_CURRENT_SOURCE_DIR}/include # Your local headers
      ${CUTLASS_INCLUDE_DIRS}             # CUTLASS & CuTe headers
  )
  # target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20 cuda_std_20)

  # Standard flags needed for CUTLASS/CuTe
  target_compile_options(${TARGET_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
  )
  target_link_libraries(${TARGET_NAME} PRIVATE utils)

  # target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cublas)
  # Derive the real library directory from the cublas import target and embed it
  # as an rpath in the ELF header so the binary runs without LD_LIBRARY_PATH.
  get_target_property(_cublas_loc CUDA::cublas IMPORTED_LOCATION)
  get_filename_component(_cublas_dir "${_cublas_loc}" DIRECTORY)
  set_target_properties(${TARGET_NAME} PROPERTIES
      BUILD_RPATH  "${_cublas_dir}"
      INSTALL_RPATH "${_cublas_dir}"
  )
endfunction()

# --- 4. Define Your Targets ---

# Example 1: Basic SGEMM
add_cute_example(sgemm_1 src/examples/sgemm_1.cu)
add_cute_example(sgemm_2 src/examples/sgemm_2.cu)
add_cute_example(sgemm_sm80 src/examples/sgemm_sm80.cu)


# Example 2: Hopper Specific (Ping Pong)
# You can override properties for specific targets if needed
# add_cute_example(hopper_pingpong src/03_hopper/pingpong_gemm.cu)
# set_target_properties(hopper_pingpong PROPERTIES CUDA_ARCHITECTURES 90a)