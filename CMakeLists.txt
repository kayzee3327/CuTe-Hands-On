cmake_minimum_required(VERSION 3.25.2 FATAL_ERROR) # cuda c++ 20
project(CuTe-Hands-On LANGUAGES CXX CUDA)

set(CLEAN_SCRIPT ${CMAKE_BINARY_DIR}/clean_non_ptx.cmake)
file(WRITE ${CLEAN_SCRIPT}
"file(GLOB files \"\${CLEAN_DIR}/*\")
foreach(f \${files})
  # If the file does NOT end in .ptx, delete it
  if(NOT f MATCHES \"\\\\.ptx$\")
      file(REMOVE \"\${f}\")
  endif()
endforeach()
")

# Use -DKEEP_PTX=ON to trigger this
option(KEEP_PTX "Keep intermediate PTX files" OFF)
# Use -DKEEP_PTX_TARGET="sgemm_sm80" to limit PTX generation to one target
set(KEEP_PTX_TARGET "" CACHE STRING "Specific target to keep PTX for (leave empty for all)")
option(BUILD_EXAMPLES "Build src/exmaples" OFF)
option(BUILD_DEMOS "Build demos" OFF)
option(BUILD_TESTS "Build tests" OFF)

# --- 1. Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# old versions of cmake does not know how to compile cuda >=c++20
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_ARCH_AMPERE 86)
set(CUDA_ARCH_HOPPER 90a)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_AMPERE})
endif()

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
target_link_libraries(utils PUBLIC CUDA::cublas CUDA::cublasLt)
set_target_properties(utils PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_AMPERE})

add_library(cutlass_fp8_gemm STATIC
  src/cutlass_fp8_gemm.cu
)
target_include_directories(cutlass_fp8_gemm PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUTLASS_INCLUDE_DIRS}
)
target_compile_options(cutlass_fp8_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
)
set_target_properties(cutlass_fp8_gemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH_HOPPER})

# --- 3. Helper Function to Add Examples ---
# This macro saves you from writing 5 lines of cmake for every new file you create
function(add_cute_example TARGET_NAME SOURCE_FILE)
  add_executable(${TARGET_NAME} ${SOURCE_FILE})

  set(TARGET_CUDA_ARCH ${CUDA_ARCH_AMPERE})
  if(ARGC GREATER 2)
    set(TARGET_CUDA_ARCH "${ARGV2}")
  endif()
  set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${TARGET_CUDA_ARCH}")
    
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
  target_compile_options(${TARGET_NAME} PRIVATE -lineinfo)
  # Check if KEEP_PTX is ON, and if the target matches (or if no specific target was provided)
  # --- UPDATED: Target-Specific PTX Output & Cleanup ---
  if(KEEP_PTX AND (KEEP_PTX_TARGET STREQUAL "" OR KEEP_PTX_TARGET STREQUAL TARGET_NAME))
      
    # 1. Isolate the temp files in a TARGET-SPECIFIC directory to prevent race conditions
    set(TARGET_PTX_DIR ${CMAKE_BINARY_DIR}/ptx_output/${TARGET_NAME})
    file(MAKE_DIRECTORY ${TARGET_PTX_DIR})
    
    target_compile_options(${TARGET_NAME} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:
        --keep 
        --keep-dir=${TARGET_PTX_DIR}
      >
    )
      
    # 2. Add a POST_BUILD step to run our cleanup script
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -DCLEAN_DIR=${TARGET_PTX_DIR} -P ${CLEAN_SCRIPT}
      COMMENT "Cleaning up non-PTX intermediate files for ${TARGET_NAME}..."
    )
      
    message(STATUS "Enabled PTX generation for ${TARGET_NAME} (Saved to ptx_output/${TARGET_NAME}/)")
  endif()

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

function(add_cute_demo TARGET_NAME SOURCE_FILE)
  add_executable(${TARGET_NAME} ${SOURCE_FILE})

  set(TARGET_CUDA_ARCH ${CUDA_ARCH_AMPERE})
  if(ARGC GREATER 2)
    set(TARGET_CUDA_ARCH "${ARGV2}")
  endif()
  set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${TARGET_CUDA_ARCH}")

  target_include_directories(${TARGET_NAME} PRIVATE 
      ${CMAKE_CURRENT_SOURCE_DIR}/include # Your local headers
      ${CUTLASS_INCLUDE_DIRS}             # CUTLASS & CuTe headers
  )
  target_compile_options(${TARGET_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
  )
endfunction()

# --- 4. Define Your Targets ---

add_cute_example(sgemm_1 src/examples/sgemm_1.cu ${CUDA_ARCH_AMPERE})
add_cute_example(sgemm_2 src/examples/sgemm_2.cu ${CUDA_ARCH_AMPERE})
add_cute_example(sgemm_sm80 src/examples/sgemm_sm80.cu ${CUDA_ARCH_AMPERE})
add_cute_example(sgemm_opt86 src/sgemm_opt86.cu ${CUDA_ARCH_AMPERE})

if(BUILD_TESTS)
add_cute_example(cublaslt_fp8_bf16_reference_test tests/cublaslt_fp8_bf16_reference_test.cu ${CUDA_ARCH_HOPPER})
add_cute_example(cutlass_fp8_bf16_tnn_test tests/cutlass_fp8_bf16_tnn_test.cu ${CUDA_ARCH_HOPPER})
target_link_libraries(cutlass_fp8_bf16_tnn_test PRIVATE cutlass_fp8_gemm)
add_cute_example(fp8_reference_speed_test tests/fp8_reference_speed_test.cu ${CUDA_ARCH_HOPPER})
target_link_libraries(fp8_reference_speed_test PRIVATE cutlass_fp8_gemm)
endif()

if(BUILD_DEMOS)
add_cute_demo(PermutationMNK demos/PermutationMNK.cu ${CUDA_ARCH_AMPERE})
add_cute_demo(epilogue_axpby demos/epilogue_axpby.cu ${CUDA_ARCH_AMPERE})
add_cute_demo(tma_desc demos/tma_desc.cu ${CUDA_ARCH_HOPPER})
endif()
