#!/bin/bash

#SBATCH --job-name=compile      # Job name
#SBATCH --output=compile.out  # Standard output and error log (%j expands to jobID)
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --mem=64G                   # Job memory request
#SBATCH --time=00:15:00             # Time limit hrs:min:sec

source /home/spack/spack/share/spack/setup-env.sh
spack load cmake@3.28.6
spack load cuda@12.9.0
spack load python@3.13.0

CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
CUDA_INCLUDE="${CUDA_ROOT}/targets/x86_64-linux/include"
CUDA_COMPAT="${PWD}/.cuda-12.9-compat"

# create temporary workaround
# rm -rf "${CUDA_COMPAT}"
# mkdir -p "${CUDA_COMPAT}/crt"

# cp -a "${CUDA_INCLUDE}/." "${CUDA_COMPAT}/"

# chmod u+w "${CUDA_COMPAT}/crt/math_functions.h"

# python3 - "${CUDA_COMPAT}/crt/math_functions.h" <<'PY'
# import re
# import sys
# from pathlib import Path

# path = Path(sys.argv[1])
# text = path.read_text()

# functions = {
#     "sinpi":  ("double", "double"),
#     "sinpif": ("float",  "float"),
#     "cospi":  ("double", "double"),
#     "cospif": ("float",  "float"),
# }

# for name, (return_type, argument_type) in functions.items():
#     pattern = re.compile(
#         rf"(^\s*extern\s+__DEVICE_FUNCTIONS_DECL__\s+"
#         rf"__device_builtin__\s+{return_type}\s+"
#         rf"{name}\s*\(\s*{argument_type}\s+x\s*\))\s*;",
#         re.MULTILINE,
#     )

#     text, count = pattern.subn(
#         rf"\1 noexcept (true);",
#         text,
#         count=1,
#     )

#     already_patched = re.search(
#         rf"{name}\s*\(\s*{argument_type}\s+x\s*\)\s*noexcept",
#         text,
#     )

#     if count != 1 and not already_patched:
#         raise RuntimeError(f"Failed to patch {name}")

# path.write_text(text)
# PY

# unset NVCC_PREPEND_FLAGS
# export NVCC_PREPEND_FLAGS="-I${CUDA_COMPAT}"

# echo "Checking patched declarations..."

# grep -nE \
#     'sinpi\(double x\).*noexcept|sinpif\(float x\).*noexcept|cospi\(double x\).*noexcept|cospif\(float x\).*noexcept' \
#     "${CUDA_COMPAT}/crt/math_functions.h"

# cat > cuda_smoke_test.cu <<'EOF'
# int main() {
#     return 0;
# }
# EOF

# echo "Testing nvcc with patched headers..."
# nvcc -c cuda_smoke_test.cu -o cuda_smoke_test.o
# rm -f cuda_smoke_test.cu cuda_smoke_test.o

# echo "CUDA header workaround succeeded."

# compile work below
unset NVCC_PREPEND_FLAGS
export NVCC_PREPEND_FLAGS="-I${CUDA_COMPAT}"

echo "Node:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "CUDA root:  ${CUDA_ROOT}"
echo "nvcc:       $(command -v nvcc)"
nvcc --version
gcc --version | head -n 1
ldd --version | head -n 1

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
    -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_TESTS=ON \
&& \
cmake --build build \
    --target cutlass_fp8_bf16_tnn_test \
    --target fp8_reference_speed_test \
    -j "${SLURM_CPUS_PER_TASK}"


echo "Job finished."
