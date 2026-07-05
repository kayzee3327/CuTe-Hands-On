#!/bin/bash

#SBATCH --job-name=fp8_tnn_profile
#SBATCH --output=fp8_tnn_profile.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:H100:1

SCRIPT_DIR=${PWD}
SCRIPTS_DIR="${SCRIPT_DIR}/src/scripts"
CUDA_COMPAT="${SCRIPT_DIR}/.cuda-12.9-compat"

if [[ -f /home/spack/spack/share/spack/setup-env.sh ]]; then
  source /home/spack/spack/share/spack/setup-env.sh
  spack load cmake@3.28.6
  spack load cuda@12.9.0
  spack load python@3.13.0
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc was not found. Load CUDA before running this script." >&2
  exit 1
fi

source /home/wangkaize/CuTe-Hands-On/.venv/bin/activate

if ! python3 -c "import nvMatmulHeuristics" >/dev/null 2>&1
then
  cat >&2 <<'EOF'
ERROR: Python cannot import nvMatmulHeuristics.
Install or load the nvidia-matmul-heuristics package in the server environment,
then rerun this script.
EOF
  exit 1
fi

CUDA_ROOT="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
PROBLEMS_FILE="${CUTLASS_FP8_PROBLEMS_FILE:-${SCRIPT_DIR}/fp8_tnn_problems.json}"
CUTLASS_DIR="${CUTLASS_FP8_CUTLASS_DIR:-${SCRIPT_DIR}/extern/cutlass}"
BUILD_DIR="${CUTLASS_FP8_BUILD_DIR:-${SCRIPT_DIR}/build/cutlass_fp8_tnn_profiler}"
RESULTS_DIR="${CUTLASS_FP8_RESULTS_DIR:-${SCRIPT_DIR}/profiles/fp8_tnn/$(date +%Y%m%d_%H%M%S)}"

CONFIGS_PER_PROBLEM="${CUTLASS_FP8_CONFIGS_PER_PROBLEM:-64}"
HEURISTICS_GPU="${CUTLASS_FP8_HEURISTICS_GPU:-H100_PCIE}"
RASTER_ORDERS="${CUTLASS_FP8_RASTER_ORDERS:-heuristic,along_n,along_m}"
SWIZZLE_SIZES="${CUTLASS_FP8_SWIZZLE_SIZES:-1,2,4,8}"
PROFILE_DURATION_MS="${CUTLASS_FP8_PROFILE_DURATION_MS:-50}"
MIN_ITERATIONS="${CUTLASS_FP8_MIN_ITERATIONS:-10}"
WARMUP_ITERATIONS="${CUTLASS_FP8_WARMUP_ITERATIONS:-10}"
BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"
RESTRICT_KERNELS="${CUTLASS_FP8_RESTRICT_KERNELS:-OFF}"

mkdir -p "${RESULTS_DIR}"

TESTLIST_FILE="${RESULTS_DIR}/fp8_tnn_heuristics_testlist.csv"
SWEEP_TESTLIST_FILE="${RESULTS_DIR}/fp8_tnn_scheduler_sweep.csv"
PROFILE_OUTPUT_PREFIX="${RESULTS_DIR}/fp8_tnn_profile"
PROFILE_CSV="${PROFILE_OUTPUT_PREFIX}.gemm.csv"
WINNERS_CSV="${RESULTS_DIR}/fp8_tnn_winners.csv"
WINNERS_MD="${RESULTS_DIR}/fp8_tnn_winners.md"

echo "Node:                $(hostname)"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "CUDA root:           ${CUDA_ROOT}"
echo "nvcc:                $(command -v nvcc)"
echo "CUTLASS dir:         ${CUTLASS_DIR}"
echo "Problems:            ${PROBLEMS_FILE}"
echo "Build dir:           ${BUILD_DIR}"
echo "Results dir:         ${RESULTS_DIR}"
echo "Configs/problem:     ${CONFIGS_PER_PROBLEM}"
echo "Heuristics GPU:      ${HEURISTICS_GPU}"
echo "Raster orders:       ${RASTER_ORDERS}"
echo "Swizzle sizes:       ${SWIZZLE_SIZES}"
echo "Profile duration ms: ${PROFILE_DURATION_MS}"
nvcc --version
gcc --version | head -n 1
ldd --version | head -n 1

unset NVCC_PREPEND_FLAGS
export NVCC_PREPEND_FLAGS="-I${CUDA_COMPAT}"
echo "NVCC_PREPEND_FLAGS changes."

python3 "${SCRIPTS_DIR}/validate_fp8_tnn_problems.py" "${PROBLEMS_FILE}"\
&&\
cmake -S "${CUTLASS_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
  -DCUTLASS_NVCC_ARCHS=90a \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_ENABLE_LIBRARY=ON \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_ENABLE_CUBLAS=OFF \
  -DCUTLASS_ENABLE_CUDNN=OFF \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm \
  -DCUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE="${PROBLEMS_FILE}" \
  -DCUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM="${CONFIGS_PER_PROBLEM}" \
  -DCUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE="${TESTLIST_FILE}" \
  -DCUTLASS_LIBRARY_HEURISTICS_GPU="${HEURISTICS_GPU}" \
  -DCUTLASS_LIBRARY_HEURISTICS_RESTRICT_KERNELS="${RESTRICT_KERNELS}"\
&& \
cmake --build "${BUILD_DIR}" --target cutlass_profiler -j "${BUILD_JOBS}"

if [[ ! -s "${TESTLIST_FILE}" ]]; then
  echo "ERROR: CUTLASS did not emit a non-empty heuristics testlist at ${TESTLIST_FILE}" >&2
  exit 1
fi

python3 "${SCRIPTS_DIR}/expand_cutlass_fp8_tnn_testlist.py" \
  "${TESTLIST_FILE}" \
  "${SWEEP_TESTLIST_FILE}" \
  --raster-orders "${RASTER_ORDERS}" \
  --swizzle-sizes "${SWIZZLE_SIZES}" \
&& \
"${BUILD_DIR}/tools/profiler/cutlass_profiler" \
  --operation=Gemm \
  --providers=cutlass \
  --testlist-file="${SWEEP_TESTLIST_FILE}" \
  --profiling-iterations=0 \
  --profiling-duration="${PROFILE_DURATION_MS}" \
  --min-iterations="${MIN_ITERATIONS}" \
  --warmup-iterations="${WARMUP_ITERATIONS}" \
  --verification-enabled=false \
  --print-kernel-before-running=true \
  --output="${PROFILE_OUTPUT_PREFIX}"

if [[ ! -s "${PROFILE_CSV}" ]]; then
  echo "ERROR: profiler output was not found at ${PROFILE_CSV}" >&2
  exit 1
fi

python3 "${SCRIPTS_DIR}/parse_cutlass_fp8_tnn_results.py" \
  "${PROFILE_CSV}" \
  "${WINNERS_CSV}" \
  "${WINNERS_MD}"

cat <<EOF

Profiling complete.

Main artifacts:
  Heuristics testlist: ${TESTLIST_FILE}
  Scheduler sweep:     ${SWEEP_TESTLIST_FILE}
  Raw profiler CSV:    ${PROFILE_CSV}
  Winners CSV:         ${WINNERS_CSV}
  Winners Markdown:    ${WINNERS_MD}

Use the winner rows to update the TileShape, ClusterShape, KernelSchedule,
EpilogueSchedule, raster_order, and swizzle_size choices in:
  include/cutlass_fp8_gemm.h
  src/cutlass_fp8_gemm.cu

EOF

deactivate
