#!/bin/bash

#SBATCH --job-name=fp8_cutlass_h100
#SBATCH --output=profiles/fp8_h100/fp8_cutlass_h100_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:H100:1
#SBATCH --time=02:00:00

set -euo pipefail

source /home/spack/spack/share/spack/setup-env.sh
spack load cmake@3.28.6
spack load cuda@12.9.0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="${ROOT_DIR}/extern/cutlass"
PROFILE_DIR="${ROOT_DIR}/profiles/fp8_h100"
PROBLEMS_FILE="${PROFILE_DIR}/fp8_e4m3_bf16_problems.json"
TESTLIST_RAW_FILE="${PROFILE_DIR}/fp8_e4m3_bf16_h100_heuristics_raw.csv"
TESTLIST_FILE="${PROFILE_DIR}/fp8_e4m3_bf16_h100_scheduler_sweep.csv"
RESULT_PREFIX="${PROFILE_DIR}/fp8_e4m3_bf16_h100_results"
WINNERS_CSV="${PROFILE_DIR}/fp8_e4m3_bf16_h100_winners.csv"
WINNERS_MD="${PROFILE_DIR}/fp8_e4m3_bf16_h100_winners.md"
CUTLASS_BUILD_DIR="${CUTLASS_DIR}/build_fp8_h100"
LOCAL_BUILD_DIR="${ROOT_DIR}/build_fp8_ref_h100"
CONFIGS_PER_PROBLEM="${CUTLASS_FP8_CONFIGS_PER_PROBLEM:-64}"
RASTER_ORDERS="${CUTLASS_FP8_RASTER_ORDERS:-along_n,along_m}"
SWIZZLE_SIZES="${CUTLASS_FP8_SWIZZLE_SIZES:-1,2,4,8}"

mkdir -p "${PROFILE_DIR}"

echo "Job started on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "CUDA compiler: $(command -v nvcc)"
nvcc --version

if ! python3 -c "import nvMatmulHeuristics" >/dev/null 2>&1; then
  if [[ -z "${CUTLASS_NVMMH_PATH:-}" && -z "${CUTLASS_NVMMH_URL:-}" ]]; then
    echo "Missing Python package nvidia-matmul-heuristics."
    echo "Install it in the server environment, or set CUTLASS_NVMMH_PATH/CUTLASS_NVMMH_URL before submitting."
    exit 1
  fi
fi

cmake -S "${CUTLASS_DIR}" -B "${CUTLASS_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS=90a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm \
  -DCUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE="${PROBLEMS_FILE}" \
  -DCUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM="${CONFIGS_PER_PROBLEM}" \
  -DCUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE="${TESTLIST_RAW_FILE}" \
  -DCUTLASS_LIBRARY_HEURISTICS_GPU=H100_PCIE \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON

cmake --build "${CUTLASS_BUILD_DIR}" --target cutlass_profiler -j "${SLURM_CPUS_PER_TASK:-16}"

python3 "${PROFILE_DIR}/expand_heuristics_testlist.py" "${TESTLIST_RAW_FILE}" "${TESTLIST_FILE}" \
  --raster-orders "${RASTER_ORDERS}" \
  --swizzle-sizes "${SWIZZLE_SIZES}"

"${CUTLASS_BUILD_DIR}/tools/profiler/cutlass_profiler" \
  --operation=Gemm \
  --providers=cutlass \
  --testlist-file="${TESTLIST_FILE}" \
  --profiling-iterations=0 \
  --profiling-duration=100 \
  --warmup-iterations=10 \
  --verification-enabled=false \
  --output="${RESULT_PREFIX}"

python3 "${PROFILE_DIR}/parse_cutlass_fp8_results.py" "${RESULT_PREFIX}*.csv" \
  --output-csv "${WINNERS_CSV}" \
  --output-md "${WINNERS_MD}"

cmake -S "${ROOT_DIR}" -B "${LOCAL_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DBUILD_FP8_CUTLASS_REF=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90a

cmake --build "${LOCAL_BUILD_DIR}" --target fp8_cutlass_reference_check -j "${SLURM_CPUS_PER_TASK:-16}"

"${LOCAL_BUILD_DIR}/fp8_cutlass_reference_check" 4096 4096 4096 1 10

echo "Profiler results prefix: ${RESULT_PREFIX}"
echo "Raw heuristic testlist: ${TESTLIST_RAW_FILE}"
echo "Scheduler-sweep testlist: ${TESTLIST_FILE}"
echo "Winner CSV: ${WINNERS_CSV}"
echo "Winner Markdown: ${WINNERS_MD}"
echo "Job finished."
