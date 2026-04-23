#!/usr/bin/env bash
# Profile the RMSNorm CUDA kernel with Nsight Compute.
#
# Usage (from repo root):
#   bash scripts/profile_rmsnorm.sh
#
# Output: profiles/rmsnorm_<timestamp>.ncu-rep (open in Nsight Compute UI)
#         Text summary printed to stdout.
#
# Requires: ncu in PATH and CUDA_VISIBLE_DEVICES=8

set -euo pipefail

PROFILE_DIR="profiles"
mkdir -p "$PROFILE_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="${PROFILE_DIR}/rmsnorm_${TIMESTAMP}"

CONDA_ENV="/data/vineel/conda-envs/inference-engine"
PYTHON="${CONDA_ENV}/bin/python"
NCU=$(which ncu 2>/dev/null || echo "${CONDA_ENV}/bin/ncu")

if [[ ! -x "$NCU" ]]; then
    echo "ERROR: ncu not found. Install with: conda install nsight-compute -c nvidia"
    exit 1
fi

echo "Profiling rmsnorm_kernel with Nsight Compute..."
echo "Report will be saved to: ${REPORT}.ncu-rep"
echo ""

CUDA_VISIBLE_DEVICES=8 "$NCU" \
    --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --kernel-name "regex:rmsnorm" \
    --launch-count 3 \
    --clock-control none \
    --export "$REPORT" \
    "$PYTHON" scripts/benchmark_rmsnorm.py

echo ""
echo "=== Nsight Compute Summary ==="
"$NCU" --import "${REPORT}.ncu-rep" --page summary
