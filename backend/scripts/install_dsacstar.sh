#!/usr/bin/env bash
# Build and install ACE dsacstar extension (requires conda OpenCV dev headers).
set -euo pipefail

ACE_ROOT="$(cd "$(dirname "$0")/../../ace-main" && pwd)"
CONDA_PREFIX="${CONDA_PREFIX:-$(conda info --base 2>/dev/null || echo "")}"

if [[ -z "${CONDA_PREFIX}" ]]; then
  echo "Error: activate conda or set CONDA_PREFIX before running this script."
  exit 1
fi

if [[ ! -d "${CONDA_PREFIX}/include/opencv4" ]]; then
  echo "Installing OpenCV (conda-forge) for headers/libs..."
  conda install -y -c conda-forge opencv
fi

export CONDA_PREFIX
pip install "${ACE_ROOT}/dsacstar"
python -c "import dsacstar; print('dsacstar OK')"
