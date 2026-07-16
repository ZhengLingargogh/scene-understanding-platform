#!/usr/bin/env bash
# Repair OpenCV in the sup conda env: remove hybrid cv2 and pin a working pip wheel.
# Note: conda-forge opencv 4.13 breaks findContours with numpy 1.26 (ABI mismatch).
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Error: activate sup first, e.g.  conda activate sup"
  exit 1
fi

PYVER="$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
SP="${CONDA_PREFIX}/lib/python${PYVER}/site-packages"

echo "==> Python ${PYVER} in ${CONDA_PREFIX}"

echo "==> Removing conda / pip OpenCV overlays"
conda remove -y opencv py-opencv libopencv 2>/dev/null || true
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y 2>/dev/null || true
rm -rf "${SP}"/opencv_python*.dist-info "${SP}"/opencv_python_headless*.dist-info "${SP}"/cv2

echo "==> Installing pip opencv-python-headless 4.10 (compatible with numpy 1.26)"
pip install "opencv-python-headless==4.10.0.84"

echo "==> Pinning numpy<2 for PyTorch / torchvision compatibility"
pip install "numpy>=1.24,<2" --force-reinstall

echo "==> Verifying cv2 + findContours"
python - <<'PY'
import numpy as np
import cv2
import torch

mask = np.zeros((64, 64), dtype=np.uint8)
mask[16:48, 16:48] = 1
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("torch+cv2 OK", torch.__version__, cv2.__version__, "contours", len(contours))
PY

BACKEND="$(cd "$(dirname "$0")/.." && pwd)"
cd "${BACKEND}"
python -c "from app.services.localization import _get_cv2; print('ACE cv2 OK', _get_cv2().__version__)"

echo "==> Done. Restart uvicorn (Ctrl+C first if already running):"
echo "    cd ${BACKEND}"
echo "    ${CONDA_PREFIX}/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
