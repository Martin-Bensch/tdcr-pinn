#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[bootstrap] Root: ${ROOT_DIR}"
echo "[bootstrap] Python: $(${PYTHON_BIN} --version)"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[bootstrap] Python executable '${PYTHON_BIN}' not found." >&2
  exit 1
fi

if ! command -v c++ >/dev/null 2>&1; then
  echo "[bootstrap] C++ compiler not found. Install clang/gcc first." >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "[bootstrap] Homebrew not found. Install Homebrew and then run:"
    echo "            brew install gsl eigen"
  else
    echo "[bootstrap] Hint: ensure dependencies are installed:"
    echo "            brew install gsl eigen"
  fi
elif [[ "$(uname -s)" == "Linux" ]]; then
  echo "[bootstrap] Ensure system packages are installed (example for Debian/Ubuntu):"
  echo "            sudo apt-get update && sudo apt-get install -y libgsl-dev libeigen3-dev"
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install -e "./Required/tdcr-lilge-binding"
"${PYTHON_BIN}" -m pip install -e ".[dev]"

echo "[bootstrap] Done."
