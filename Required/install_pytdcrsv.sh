#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${PROJECT_ROOT}/Required/tdcr-lilge-binding"