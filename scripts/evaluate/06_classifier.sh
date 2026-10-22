#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: $0 RUN_ROOT [forward|reverse] [classifier options]"
  echo "RUN_ROOT is the concrete export directory containing 01_predictions/."
  exec "${PYTHON_BIN}" -m bloom.evaluation classifier --help
fi
RUN_ROOT="$1"
shift
DIRECTION="forward"
if [[ "${1:-}" == "forward" || "${1:-}" == "reverse" ]]; then
  DIRECTION="$1"
  shift
fi
exec "${PYTHON_BIN}" -m bloom.evaluation classifier \
  --prediction-h5 "${RUN_ROOT}/01_predictions/measured_${DIRECTION}.h5" \
  --output "${RUN_ROOT}/07_classifier_${DIRECTION}" "$@"
