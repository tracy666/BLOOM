#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
  echo "Usage: $0 RUN_ROOT [biology-report options]"
  echo "Discovers measured and supported scenario exports within one run."
  exec "${PYTHON_BIN}" -m bloom.evaluation biology-report --help
fi
RUN_ROOT="$1"
shift
INPUT_ARGS=()
for DIRECTION in forward reverse; do
  CANDIDATE="${RUN_ROOT}/01_predictions/measured_${DIRECTION}.h5"
  if [[ -f "${CANDIDATE}" ]]; then INPUT_ARGS+=(--h5 "${CANDIDATE}"); fi
done
for SCENARIO in \
  05_interpolation/dense_developmental_time.h5 \
  08_perturbation/perturbation_compact_response.h5 \
  09_extrapolation/autoregressive_future.h5 \
  10_reverse_history/autoregressive_history.h5; do
  for RELATIVE in \
    "01_predictions/${SCENARIO##*/}" \
    "01_predictions/scenarios/${SCENARIO}" \
    "00_run_control/immutable_inputs/generated_scenarios/${SCENARIO}"; do
    CANDIDATE="${RUN_ROOT}/${RELATIVE}"
    if [[ -f "${CANDIDATE}" ]]; then
      INPUT_ARGS+=(--h5 "${CANDIDATE}")
      break
    fi
  done
done
if [[ ${#INPUT_ARGS[@]} -eq 0 ]]; then
  echo "No supported prediction H5 was found below ${RUN_ROOT}." >&2
  echo "For explicitly relocated files, use python -m bloom.evaluation biology-report --h5 PATH --out PATH." >&2
  exit 2
fi
exec "${PYTHON_BIN}" -m bloom.evaluation biology-report \
  --out "${RUN_ROOT}/06_biological_questions" "${INPUT_ARGS[@]}" "$@"
