# Shared portable paths for the 24 model-specific launchers.
BLOOM_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-${BLOOM_PYTHON:-python3}}
export PYTHONPATH="${BLOOM_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
DATASET=${DATASET:-cortex}
DATA_ROOT=${DATA_ROOT:-${BLOOM_REPO_ROOT}/data}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BLOOM_REPO_ROOT}/outputs/training}
PANEL=${PANEL:-author}
HVG_GENES=${HVG_GENES:-200}
TS_LIBRARY_ROOT=${TS_LIBRARY_ROOT:-${BLOOM_TSL_ROOT:-${BLOOM_REPO_ROOT}/external/Time-Series-Library}}

training_bool_flag() {
  local name="$1" value="$2"
  case "$value" in
    1|true|True|TRUE|yes) printf '%s' "--${name}" ;;
    0|false|False|FALSE|no) printf '%s' "--no-${name}" ;;
    *) printf 'Invalid Boolean %s=%s\n' "$name" "$value" >&2; return 2 ;;
  esac
}

run_training() {
  local model="$1"
  shift
  "$PYTHON" -m bloom.training \
    --model "$model" \
    --dataset "$DATASET" \
    --panel "$PANEL" \
    --hvg-genes "$HVG_GENES" \
    --data-root "$DATA_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --tsl-root "$TS_LIBRARY_ROOT" \
    "$@"
}
