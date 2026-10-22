#!/usr/bin/env bash
# Shared paths for preprocessing commands; source this file from a launcher.
set -euo pipefail
BLOOM_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-python}
DATASET=${DATASET:?Set DATASET=prenatal, cortex, or sted}
case "$DATASET" in prenatal|cortex|sted) ;; *) echo "Unknown DATASET=$DATASET" >&2; exit 2 ;; esac
case "$DATASET" in
  prenatal) TIME_COLUMN=${TIME_COLUMN:-time}; CELLTYPE_COLUMN=${CELLTYPE_COLUMN:-major_trajectory} ;;
  cortex) TIME_COLUMN=${TIME_COLUMN:-time}; CELLTYPE_COLUMN=${CELLTYPE_COLUMN:-class_label} ;;
  sted) TIME_COLUMN=${TIME_COLUMN:-day}; CELLTYPE_COLUMN=${CELLTYPE_COLUMN:-label} ;;
esac
DATA_ROOT=${DATA_ROOT:-"$BLOOM_ROOT/data"}
DATASET_ROOT=${DATASET_ROOT:-"$DATA_ROOT/$DATASET"}
PREPARED_DIR=${PREPARED_DIR:-"$DATASET_ROOT/prepared"}
PRIORS_DIR=${PRIORS_DIR:-"$DATASET_ROOT/priors"}
export PYTHONPATH="$BLOOM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
