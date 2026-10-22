#!/usr/bin/env bash
# Solve sequential OT on an existing PCA representation (or explicitly compute it).
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
args=(--input "${H5AD:-$DATASET_ROOT/raw/atlas.h5ad}"
      --output-dir "${TRANSPORT_DIR:-$DATASET_ROOT/transport}"
      --time-column "${TIME_COLUMN:-time}" --pca-key "${PCA_KEY:-X_pca}"
      --epsilon "${OT_EPSILON:-0.001}" --rank "${OT_RANK:--1}"
      --tau-a "${OT_TAU_A:-1.0}" --tau-b "${OT_TAU_B:-1.0}"
      --scale-cost mean --device "${OT_DEVICE:-cpu}" --seed "${OT_SEED:-0}")
if [[ ${COMPUTE_PCA:-0} == 1 ]]; then
  args+=(--compute-pca --pca-components "${PCA_COMPONENTS:-50}")
  [[ -z ${PCA_MASK_COLUMN:-} ]] || args+=(--pca-mask-column "$PCA_MASK_COLUMN")
  [[ -z ${PCA_LAYER:-} ]] || args+=(--pca-layer "$PCA_LAYER")
fi
exec "$PYTHON" -m bloom.preprocessing transport "${args[@]}" "$@"
