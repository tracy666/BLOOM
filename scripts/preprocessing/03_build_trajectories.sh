#!/usr/bin/env bash
# Retain source-defined OT candidates and assemble complete anchored paths.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON" -m bloom.preprocessing trajectories \
  --ADATA_OBS_PATH "$PREPARED_DIR/obs.parquet" \
  --transitions_dir "${TRANSPORT_DIR:-$DATASET_ROOT/transport}" \
  --transitions_origin "${PCA_KEY:-X_pca}" \
  --time_key "${TIME_COLUMN:-time}" --celltype_key "${CELLTYPE_COLUMN:-class_label}" \
  --edge_mode filtered --top_k "${TOP_K:-10}" --cum_mass "${CUM_MASS:-0.95}" \
  --anchors_direction both --min_anchor_support "${MIN_ANCHOR_SUPPORT:-2}" \
  --seed "${TRAJECTORY_SEED:-0}" --out_dir "${TRAJECTORY_DIR:-$DATASET_ROOT/trajectories}" \
  --canonical-output "$PREPARED_DIR/trajectories.npy" "$@"
