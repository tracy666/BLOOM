#!/usr/bin/env bash
# The paper's weighted GRN prior uses adjacencies.tsv from the grn stage.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
TF_NAMES=${TF_NAMES:?Set TF_NAMES to the external mouse TF-list file}
exec "$PYTHON" -m bloom.preprocessing grn \
  --adata_x_path "$PREPARED_DIR/X.npz" \
  --adata_obs_path "$PREPARED_DIR/obs.parquet" \
  --adata_var_path "${GRN_VAR_PATH:-$PREPARED_DIR/genes_transport_hvg_2000.parquet}" \
  --model_var_path "${MODEL_VAR_PATH:-$PREPARED_DIR/genes_author.parquet}" \
  --tf_names_path "$TF_NAMES" --out_dir "${GRN_DIR:-$DATASET_ROOT/grn}" \
  --gene_name_col "${GENE_COLUMN:-gene_short_name}" \
  --time_col "${TIME_COLUMN:-time}" --celltype_col "${CELLTYPE_COLUMN:-class_label}" \
  --sample_strategy stratified_time_celltype --max_cells_total "${GRN_MAX_CELLS:-50000}" \
  --max_cells_per_time_celltype "${GRN_CELLS_PER_STRATUM:-500}" \
  --method grnboost2 --seed "${GRN_SEED:-42}" --num_workers "${GRN_WORKERS:-16}" \
  --use_dask "${GRN_USE_DASK:-True}" --steps grn "$@"
