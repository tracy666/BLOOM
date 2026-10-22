#!/usr/bin/env bash
# Convert the dataset's released graphs/programmes, then publish stable filenames.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
AUTHOR_PRIOR_DIR=${AUTHOR_PRIOR_DIR:?Set AUTHOR_PRIOR_DIR to the original dataset author resources}
CONVERTED_DIR=${CONVERTED_DIR:-"$PRIORS_DIR/converted"}
args=(--author-prior-dir "$AUTHOR_PRIOR_DIR" --output-dir "$CONVERTED_DIR"
      --model-var-path "${MODEL_VAR_PATH:-$PREPARED_DIR/genes_author.parquet}"
      --model-gene-column "${GENE_COLUMN:-gene_short_name}" --obs-path "$PREPARED_DIR/obs.parquet")
case "$DATASET" in
  cortex)
    "$PYTHON" -m bloom.preprocessing author-priors --dataset small "${args[@]}" \
      --small-obs-cluster-col "${CLUSTER_COLUMN:-cluster_label}" "$@"
    ;;
  prenatal)
    "$PYTHON" -m bloom.preprocessing author-priors --dataset big "${args[@]}" \
      --big-cell-graph-rds "${CELL_GRAPH_RDS:-$AUTHOR_PRIOR_DIR/df_cell_graph.rds}" "$@"
    ;;
  sted)
    "$PYTHON" -m bloom.preprocessing sted-priors "${args[@]}" \
      --obs-label-col "${CELLTYPE_COLUMN:-label}" \
      --obs-period-col "${PERIOD_COLUMN:-embryonic_period}" --obs-day-col "${TIME_COLUMN:-day}" "$@"
    ;;
esac
canonical=(--dataset "$DATASET" --input-dir "$CONVERTED_DIR" --output-dir "$PRIORS_DIR")
[[ ${CANONICAL_OVERWRITE:-0} != 1 ]] || canonical+=(--overwrite)
"$PYTHON" -m bloom.preprocessing canonicalize-priors "${canonical[@]}"
