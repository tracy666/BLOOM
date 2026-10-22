#!/usr/bin/env bash
# Export a prepared atlas and explicitly selected model/transport gene panels.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
H5AD=${H5AD:-"$DATASET_ROOT/raw/atlas.h5ad"}
args=(--input "$H5AD" --output-dir "$PREPARED_DIR"
      --time-column "${TIME_COLUMN:-time}" --celltype-column "${CELLTYPE_COLUMN:-class_label}"
      --gene-column "${GENE_COLUMN:-gene_short_name}")
[[ -z ${EXPRESSION_LAYER:-} ]] || args+=(--layer "$EXPRESSION_LAYER")
[[ -z ${AUTHOR_GENES:-} ]] || args+=(--author-gene-list "$AUTHOR_GENES")
[[ -z ${AUTHOR_MASK_COLUMN:-} ]] || args+=(--author-mask-column "$AUTHOR_MASK_COLUMN")
[[ -z ${HVG_GENES:-} ]] || args+=(--hvg-gene-list "$HVG_GENES")
[[ -z ${HVG_MASK_COLUMN:-} ]] || args+=(--hvg-mask-column "$HVG_MASK_COLUMN")
[[ -z ${TRANSPORT_GENES:-} ]] || args+=(--transport-gene-list "$TRANSPORT_GENES")
[[ -z ${TRANSPORT_MASK_COLUMN:-} ]] || args+=(--transport-mask-column "$TRANSPORT_MASK_COLUMN")
exec "$PYTHON" -m bloom.preprocessing export-h5ad "${args[@]}" "$@"
