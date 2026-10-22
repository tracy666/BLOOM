#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

# Author genes are selected by default. PANEL=hvg selects HVG_GENES (default 200).
# DATASET=prenatal|cortex|sted chooses the dataset-specific observation columns.
# Extra CLI arguments are appended last and override the explicit settings below.
# Reformer: the supplied v5 forecasting optimization recipe.
# Upstream model/layer implementations are loaded from TS_LIBRARY_ROOT.
run_training "Reformer" \
  --SEQ_LEN "${SEQ_LEN:-8}" \
  --TRANS_LABEL_LEN "${LABEL_LEN:-4}" \
  --PRED_LEN "${PRED_LEN:-1}" \
  --SEED "${SEED:-42}" \
  --DATASET_RANDOM_SEED "${DATASET_RANDOM_SEED:-42}" \
  --CUDA_VISIBLE_DEVICES "${GPU:-0}" \
  --SPLIT_METHOD "${SPLIT_METHOD:-random_proportional_split}" \
  --TRAIN_RATIO "${TRAIN_RATIO:-0.7}" \
  --VALID_RATIO "${VALID_RATIO:-0.1}" \
  --TRANS_BATCH "${BATCH_SIZE:-32768}" \
  --TRANS_EPOCHS "${EPOCHS:-100}" \
  --TRANS_LR "${LEARNING_RATE:-1e-4}" \
  --TRANS_D_MODEL "${D_MODEL:-64}" \
  --TRANS_GRAD_CLIP "${GRAD_CLIP:-1.0}" \
  --TRANS_STOP_METRIC "${STOP_METRIC:-val_loss}" \
  --TRANS_EARLY_STOPPING_PATIENCE "${PATIENCE:-5}" \
  --TRANS_EARLY_STOPPING_MIN_DELTA "${MIN_DELTA:-0.0}" \
  "$(training_bool_flag TRANS_USE_CELLTYPE_CONDITIONING "${CELLTYPE_CONDITIONING:-False}")" \
  "$(training_bool_flag FP16 "${FP16:-False}")" \
  --TRANS_EVAL_BOTH_WEIGHTINGS \
  --SAVE_CHECKPOINTS \
  "$@"
