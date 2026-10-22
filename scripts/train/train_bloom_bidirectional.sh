#!/usr/bin/env bash
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.bash"

# Author genes are selected by default. PANEL=hvg selects HVG_GENES (default 200).
# DATASET=prenatal|cortex|sted chooses the dataset-specific observation columns.
# Extra CLI arguments are appended last and override the explicit settings below.
# Bidirectional BLOOM: source-derived adaptive-prior release recipe.
# Forward/reverse unidirectional models are separate runs; set
# FM_TEMPORAL_DIRECTION=reverse to train the latter with the same script.
# All retained native options are listed by: python -m bloom.training --model bloom-bidirectional --native-help
run_training "bloom-bidirectional" \
  --direction "${FM_TEMPORAL_DIRECTION:-bidirectional}" \
  --SEQ_LEN "${SEQ_LEN:-8}" \
  --PRED_LEN "${PRED_LEN:-1}" \
  --SEED "${SEED:-42}" \
  --DATASET_RANDOM_SEED "${DATASET_RANDOM_SEED:-42}" \
  --CUDA_VISIBLE_DEVICES "${GPU:-0}" \
  --SPLIT_METHOD "${SPLIT_METHOD:-random_proportional_split}" \
  --TRAIN_RATIO "${TRAIN_RATIO:-0.7}" \
  --VALID_RATIO "${VALID_RATIO:-0.1}" \
  --FM_BATCH "${BATCH_SIZE:-128}" \
  --FM_EPOCHS "${EPOCHS:-100}" \
  --FM_LR "${LEARNING_RATE:-1e-4}" \
  --FM_WEIGHT_DECAY "${WEIGHT_DECAY:-1e-4}" \
  --FM_GRAD_CLIP "${GRAD_CLIP:-1.0}" \
  --FM_STOP_METRIC "${STOP_METRIC:-val_loss}" \
  --FM_EARLY_STOPPING_PATIENCE "${PATIENCE:-5}" \
  --FM_EARLY_STOPPING_MIN_DELTA "${MIN_DELTA:-0.0001}" \
  --FM_EARLY_STOPPING_START_EPOCH "${EARLY_STOP_START:-25}" \
  --MAX_SOURCE_CELLS_PER_DAY "${SOURCE_CELL_CAP:-2048}" \
  --CELL_ENCODER_HIDDEN "${CELL_HIDDEN:-256}" \
  --DAY_ENCODER_HIDDEN "${DAY_HIDDEN:-256}" \
  --TEMPORAL_HIDDEN "${TEMPORAL_HIDDEN:-256}" \
  --TYPE_EMB_DIM "${TYPE_EMB_DIM:-32}" \
  --TRAJ_HINT_HIDDEN "${TRAJ_HINT_HIDDEN:-128}" \
  --FLOW_CONTEXT_DIM "${FLOW_CONTEXT_DIM:-256}" \
  --FLOW_HIDDEN_DIM "${FLOW_HIDDEN_DIM:-256}" \
  --FM_BIO_ABLATIONS "${BIO_MODE:-bio7_adaptive_full}" \
  --FM_VARIANTS FM_context_gaussian_hist1_ctype_soft_traj1 \
  --FLOW_LOSS_WEIGHT "${FLOW_LOSS_WEIGHT:-1.0}" \
  --AUX_POINT_LOSS_WEIGHT "${AUX_POINT_LOSS_WEIGHT:-0.05}" \
  --TRAJ_SOFT_WEIGHT "${TRAJ_SOFT_WEIGHT:-0.0}" \
  --CURATED_GRAPH_REG_WEIGHT "${GRAPH_REG_WEIGHT:-1e-4}" \
  --FLOW_NUM_STEPS "${FLOW_STEPS:-8}" \
  --FLOW_NUM_SAMPLES "${FLOW_SAMPLES:-4}" \
  --FLOW_SIGMA "${FLOW_SIGMA:-1.0}" \
  --BASE_LOGVAR_MIN "${BASE_LOGVAR_MIN:--8.0}" \
  --BASE_LOGVAR_MAX "${BASE_LOGVAR_MAX:-3.0}" \
  --FM_PRIOR_GATE_MODE "${PRIOR_GATE_MODE:-conditional}" \
  --FM_PRIOR_GATE_INIT "${PRIOR_GATE_INIT:-0.10}" \
  --FM_PRIOR_DROPOUT "${PRIOR_DROPOUT:-0.10}" \
  --FM_PRIOR_WARMUP_EPOCHS "${PRIOR_WARMUP:-5}" \
  --FM_PRIOR_RAMP_EPOCHS "${PRIOR_RAMP:-10}" \
  --FM_CONTINUOUS_TIME_WEIGHT "${CONTINUOUS_TIME_WEIGHT:-0.10}" \
  --FM_CONTINUOUS_TIME_MIN_FRACTION "${CONTINUOUS_MIN:-0.05}" \
  --FM_CONTINUOUS_TIME_MAX_FRACTION "${CONTINUOUS_MAX:-0.95}" \
  --FM_COMPOSITION_LOSS_WEIGHT "${COMPOSITION_WEIGHT:-0.05}" \
  --FM_EVAL_READOUTS flow_sample point_head blend \
  --FM_BLEND_WEIGHTS 0.50 0.70 0.85 1.00 \
  --FM_PRIMARY_READOUT point_head \
  "$(training_bool_flag FP16 "${FP16:-False}")" \
  --SAVE_CHECKPOINTS \
  "$@"
