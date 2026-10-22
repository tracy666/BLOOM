"""Shared training arguments for BLOOM and forecasting baselines.

The common schema is importable without scientific Python dependencies. Model
families select the few historical optimization defaults that actually differ;
all datasets and model directions use this single parser.
"""

import argparse
import sys

DEFAULT_TRANS_BASE_MODELS = [
    "LightTS",
    "Crossformer",
    "TSMixer",
    "FreTS",
    "Autoformer",
    "DLinear",
    "FEDformer",
    "FiLM",
    "Informer",
    "MICN",
    "Nonstationary_Transformer",
    "PAttn",
    "PatchTST",
    "Reformer",
    "SegRNN",
    "TiDE",
    "TimeFilter",
    "TimeXer",
    "TimesNet",
    "Transformer",
    "WPMixer",
    "iTransformer",
    "Mamba",
    "Chronos",
    "Chronos2",
    "Sundial",
    "TiRex",
    "TimeMoE",
]

DEFAULT_TRANS_FOUNDATION_MODELS = [
    "Chronos",
    "Chronos2",
    "Sundial",
    "TiRex",
    "TimeMoE",
]

DEFAULT_FOUNDATION_CHECKPOINT_SPECS = []

FOUNDATION_CHECKPOINT_EXAMPLES = [
    "Chronos::bolt_tiny::amazon/chronos-bolt-tiny",
    "Chronos::bolt_mini::amazon/chronos-bolt-mini",
    "Chronos::bolt_small::amazon/chronos-bolt-small",
    "Chronos::bolt_base::amazon/chronos-bolt-base",
    "Chronos::t5_tiny::amazon/chronos-t5-tiny",
    "Chronos::t5_mini::amazon/chronos-t5-mini",
    "Chronos::t5_small::amazon/chronos-t5-small",
    "Chronos::t5_base::amazon/chronos-t5-base",
    "Chronos::t5_large::amazon/chronos-t5-large",
    "Chronos2::chronos2::amazon/chronos-2",
    "Chronos2::chronos2_synth::autogluon/chronos-2-synth",
    "Chronos2::chronos2_small::autogluon/chronos-2-small",
    "Sundial::base128m::thuml/sundial-base-128m",
    "TiRex::base::NX-AI/TiRex",
    "TimeMoE::50m::Maple728/TimeMoE-50M",
    "TimeMoE::200m::Maple728/TimeMoE-200M",
]

MODEL_FAMILIES = ("baseline", "unidirectional", "bidirectional")

DEFAULT_FM_VARIANT_SPECS = [
    "FM_context_gaussian_hist1_ctype_soft_traj1",
    "FM_context_gaussian_hist0_ctype_soft_traj1",
    "FM_context_gaussian_hist1_ctype_hard_traj1",
    "FM_context_gaussian_hist1_ctype_soft_traj0",
    "FM_context_gaussian_hist1_ctype_hard_traj0",
    "FM_context_gaussian_hist0_ctype_hard_traj1",
    "FM_context_gaussian_hist0_ctype_soft_traj0",
    "FM_context_gaussian_hist0_ctype_hard_traj0",
]


def build_parser(model_family: str = "unidirectional") -> argparse.ArgumentParser:
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {model_family!r}")
    parser = argparse.ArgumentParser(
        description="Train BLOOM or forecasting baselines on one prepared expression panel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES,
        default=model_family,
        help="Select forecasting or directional BLOOM training defaults.",
    )
    parser.add_argument("--ADATA_X_PATH", type=str, required=True)
    parser.add_argument("--ADATA_OBS_PATH", type=str, required=True)
    parser.add_argument("--ADATA_VAR_PATH", type=str, required=True)
    parser.add_argument("--TRAJ_PATH", type=str, default=None)
    parser.add_argument("--REPO_DIR", type=str, required=True)
    parser.add_argument("--OUTPUT_DIR", type=str, default=None)
    parser.add_argument("--MSG", type=str, default="unified_trans_fm_benchmark")

    parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0")
    parser.add_argument("--SEED", type=int, default=42)
    parser.add_argument("--FP16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--NUM_WORKERS", type=int, default=0)
    parser.add_argument(
        "--DETERMINISTIC_CUDNN", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--TORCH_MATMUL_PRECISION", type=str, default="high", choices=["highest", "high", "medium"]
    )

    parser.add_argument("--TIME_COL", type=str, default="time")
    parser.add_argument("--CELLTYPE_COL", type=str, default="class_label")
    parser.add_argument(
        "--SPLIT_METHOD",
        type=str,
        default="random_proportional_split",
        choices=[
            "deterministic_inorder_split",
            "random_proportional_split",
            "prespecified_test_timestamp_split",
        ],
    )
    parser.add_argument("--TRAIN_RATIO", type=float, default=0.7)
    parser.add_argument("--VALID_RATIO", type=float, default=0.1)
    parser.add_argument("--DATASET_RANDOM_SEED", type=int, default=42)
    parser.add_argument(
        "--PRESPECIFIED_TEST_TIMESTAMPS",
        type=int,
        nargs="*",
        default=[27, 28, 29, 30, 31, 32, 33, 34],
    )

    parser.add_argument(
        "--EXECUTION_MODE",
        type=str,
        default="train_eval",
        choices=["train_eval", "train_only", "eval_only"],
    )
    parser.add_argument("--EVAL_SPLIT", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--SAVE_CHECKPOINTS", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--CHECKPOINT_SUBDIR", type=str, default="checkpoints")
    parser.add_argument("--SUMMARY_SUBDIR", type=str, default="summaries")
    parser.add_argument("--LOG_SUBDIR", type=str, default="logs")

    parser.add_argument(
        "--SEQ_LEN",
        type=int,
        default=1,
        choices=[0, 1, 2, 4, 8],
        help=(
            "Unified history choice for this run. For TRANS: true encoder SEQ_LEN and must be >=1. "
            "For FM: hist0 means only previous day; hist1 uses exactly SEQ_LEN history days. "
            "If SEQ_LEN=0, hist1 FM variants are skipped by design."
        ),
    )
    parser.add_argument("--TRANS_LABEL_LEN", type=int, default=4)
    parser.add_argument("--PRED_LEN", type=int, default=1)

    parser.add_argument("--RUN_TRANS", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--TRANS_BATCH", type=int, default=256)
    parser.add_argument("--TRANS_EPOCHS", type=int, default=20)
    parser.add_argument("--TRANS_LR", type=float, default=1e-4)
    parser.add_argument("--TRANS_D_MODEL", type=int, default=128)
    parser.add_argument(
        "--TRANS_N_HEADS",
        type=int,
        default=8,
        help="Shared Time-Series-Library attention-head setting; models without attention may ignore it.",
    )
    parser.add_argument("--TRANS_E_LAYERS", type=int, default=2)
    parser.add_argument("--TRANS_D_LAYERS", type=int, default=1)
    parser.add_argument(
        "--TRANS_D_FF",
        type=int,
        default=16,
        help="Shared feed-forward width for forecasting model construction.",
    )
    parser.add_argument("--TRANS_GRAD_CLIP", type=float, default=1.0)
    parser.add_argument(
        "--TRANS_STOP_METRIC",
        type=str,
        default="val_loss",
        choices=[
            "val_loss",
            "val_pcc_cell",
            "val_pcc_gene",
            "val_pcc_cell_unique",
            "val_pcc_gene_unique",
        ],
    )
    parser.add_argument("--TRANS_EARLY_STOPPING_PATIENCE", type=int, default=5)
    parser.add_argument("--TRANS_EARLY_STOPPING_MIN_DELTA", type=float, default=0.0)
    parser.add_argument(
        "--TRANS_LOAD_BEST_AT_END", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--TRANS_BASE_MODELS",
        type=str,
        nargs="*",
        default=DEFAULT_TRANS_BASE_MODELS,
    )
    parser.add_argument(
        "--TRANS_FOUNDATION_MODELS", type=str, nargs="*", default=DEFAULT_TRANS_FOUNDATION_MODELS
    )
    parser.add_argument(
        "--FOUNDATION_RUN_MODES",
        type=str,
        nargs="*",
        default=["scratch", "zero_shot_no_ckpt"],
        choices=["scratch", "zero_shot_no_ckpt", "zero_shot_ckpt", "finetune_ckpt"],
    )
    parser.add_argument(
        "--FOUNDATION_CHECKPOINT_SPECS",
        type=str,
        nargs="*",
        default=DEFAULT_FOUNDATION_CHECKPOINT_SPECS,
        help=(
            "Repeated specs with format ModelName::Label::CheckpointPathOrHFId. "
            "The third field can be a HuggingFace repo id (for example amazon/chronos-bolt-tiny), "
            "a local HuggingFace snapshot folder, a .pt/.pth/.ckpt file, or a .safetensors file."
        ),
    )

    parser.add_argument(
        "--FOUNDATION_ENABLE_SCRATCH", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--FOUNDATION_ENABLE_ZERO_SHOT_NO_CKPT", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--FOUNDATION_LOCAL_FILES_ONLY", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--HF_CACHE_DIR", type=str, default=None)
    parser.add_argument(
        "--FOUNDATION_AUTO_SNAPSHOT_DOWNLOAD",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If a foundation checkpoint source looks like a HuggingFace repo id, try to resolve it with "
            "huggingface_hub.snapshot_download before model construction. This is useful when the upstream "
            "model accepts a local snapshot folder."
        ),
    )
    parser.add_argument(
        "--FOUNDATION_FORCE_GENERIC_LOAD",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Force generic state_dict/from_pretrained loading after model construction. Default False because "
            "many Time-Series-Library foundation wrappers load pretrained weights during __init__ from config fields; "
            "generic loading a HuggingFace repo id often fails."
        ),
    )
    parser.add_argument("--TRANS_SKIP_MODELS", type=str, nargs="*", default=[])
    parser.add_argument(
        "--TRANS_USE_CELLTYPE_CONDITIONING", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--TRANS_CELLTYPE_EMB_INIT_STD", type=float, default=0.0)
    parser.add_argument("--TRANS_CELLTYPE_EMB_SCALE", type=float, default=1.0)
    parser.add_argument(
        "--TRANS_RETRIEVAL_ENABLED", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--TRANS_RETRIEVAL_POOL_MODE",
        type=str,
        default="oracle_target_time",
        choices=["oracle_target_time", "train_time_only", "all_times"],
    )
    parser.add_argument(
        "--TRANS_RETRIEVAL_METRIC", type=str, default="cosine", choices=["cosine", "l2"]
    )
    parser.add_argument("--TRANS_RETRIEVAL_TOPK", type=int, default=1)
    parser.add_argument("--TRANS_RETRIEVAL_MAX_POOL_PER_TS", type=int, default=None)
    parser.add_argument(
        "--TRANS_EVAL_BOTH_WEIGHTINGS",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate TRANS in two ways: (1) the original trajectory-occurrence-weighted metric and "
            "(2) a unique-target-cell metric that averages duplicate predictions for the same target cell first. "
            "The original mean_cell_pcc/mean_gene_pcc fields are preserved; unique-cell fields are added."
        ),
    )
    parser.add_argument(
        "--EXISTING_CHECKPOINT_ROOT",
        type=str,
        default=None,
        help="Optional checkpoint root used by EXECUTION_MODE=eval_only. This lets a new evaluation run load checkpoints from an older run folder.",
    )
    parser.add_argument(
        "--ONLY_MODEL_TAGS",
        type=str,
        nargs="*",
        default=[],
        help="Optional exact model-tag filter, mainly used by the standalone aligned re-evaluation helper.",
    )

    parser.add_argument("--RUN_FM", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--FM_VARIANTS",
        type=str,
        nargs="*",
        default=DEFAULT_FM_VARIANT_SPECS,
        help=(
            "Explicit FM variant list controlling FM run order. "
            "Format: FM_<standard|context_gaussian>_hist<0|1>_ctype_<none|soft|hard>_traj<0|1>. "
            "Items starting with '#' are ignored. If this list is empty, the legacy Cartesian-product "
            "construction from FLOW_BASE_MODES/FM_CELLTYPE_MODES/FM_USE_HISTORY_OPTIONS/FM_USE_TRAJ_SOFT_OPTIONS is used."
        ),
    )
    parser.add_argument("--FM_BATCH", type=int, default=256)
    parser.add_argument("--FM_EPOCHS", type=int, default=20)
    parser.add_argument("--FM_LR", type=float, default=1e-4)
    parser.add_argument("--FM_WEIGHT_DECAY", type=float, default=1e-4)
    parser.add_argument("--FM_GRAD_CLIP", type=float, default=1.0)
    parser.add_argument(
        "--FM_STOP_METRIC",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_pcc_cell", "val_pcc_gene"],
    )
    parser.add_argument("--FM_EARLY_STOPPING_PATIENCE", type=int, default=5)
    parser.add_argument("--FM_EARLY_STOPPING_MIN_DELTA", type=float, default=0.0)
    parser.add_argument(
        "--FM_EARLY_STOPPING_START_EPOCH",
        type=int,
        default=1,
        help=(
            "First epoch at which FM early-stopping patience is allowed to accumulate. "
            "Best-checkpoint tracking remains active from epoch 1. The default 1 preserves "
            "the historical behavior; use 25 for the final bio7 warm-up/ramp protocol. "
            "If this exceeds FM_EPOCHS, training simply runs for all requested epochs."
        ),
    )
    parser.add_argument(
        "--FM_LOAD_BEST_AT_END", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--MAX_SOURCE_CELLS_PER_DAY", type=int, default=128)
    parser.add_argument("--CELL_ENCODER_HIDDEN", type=int, default=256)
    parser.add_argument("--DAY_ENCODER_HIDDEN", type=int, default=256)
    parser.add_argument("--TEMPORAL_HIDDEN", type=int, default=256)
    parser.add_argument("--TYPE_EMB_DIM", type=int, default=32)
    parser.add_argument(
        "--FM_TYPE_EMB_INIT_STD",
        type=float,
        default=0.0,
        help="Std for FM target cell-type embedding initialization. Default 0.0 preserves old behavior; 0.02 is useful for big data.",
    )
    parser.add_argument("--TRAJ_HINT_HIDDEN", type=int, default=128)
    parser.add_argument("--FLOW_CONTEXT_DIM", type=int, default=256)
    parser.add_argument("--FLOW_HIDDEN_DIM", type=int, default=256)
    parser.add_argument(
        "--FLOW_BASE_MODES",
        type=str,
        nargs="*",
        default=["standard", "context_gaussian"],
        choices=["standard", "context_gaussian"],
    )
    parser.add_argument("--FLOW_SIGMA", type=float, default=1.0)
    parser.add_argument("--FLOW_NUM_STEPS", type=int, default=8)
    parser.add_argument("--FLOW_NUM_SAMPLES", type=int, default=4)
    parser.add_argument("--FLOW_LOSS_WEIGHT", type=float, default=1.0)
    parser.add_argument("--AUX_POINT_LOSS_WEIGHT", type=float, default=0.05)
    parser.add_argument("--BASE_LOGVAR_MIN", type=float, default=-8.0)
    parser.add_argument("--BASE_LOGVAR_MAX", type=float, default=3.0)
    parser.add_argument(
        "--FM_CELLTYPE_MODES",
        type=str,
        nargs="*",
        default=["none", "soft", "hard"],
        choices=["none", "soft", "hard"],
    )
    parser.add_argument(
        "--FM_USE_HISTORY_OPTIONS", type=int, nargs="*", default=[0, 1], choices=[0, 1]
    )
    parser.add_argument(
        "--FM_USE_TRAJ_SOFT_OPTIONS", type=int, nargs="*", default=[0, 1], choices=[0, 1]
    )
    parser.add_argument("--TRAJ_SOFT_WEIGHT", type=float, default=0.05)
    parser.add_argument(
        "--ORACLE_TARGET_CELLTYPE_AT_EVAL", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument(
        "--FM_BIO_ABLATIONS",
        type=str,
        nargs="*",
        default=["bio0_baseline"],
        choices=[
            "bio0_baseline",
            "bio1_actual_time",
            "bio2_hierarchy",
            "bio3_curated_graph",
            "bio4_anatomy",
            "bio5_drivers",
            "bio6_grn",
            "bio_grn_only",
            "bio2_hierarchy_grn",
            "bio7_adaptive_full",
        ],
        help=(
            "Cumulative FM-only ablations. bio2 contains bio1; bio3 contains bio1+bio2; and so on. "
            "bio_grn_only and bio2_hierarchy_grn are focused GRN controls. bio7_adaptive_full contains the "
            "same six inputs as bio6, but fuses them through learned residual trust gates. The list is crossed "
            "with FM_VARIANTS. bio0 exactly keeps the old scalable-v3 encoder path."
        ),
    )
    parser.add_argument(
        "--BIO_HIERARCHY_COLS",
        type=str,
        nargs="*",
        default=[],
        help="Fine-to-coarse obs columns used as target ontology covariates. Missing values receive padding code 0.",
    )
    parser.add_argument(
        "--BIO_ACTUAL_TIME_COL",
        type=str,
        default="",
        help="Continuous/physical-time obs column. Empty falls back to TIME_COL; examples: Age or day.",
    )
    parser.add_argument(
        "--BIO_ANATOMY_COLS",
        type=str,
        nargs="*",
        default=[],
        help="Categorical anatomical/context obs columns, for example roi.",
    )
    parser.add_argument(
        "--BIO_NUMERIC_CONTEXT_COLS",
        type=str,
        nargs="*",
        default=[],
        help="Numeric or numeric-like context columns, for example somite_count. A missingness mask is added.",
    )
    parser.add_argument("--BIO_EMB_DIM", type=int, default=16)
    parser.add_argument("--BIO_PRIOR_HIDDEN", type=int, default=64)
    parser.add_argument(
        "--BIO_ALLOW_MISSING_COMPONENTS",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If false, a requested ablation is skipped when its required resource is absent; prevents duplicate pseudo-ablations.",
    )
    parser.add_argument(
        "--CURATED_GRAPH_PATH",
        type=str,
        default="",
        help="Optional CSV/TSV/Parquet/XLSX long edge table from the original study. Empty means unavailable.",
    )
    parser.add_argument("--CURATED_GRAPH_SHEET", type=str, default="")
    parser.add_argument("--CURATED_GRAPH_SOURCE_COL", type=str, default="source")
    parser.add_argument("--CURATED_GRAPH_TARGET_COL", type=str, default="target")
    parser.add_argument("--CURATED_GRAPH_WEIGHT_COL", type=str, default="")
    parser.add_argument(
        "--CURATED_GRAPH_NODES_PATH",
        type=str,
        default="",
        help="Optional node table from the author-prior conversion; includes isolated nodes absent from the edge table.",
    )
    parser.add_argument("--CURATED_GRAPH_NODE_COL", type=str, default="")
    parser.add_argument(
        "--CURATED_GRAPH_NODE_OBS_COLS",
        type=str,
        nargs="*",
        default=[],
        help="Obs columns composed into each cell's graph-node label. Empty uses CELLTYPE_COL.",
    )
    parser.add_argument("--CURATED_GRAPH_NODE_SEPARATOR", type=str, default="::")
    parser.add_argument(
        "--CURATED_GRAPH_CELL_MAP_PATH",
        type=str,
        default="",
        help=(
            "Optional author-prior cell-to-graph-node table. When supplied, this exact mapping takes precedence "
            "over composing labels from CURATED_GRAPH_NODE_OBS_COLS. TSV/TSV.GZ is streamed in chunks."
        ),
    )
    parser.add_argument("--CURATED_GRAPH_CELL_MAP_CELL_ID_COL", type=str, default="cell_id")
    parser.add_argument("--CURATED_GRAPH_CELL_MAP_NODE_COL", type=str, default="graph_node")
    parser.add_argument(
        "--CURATED_GRAPH_OBS_CELL_ID_COL",
        type=str,
        default="",
        help="Obs column containing cell IDs. Empty uses the obs index.",
    )
    parser.add_argument("--CURATED_GRAPH_CELL_MAP_CHUNKSIZE", type=int, default=1_000_000)
    parser.add_argument("--CURATED_GRAPH_MIN_CELL_COVERAGE", type=float, default=0.01)
    parser.add_argument("--CURATED_GRAPH_REG_WEIGHT", type=float, default=1e-4)
    parser.add_argument(
        "--DRIVER_PROGRAM_PATH",
        type=str,
        default="",
        help="Optional long table with graph node, gene, and optional score columns.",
    )
    parser.add_argument("--DRIVER_PROGRAM_SHEET", type=str, default="")
    parser.add_argument("--DRIVER_NODE_COL", type=str, default="node")
    parser.add_argument("--DRIVER_GENE_COL", type=str, default="gene")
    parser.add_argument("--DRIVER_SCORE_COL", type=str, default="")
    parser.add_argument(
        "--GRN_PATH",
        type=str,
        default="",
        help="Optional pySCENIC-style adjacency table (TF, target, importance) or compatible CSV/TSV/Parquet.",
    )
    parser.add_argument("--GRN_SOURCE_COL", type=str, default="")
    parser.add_argument("--GRN_TARGET_COL", type=str, default="")
    parser.add_argument("--GRN_WEIGHT_COL", type=str, default="")
    parser.add_argument("--GRN_TOPK_PER_TARGET", type=int, default=32)
    parser.add_argument("--GRN_MIN_WEIGHT", type=float, default=0.0)
    parser.add_argument(
        "--GRN_EXTERNAL_TF_MODE",
        type=str,
        choices=["drop", "target_gate"],
        default="target_gate",
        help=(
            "drop reproduces v4 by discarding regulators outside the FM genes. target_gate additionally retains "
            "their target connectivity as a fixed per-target structural gate without pretending their expression is observed."
        ),
    )
    parser.add_argument(
        "--BIO_PRIOR_VALIDATE_ONLY",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load, align, validate, and report biological priors, then exit before trajectory loading/training.",
    )

    parser.add_argument(
        "--FM_TEMPORAL_DIRECTION",
        type=str,
        default="forward",
        choices=["forward", "reverse", "bidirectional"],
        help=(
            "Direction used only by FM. forward predicts later cells from earlier context. reverse creates a "
            "separate control model that predicts earlier cells from later context. bidirectional trains one "
            "shared, direction-conditioned checkpoint on both causal directions. TRANS is never reversed."
        ),
    )
    parser.add_argument(
        "--FM_BIDIRECTIONAL_CONDITIONING",
        type=str,
        default="embedding_adapter",
        choices=["embedding", "embedding_adapter"],
        help=(
            "How a bidirectional FM distinguishes developmental direction. embedding adds a learned direction "
            "token to the shared context; embedding_adapter also adds a small zero-initialized residual adapter."
        ),
    )
    parser.add_argument(
        "--FM_BIDIRECTIONAL_ADAPTER_HIDDEN",
        type=int,
        default=64,
        help="Hidden width of the optional direction residual adapter.",
    )
    parser.add_argument(
        "--FM_BIDIRECTIONAL_FORWARD_FRACTION",
        type=float,
        default=0.50,
        help=(
            "Forward-example fraction in each bidirectional training epoch. Exact-ratio sampling prevents the "
            "direction with more target cells from dominating."
        ),
    )
    parser.add_argument(
        "--FM_BIDIRECTIONAL_SPLIT_POLICY",
        type=str,
        default="global_biological_targets",
        choices=["global_biological_targets"],
        help=(
            "Leakage-safe split policy. Biological stages are partitioned once, before direction-specific context "
            "eligibility is applied. For random_proportional_split only, a deterministic size-preserving repair "
            "supplies both-direction coverage when needed. A held-out biological target can never become an "
            "opposite-direction training target."
        ),
    )
    parser.add_argument(
        "--FM_PRIOR_GATE_MODE",
        type=str,
        default="conditional",
        choices=["global", "conditional"],
        help="Gate type used by bio7_adaptive_full; ignored by bio0-bio6.",
    )
    parser.add_argument(
        "--FM_PRIOR_GATE_INIT",
        type=float,
        default=0.10,
        help="Initial sigmoid trust probability for every bio7 prior gate.",
    )
    parser.add_argument(
        "--FM_PRIOR_DROPOUT",
        type=float,
        default=0.0,
        help="Whole-component dropout probability for bio7 residual priors during training.",
    )
    parser.add_argument(
        "--FM_PRIOR_GATE_L1_WEIGHT",
        type=float,
        default=0.0,
        help="Optional mean-gate penalty for bio7. Keep at 0 for the first final-setting screen.",
    )
    parser.add_argument(
        "--FM_PRIOR_WARMUP_EPOCHS",
        type=int,
        default=0,
        help="Epochs that train the bio0-compatible core before bio7 residual priors are enabled.",
    )
    parser.add_argument(
        "--FM_PRIOR_RAMP_EPOCHS",
        type=int,
        default=0,
        help="Epochs over which bio7 prior residual strength ramps from 0 to 1 after warm-up.",
    )
    parser.add_argument(
        "--FM_CONTINUOUS_TIME_WEIGHT",
        type=float,
        default=0.0,
        help=(
            "Weight of trajectory-linked developmental-time interpolation augmentation. A value >0 requires "
            "actual-time input and trajectory hints. It does not change FLOW_NUM_STEPS or the FM ODE tau."
        ),
    )
    parser.add_argument("--FM_CONTINUOUS_TIME_MIN_FRACTION", type=float, default=0.05)
    parser.add_argument("--FM_CONTINUOUS_TIME_MAX_FRACTION", type=float, default=0.95)
    parser.add_argument(
        "--FM_CONTEXT_GENE_DROPOUT",
        type=float,
        default=0.0,
        help="Optional per-sample gene-channel dropout on FM history during training; useful only as a robustness ablation.",
    )
    parser.add_argument(
        "--FM_CONTEXT_CELL_DROPOUT",
        type=float,
        default=0.0,
        help="Optional context-cell dropout during FM training; useful for sampling/composition robustness.",
    )
    parser.add_argument(
        "--FM_COMPOSITION_LOSS_WEIGHT",
        type=float,
        default=0.0,
        help=(
            "Weight for an unconditional next-stage cell-type composition head. This head uses population history "
            "without oracle target-cell labels and is separate from row-aligned expression evaluation."
        ),
    )
    parser.add_argument(
        "--FM_INIT_CHECKPOINT",
        type=str,
        default="",
        help=(
            "Optional best.pt used to warm-start a new FM. Shape-compatible keys are loaded and new conditioning modules "
            "remain initialized. Shared core weights can initialize an adaptive-prior model when their names and shapes match."
        ),
    )

    parser.add_argument(
        "--FM_RESIDUAL_MODE",
        type=str,
        default="none",
        choices=["none", "trajectory"],
        help=(
            "If 'trajectory', FM learns the residual target_x - trajectory_anchor instead of target_x. "
            "The final prediction is anchor + predicted_residual. This makes FM more cell-specific and closer "
            "to the TRANS trajectory-window task. Default 'none' keeps old behavior."
        ),
    )
    parser.add_argument(
        "--FM_PREDICTION_MODE",
        type=str,
        default="flow_sample",
        choices=["flow_sample", "point_head", "residual_point", "blend"],
        help=(
            "Prediction used for validation/test. 'flow_sample' preserves old stochastic FM inference. "
            "'point_head' is deterministic. 'residual_point' is deterministic and intended for "
            "FM_RESIDUAL_MODE=trajectory. 'blend' combines deterministic point prediction and flow-sample prediction."
        ),
    )
    parser.add_argument(
        "--FM_POINT_BLEND_WEIGHT",
        type=float,
        default=0.7,
        help="For --FM_PREDICTION_MODE blend: final = weight * point_prediction + (1-weight) * flow_prediction.",
    )
    parser.add_argument(
        "--FM_EVAL_READOUTS",
        type=str,
        nargs="*",
        default=[],
        help=(
            "Optional multi-readout evaluation from ONE trained FM. Supported entries: flow_sample, point_head, "
            "residual_point, blend, or explicit blend_<weight> such as blend_0.75. If 'blend' is present, "
            "all --FM_BLEND_WEIGHTS are evaluated. Empty preserves the old single FM_PREDICTION_MODE behavior."
        ),
    )
    parser.add_argument(
        "--FM_BLEND_WEIGHTS",
        type=float,
        nargs="*",
        default=[0.50, 0.70, 0.85, 1.00],
        help="Blend weights expanded when FM_EVAL_READOUTS contains 'blend'. Add/delete values freely.",
    )
    parser.add_argument(
        "--FM_PRIMARY_READOUT",
        type=str,
        default="point_head",
        help=(
            "Backward-compatible primary FM readout used for the legacy variant-level checkpoint/reporting fields. "
            "In multi-readout mode it NO LONGER controls early stopping: training stops only after no requested "
            "readout has improved for FM_EARLY_STOPPING_PATIENCE consecutive validation epochs."
        ),
    )
    parser.add_argument(
        "--FM_SAVE_BEST_PER_READOUT",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the best validation checkpoint separately for every requested FM readout while training only once.",
    )
    parser.add_argument(
        "--FM_RESIDUAL_REQUIRE_TRAJ_HINT",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If enabled with FM_RESIDUAL_MODE=trajectory, crash when a batch has no trajectory anchor. "
            "Default False safely falls back to non-residual prediction for those cells."
        ),
    )

    parser.add_argument(
        "--TRAJECTORY_SUBSAMPLE_MODE", type=str, default="none", choices=["none", "trajectory_rows"]
    )
    parser.add_argument("--TRAJECTORY_SUBSAMPLE_FRACTION", type=float, default=1.0)
    parser.add_argument("--TRAJECTORY_SUBSAMPLE_SEED", type=int, default=42)
    parser.add_argument("--TRAJECTORY_SUBSAMPLE_MIN_ROWS", type=int, default=1)
    parser.add_argument(
        "--TRAJECTORY_SUBSAMPLE_RESTRICT_FM_TARGETS",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--TRAJECTORY_SUBSAMPLE_RESTRICT_CONTEXT",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--FM_SCALE_EXPERIMENT", type=str, default="none", choices=["none", "A", "B", "C", "D", "E"]
    )
    parser.add_argument("--FM_SCALE_FRACTION", type=float, default=0.10)
    parser.add_argument("--FM_SCALE_SEED", type=int, default=42)
    parser.add_argument(
        "--FM_MAX_TRAIN_STEPS_PER_EPOCH",
        type=int,
        default=0,
        help="0 means exhaust the current FM train loader. A positive value caps optimizer steps per epoch; intended for experiment D.",
    )
    parser.add_argument(
        "--FM_SAVE_EPOCH_SUBSAMPLE_INDICES",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For rotating experiment C, save deterministic trajectory-row indices selected at every epoch.",
    )

    parser.add_argument("--SAVE_CSV", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--PCC_BACKEND", type=str, default="fast", choices=["fast", "scipy"])
    parser.add_argument("--PCC_NAN_POLICY", type=str, default="zero", choices=["zero", "skip"])
    parser.add_argument("--PCC_EPS", type=float, default=1e-12)
    parser.add_argument("--PCC_MIN_VALID_FRACTION", type=float, default=0.05)
    if model_family == "baseline":
        parser.set_defaults(
            TRANS_D_MODEL=64,
            TRANS_N_HEADS=4,
            TRANS_E_LAYERS=1,
            TRANS_D_LAYERS=1,
            TRANS_D_FF=16,
            TRANS_USE_CELLTYPE_CONDITIONING=False,
            FOUNDATION_CHECKPOINT_SPECS=list(FOUNDATION_CHECKPOINT_EXAMPLES),
        )
    return parser


def parse_args(
    argv: list[str] | None = None, model_family: str | None = None
) -> argparse.Namespace:
    """Parse the shared schema with an explicit or inferred model family."""
    values = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--model-family", choices=MODEL_FAMILIES)
    selector.add_argument(
        "--FM_TEMPORAL_DIRECTION", choices=["forward", "reverse", "bidirectional"]
    )
    selected, _ = selector.parse_known_args(values)
    family = (
        model_family
        or selected.model_family
        or (
            "bidirectional"
            if selected.FM_TEMPORAL_DIRECTION == "bidirectional"
            else "unidirectional"
        )
    )
    if model_family is not None and selected.model_family not in (None, model_family):
        raise ValueError("The requested model family conflicts with --model-family.")
    parser = build_parser(family)
    parsed = parser.parse_args(values)
    if family == "unidirectional" and parsed.FM_TEMPORAL_DIRECTION == "bidirectional":
        parser.error("Shared bidirectional training requires --model-family bidirectional.")
    return parsed
