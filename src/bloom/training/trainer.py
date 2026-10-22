"""Shared data preparation, optimization and checkpointing for BLOOM and baselines.

Importing this module does not parse command-line arguments, load expression
matrices, create output folders or begin training. ``load_runtime`` creates an
isolated run namespace; ``run`` executes the selected model family in it.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
import importlib.util
import uuid

from bloom.training.arguments import build_parser, parse_args

import argparse
import copy
import gc
import importlib
import json
import logging
import math
import os
import pickle
import random
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from bloom.models.transformer import TRANS_MODEL_REGISTRY
from bloom.models.transformer import CellTypeConditionWrapper
from bloom.models.bloom import bind_native_models

MODEL_FAMILY: str
ARGS: argparse.Namespace
DEVICE: torch.device
LOGGER: logging.Logger
BIO_PRIOR_DATA: Any
FM_DATA: Any
RESULT_ROWS: list[dict]

# Shared defaults; public recipes select one descriptive model family.
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
DEFAULT_TRANS_FOUNDATION_MODELS = ["Chronos", "Chronos2", "Sundial", "TiRex", "TimeMoE"]
DEFAULT_FOUNDATION_CHECKPOINT_SPECS = []
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
BIO_ABLATION_LEVEL = {
    "bio0_baseline": 0,
    "bio1_actual_time": 1,
    "bio2_hierarchy": 2,
    "bio3_curated_graph": 3,
    "bio4_anatomy": 4,
    "bio5_drivers": 5,
    "bio6_grn": 6,
    "bio_grn_only": 6,
    "bio2_hierarchy_grn": 6,
    "bio7_adaptive_full": 7,
}
BIO_ABLATION_COMPONENTS: Dict[str, Set[str]] = {
    "bio0_baseline": set(),
    "bio1_actual_time": {"actual_time"},
    "bio2_hierarchy": {"actual_time", "hierarchy"},
    "bio3_curated_graph": {"actual_time", "hierarchy", "graph"},
    "bio4_anatomy": {"actual_time", "hierarchy", "graph", "anatomy"},
    "bio5_drivers": {"actual_time", "hierarchy", "graph", "anatomy", "drivers"},
    "bio6_grn": {"actual_time", "hierarchy", "graph", "anatomy", "drivers", "grn"},
    "bio_grn_only": {"grn"},
    "bio2_hierarchy_grn": {"actual_time", "hierarchy", "grn"},
    "bio7_adaptive_full": {"actual_time", "hierarchy", "graph", "anatomy", "drivers", "grn"},
}
TRANS_THU_DEFAULTS = dict(
    task_name="long_term_forecast",
    is_training=True,
    model_id="gene",
    features="M",
    target="0",
    freq="h",
    inverse=False,
    seq_len=None,
    label_len=None,
    pred_len=None,
    enc_in=None,
    dec_in=None,
    c_out=None,
    d_model=64,
    n_heads=4,
    e_layers=1,
    d_layers=1,
    d_ff=32,
    factor=3,
    distil=True,
    embed="timeF",
    dropout=0.1,
    attn="prob",
    activation="gelu",
    output_attention=False,
    individual=False,
    mix=True,
    root_path="./",
    data_path="none",
    num_workers=4,
    moving_avg=25,
    top_k=5,
    num_kernels=6,
)


def _drop_commented_or_empty_items(x: Sequence[str]) -> List[str]:
    """
    Remove empty strings and strings that begin with '#'.

    This lets you do either of these safely:
      DEFAULT_TRANS_BASE_MODELS = [
          "TSMixer",
          # "Chronos",
      ]

    or, from shell arrays:
      TRANS_MODELS=("TSMixer" "#Chronos")
    """
    out: List[str] = []
    for item in list(x or []):
        item = str(item).strip()
        if item == "":
            continue
        if item.startswith("#"):
            continue
        out.append(item)
    return out


def effective_trans_label_len() -> int:
    """
    Return a label length that is safe for the current TRANS encoder length.

    Reason:
        The decoder-known segment is copied from the tail of the encoder.
        Therefore TRANS_LABEL_LEN must not exceed SEQ_LEN.  Instead of crashing,
        we clamp it here and log the effective value in the run config.

    Examples:
        SEQ_LEN=1  TRANS_LABEL_LEN=1  -> 1
        SEQ_LEN=2  TRANS_LABEL_LEN=1  -> 1
        SEQ_LEN=4  TRANS_LABEL_LEN=2  -> 2
        SEQ_LEN=8  TRANS_LABEL_LEN=4  -> 4
        SEQ_LEN=0  -> 0 because TRANS is skipped entirely.
    """
    if int(ARGS.SEQ_LEN) <= 0:
        return 0
    return int(max(1, min(int(ARGS.TRANS_LABEL_LEN), int(ARGS.SEQ_LEN))))


def set_global_seed(*, seed: int, deterministic_cudnn: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _script_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def setup_logger() -> Tuple[logging.Logger, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{ARGS.MSG}_log_{ts}.log")
    logger = logging.getLogger("unified_trans_fm_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.info("============================================================")
    logger.info("Unified TRANS + FM benchmark run started")
    logger.info(f"Log file      : {log_path}")
    logger.info(f"Run root      : {RUN_ROOT}")
    logger.info(f"Checkpoint dir: {CKPT_DIR}")
    logger.info(f"Summary dir   : {SUMMARY_DIR}")
    logger.info("============================================================")
    return (logger, log_path)


def pretty_md_table(rows: List[Dict[str, Any]], columns: List[str], floatfmt: str = ".4f") -> str:
    if not rows:
        return "(empty)"
    df = pd.DataFrame(rows)
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    df = df.loc[:, columns]
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return str(df)


def log_config(logger: logging.Logger) -> None:
    logger.info("----- Configuration -----")
    cfg_items = vars(ARGS)
    for k in sorted(cfg_items.keys()):
        logger.info(f"{k:>35}: {cfg_items[k]}")
    logger.info("-------------------------")


def make_worker_init_fn(base_seed: int):

    def _init_fn(worker_id: int):
        seed = int(base_seed) + int(worker_id)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    return _init_fn


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return pearsonr(x, y)[0]


def _nan_robust_mean(v: np.ndarray, *, policy: str, min_valid_fraction: float) -> float:
    v = np.asarray(v, dtype=np.float64)
    finite = np.isfinite(v)
    n = int(v.size)
    n_valid = int(finite.sum())
    if n == 0:
        return 0.0
    valid_fraction = float(n_valid / n)
    if policy == "zero":
        v2 = np.where(finite, v, 0.0)
        out = float(v2.mean())
        return out if np.isfinite(out) else 0.0
    if policy == "skip":
        if n_valid == 0 or valid_fraction < float(min_valid_fraction):
            return 0.0
        out = float(v[finite].mean())
        return out if np.isfinite(out) else 0.0
    raise ValueError(policy)


def pcc_cells_and_genes_fast(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    xn = x.detach().cpu().numpy().astype(np.float64, copy=False)
    yn = y.detach().cpu().numpy().astype(np.float64, copy=False)
    xc = xn - xn.mean(axis=1, keepdims=True)
    yc = yn - yn.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", xc, yc)
    den = np.sqrt(np.einsum("ij,ij->i", xc, xc) * np.einsum("ij,ij->i", yc, yc))
    cell_corr = np.empty_like(num, dtype=np.float64)
    valid = (den > float(ARGS.PCC_EPS)) & np.isfinite(den) & np.isfinite(num)
    cell_corr[valid] = num[valid] / den[valid]
    cell_corr[~valid] = np.nan
    xg = xn - xn.mean(axis=0, keepdims=True)
    yg = yn - yn.mean(axis=0, keepdims=True)
    numg = np.einsum("ij,ij->j", xg, yg)
    deng = np.sqrt(np.einsum("ij,ij->j", xg, xg) * np.einsum("ij,ij->j", yg, yg))
    gene_corr = np.empty_like(numg, dtype=np.float64)
    validg = (deng > float(ARGS.PCC_EPS)) & np.isfinite(deng) & np.isfinite(numg)
    gene_corr[validg] = numg[validg] / deng[validg]
    gene_corr[~validg] = np.nan
    mean_cell = _nan_robust_mean(
        cell_corr,
        policy=str(ARGS.PCC_NAN_POLICY),
        min_valid_fraction=float(ARGS.PCC_MIN_VALID_FRACTION),
    )
    mean_gene = _nan_robust_mean(
        gene_corr,
        policy=str(ARGS.PCC_NAN_POLICY),
        min_valid_fraction=float(ARGS.PCC_MIN_VALID_FRACTION),
    )
    return (float(mean_cell), float(mean_gene))


def pcc_cells_and_genes_scipy(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    xn = x.detach().cpu().numpy()
    yn = y.detach().cpu().numpy()
    cell_corr = np.array([safe_pearson(xn[i], yn[i]) for i in range(xn.shape[0])], dtype=np.float64)
    gene_corr = np.array(
        [safe_pearson(xn[:, j], yn[:, j]) for j in range(xn.shape[1])], dtype=np.float64
    )
    mean_cell = _nan_robust_mean(
        cell_corr,
        policy=str(ARGS.PCC_NAN_POLICY),
        min_valid_fraction=float(ARGS.PCC_MIN_VALID_FRACTION),
    )
    mean_gene = _nan_robust_mean(
        gene_corr,
        policy=str(ARGS.PCC_NAN_POLICY),
        min_valid_fraction=float(ARGS.PCC_MIN_VALID_FRACTION),
    )
    return (float(mean_cell), float(mean_gene))


def pcc_cells_and_genes(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    if str(ARGS.PCC_BACKEND) == "scipy":
        return pcc_cells_and_genes_scipy(x, y)
    return pcc_cells_and_genes_fast(x, y)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask_f = mask.to(x.dtype)
    while mask_f.ndim < x.ndim:
        mask_f = mask_f.unsqueeze(-1)
    num = (x * mask_f).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp_min(1.0)
    return num / den


def deterministic_sample(indices: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= int(max_n):
        return indices
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(indices, size=int(max_n), replace=False)
    sampled = np.asarray(sampled, dtype=np.int64)
    sampled.sort()
    return sampled


def is_memory_related_runtime_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "out of memory",
        "cuda out of memory",
        "cudnn_status_alloc_failed",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "cuda error: memory allocation",
        "can't allocate memory",
        "cannot allocate memory",
    ]
    return any((marker in text for marker in markers))


def release_memory_after_failure() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def make_zero_mark(x: torch.Tensor, mark_dim: int = 4) -> torch.Tensor:
    b, t, _ = x.shape
    return torch.zeros(b, t, mark_dim, dtype=torch.float32, device=x.device)


def load_pickle_or_npy(path: str):
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return list(arr)
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported file extension: {path}")


# Data alignment and developmental-stage partitions.
def make_timestamp_splits(
    *,
    total_timestamp: int,
    min_target_ts: int,
    split_method: str,
    train_ratio: float,
    valid_ratio: float,
    prespecified_test_timestamps: Sequence[int],
    dataset_random_seed: int,
) -> Tuple[Set[int], Set[int], Set[int]]:
    rng = random.Random(int(dataset_random_seed))
    if split_method == "deterministic_inorder_split":
        train_end = int(total_timestamp * float(train_ratio))
        val_end = train_end + int(total_timestamp * float(valid_ratio))
        train = set(range(max(int(min_target_ts), 0), train_end))
        val = set(range(max(int(min_target_ts), train_end), val_end))
        test = set(range(max(int(min_target_ts), val_end), int(total_timestamp)))
        return (train, val, test)
    if split_method == "random_proportional_split":
        timestamps = list(range(int(min_target_ts), int(total_timestamp)))
        n_test = int(total_timestamp * (1.0 - float(train_ratio) - float(valid_ratio)))
        n_val = int(total_timestamp * float(valid_ratio))
        n_test = max(1, min(n_test, len(timestamps))) if len(timestamps) > 0 else 0
        test = set(rng.sample(timestamps, k=n_test)) if n_test > 0 else set()
        val: Set[int] = set()
        distance = 1
        while len(val) < n_val and distance <= int(total_timestamp):
            for t in sorted(test):
                cand = int(t) - int(distance)
                if cand >= int(min_target_ts) and cand not in test and (cand not in val):
                    val.add(cand)
                    if len(val) >= n_val:
                        break
            distance += 1
        if len(val) < n_val:
            remaining = [ts for ts in timestamps if ts not in test and ts not in val]
            if test:
                remaining.sort(key=lambda ts: min((abs(int(ts) - int(t)) for t in test)))
            val.update(remaining[: max(0, n_val - len(val))])
        train = set((ts for ts in timestamps if ts not in val and ts not in test))
        return (train, val, test)
    if split_method == "prespecified_test_timestamp_split":
        if not prespecified_test_timestamps:
            raise ValueError("prespecified_test_timestamps required")
        test = {int(t) for t in prespecified_test_timestamps}
        if min(test) < int(min_target_ts):
            raise ValueError("Provided test timestamp < required minimum target timestamp")
        n_val = int(total_timestamp * float(valid_ratio))
        val: Set[int] = set()
        distance = 1
        while len(val) < n_val and distance <= int(total_timestamp):
            for t in sorted(test):
                cand = int(t) - int(distance)
                if cand >= int(min_target_ts) and cand not in test and (cand not in val):
                    val.add(cand)
                    if len(val) >= n_val:
                        break
            distance += 1
        if len(val) < n_val:
            distance = 1
            while len(val) < n_val and distance <= int(total_timestamp):
                for t in sorted(test):
                    cand = int(t) + int(distance)
                    if cand < int(total_timestamp) and cand not in test and (cand not in val):
                        val.add(cand)
                        if len(val) >= n_val:
                            break
                distance += 1
        train = {
            ts
            for ts in range(int(min_target_ts), int(total_timestamp))
            if ts not in test and ts not in val
        }
        return (train, val, test)
    raise ValueError(f"Unknown split_method={split_method!r}")


def repair_bidirectional_stage_split_coverage(
    *,
    train_timestamps: Set[int],
    val_timestamps: Set[int],
    test_timestamps: Set[int],
    total_timestamp: int,
    history_days: int,
) -> Tuple[Set[int], Set[int], Set[int], Dict[str, Any]]:
    """
    Preserve split sizes while making every partition evaluable in both
    developmental directions.

    A stage is forward-eligible when L earlier stages exist and reverse-eligible
    when L later stages exist. Whenever possible, one interior stage eligible
    in both directions is anchored in train, validation, and test. Remaining
    stages preferentially stay in their initial partition. No stage is
    duplicated, so biological target-stage disjointness is preserved.

    If a future dataset lacks enough interior stages, failing early is safer
    than selecting a "bidirectional" checkpoint from one-direction validation.
    """
    split_names = ("train", "val", "test")
    original: Dict[str, Set[int]] = {
        "train": {int(x) for x in train_timestamps},
        "val": {int(x) for x in val_timestamps},
        "test": {int(x) for x in test_timestamps},
    }
    sizes = {name: len(original[name]) for name in split_names}
    if any((sizes[name] <= 0 for name in split_names)):
        raise RuntimeError(
            f"Bidirectional FM requires non-empty train/val/test stage sets, got {sizes}"
        )
    expected_stages = set(range(int(total_timestamp)))
    observed_stages = set().union(*(original[name] for name in split_names))
    if observed_stages != expected_stages:
        missing = sorted(expected_stages - observed_stages)
        extra = sorted(observed_stages - expected_stages)
        raise RuntimeError(
            f"Bidirectional split repair expects one complete global stage partition; missing={missing}, extra={extra}"
        )
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            if original[left] & original[right]:
                raise RuntimeError(
                    f"Bidirectional stage partitions overlap: {left}/{right}={sorted(original[left] & original[right])}"
                )
    l = int(history_days)
    t_total = int(total_timestamp)

    def coverage(stages: Set[int]) -> Tuple[bool, bool]:
        has_forward = any((int(t) >= l for t in stages))
        has_reverse = any((int(t) <= t_total - 1 - l for t in stages))
        return (bool(has_forward), bool(has_reverse))

    before_coverage = {name: coverage(original[name]) for name in split_names}
    if all((all(flags) for flags in before_coverage.values())):
        report = {
            "applied": False,
            "history_days": l,
            "before": {name: sorted(original[name]) for name in split_names},
            "after": {name: sorted(original[name]) for name in split_names},
            "moved_in": {name: [] for name in split_names},
            "moved_out": {name: [] for name in split_names},
        }
        return (original["train"], original["val"], original["test"], report)
    both_eligible = set(range(l, t_total - l))
    if len(both_eligible) < len(split_names):
        raise RuntimeError(
            f"Cannot construct three bidirectionally evaluable stage partitions: TOTAL_TIMESTAMP={t_total}, history_days={l}, both-direction-eligible stages={sorted(both_eligible)}. Reduce SEQ_LEN or use separate forward/reverse models."
        )
    anchors: Dict[str, int] = {}
    used_anchors: Set[int] = set()
    for name in split_names:
        candidates = sorted((original[name] & both_eligible) - used_anchors)
        if candidates:
            anchors[name] = int(candidates[0])
            used_anchors.add(int(candidates[0]))
    for name in split_names:
        if name in anchors:
            continue
        candidates = sorted(both_eligible - used_anchors)
        if not candidates:
            raise RuntimeError("Bidirectional split repair ran out of unique interior anchors")
        candidates.sort(
            key=lambda stage: (
                -max(
                    (
                        len(original[owner] & both_eligible)
                        for owner in split_names
                        if int(stage) in original[owner]
                    )
                ),
                int(stage),
            )
        )
        anchors[name] = int(candidates[0])
        used_anchors.add(int(candidates[0]))
    repaired: Dict[str, Set[int]] = {name: {int(anchors[name])} for name in split_names}
    remaining = expected_stages - used_anchors
    for name in split_names:
        capacity = int(sizes[name] - len(repaired[name]))
        keep = sorted(original[name] & remaining)[:capacity]
        repaired[name].update((int(x) for x in keep))
        remaining.difference_update(keep)
    for name in split_names:
        capacity = int(sizes[name] - len(repaired[name]))
        take = sorted(remaining)[:capacity]
        repaired[name].update((int(x) for x in take))
        remaining.difference_update(take)
    if remaining:
        raise RuntimeError(
            f"Bidirectional split repair left unassigned stages: {sorted(remaining)}"
        )
    after_sizes = {name: len(repaired[name]) for name in split_names}
    if after_sizes != sizes:
        raise RuntimeError(
            f"Bidirectional split repair changed partition sizes: before={sizes}, after={after_sizes}"
        )
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            if repaired[left] & repaired[right]:
                raise RuntimeError("Bidirectional split repair produced overlapping partitions")
    if set().union(*(repaired[name] for name in split_names)) != expected_stages:
        raise RuntimeError(
            "Bidirectional split repair did not preserve the complete stage universe"
        )
    after_coverage = {name: coverage(repaired[name]) for name in split_names}
    if not all((all(flags) for flags in after_coverage.values())):
        raise RuntimeError(
            f"Bidirectional split repair failed direction coverage: {after_coverage}"
        )
    report = {
        "applied": True,
        "history_days": l,
        "before": {name: sorted(original[name]) for name in split_names},
        "after": {name: sorted(repaired[name]) for name in split_names},
        "moved_in": {name: sorted(repaired[name] - original[name]) for name in split_names},
        "moved_out": {name: sorted(original[name] - repaired[name]) for name in split_names},
    }
    return (repaired["train"], repaired["val"], repaired["test"], report)


def read_obs_or_var_table(path: str) -> pd.DataFrame:
    """Read the existing Parquet format, with CSV/TSV support for portable smoke tests."""
    path = str(path)
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path, sep="\t", index_col=0)
    if lower.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, index_col=0)
    raise ValueError(f"ADATA_OBS_PATH/ADATA_VAR_PATH must be Parquet, CSV, or TSV: {path}")


@dataclass
class BiologicalPriorData:
    gene_names: List[str]
    actual_time_by_ordinal: np.ndarray
    hierarchy_codes: List[np.ndarray]
    hierarchy_cardinalities: List[int]
    hierarchy_columns: List[str]
    anatomy_codes: List[np.ndarray]
    anatomy_cardinalities: List[int]
    anatomy_columns: List[str]
    numeric_context: np.ndarray
    numeric_context_columns: List[str]
    graph_cell_codes: np.ndarray
    graph_node_features: np.ndarray
    graph_edges: np.ndarray
    graph_edge_weights: np.ndarray
    graph_node_names: List[str]
    driver_program_by_node: np.ndarray
    grn_adjacency: np.ndarray
    grn_target_support: np.ndarray
    available: Dict[str, bool]
    reports: Dict[str, Any]


def _read_prior_table(path: str, sheet_name: str = "") -> pd.DataFrame:
    path = str(path or "").strip()
    if path == "":
        raise ValueError("Empty prior-table path")
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name if sheet_name else 0)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if path.lower().endswith((".tsv.gz", ".txt.gz")):
        return pd.read_csv(path, sep="\t")
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported prior table extension for {path!r}")


def _resolve_column(
    df: pd.DataFrame, requested: str, candidates: Sequence[str], *, required: bool
) -> Optional[str]:
    lower_to_original = {str(c).lower(): str(c) for c in df.columns}
    requested = str(requested or "").strip()
    if requested:
        if requested in df.columns:
            return requested
        if requested.lower() in lower_to_original:
            return lower_to_original[requested.lower()]
        if required:
            raise KeyError(
                f"Requested column {requested!r} not found; available={list(df.columns)}"
            )
        return None
    for cand in candidates:
        if str(cand).lower() in lower_to_original:
            return lower_to_original[str(cand).lower()]
    if required:
        raise KeyError(
            f"Could not infer required column from {list(candidates)}; available={list(df.columns)}"
        )
    return None


def _hvg_gene_names() -> List[str]:
    hvg_var = ADATA_VAR.loc[HVG_MASK]
    candidate_columns = (
        "gene_short_name",
        "gene_name",
        "gene",
        "gene_symbol",
        "symbol",
        "feature_name",
        "features",
        "gene_id",
        "ensembl_id",
        "_index",
        "index",
    )
    for candidate_column in candidate_columns:
        if candidate_column not in hvg_var.columns:
            continue
        values = hvg_var[candidate_column].astype(str).tolist()
        if len(values) != len(set(values)):
            LOGGER.warning(f"[bio][gene] ignoring non-unique gene-name column {candidate_column!r}")
            continue
        LOGGER.info(f"[bio][gene] using ADATA_VAR[{candidate_column!r}] for {len(values)} FM genes")
        return values
    values = [str(value) for value in hvg_var.index.tolist()]
    if isinstance(hvg_var.index, pd.RangeIndex) or len(values) != len(set(values)):
        raise ValueError(
            "Could not obtain unique FM gene names. The big dataset requires ADATA_VAR['gene_short_name'] to align with the pySCENIC adjacency."
        )
    LOGGER.info(f"[bio][gene] using ADATA_VAR index for {len(values)} FM genes")
    return values


def _encode_obs_category(column: str) -> Tuple[np.ndarray, int]:
    if column not in ADATA_OBS.columns:
        raise KeyError(f"Biological-prior column {column!r} not found in obs")
    cat = ADATA_OBS[column].astype("category")
    codes = cat.cat.codes.to_numpy(dtype=np.int32, copy=True) + 1
    cardinality = int(len(cat.cat.categories) + 1)
    return (codes, cardinality)


def _numeric_series_memory_safe(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Parse numeric-like obs values without expanding a categorical column to millions of strings."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = series.cat.categories
        parsed_categories = pd.to_numeric(pd.Series(categories), errors="coerce").to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(parsed_categories).any():
            parsed_categories = (
                pd.Series(categories.astype(str))
                .str.extract("([-+]?\\d*\\.?\\d+)", expand=False)
                .astype(float)
                .to_numpy(dtype=np.float32)
            )
        cat_codes = series.cat.codes.to_numpy(dtype=np.int64, copy=False)
        out = np.full((len(series),), np.nan, dtype=np.float32)
        valid_code = cat_codes >= 0
        out[valid_code] = parsed_categories[cat_codes[valid_code]]
    else:
        parsed = pd.to_numeric(series, errors="coerce")
        if int(parsed.notna().sum()) == 0:
            parsed = (
                series.astype(str).str.extract("([-+]?\\d*\\.?\\d+)", expand=False).astype(float)
            )
        out = parsed.to_numpy(dtype=np.float32, copy=True)
    mask = np.isfinite(out).astype(np.float32)
    return (out, mask)


def _normalized_numeric_obs(column: str) -> Tuple[np.ndarray, np.ndarray]:
    values, mask = _numeric_series_memory_safe(ADATA_OBS[column])
    finite = mask > 0
    if np.any(finite):
        center = float(np.nanmean(values[finite]))
        scale = float(np.nanstd(values[finite]))
        if not np.isfinite(scale) or scale < 1e-08:
            scale = 1.0
        values[finite] = (values[finite] - center) / scale
    values[~finite] = 0.0
    return (values.astype(np.float32, copy=False), mask.astype(np.float32, copy=False))


def _actual_time_by_ordinal() -> np.ndarray:
    actual_col = str(ARGS.BIO_ACTUAL_TIME_COL).strip() or str(ARGS.TIME_COL)
    if actual_col not in ADATA_OBS.columns:
        raise KeyError(f"BIO_ACTUAL_TIME_COL={actual_col!r} not found in obs")
    raw, mask = _numeric_series_memory_safe(ADATA_OBS[actual_col])
    result = np.arange(TOTAL_TIMESTAMP, dtype=np.float32)
    for t in range(TOTAL_TIMESTAMP):
        local = raw[(ORDINAL_TIME == int(t)) & (mask > 0)]
        if local.size > 0:
            result[t] = float(np.median(local))
    center = float(np.mean(result))
    scale = float(np.std(result))
    if scale < 1e-08:
        scale = 1.0
    return ((result - center) / scale).astype(np.float32)


def _map_obs_labels_to_graph_codes(
    node_to_code: Dict[str, int], obs_columns: Sequence[str], separator: str
) -> np.ndarray:
    columns = list(obs_columns) if len(obs_columns) > 0 else [CELLTYPE_COL]
    for col in columns:
        if col not in ADATA_OBS.columns:
            raise KeyError(f"CURATED_GRAPH_NODE_OBS_COLS contains missing obs column {col!r}")
    if len(columns) == 1:
        series = ADATA_OBS[columns[0]]
        if isinstance(series.dtype, pd.CategoricalDtype):
            lookup = np.asarray(
                [node_to_code.get(str(x), 0) for x in series.cat.categories], dtype=np.int32
            )
            raw_codes = series.cat.codes.to_numpy(dtype=np.int64, copy=False)
            out = np.zeros((len(series),), dtype=np.int32)
            valid = raw_codes >= 0
            out[valid] = lookup[raw_codes[valid]]
            return out
        mapped = series.map(lambda x: node_to_code.get(str(x), 0))
        return mapped.to_numpy(dtype=np.int32, copy=True)
    LOGGER.warning(
        "Composing multiple graph-node obs columns can require substantial temporary memory on very large datasets."
    )
    labels = ADATA_OBS[columns[0]].astype(str)
    for col in columns[1:]:
        labels = labels.str.cat(ADATA_OBS[col].astype(str), sep=str(separator))
    return labels.map(lambda x: node_to_code.get(str(x), 0)).to_numpy(dtype=np.int32, copy=True)


def _iter_cell_map_chunks(path: str, cell_id_col: str, node_col: str, chunksize: int):
    """Yield only the two required columns; stream text maps to cap peak memory."""
    lower = str(path).lower()
    usecols = [str(cell_id_col), str(node_col)]
    if lower.endswith(".parquet"):
        yield pd.read_parquet(path, columns=usecols)
        return
    if lower.endswith((".xlsx", ".xls")):
        yield pd.read_excel(path, usecols=usecols)
        return
    if lower.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")):
        reader = pd.read_csv(path, sep="\t", usecols=usecols, chunksize=max(1, int(chunksize)))
    elif lower.endswith((".csv", ".csv.gz", ".gz")):
        reader = pd.read_csv(path, usecols=usecols, chunksize=max(1, int(chunksize)))
    else:
        raise ValueError(f"Unsupported cell-map extension for {path!r}")
    yield from reader


def _map_external_cell_ids_to_graph_codes(
    node_to_code: Dict[str, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Align the conversion script's exact cell_id -> graph_node mapping to obs.

    The usual fast path detects that the generated map is already in obs row
    order and avoids constructing a hash table for all 11 million cell IDs. If
    any chunk is out of order, exact ID alignment is enabled lazily.
    """
    path = str(ARGS.CURATED_GRAPH_CELL_MAP_PATH).strip()
    cell_id_col = str(ARGS.CURATED_GRAPH_CELL_MAP_CELL_ID_COL)
    node_col = str(ARGS.CURATED_GRAPH_CELL_MAP_NODE_COL)
    obs_cell_id_col = str(ARGS.CURATED_GRAPH_OBS_CELL_ID_COL).strip()
    if obs_cell_id_col:
        if obs_cell_id_col not in ADATA_OBS.columns:
            raise KeyError(f"CURATED_GRAPH_OBS_CELL_ID_COL={obs_cell_id_col!r} not found in obs")
        obs_ids = pd.Index(ADATA_OBS[obs_cell_id_col], name=obs_cell_id_col)
    else:
        obs_ids = ADATA_OBS.index
    if len(obs_ids) != N_CELLS:
        raise ValueError(f"obs cell-ID length {len(obs_ids):,} != N_CELLS {N_CELLS:,}")
    if not obs_ids.is_unique:
        examples = obs_ids[obs_ids.duplicated()].unique()[:10].tolist()
        raise ValueError(
            f"Obs cell IDs must be unique for exact author-map alignment; examples={examples}"
        )
    codes = np.zeros((N_CELLS,), dtype=np.int32)
    assigned = np.zeros((N_CELLS,), dtype=np.bool_)
    row_cursor = 0
    total_rows = 0
    id_matches = 0
    duplicate_rows = 0
    conflicting_rows = 0
    unknown_nodes = 0
    fallback_id_alignment = False
    for chunk in _iter_cell_map_chunks(
        path, cell_id_col, node_col, int(ARGS.CURATED_GRAPH_CELL_MAP_CHUNKSIZE)
    ):
        if cell_id_col not in chunk.columns or node_col not in chunk.columns:
            raise KeyError(
                f"Cell map requires columns {cell_id_col!r}, {node_col!r}; available={list(chunk.columns)}"
            )
        n_chunk = int(len(chunk))
        if n_chunk == 0:
            continue
        map_ids = chunk[cell_id_col].to_numpy(copy=False)
        can_try_ordered = not fallback_id_alignment and row_cursor + n_chunk <= N_CELLS
        ordered = False
        if can_try_ordered:
            expected = np.asarray(obs_ids[row_cursor : row_cursor + n_chunk])
            if expected.dtype == map_ids.dtype:
                ordered = bool(np.array_equal(expected, map_ids))
            else:
                ordered = bool(
                    np.array_equal(expected.astype(str), np.asarray(map_ids).astype(str))
                )
        if ordered:
            positions = np.arange(row_cursor, row_cursor + n_chunk, dtype=np.int64)
        else:
            if not fallback_id_alignment:
                LOGGER.warning(
                    "[bio][graph-map] row order differs from obs; switching to exact cell-ID alignment"
                )
                fallback_id_alignment = True
            positions = obs_ids.get_indexer(pd.Index(map_ids))
        node_values = chunk[node_col].to_numpy(copy=False)
        node_codes = np.fromiter(
            (node_to_code.get(str(value), 0) if pd.notna(value) else 0 for value in node_values),
            dtype=np.int32,
            count=n_chunk,
        )
        valid_pos = positions >= 0
        id_matches += int(np.sum(valid_pos))
        unknown_nodes += int(np.sum(valid_pos & (node_codes <= 0)))
        for pos, code in zip(positions[valid_pos], node_codes[valid_pos]):
            pos_i = int(pos)
            code_i = int(code)
            if assigned[pos_i]:
                duplicate_rows += 1
                if int(codes[pos_i]) != code_i:
                    conflicting_rows += 1
                continue
            codes[pos_i] = code_i
            assigned[pos_i] = True
        total_rows += n_chunk
        row_cursor += n_chunk
    if total_rows == 0:
        raise ValueError(f"Author cell-map is empty: {path}")
    if conflicting_rows > 0:
        raise ValueError(
            f"Author cell-map contains {conflicting_rows:,} conflicting duplicate cell IDs"
        )
    report = {
        "cell_map_path": path,
        "cell_map_rows": int(total_rows),
        "cell_map_id_matches": int(id_matches),
        "cell_map_id_coverage": float(id_matches / max(1, N_CELLS)),
        "cell_map_assigned_cells": int(np.sum(assigned)),
        "cell_map_known_node_cells": int(np.sum(codes > 0)),
        "cell_map_unknown_node_rows": int(unknown_nodes),
        "cell_map_duplicate_rows": int(duplicate_rows),
        "cell_map_used_id_fallback": bool(fallback_id_alignment),
        "obs_cell_id_source": obs_cell_id_col if obs_cell_id_col else "obs.index",
    }
    return (codes, report)


def _directed_graph_features(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    """Return [depth, indegree, outdegree, ancestors, descendants, is_terminal] for codes 0..N."""
    parents: List[Set[int]] = [set() for _ in range(n_nodes + 1)]
    children: List[Set[int]] = [set() for _ in range(n_nodes + 1)]
    for src, dst in np.asarray(edges, dtype=np.int64):
        if src <= 0 or dst <= 0 or src == dst:
            continue
        children[int(src)].add(int(dst))
        parents[int(dst)].add(int(src))
    depth = np.zeros((n_nodes + 1,), dtype=np.float32)
    roots = [i for i in range(1, n_nodes + 1) if len(parents[i]) == 0]
    queue: List[int] = list(roots)
    seen = set(roots)
    while queue:
        node = queue.pop(0)
        for child in children[node]:
            depth[child] = max(depth[child], depth[node] + 1.0)
            if child not in seen:
                queue.append(child)
                seen.add(child)

    def reachable(start: int, links: List[Set[int]]) -> int:
        visited: Set[int] = set()
        stack = list(links[start])
        while stack:
            node = stack.pop()
            if node in visited or node == start:
                continue
            visited.add(node)
            stack.extend(links[node])
        return len(visited)

    features = np.zeros((n_nodes + 1, 6), dtype=np.float32)
    for node in range(1, n_nodes + 1):
        features[node] = np.asarray(
            [
                depth[node],
                len(parents[node]),
                len(children[node]),
                reachable(node, parents),
                reachable(node, children),
                float(len(children[node]) == 0),
            ],
            dtype=np.float32,
        )
    for col in range(5):
        vmax = float(np.max(features[:, col]))
        if vmax > 0:
            features[:, col] /= vmax
    return features


def build_biological_prior_data() -> BiologicalPriorData:
    requested_level = max([BIO_ABLATION_LEVEL[x] for x in ARGS.FM_BIO_ABLATIONS], default=0)
    requested_components: Set[str] = set()
    for tag in ARGS.FM_BIO_ABLATIONS:
        requested_components.update(BIO_ABLATION_COMPONENTS[str(tag)])
    gene_names = _hvg_gene_names()
    reports: Dict[str, Any] = {"requested_level": int(requested_level)}
    available = {
        "actual_time": False,
        "hierarchy": False,
        "graph": False,
        "anatomy": False,
        "drivers": False,
        "grn": False,
    }
    actual_time = (
        _actual_time_by_ordinal()
        if "actual_time" in requested_components
        else np.arange(TOTAL_TIMESTAMP, dtype=np.float32)
    )
    available["actual_time"] = "actual_time" in requested_components
    hierarchy_codes: List[np.ndarray] = []
    hierarchy_cards: List[int] = []
    hierarchy_cols: List[str] = []
    if "hierarchy" in requested_components:
        for col in ARGS.BIO_HIERARCHY_COLS:
            if col not in ADATA_OBS.columns:
                LOGGER.warning(f"[bio] hierarchy column unavailable and ignored: {col}")
                continue
            codes, card = _encode_obs_category(col)
            hierarchy_codes.append(codes)
            hierarchy_cards.append(card)
            hierarchy_cols.append(str(col))
        available["hierarchy"] = len(hierarchy_codes) > 0
        reports["hierarchy_cardinalities"] = dict(zip(hierarchy_cols, hierarchy_cards))
    graph_cell_codes = np.zeros((N_CELLS,), dtype=np.int32)
    graph_node_features = np.zeros((1, 6), dtype=np.float32)
    graph_edges = np.zeros((0, 2), dtype=np.int64)
    graph_edge_weights = np.zeros((0,), dtype=np.float32)
    graph_node_names = ["__MISSING__"]
    node_to_code: Dict[str, int] = {}
    if "graph" in requested_components and str(ARGS.CURATED_GRAPH_PATH).strip():
        graph_df = _read_prior_table(str(ARGS.CURATED_GRAPH_PATH), str(ARGS.CURATED_GRAPH_SHEET))
        src_col = _resolve_column(
            graph_df,
            str(ARGS.CURATED_GRAPH_SOURCE_COL),
            ["source", "from", "parent", "antecedent"],
            required=True,
        )
        dst_col = _resolve_column(
            graph_df,
            str(ARGS.CURATED_GRAPH_TARGET_COL),
            ["target", "to", "child", "descendant"],
            required=True,
        )
        weight_col = _resolve_column(
            graph_df,
            str(ARGS.CURATED_GRAPH_WEIGHT_COL),
            ["weight", "score", "confidence"],
            required=False,
        )
        valid = graph_df[src_col].notna() & graph_df[dst_col].notna()
        graph_df = graph_df.loc[
            valid, [src_col, dst_col] + ([weight_col] if weight_col else [])
        ].copy()
        src_names = graph_df[src_col].astype(str).to_numpy()
        dst_names = graph_df[dst_col].astype(str).to_numpy()
        unique_node_set = set(src_names.tolist()) | set(dst_names.tolist())
        if str(ARGS.CURATED_GRAPH_NODES_PATH).strip():
            graph_nodes_df = _read_prior_table(str(ARGS.CURATED_GRAPH_NODES_PATH))
            graph_node_col = _resolve_column(
                graph_nodes_df,
                str(ARGS.CURATED_GRAPH_NODE_COL),
                ["node", "graph_node", "name", "label"],
                required=True,
            )
            unique_node_set.update(graph_nodes_df[graph_node_col].dropna().astype(str).tolist())
            reports["graph_nodes_table_rows"] = int(len(graph_nodes_df))
        unique_nodes = sorted(unique_node_set)
        graph_node_names = ["__MISSING__"] + unique_nodes
        node_to_code = {name: i + 1 for i, name in enumerate(unique_nodes)}
        graph_edges = np.asarray(
            [(node_to_code[s], node_to_code[t]) for s, t in zip(src_names, dst_names)],
            dtype=np.int64,
        )
        if weight_col:
            graph_edge_weights = (
                pd.to_numeric(graph_df[weight_col], errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=np.float32)
            )
        else:
            graph_edge_weights = np.ones((graph_edges.shape[0],), dtype=np.float32)
        if str(ARGS.CURATED_GRAPH_CELL_MAP_PATH).strip():
            graph_cell_codes, cell_map_report = _map_external_cell_ids_to_graph_codes(node_to_code)
            reports["graph_cell_map"] = cell_map_report
        else:
            LOGGER.warning(
                "[bio][graph] no exact cell map supplied; using the v4 obs-label composition fallback"
            )
            graph_cell_codes = _map_obs_labels_to_graph_codes(
                node_to_code,
                ARGS.CURATED_GRAPH_NODE_OBS_COLS,
                str(ARGS.CURATED_GRAPH_NODE_SEPARATOR),
            )
        graph_node_features = _directed_graph_features(len(unique_nodes), graph_edges)
        coverage = float(np.mean(graph_cell_codes > 0))
        available["graph"] = coverage >= float(ARGS.CURATED_GRAPH_MIN_CELL_COVERAGE)
        reports.update(
            {
                "graph_nodes": len(unique_nodes),
                "graph_edges": int(graph_edges.shape[0]),
                "graph_cell_coverage": coverage,
            }
        )
        LOGGER.info(
            f"[bio][graph] nodes={len(unique_nodes)} edges={graph_edges.shape[0]} cell_coverage={coverage:.4f}"
        )
    elif "graph" in requested_components:
        LOGGER.warning("[bio][graph] CURATED_GRAPH_PATH is empty")
    anatomy_codes: List[np.ndarray] = []
    anatomy_cards: List[int] = []
    anatomy_cols: List[str] = []
    numeric_parts: List[np.ndarray] = []
    numeric_cols: List[str] = []
    if "anatomy" in requested_components:
        for col in ARGS.BIO_ANATOMY_COLS:
            if col not in ADATA_OBS.columns:
                LOGGER.warning(f"[bio] anatomy column unavailable and ignored: {col}")
                continue
            codes, card = _encode_obs_category(col)
            anatomy_codes.append(codes)
            anatomy_cards.append(card)
            anatomy_cols.append(str(col))
        for col in ARGS.BIO_NUMERIC_CONTEXT_COLS:
            if col not in ADATA_OBS.columns:
                LOGGER.warning(f"[bio] numeric context column unavailable and ignored: {col}")
                continue
            values, mask = _normalized_numeric_obs(col)
            numeric_parts.extend([values, mask])
            numeric_cols.append(str(col))
        available["anatomy"] = len(anatomy_codes) > 0 or len(numeric_cols) > 0
    numeric_context = (
        np.stack(numeric_parts, axis=1).astype(np.float32, copy=False)
        if numeric_parts
        else np.zeros((N_CELLS, 0), dtype=np.float32)
    )
    driver_by_node = np.zeros((len(graph_node_names), N_GENES), dtype=np.float32)
    if (
        "drivers" in requested_components
        and available["graph"]
        and str(ARGS.DRIVER_PROGRAM_PATH).strip()
    ):
        driver_df = _read_prior_table(str(ARGS.DRIVER_PROGRAM_PATH), str(ARGS.DRIVER_PROGRAM_SHEET))
        node_col = _resolve_column(
            driver_df,
            str(ARGS.DRIVER_NODE_COL),
            ["node", "target_node", "target", "celltype", "cell_type"],
            required=True,
        )
        gene_col = _resolve_column(
            driver_df,
            str(ARGS.DRIVER_GENE_COL),
            ["gene", "gene_name", "symbol", "driver"],
            required=True,
        )
        score_col = _resolve_column(
            driver_df,
            str(ARGS.DRIVER_SCORE_COL),
            ["score", "weight", "importance", "logfc"],
            required=False,
        )
        gene_to_idx = {str(g): i for i, g in enumerate(gene_names)}
        n_matched = 0
        driver_nodes = driver_df[node_col].astype(str).to_numpy()
        driver_genes = driver_df[gene_col].astype(str).to_numpy()
        driver_scores = (
            pd.to_numeric(driver_df[score_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            if score_col is not None
            else np.ones((len(driver_df),), dtype=np.float32)
        )
        for node_name, gene_name, score in zip(driver_nodes, driver_genes, driver_scores):
            node_code = node_to_code.get(str(node_name), 0)
            gene_idx = gene_to_idx.get(str(gene_name), None)
            if node_code <= 0 or gene_idx is None:
                continue
            driver_by_node[node_code, int(gene_idx)] += float(score)
            n_matched += 1
        row_scale = np.max(np.abs(driver_by_node), axis=1, keepdims=True)
        row_scale[row_scale < 1e-08] = 1.0
        driver_by_node /= row_scale
        available["drivers"] = n_matched > 0
        reports["drivers"] = {
            "input_rows": int(len(driver_df)),
            "matched_node_gene_rows": int(n_matched),
            "matched_fraction": float(n_matched / max(1, len(driver_df))),
            "nodes_with_matched_drivers": int(np.sum(np.any(np.abs(driver_by_node) > 0, axis=1))),
            "model_genes_with_matched_drivers": int(
                np.sum(np.any(np.abs(driver_by_node) > 0, axis=0))
            ),
        }
        LOGGER.info(f"[bio][drivers] matched node-gene rows={n_matched}/{len(driver_df)}")
    elif "drivers" in requested_components:
        LOGGER.warning(
            "[bio][drivers] DRIVER_PROGRAM_PATH is empty or curated graph is unavailable"
        )
    grn_adjacency = np.zeros((N_GENES, N_GENES), dtype=np.float32)
    grn_target_support = np.zeros((N_GENES,), dtype=np.float32)
    if "grn" in requested_components and str(ARGS.GRN_PATH).strip():
        grn_df = _read_prior_table(str(ARGS.GRN_PATH))
        source_col = _resolve_column(
            grn_df, str(ARGS.GRN_SOURCE_COL), ["tf", "regulator", "source", "from"], required=True
        )
        target_col = _resolve_column(
            grn_df, str(ARGS.GRN_TARGET_COL), ["target", "gene", "to"], required=True
        )
        weight_col = _resolve_column(
            grn_df, str(ARGS.GRN_WEIGHT_COL), ["importance", "weight", "score"], required=False
        )
        gene_to_idx = {str(g): i for i, g in enumerate(gene_names)}
        target_rows: List[Tuple[str, int, float]] = []
        grn_sources = grn_df[source_col].astype(str).to_numpy()
        grn_targets = grn_df[target_col].astype(str).to_numpy()
        grn_weights = (
            pd.to_numeric(grn_df[weight_col], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            if weight_col is not None
            else np.ones((len(grn_df),), dtype=np.float32)
        )
        for source_name, target_name, weight in zip(grn_sources, grn_targets, grn_weights):
            dst = gene_to_idx.get(str(target_name), None)
            if dst is None:
                continue
            numeric_weight = float(weight)
            if not np.isfinite(numeric_weight) or abs(numeric_weight) < float(ARGS.GRN_MIN_WEIGHT):
                continue
            if str(ARGS.GRN_EXTERNAL_TF_MODE) == "drop" and str(source_name) not in gene_to_idx:
                continue
            target_rows.append((str(source_name), int(dst), numeric_weight))
        if int(ARGS.GRN_TOPK_PER_TARGET) > 0 and target_rows:
            by_target: Dict[int, List[Tuple[str, int, float]]] = defaultdict(list)
            for edge in target_rows:
                by_target[edge[1]].append(edge)
            target_rows = [
                edge
                for rows in by_target.values()
                for edge in sorted(rows, key=lambda x: abs(x[2]), reverse=True)[
                    : int(ARGS.GRN_TOPK_PER_TARGET)
                ]
            ]
        dynamic_edges = 0
        external_edges = 0
        for source_name, dst, weight in target_rows:
            src = gene_to_idx.get(str(source_name), None)
            if src is not None:
                grn_adjacency[int(src), int(dst)] += float(weight)
                dynamic_edges += 1
            else:
                external_edges += 1
            if str(ARGS.GRN_EXTERNAL_TF_MODE) == "target_gate":
                grn_target_support[int(dst)] += abs(float(weight))
        col_scale = np.sum(np.abs(grn_adjacency), axis=0, keepdims=True)
        col_scale[col_scale < 1e-08] = 1.0
        grn_adjacency /= col_scale
        support_max = float(np.max(grn_target_support)) if grn_target_support.size else 0.0
        if support_max > 0:
            grn_target_support /= support_max
        available["grn"] = len(target_rows) > 0 and (dynamic_edges > 0 or support_max > 0)
        reports["grn"] = {
            "input_rows": int(len(grn_df)),
            "selected_edges_targeting_model_genes": int(len(target_rows)),
            "dynamic_edges_regulator_and_target_in_model": int(dynamic_edges),
            "external_regulator_edges_retained_as_target_support": int(
                external_edges if str(ARGS.GRN_EXTERNAL_TF_MODE) == "target_gate" else 0
            ),
            "model_targets_covered": int(
                np.sum((np.sum(np.abs(grn_adjacency), axis=0) > 0) | (grn_target_support > 0))
            ),
            "external_tf_mode": str(ARGS.GRN_EXTERNAL_TF_MODE),
        }
        LOGGER.info(
            f"[bio][grn] selected={len(target_rows)} dynamic={dynamic_edges} external_target_support={external_edges}"
        )
    elif "grn" in requested_components:
        LOGGER.warning("[bio][grn] GRN_PATH is empty")
    reports["available"] = dict(available)
    try:
        with open(os.path.join(SUMMARY_DIR, f"{ARGS.MSG}_biological_prior_report.json"), "w") as f:
            json.dump(reports, f, indent=2)
    except Exception as exc:
        LOGGER.warning(f"[bio] could not save prior report: {exc}")
    return BiologicalPriorData(
        gene_names=gene_names,
        actual_time_by_ordinal=actual_time,
        hierarchy_codes=hierarchy_codes,
        hierarchy_cardinalities=hierarchy_cards,
        hierarchy_columns=hierarchy_cols,
        anatomy_codes=anatomy_codes,
        anatomy_cardinalities=anatomy_cards,
        anatomy_columns=anatomy_cols,
        numeric_context=numeric_context,
        numeric_context_columns=numeric_cols,
        graph_cell_codes=graph_cell_codes,
        graph_node_features=graph_node_features,
        graph_edges=graph_edges,
        graph_edge_weights=graph_edge_weights,
        graph_node_names=graph_node_names,
        driver_program_by_node=driver_by_node,
        grn_adjacency=grn_adjacency,
        grn_target_support=grn_target_support,
        available=available,
        reports=reports,
    )


def apply_deterministic_trajectory_subsampling_if_requested() -> None:
    global TRAJECTORIES
    global CELLS_BY_TIME
    global CELLS_BY_TIME_TYPE
    global TIME_TYPE_COUNTS
    global SUBSAMPLED_TRAJECTORY_ROW_INDICES
    global SUBSAMPLED_TRAJ_CELL_MASK
    global SUBSAMPLED_TRAJ_CELL_INDICES_BY_TIME
    mode = str(ARGS.TRAJECTORY_SUBSAMPLE_MODE)
    fraction = float(ARGS.TRAJECTORY_SUBSAMPLE_FRACTION)
    if mode == "none" or fraction >= 1.0:
        LOGGER.info("[trajectory_subsample] disabled")
        return
    if TRAJECTORIES is None:
        raise RuntimeError("TRAJECTORY_SUBSAMPLE_MODE was requested, but TRAJ_PATH is unavailable.")
    if mode != "trajectory_rows":
        raise ValueError(f"Unsupported TRAJECTORY_SUBSAMPLE_MODE={mode!r}")
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("TRAJECTORY_SUBSAMPLE_FRACTION must be in (0, 1].")
    traj_arr = np.asarray(TRAJECTORIES, dtype=np.int64)
    if traj_arr.ndim != 2:
        raise ValueError(f"Expected TRAJECTORIES to be a 2D array/list, got shape={traj_arr.shape}")
    n_traj = int(traj_arr.shape[0])
    n_keep = int(round(float(fraction) * float(n_traj)))
    n_keep = max(int(ARGS.TRAJECTORY_SUBSAMPLE_MIN_ROWS), n_keep)
    n_keep = min(n_keep, n_traj)
    rng = np.random.default_rng(int(ARGS.TRAJECTORY_SUBSAMPLE_SEED))
    row_indices = rng.choice(np.arange(n_traj, dtype=np.int64), size=n_keep, replace=False)
    row_indices = np.asarray(row_indices, dtype=np.int64)
    row_indices.sort()
    sampled_traj_arr = traj_arr[row_indices, :]
    sampled_cell_indices = np.unique(sampled_traj_arr.reshape(-1)).astype(np.int64)
    cell_mask = np.zeros((int(N_CELLS),), dtype=bool)
    valid_cell_indices = sampled_cell_indices[
        (sampled_cell_indices >= 0) & (sampled_cell_indices < int(N_CELLS))
    ]
    cell_mask[valid_cell_indices] = True
    SUBSAMPLED_TRAJECTORY_ROW_INDICES = row_indices
    SUBSAMPLED_TRAJ_CELL_MASK = cell_mask
    TRAJECTORIES = sampled_traj_arr.tolist()
    try:
        np.save(
            os.path.join(SUMMARY_DIR, f"{ARGS.MSG}_trajectory_subsample_row_indices.npy"),
            row_indices,
        )
        np.save(
            os.path.join(SUMMARY_DIR, f"{ARGS.MSG}_trajectory_subsample_cell_indices.npy"),
            valid_cell_indices,
        )
    except Exception as exc:
        LOGGER.info(f"[trajectory_subsample] could not save subsample index arrays: {exc}")
    LOGGER.info(
        f"[trajectory_subsample] sampled trajectory rows: mode={mode} fraction={fraction} seed={ARGS.TRAJECTORY_SUBSAMPLE_SEED} rows={n_keep}/{n_traj} unique_cells={valid_cell_indices.size}/{N_CELLS}"
    )
    if bool(ARGS.TRAJECTORY_SUBSAMPLE_RESTRICT_CONTEXT):
        new_cells_by_time: List[np.ndarray] = []
        new_cells_by_time_type: List[Dict[int, np.ndarray]] = []
        new_time_type_counts = np.zeros((TOTAL_TIMESTAMP, N_CELLTYPES), dtype=np.int64)
        sampled_by_time: List[np.ndarray] = []
        for t in range(TOTAL_TIMESTAMP):
            idx = np.flatnonzero((ORDINAL_TIME == int(t)) & cell_mask).astype(np.int64)
            new_cells_by_time.append(idx)
            sampled_by_time.append(idx)
            by_type: Dict[int, np.ndarray] = {}
            if idx.size > 0:
                local_types = CELLTYPE_CODES[idx]
                for ty in np.unique(local_types):
                    ty = int(ty)
                    ids = idx[local_types == ty].astype(np.int64)
                    by_type[ty] = ids
                    new_time_type_counts[t, ty] = int(ids.size)
            new_cells_by_time_type.append(by_type)
        CELLS_BY_TIME = new_cells_by_time
        CELLS_BY_TIME_TYPE = new_cells_by_time_type
        TIME_TYPE_COUNTS = new_time_type_counts
        SUBSAMPLED_TRAJ_CELL_INDICES_BY_TIME = sampled_by_time
        per_time_counts = [int(x.size) for x in CELLS_BY_TIME]
        LOGGER.info(
            f"[trajectory_subsample] restricted FM context pools to sampled trajectory cells. min_per_time={(min(per_time_counts) if per_time_counts else 0)} max_per_time={(max(per_time_counts) if per_time_counts else 0)} mean_per_time={(float(np.mean(per_time_counts)) if per_time_counts else 0.0):.1f}"
        )


def sample_full_trajectory_rows_and_cell_mask(
    *, fraction: float, seed: int, min_rows: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministically sample rows from the ORIGINAL full trajectory array and
    return both the selected row indices and a Boolean cell mask.

    This helper does not mutate TRAJECTORIES or the global context pools. It is
    used by FM scale experiments B/C/E so training can be reduced while
    validation/test remain on the full population.
    """
    if TRAJECTORIES_FULL_ARRAY is None:
        raise RuntimeError(
            "Trajectory sampling requested but the full trajectory array is unavailable."
        )
    frac = float(fraction)
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {frac}")
    n_traj = int(TRAJECTORIES_FULL_ARRAY.shape[0])
    n_keep = int(round(frac * float(n_traj)))
    n_keep = max(int(min_rows), min(n_keep, n_traj))
    rng = np.random.default_rng(int(seed))
    row_indices = rng.choice(np.arange(n_traj, dtype=np.int64), size=n_keep, replace=False)
    row_indices = np.asarray(row_indices, dtype=np.int64)
    row_indices.sort()
    sampled_cells = np.unique(TRAJECTORIES_FULL_ARRAY[row_indices, :].reshape(-1)).astype(np.int64)
    sampled_cells = sampled_cells[(sampled_cells >= 0) & (sampled_cells < int(N_CELLS))]
    cell_mask = np.zeros((int(N_CELLS),), dtype=bool)
    cell_mask[sampled_cells] = True
    return (row_indices, cell_mask)


def build_context_pools_from_cell_mask(
    cell_mask: np.ndarray,
) -> Tuple[List[np.ndarray], List[Dict[int, np.ndarray]], np.ndarray]:
    """Build timestamp/type context pools restricted to a Boolean global cell mask."""
    mask = np.asarray(cell_mask, dtype=bool)
    cells_by_time: List[np.ndarray] = []
    cells_by_time_type: List[Dict[int, np.ndarray]] = []
    counts = np.zeros((TOTAL_TIMESTAMP, N_CELLTYPES), dtype=np.int64)
    for t in range(TOTAL_TIMESTAMP):
        idx = np.flatnonzero((ORDINAL_TIME == int(t)) & mask).astype(np.int64)
        cells_by_time.append(idx)
        by_type: Dict[int, np.ndarray] = {}
        if idx.size > 0:
            local_types = CELLTYPE_CODES[idx]
            for ty in np.unique(local_types):
                ty = int(ty)
                ids = idx[local_types == ty].astype(np.int64)
                by_type[ty] = ids
                counts[t, ty] = int(ids.size)
        cells_by_time_type.append(by_type)
    return (cells_by_time, cells_by_time_type, counts)


def build_traj_prev_map_from_array(traj_array: np.ndarray) -> Dict[int, np.ndarray]:
    """Build the same previous-cell hint map as the legacy helper, from an explicit trajectory array."""
    prev_map_list: Dict[int, List[int]] = defaultdict(list)
    arr = np.asarray(traj_array, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D trajectory array, got shape={arr.shape}")
    for row in arr:
        for pos in range(1, int(row.size)):
            prev_map_list[int(row[pos])].append(int(row[pos - 1]))
    return {k: np.asarray(v, dtype=np.int64) for k, v in prev_map_list.items()}


@dataclass
class UnifiedEvalResult:
    mean_cell_pcc: float
    mean_gene_pcc: float
    day_metrics: Dict[str, float]
    n_cells: int


def evaluate_predictions_unified(
    *, y_pred: torch.Tensor, y_true: torch.Tensor, time_idx: torch.Tensor
) -> UnifiedEvalResult:
    unique_days = sorted(torch.unique(time_idx).tolist())
    daily_cell_pcc = []
    daily_gene_pcc = []
    day_metrics: Dict[str, float] = {}
    for d in unique_days:
        mask = time_idx == int(d)
        cp, gp = pcc_cells_and_genes(y_pred[mask], y_true[mask])
        daily_cell_pcc.append(cp)
        daily_gene_pcc.append(gp)
        day_metrics[f"day{int(d)}_cell_pcc"] = float(cp)
        day_metrics[f"day{int(d)}_gene_pcc"] = float(gp)
    mean_cell_pcc = (
        float(np.mean(np.asarray(daily_cell_pcc, dtype=np.float64))) if daily_cell_pcc else 0.0
    )
    mean_gene_pcc = (
        float(np.mean(np.asarray(daily_gene_pcc, dtype=np.float64))) if daily_gene_pcc else 0.0
    )
    return UnifiedEvalResult(
        mean_cell_pcc=mean_cell_pcc,
        mean_gene_pcc=mean_gene_pcc,
        day_metrics=day_metrics,
        n_cells=int(y_pred.shape[0]),
    )


# Expression-agreement metrics accumulated without retaining every prediction.
class StreamingDayPCCAccumulator:
    """
    Memory-efficient day-wise PCC/MSE accumulator.

    It reproduces the benchmark's day-wise averaging semantics without storing
    all predictions. This is important when one FM checkpoint is evaluated with
    many readouts and when TRANS duplicate target-cell predictions are aligned.
    """

    def __init__(self, n_genes: int):
        self.n_genes = int(n_genes)
        self.stats: Dict[int, Dict[str, Any]] = {}

    def _ensure_day(self, day: int) -> Dict[str, Any]:
        day = int(day)
        if day not in self.stats:
            self.stats[day] = {
                "n": 0,
                "cell_total": 0,
                "cell_valid": 0,
                "cell_sum": 0.0,
                "sum_x": np.zeros((self.n_genes,), dtype=np.float64),
                "sum_y": np.zeros((self.n_genes,), dtype=np.float64),
                "sum_x2": np.zeros((self.n_genes,), dtype=np.float64),
                "sum_y2": np.zeros((self.n_genes,), dtype=np.float64),
                "sum_xy": np.zeros((self.n_genes,), dtype=np.float64),
                "mse_sse": 0.0,
                "mse_count": 0,
            }
        return self.stats[day]

    def update(self, *, y_pred: torch.Tensor, y_true: torch.Tensor, time_idx: torch.Tensor) -> None:
        pred_np = y_pred.detach().cpu().numpy().astype(np.float64, copy=False)
        true_np = y_true.detach().cpu().numpy().astype(np.float64, copy=False)
        day_np = time_idx.detach().cpu().numpy().astype(np.int64, copy=False)
        for day in np.unique(day_np):
            mask = day_np == int(day)
            x = pred_np[mask]
            y = true_np[mask]
            if x.shape[0] == 0:
                continue
            st = self._ensure_day(int(day))
            st["n"] += int(x.shape[0])
            xc = x - x.mean(axis=1, keepdims=True)
            yc = y - y.mean(axis=1, keepdims=True)
            num = np.einsum("ij,ij->i", xc, yc)
            den = np.sqrt(np.einsum("ij,ij->i", xc, xc) * np.einsum("ij,ij->i", yc, yc))
            valid = (den > float(ARGS.PCC_EPS)) & np.isfinite(den) & np.isfinite(num)
            corr = np.zeros_like(num, dtype=np.float64)
            corr[valid] = num[valid] / den[valid]
            st["cell_total"] += int(corr.size)
            st["cell_valid"] += int(valid.sum())
            st["cell_sum"] += float(corr[valid].sum())
            st["sum_x"] += x.sum(axis=0)
            st["sum_y"] += y.sum(axis=0)
            st["sum_x2"] += np.einsum("ij,ij->j", x, x)
            st["sum_y2"] += np.einsum("ij,ij->j", y, y)
            st["sum_xy"] += np.einsum("ij,ij->j", x, y)
            diff = x - y
            st["mse_sse"] += float(np.einsum("ij,ij->", diff, diff))
            st["mse_count"] += int(diff.size)

    def finalize(self) -> Tuple[UnifiedEvalResult, float]:
        daily_cell_pcc: List[float] = []
        daily_gene_pcc: List[float] = []
        day_metrics: Dict[str, float] = {}
        total_mse_sse = 0.0
        total_mse_count = 0
        total_cells = 0
        for day in sorted(self.stats.keys()):
            st = self.stats[day]
            n = int(st["n"])
            total_cells += n
            if str(ARGS.PCC_NAN_POLICY) == "zero":
                cp = float(st["cell_sum"] / max(1, int(st["cell_total"])))
            else:
                valid_fraction = float(st["cell_valid"] / max(1, int(st["cell_total"])))
                if int(st["cell_valid"]) == 0 or valid_fraction < float(
                    ARGS.PCC_MIN_VALID_FRACTION
                ):
                    cp = 0.0
                else:
                    cp = float(st["cell_sum"] / int(st["cell_valid"]))
            if n <= 1:
                gene_corr = np.full((self.n_genes,), np.nan, dtype=np.float64)
            else:
                sx = st["sum_x"]
                sy = st["sum_y"]
                sxx = st["sum_x2"] - sx * sx / float(n)
                syy = st["sum_y2"] - sy * sy / float(n)
                sxy = st["sum_xy"] - sx * sy / float(n)
                den = np.sqrt(np.maximum(sxx, 0.0) * np.maximum(syy, 0.0))
                gene_corr = np.full((self.n_genes,), np.nan, dtype=np.float64)
                valid = (den > float(ARGS.PCC_EPS)) & np.isfinite(den) & np.isfinite(sxy)
                gene_corr[valid] = sxy[valid] / den[valid]
            gp = _nan_robust_mean(
                gene_corr,
                policy=str(ARGS.PCC_NAN_POLICY),
                min_valid_fraction=float(ARGS.PCC_MIN_VALID_FRACTION),
            )
            daily_cell_pcc.append(float(cp))
            daily_gene_pcc.append(float(gp))
            day_metrics[f"day{int(day)}_cell_pcc"] = float(cp)
            day_metrics[f"day{int(day)}_gene_pcc"] = float(gp)
            total_mse_sse += float(st["mse_sse"])
            total_mse_count += int(st["mse_count"])
        mean_cell = (
            float(np.mean(np.asarray(daily_cell_pcc, dtype=np.float64))) if daily_cell_pcc else 0.0
        )
        mean_gene = (
            float(np.mean(np.asarray(daily_gene_pcc, dtype=np.float64))) if daily_gene_pcc else 0.0
        )
        mse = float(total_mse_sse / max(1, total_mse_count))
        return (
            UnifiedEvalResult(
                mean_cell_pcc=mean_cell,
                mean_gene_pcc=mean_gene,
                day_metrics=day_metrics,
                n_cells=int(total_cells),
            ),
            mse,
        )


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {str(k): v.detach().cpu().clone() for k, v in model.state_dict().items()}


@dataclass(frozen=True)
class FoundationCheckpointSpec:
    model_name: str
    label: str
    source: str


def parse_foundation_checkpoint_specs(
    specs: Sequence[str],
) -> Dict[str, List[FoundationCheckpointSpec]]:
    out: Dict[str, List[FoundationCheckpointSpec]] = defaultdict(list)
    for spec in specs:
        spec = str(spec).strip()
        if spec == "" or spec.startswith("#"):
            continue
        parts = str(spec).split("::")
        if len(parts) != 3:
            raise ValueError(
                "Each FOUNDATION_CHECKPOINT_SPECS entry must have format ModelName::Label::CheckpointPathOrHFId"
            )
        model_name, label, source = parts
        out[str(model_name)].append(
            FoundationCheckpointSpec(
                model_name=str(model_name), label=str(label), source=str(source)
            )
        )
    return out


def _looks_like_hf_repo_id(source: str) -> bool:
    """
    Heuristic: HuggingFace repo ids look like "org/name" and are not local paths.
    """
    source = str(source)
    if os.path.exists(source):
        return False
    if source.startswith("/") or source.startswith("./") or source.startswith("../"):
        return False
    if ":" in source:
        return False
    return "/" in source and (not source.endswith((".pt", ".pth", ".ckpt", ".bin", ".safetensors")))


def resolve_foundation_checkpoint_source(
    *, source: Optional[str], logger: logging.Logger, model_tag: str
) -> Optional[str]:
    """
    Convert a HuggingFace repo id into a local snapshot folder when possible.

    Why this is needed:
        Your previous error happened because the benchmark treated
        'amazon/chronos-bolt-tiny' like a local checkpoint file.  A HuggingFace
        repo id is not a file; it must either be loaded by a model-specific
        from_pretrained(...) method, or downloaded to a snapshot folder first.
    """
    if source is None:
        return None
    source = str(source)
    if os.path.exists(source):
        return source
    if not bool(ARGS.FOUNDATION_AUTO_SNAPSHOT_DOWNLOAD):
        return source
    if not _looks_like_hf_repo_id(source):
        return source
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        logger.info(
            f"[{model_tag}] huggingface_hub is not installed, so repo id {source!r} cannot be auto-downloaded. Install with: pip install -U huggingface_hub. Error: {exc}"
        )
        return source
    try:
        local_path = snapshot_download(
            repo_id=source,
            cache_dir=ARGS.HF_CACHE_DIR,
            local_files_only=bool(ARGS.FOUNDATION_LOCAL_FILES_ONLY),
        )
        logger.info(
            f"[{model_tag}] resolved HF repo id {source!r} -> local snapshot {local_path!r}"
        )
        return str(local_path)
    except Exception as exc:
        logger.info(
            f"[{model_tag}] could not auto-resolve HF repo id {source!r}. Continuing with original source. If this is an offline/HPC node, run `hf download {source} --local-dir <folder>` first. Error: {exc}"
        )
        return source


@dataclass(frozen=True)
class TransformRunSpec:
    model_name: str
    run_mode: str
    checkpoint_label: Optional[str] = None
    checkpoint_source: Optional[str] = None

    @property
    def tag(self) -> str:
        base = f"TRANS_{self.model_name}_{self.run_mode}"
        if self.checkpoint_label is not None:
            base = f"{base}_{self.checkpoint_label}"
        return base


def build_transform_specs() -> List[TransformRunSpec]:
    specs: List[TransformRunSpec] = []
    for model_name in _drop_commented_or_empty_items(ARGS.TRANS_BASE_MODELS):
        model_name = str(model_name)
        if model_name not in TRANS_FOUNDATION_MODEL_SET:
            specs.append(TransformRunSpec(model_name=model_name, run_mode="scratch"))
            continue
        for mode in ARGS.FOUNDATION_RUN_MODES:
            mode = str(mode)
            if mode in {"scratch", "zero_shot_no_ckpt"}:
                specs.append(TransformRunSpec(model_name=model_name, run_mode=mode))
                continue
            if mode in {"zero_shot_ckpt", "finetune_ckpt"}:
                ckpt_list = FOUNDATION_CKPT_SPECS.get(model_name, [])
                if len(ckpt_list) == 0:
                    specs.append(
                        TransformRunSpec(
                            model_name=model_name,
                            run_mode=mode,
                            checkpoint_label="NO_CKPT_SPEC",
                            checkpoint_source=None,
                        )
                    )
                else:
                    for ckpt_spec in ckpt_list:
                        specs.append(
                            TransformRunSpec(
                                model_name=model_name,
                                run_mode=mode,
                                checkpoint_label=str(ckpt_spec.label),
                                checkpoint_source=str(ckpt_spec.source),
                            )
                        )
                continue
            raise ValueError(mode)
    return specs


class TrajectoryWindowDataset(Dataset):

    def __init__(
        self,
        *,
        adata_X: sp.csr_matrix,
        trajectories: List[List[int]],
        ordinal_time: np.ndarray,
        celltype_codes: np.ndarray,
        split_name: Literal["train", "val", "test"],
        seq_len: int,
        label_len: int,
        pred_len: int,
        train_timestamps: Set[int],
        val_timestamps: Set[int],
        test_timestamps: Set[int],
        n_genes: int,
    ):
        super().__init__()
        self.adata_X = adata_X
        self.trajectories = np.asarray(trajectories, dtype=np.int32)
        self.ordinal_time = np.asarray(ordinal_time, dtype=np.int32)
        self.celltype_codes = np.asarray(celltype_codes, dtype=np.int32)
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.n_genes = int(n_genes)
        if split_name == "train":
            allowed = train_timestamps
        elif split_name == "val":
            allowed = val_timestamps
        elif split_name == "test":
            allowed = test_timestamps
        else:
            raise ValueError(split_name)
        self.allowed_target_timestamps = set((int(x) for x in allowed))
        self.window_length = self.seq_len + self.pred_len
        self.total_timestamp = int(np.max(self.ordinal_time) + 1)
        first_target_offset = self.seq_len
        self.index_pairs: List[Tuple[int, int]] = []
        for traj_idx in range(self.trajectories.shape[0]):
            for window_start in range(0, self.total_timestamp - self.window_length + 1):
                target_timestamp = int(window_start + first_target_offset)
                if target_timestamp in self.allowed_target_timestamps:
                    self.index_pairs.append((int(traj_idx), int(window_start)))
        if len(self.index_pairs) == 0:
            raise RuntimeError("No windows found for the given TRANS split.")

    def __len__(self) -> int:
        return len(self.index_pairs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        traj_idx, window_start = self.index_pairs[int(index)]
        enc_st = int(window_start)
        enc_ed = enc_st + int(self.seq_len)
        dec_st = enc_ed
        dec_ed = dec_st + int(self.pred_len)
        encoder_cell_index = self.trajectories[traj_idx, enc_st:enc_ed]
        decoder_cell_index = self.trajectories[traj_idx, dec_st:dec_ed]
        encoder_tensor = torch.tensor(
            self.adata_X[encoder_cell_index, :].toarray(), dtype=torch.float32
        )
        dec_known = encoder_tensor[-int(self.label_len) :, :].clone()
        dec_zeros = torch.zeros((int(self.pred_len), int(self.n_genes)), dtype=torch.float32)
        decoder_tensor = torch.cat([dec_known, dec_zeros], dim=0)
        target_tensor = torch.tensor(
            self.adata_X[decoder_cell_index, :].toarray(), dtype=torch.float32
        )
        enc_type_ids = torch.tensor(self.celltype_codes[encoder_cell_index], dtype=torch.long)
        dec_type_known = enc_type_ids[-int(self.label_len) :].clone()
        dec_type_pred = enc_type_ids[-1:].repeat(int(self.pred_len))
        decoder_type_ids = torch.cat([dec_type_known, dec_type_pred], dim=0)
        return {
            "encoder_tensor": encoder_tensor,
            "decoder_tensor": decoder_tensor,
            "target_tensor": target_tensor,
            "encoder_type_ids": enc_type_ids,
            "decoder_type_ids": decoder_type_ids,
            "trajectory_index": int(traj_idx),
            "target_cell_index": int(decoder_cell_index[-1]),
            "target_timestamp_index": int(self.ordinal_time[int(decoder_cell_index[0])]),
        }


class TransformDataBundle:

    def __init__(self):
        if TRAJECTORIES is None:
            raise RuntimeError("TRANS requested but TRAJ_PATH is unavailable.")
        if int(ARGS.SEQ_LEN) <= 0:
            raise RuntimeError("TRANS does not support SEQ_LEN=0. Use SEQ_LEN>=1.")
        min_target_ts = int(ARGS.SEQ_LEN)
        self.train_timestamps, self.val_timestamps, self.test_timestamps = make_timestamp_splits(
            total_timestamp=TOTAL_TIMESTAMP,
            min_target_ts=min_target_ts,
            split_method=str(ARGS.SPLIT_METHOD),
            train_ratio=float(ARGS.TRAIN_RATIO),
            valid_ratio=float(ARGS.VALID_RATIO),
            prespecified_test_timestamps=ARGS.PRESPECIFIED_TEST_TIMESTAMPS,
            dataset_random_seed=int(ARGS.DATASET_RANDOM_SEED),
        )
        LOGGER.info(
            f"[TRANS] split\n  train timestamps: {sorted(self.train_timestamps)}\n  val   timestamps: {sorted(self.val_timestamps)}\n  test  timestamps: {sorted(self.test_timestamps)}"
        )
        common = dict(
            adata_X=ADATA_X,
            trajectories=TRAJECTORIES,
            ordinal_time=ORDINAL_TIME,
            celltype_codes=CELLTYPE_CODES,
            seq_len=int(ARGS.SEQ_LEN),
            label_len=effective_trans_label_len(),
            pred_len=int(ARGS.PRED_LEN),
            train_timestamps=self.train_timestamps,
            val_timestamps=self.val_timestamps,
            test_timestamps=self.test_timestamps,
            n_genes=int(N_GENES),
        )
        self.train_dataset = TrajectoryWindowDataset(split_name="train", **common)
        self.val_dataset = TrajectoryWindowDataset(split_name="val", **common)
        self.test_dataset = TrajectoryWindowDataset(split_name="test", **common)
        LOGGER.info(
            f"[TRANS] dataset sizes train={len(self.train_dataset)} val={len(self.val_dataset)} test={len(self.test_dataset)}"
        )

    def make_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        worker_init = make_worker_init_fn(int(ARGS.SEED) + 1000)
        generator = make_torch_generator(int(ARGS.SEED) + 2000 + int(batch_size))
        dl_kw = dict(
            batch_size=int(batch_size),
            num_workers=int(ARGS.NUM_WORKERS),
            pin_memory=True,
            worker_init_fn=worker_init,
        )
        tr = DataLoader(
            self.train_dataset, shuffle=True, drop_last=True, generator=generator, **dl_kw
        )
        va = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **dl_kw)
        te = DataLoader(self.test_dataset, shuffle=False, drop_last=False, **dl_kw)
        return (tr, va, te)


class RetrievalPostProcessor:

    def __init__(
        self,
        *,
        adata_X: sp.csr_matrix,
        ordinal_time: np.ndarray,
        train_timestamps: Set[int],
        pool_mode: str,
        metric: str,
        topk: int,
        max_pool_per_ts: Optional[int],
        seed: int,
        logger: Optional[logging.Logger],
    ):
        self.adata_X = adata_X
        self.ordinal_time = np.asarray(ordinal_time, dtype=np.int32)
        self.train_timestamps = set((int(x) for x in train_timestamps))
        self.pool_mode = str(pool_mode)
        self.metric = str(metric)
        self.topk = int(topk)
        self.max_pool_per_ts = max_pool_per_ts
        self.rng = np.random.default_rng(int(seed))
        self.logger = logger
        self.pools: Dict[int, np.ndarray] = {}
        self._build_pools()

    def _subsample(self, x: np.ndarray) -> np.ndarray:
        if self.max_pool_per_ts is None:
            return x
        m = int(x.shape[0])
        if m <= int(self.max_pool_per_ts):
            return x
        idx = self.rng.choice(m, size=int(self.max_pool_per_ts), replace=False)
        return x[idx]

    def _build_pools(self) -> None:
        if self.logger is not None:
            self.logger.info(
                f"[TRANS retrieval] mode={self.pool_mode} metric={self.metric} topk={self.topk}"
            )
        if self.pool_mode == "all_times":
            x = self._subsample(self.adata_X.toarray().astype(np.float32, copy=False))
            self.pools[-1] = x
            return
        if self.pool_mode == "train_time_only":
            idx = np.flatnonzero(np.isin(self.ordinal_time, list(self.train_timestamps)))
            x = self._subsample(self.adata_X[idx, :].toarray().astype(np.float32, copy=False))
            self.pools[-1] = x
            return
        if self.pool_mode == "oracle_target_time":
            for ts in np.unique(self.ordinal_time):
                idx = np.flatnonzero(self.ordinal_time == int(ts))
                x = self._subsample(self.adata_X[idx, :].toarray().astype(np.float32, copy=False))
                self.pools[int(ts)] = x
            return
        raise ValueError(self.pool_mode)

    @staticmethod
    def _cosine_topk(query: np.ndarray, pool: np.ndarray, topk: int) -> np.ndarray:
        q = query.astype(np.float32, copy=False)
        p = pool.astype(np.float32, copy=False)
        qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-08
        pn = np.linalg.norm(p, axis=1, keepdims=True) + 1e-08
        sim = q / qn @ (p / pn).T
        kth = min(int(topk), max(1, sim.shape[1])) - 1
        idx = np.argpartition(-sim, kth=kth, axis=1)[:, : int(topk)]
        return np.stack([p[i].mean(axis=0) for i in idx], axis=0)

    @staticmethod
    def _l2_topk(query: np.ndarray, pool: np.ndarray, topk: int) -> np.ndarray:
        q = query.astype(np.float32, copy=False)
        p = pool.astype(np.float32, copy=False)
        q2 = (q * q).sum(axis=1, keepdims=True)
        p2 = (p * p).sum(axis=1, keepdims=True).T
        dist2 = q2 + p2 - 2.0 * (q @ p.T)
        kth = min(int(topk), max(1, dist2.shape[1])) - 1
        idx = np.argpartition(dist2, kth=kth, axis=1)[:, : int(topk)]
        return np.stack([p[i].mean(axis=0) for i in idx], axis=0)

    def apply(self, y_pred: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
        y = y_pred.detach().cpu().numpy().astype(np.float32, copy=False)
        t = time_idx.detach().cpu().numpy().astype(np.int32, copy=False)
        out = np.empty_like(y)
        if self.pool_mode in {"all_times", "train_time_only"}:
            pool = self.pools[-1]
            out = (
                self._cosine_topk(y, pool, self.topk)
                if self.metric == "cosine"
                else self._l2_topk(y, pool, self.topk)
            )
            return torch.from_numpy(out).to(y_pred.device, dtype=y_pred.dtype)
        for ts in np.unique(t):
            mask = t == int(ts)
            pool = self.pools.get(int(ts), None)
            if pool is None or pool.shape[0] == 0:
                out[mask] = y[mask]
                continue
            out[mask] = (
                self._cosine_topk(y[mask], pool, self.topk)
                if self.metric == "cosine"
                else self._l2_topk(y[mask], pool, self.topk)
            )
        return torch.from_numpy(out).to(y_pred.device, dtype=y_pred.dtype)


def build_trans_base_model(
    *,
    model_name: str,
    run_mode: str,
    n_genes: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    d_model: int,
    n_heads: int,
    e_layers: int,
    d_layers: int,
    d_ff: int,
    device: torch.device,
    foundation_checkpoint_source: Optional[str] = None,
    foundation_checkpoint_label: Optional[str] = None,
) -> nn.Module:
    mod_path, cls_name = TRANS_MODEL_REGISTRY[str(model_name)]
    module = importlib.import_module(mod_path)
    model_class = getattr(module, cls_name)
    hp = copy.deepcopy(TRANS_THU_DEFAULTS)
    hp.update(
        seq_len=int(seq_len),
        label_len=int(label_len),
        pred_len=int(pred_len),
        enc_in=int(n_genes),
        dec_in=int(n_genes),
        c_out=int(n_genes),
        d_model=int(d_model),
        n_heads=int(n_heads),
        e_layers=int(e_layers),
        d_layers=int(d_layers),
        d_ff=int(d_ff),
    )
    hp["device"] = device if str(model_name) == "WPMixer" else str(device)
    hp.setdefault("expand", 2)
    hp.setdefault("dt_rank", math.ceil(int(d_model) / 16))
    hp.setdefault("d_conv", 4)
    hp["d_ff"] = int(d_ff)
    if str(model_name) == "ETSformer":
        hp["d_layers"] = int(hp["e_layers"])
    if str(model_name) == "FreTS":
        hp.setdefault("channel_independence", 1)
    if str(model_name) == "Nonstationary_Transformer":
        hp.setdefault("p_hidden_dims", [64])
        hp.setdefault("p_hidden_layers", 1)
    if str(model_name) == "SegRNN":
        hp.setdefault("seg_len", 1)
    if str(model_name) in {"PAttn", "PatchTST", "TimeXer", "WPMixer", "TimeFilter"}:
        safe_patch_len = int(max(1, min(int(seq_len), 16)))
        hp["patch_len"] = safe_patch_len
        hp.setdefault("stride", 1)
    if str(model_name) == "TimeFilter":
        hp.setdefault("alpha", 0.1)
        hp.setdefault("top_p", 0.5)
        hp.setdefault("pos", 1)
    if str(model_name) == "TimeMixer":
        hp.setdefault("down_sampling_window", 1)
        hp.setdefault("channel_independence", 1)
        hp.setdefault("decomp_method", "moving_avg")
        hp.setdefault("down_sampling_layers", 0)
        hp.setdefault("use_norm", 1)
        hp.setdefault("down_sampling_method", "avg")
    if str(model_name) == "TimeXer":
        hp.setdefault("use_norm", 1)
        hp.setdefault("patch_len", 1)
    if str(model_name) == "TimesNet":
        hp["top_k"] = 1
    if str(model_name) == "WPMixer":
        hp["batch_size"] = int(ARGS.TRANS_BATCH)
        hp.setdefault("patch_len", 1)
        hp.setdefault("use_amp", bool(ARGS.FP16))
    if str(model_name) in TRANS_FOUNDATION_MODEL_SET and str(run_mode) != "scratch":
        hp["task_name"] = "zero_shot_forecast"
        hp["is_training"] = False
        if foundation_checkpoint_source is not None:
            hp["model_id"] = str(foundation_checkpoint_source)
            hp["pretrained_model_name_or_path"] = str(foundation_checkpoint_source)
            hp["model_name_or_path"] = str(foundation_checkpoint_source)
            hp["pretrained_model_path"] = str(foundation_checkpoint_source)
            hp["checkpoint_path"] = str(foundation_checkpoint_source)
            hp["ckpt_path"] = str(foundation_checkpoint_source)
            hp["model_path"] = str(foundation_checkpoint_source)
            hp["repo_id"] = str(foundation_checkpoint_source)
            hp["cache_dir"] = ARGS.HF_CACHE_DIR
            hp["local_files_only"] = bool(ARGS.FOUNDATION_LOCAL_FILES_ONLY)
            hp["trust_remote_code"] = True
            hp["token"] = os.environ.get("HF_TOKEN", None)
            hp["checkpoint_label"] = foundation_checkpoint_label
    cfg = SimpleNamespace(**hp)
    model = model_class(cfg)
    try:
        model = model.to(device)
    except Exception:
        pass
    return model


def _candidate_modules_for_checkpoint(model: nn.Module) -> List[nn.Module]:
    """
    Return possible modules that may own a checkpoint-loading hook.

    Why we need this:
        In this benchmark, the actual THU model may be wrapped by
        CellTypeConditionWrapper.  Some foundation models expose from_pretrained
        on the outer model, some on .model, and some only on the wrapped .base.
        This helper prevents one wrapping layer from hiding the loading method.
    """
    out: List[nn.Module] = []
    visited: Set[int] = set()

    def add(maybe_module: Any) -> None:
        if isinstance(maybe_module, nn.Module) and id(maybe_module) not in visited:
            visited.add(id(maybe_module))
            out.append(maybe_module)

    add(model)
    if hasattr(model, "base"):
        add(getattr(model, "base"))
    if hasattr(model, "model"):
        add(getattr(model, "model"))
    if hasattr(model, "base") and hasattr(getattr(model, "base"), "model"):
        add(getattr(getattr(model, "base"), "model"))
    return out


def _try_from_pretrained_on_candidate(
    *, candidate: nn.Module, source: str, logger: logging.Logger, model_name: str
) -> bool:
    """
    Try HuggingFace-style loading on one candidate module.

    The method can behave in two ways depending on the upstream implementation:
      1. It mutates the existing candidate and returns None.
      2. It returns a freshly loaded nn.Module.
    We support both cases.
    """
    if not hasattr(candidate, "from_pretrained"):
        return False
    fn = getattr(candidate, "from_pretrained")
    if not callable(fn):
        return False
    call_attempts = []
    call_attempts.append(dict(pretrained_model_name_or_path=source))
    call_attempts.append(dict(model_id=source))
    call_attempts.append(dict(path=source))
    call_attempts.append(dict())
    for kwargs in call_attempts:
        try:
            if kwargs:
                loaded = fn(**kwargs)
            else:
                loaded = fn(source)
            if isinstance(loaded, nn.Module):
                missing, unexpected = candidate.load_state_dict(loaded.state_dict(), strict=False)
                logger.info(
                    f"[{model_name}] loaded via from_pretrained on {type(candidate).__name__}; missing={len(missing)} unexpected={len(unexpected)}"
                )
            else:
                logger.info(
                    f"[{model_name}] from_pretrained succeeded on {type(candidate).__name__}"
                )
            return True
        except TypeError:
            continue
        except Exception as exc:
            logger.info(
                f"[{model_name}] from_pretrained attempt failed on {type(candidate).__name__}: {exc}"
            )
            continue
    return False


def _load_state_dict_from_local_file(source: str) -> Dict[str, Any]:
    """
    Load a state_dict-like object from .pt/.pth/.ckpt/.bin or .safetensors.

    For HuggingFace model folders, do NOT call this on the folder itself;
    use from_pretrained on the folder path instead.
    """
    lower = str(source).lower()
    if lower.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise ImportError(
                "The checkpoint is a .safetensors file, but safetensors is not installed. Install it with: pip install safetensors"
            ) from exc
        return load_file(source)
    return torch.load(source, map_location="cpu")


def _extract_state_dict_candidates(ckpt: Any) -> List[Dict[str, Any]]:
    candidate_state_dicts: List[Dict[str, Any]] = []
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "weights", "net", "module"]:
            maybe = ckpt.get(key, None)
            if isinstance(maybe, dict):
                candidate_state_dicts.append(maybe)
        if all((torch.is_tensor(v) for v in ckpt.values())):
            candidate_state_dicts.append(ckpt)
    return candidate_state_dicts


def _try_load_state_dict_candidates(
    *,
    model: nn.Module,
    candidate_state_dicts: List[Dict[str, Any]],
    source: str,
    model_name: str,
    logger: logging.Logger,
) -> bool:
    errors: List[str] = []
    modules = _candidate_modules_for_checkpoint(model)
    for sd in candidate_state_dicts:
        for module in modules:
            try:
                missing, unexpected = module.load_state_dict(sd, strict=False)
                logger.info(
                    f"[{model_name}] loaded state_dict into {type(module).__name__} from {source}; missing={len(missing)} unexpected={len(unexpected)}"
                )
                return True
            except Exception as exc:
                errors.append(f"{type(module).__name__}: {exc}")
        cleaned_variants: List[Dict[str, Any]] = []
        for prefix in ["model.", "base.", "module.", "net."]:
            cleaned = {
                str(k)[len(prefix) :] if str(k).startswith(prefix) else str(k): v
                for k, v in sd.items()
            }
            cleaned_variants.append(cleaned)
        for cleaned in cleaned_variants:
            for module in modules:
                try:
                    missing, unexpected = module.load_state_dict(cleaned, strict=False)
                    logger.info(
                        f"[{model_name}] loaded cleaned state_dict into {type(module).__name__} from {source}; missing={len(missing)} unexpected={len(unexpected)}"
                    )
                    return True
                except Exception as exc:
                    errors.append(f"cleaned->{type(module).__name__}: {exc}")
    logger.info(f"[{model_name}] state_dict loading attempts failed: {errors[:8]}")
    return False


def generic_load_checkpoint_into_model(
    *, model: nn.Module, checkpoint_source: str, model_name: str, logger: logging.Logger
) -> None:
    """
    Load a checkpoint into a TRANS/foundation model.

    Supported checkpoint_source values:
        1. HuggingFace repo id, e.g. 'amazon/chronos-bolt-tiny'
        2. Local HuggingFace snapshot directory, e.g. '/path/to/checkpoints/chronos-bolt-tiny'
        3. Local .pt/.pth/.ckpt/.bin file readable by torch.load
        4. Local .safetensors file readable by safetensors.torch.load_file

    Important design choice:
        If source is a directory, we try from_pretrained(source) FIRST.
        This is necessary because HuggingFace repositories are directories that
        contain config files + sharded/safetensors weights, not a single object
        that torch.load(directory) can read.
    """
    source = str(checkpoint_source)
    modules = _candidate_modules_for_checkpoint(model)
    if os.path.isdir(source):
        for module in modules:
            if _try_from_pretrained_on_candidate(
                candidate=module, source=source, logger=logger, model_name=model_name
            ):
                return
        possible_files = []
        for filename in [
            "pytorch_model.bin",
            "model.safetensors",
            "checkpoint.pt",
            "best.pt",
            "last.ckpt",
        ]:
            p = os.path.join(source, filename)
            if os.path.exists(p):
                possible_files.append(p)
        for p in possible_files:
            ckpt = _load_state_dict_from_local_file(p)
            candidates = _extract_state_dict_candidates(ckpt)
            if _try_load_state_dict_candidates(
                model=model,
                candidate_state_dicts=candidates,
                source=p,
                model_name=model_name,
                logger=logger,
            ):
                return
        raise RuntimeError(
            f"[{model_name}] {source!r} is a directory, but no from_pretrained hook or readable weight file worked."
        )
    if os.path.isfile(source):
        ckpt = _load_state_dict_from_local_file(source)
        candidates = _extract_state_dict_candidates(ckpt)
        if len(candidates) == 0:
            raise RuntimeError(
                f"[{model_name}] Could not infer a state_dict from local checkpoint {source!r}."
            )
        if _try_load_state_dict_candidates(
            model=model,
            candidate_state_dicts=candidates,
            source=source,
            model_name=model_name,
            logger=logger,
        ):
            return
        raise RuntimeError(
            f"[{model_name}] all local checkpoint loading attempts failed for {source!r}."
        )
    for module in modules:
        if _try_from_pretrained_on_candidate(
            candidate=module, source=source, logger=logger, model_name=model_name
        ):
            return
    raise FileNotFoundError(
        f"[{model_name}] checkpoint source {source!r} is not a local path and no usable from_pretrained hook succeeded. If this is a HuggingFace model, either install the required model package or download the repo with `hf download ... --local-dir ...` and pass the local folder."
    )


def _make_dec_for_transform_model(
    model_name: str,
    enc: torch.Tensor,
    dec: torch.Tensor,
    *,
    pred_len: int,
    label_len: int,
    seq_len: int,
) -> torch.Tensor:
    if str(model_name) == "MICN":
        bsz, _, ng = enc.shape
        dec_known = enc
        dec_zeros = torch.zeros((bsz, int(pred_len), ng), dtype=enc.dtype, device=enc.device)
        return torch.cat([dec_known, dec_zeros], dim=1)
    return dec


def _make_dec_types_for_transform_model(
    model_name: str,
    enc_types: torch.Tensor,
    dec_types: torch.Tensor,
    *,
    pred_len: int,
    label_len: int,
    seq_len: int,
) -> torch.Tensor:
    if str(model_name) == "MICN":
        last = enc_types[:, -1:].repeat(1, int(pred_len))
        return torch.cat([enc_types, last], dim=1)
    return dec_types


def _extract_primary_tensor_output_simple(raw_output: Any) -> torch.Tensor:
    if torch.is_tensor(raw_output):
        return raw_output
    if isinstance(raw_output, (tuple, list)):
        for item in raw_output:
            if torch.is_tensor(item):
                return item
    if isinstance(raw_output, dict):
        for key in ["output", "outputs", "prediction", "pred", "forecast", "y_hat", "logits"]:
            val = raw_output.get(key, None)
            if torch.is_tensor(val):
                return val
    for attr in ["output", "outputs", "prediction", "pred", "forecast", "y_hat", "logits"]:
        if hasattr(raw_output, attr):
            val = getattr(raw_output, attr)
            if torch.is_tensor(val):
                return val
    raise RuntimeError(f"Could not extract a tensor output from type={type(raw_output)}")


def maybe_wrap_celltype_conditioning(
    *, model: nn.Module, n_genes: int, device: torch.device
) -> nn.Module:
    if not bool(ARGS.TRANS_USE_CELLTYPE_CONDITIONING):
        return model
    wrapped = CellTypeConditionWrapper(
        model,
        n_celltypes=int(N_CELLTYPES),
        n_genes=int(n_genes),
        init_std=float(ARGS.TRANS_CELLTYPE_EMB_INIT_STD),
        scale=float(ARGS.TRANS_CELLTYPE_EMB_SCALE),
    )
    try:
        wrapped = wrapped.to(device)
    except Exception:
        pass
    return wrapped


def call_transform_model(
    *,
    model_name: str,
    model: nn.Module,
    enc: torch.Tensor,
    dec: torch.Tensor,
    enc_types: torch.Tensor,
    dec_types: torch.Tensor,
) -> torch.Tensor:
    enc_mark = make_zero_mark(enc)
    dec_mark = make_zero_mark(dec)
    try:
        raw_out = model(enc, enc_mark, dec, dec_mark, enc_types=enc_types, dec_types=dec_types)
    except TypeError as exc:
        msg = str(exc)
        if "unexpected keyword" in msg or "enc_types" in msg or "dec_types" in msg:
            raw_out = model(enc, enc_mark, dec, dec_mark)
        else:
            raise
    out = _extract_primary_tensor_output_simple(raw_out)
    if out.ndim == 3:
        out = out[:, -1, :]
    elif out.ndim != 2:
        raise RuntimeError(f"[{model_name}] Unsupported output tensor shape: {tuple(out.shape)}")
    if out.device != enc.device:
        out = out.to(enc.device)
    return out


def predict_all_transform(
    *,
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
    apply_retrieval: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds, truths, day_ids = ([], [], [])
    model.eval()
    with torch.no_grad(), autocast(enabled=bool(fp16)):
        for batch in loader:
            enc = batch["encoder_tensor"].to(device)
            dec = batch["decoder_tensor"].to(device)
            enc_types = batch["encoder_type_ids"].to(device)
            dec_types = batch["decoder_type_ids"].to(device)
            y_true = batch["target_tensor"].to(device)[:, 0, :]
            dec = _make_dec_for_transform_model(
                model_name,
                enc,
                dec,
                pred_len=int(ARGS.PRED_LEN),
                label_len=effective_trans_label_len(),
                seq_len=int(ARGS.SEQ_LEN),
            )
            dec_types = _make_dec_types_for_transform_model(
                model_name,
                enc_types,
                dec_types,
                pred_len=int(ARGS.PRED_LEN),
                label_len=effective_trans_label_len(),
                seq_len=int(ARGS.SEQ_LEN),
            )
            y_pred = call_transform_model(
                model_name=model_name,
                model=model,
                enc=enc,
                dec=dec,
                enc_types=enc_types,
                dec_types=dec_types,
            )
            if apply_retrieval and TRANS_RETRIEVER is not None:
                y_pred = TRANS_RETRIEVER.apply(y_pred, batch["target_timestamp_index"].to(device))
            preds.append(y_pred.detach().cpu())
            truths.append(y_true.detach().cpu())
            day_ids.append(batch["target_timestamp_index"].cpu())
    return (torch.cat(preds, dim=0), torch.cat(truths, dim=0), torch.cat(day_ids, dim=0).long())


def transform_unique_target_cells_from_loader(loader: DataLoader) -> np.ndarray:
    """Return sorted unique target cell indices represented by a TRANS loader."""
    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        subset_positions = np.asarray(dataset.indices, dtype=np.int64)
    else:
        base_dataset = dataset
        subset_positions = None
    if not isinstance(base_dataset, TrajectoryWindowDataset):
        raise TypeError(f"Expected TrajectoryWindowDataset, got {type(base_dataset)}")
    cache_name = "_all_target_cell_indices_cache"
    if not hasattr(base_dataset, cache_name):
        all_ids = np.empty((len(base_dataset.index_pairs),), dtype=np.int64)
        target_offset = int(base_dataset.seq_len) + int(base_dataset.pred_len) - 1
        for i, (traj_idx, window_start) in enumerate(base_dataset.index_pairs):
            all_ids[i] = int(
                base_dataset.trajectories[int(traj_idx), int(window_start) + target_offset]
            )
        setattr(base_dataset, cache_name, all_ids)
    all_ids = getattr(base_dataset, cache_name)
    if subset_positions is not None:
        all_ids = all_ids[subset_positions]
    return np.unique(np.asarray(all_ids, dtype=np.int64))


def evaluate_transform_loader_dual_weighting(
    *,
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    fp16: bool,
    apply_retrieval: bool,
) -> Tuple[UnifiedEvalResult, UnifiedEvalResult, float]:
    """
    Evaluate TRANS with both benchmark weightings in ONE forward pass.

    occurrence result:
        Original behavior. Every trajectory-window occurrence is counted.

    unique result:
        Duplicate predictions for the same target cell are averaged first, then
        each unique biological target cell contributes once.
    """
    occurrence_acc = StreamingDayPCCAccumulator(n_genes=int(N_GENES))
    unique_cells = transform_unique_target_cells_from_loader(loader)
    unique_cells = unique_cells[(unique_cells >= 0) & (unique_cells < int(N_CELLS))]
    pred_sum = np.zeros((int(unique_cells.size), int(N_GENES)), dtype=np.float32)
    pred_count = np.zeros((int(unique_cells.size),), dtype=np.int64)
    model.eval()
    with torch.no_grad(), autocast(enabled=bool(fp16)):
        for batch in loader:
            enc = batch["encoder_tensor"].to(device)
            dec = batch["decoder_tensor"].to(device)
            enc_types = batch["encoder_type_ids"].to(device)
            dec_types = batch["decoder_type_ids"].to(device)
            y_true = batch["target_tensor"].to(device)[:, 0, :]
            day_idx = batch["target_timestamp_index"].to(device)
            dec = _make_dec_for_transform_model(
                model_name,
                enc,
                dec,
                pred_len=int(ARGS.PRED_LEN),
                label_len=effective_trans_label_len(),
                seq_len=int(ARGS.SEQ_LEN),
            )
            dec_types = _make_dec_types_for_transform_model(
                model_name,
                enc_types,
                dec_types,
                pred_len=int(ARGS.PRED_LEN),
                label_len=effective_trans_label_len(),
                seq_len=int(ARGS.SEQ_LEN),
            )
            y_pred = call_transform_model(
                model_name=model_name,
                model=model,
                enc=enc,
                dec=dec,
                enc_types=enc_types,
                dec_types=dec_types,
            )
            if apply_retrieval and TRANS_RETRIEVER is not None:
                y_pred = TRANS_RETRIEVER.apply(y_pred, day_idx)
            occurrence_acc.update(y_pred=y_pred, y_true=y_true, time_idx=day_idx)
            cell_ids = (
                batch["target_cell_index"].detach().cpu().numpy().astype(np.int64, copy=False)
            )
            positions = np.searchsorted(unique_cells, cell_ids)
            valid = (positions >= 0) & (positions < unique_cells.size)
            if np.any(valid):
                valid_positions = positions[valid]
                valid_cell_ids = cell_ids[valid]
                equality = unique_cells[valid_positions] == valid_cell_ids
                valid_indices = np.flatnonzero(valid).astype(np.int64)
                valid = np.zeros_like(valid, dtype=bool)
                valid[valid_indices[equality]] = True
            if np.any(valid):
                pos = positions[valid]
                pred_np = y_pred.detach().cpu().numpy().astype(np.float32, copy=False)[valid]
                np.add.at(pred_sum, pos, pred_np)
                np.add.at(pred_count, pos, 1)
    occurrence_result, occurrence_mse = occurrence_acc.finalize()
    unique_acc = StreamingDayPCCAccumulator(n_genes=int(N_GENES))
    valid_unique = np.flatnonzero(pred_count > 0).astype(np.int64)
    chunk_size = 8192
    for start in range(0, int(valid_unique.size), chunk_size):
        ids = valid_unique[start : start + chunk_size]
        cell_ids = unique_cells[ids]
        pred = pred_sum[ids] / pred_count[ids, None].astype(np.float32)
        true = ADATA_X[cell_ids, :].toarray().astype(np.float32, copy=False)
        days = ORDINAL_TIME[cell_ids].astype(np.int64, copy=False)
        unique_acc.update(
            y_pred=torch.from_numpy(pred),
            y_true=torch.from_numpy(true),
            time_idx=torch.from_numpy(days),
        )
    unique_result, _ = unique_acc.finalize()
    return (occurrence_result, unique_result, float(occurrence_mse))


def mse_over_transform(
    *, model_name: str, model: nn.Module, loader: DataLoader, device: torch.device, fp16: bool
) -> float:
    mse = nn.MSELoss(reduction="mean")
    total, n_total = (0.0, 0)
    model.eval()
    with torch.no_grad(), autocast(enabled=bool(fp16)):
        for batch in loader:
            enc = batch["encoder_tensor"].to(device)
            dec = batch["decoder_tensor"].to(device)
            enc_types = batch["encoder_type_ids"].to(device)
            dec_types = batch["decoder_type_ids"].to(device)
            y_true = batch["target_tensor"].to(device)[:, 0, :]
            dec = _make_dec_for_transform_model(
                model_name,
                enc,
                dec,
                pred_len=int(ARGS.PRED_LEN),
                label_len=effective_trans_label_len(),
                seq_len=int(ARGS.SEQ_LEN),
            )
            dec_types = _make_dec_types_for_transform_model(
                model_name,
                enc_types,
                dec_types,
                pred_len=int(ARGS.PRED_LEN),
                label_len=effective_trans_label_len(),
                seq_len=int(ARGS.SEQ_LEN),
            )
            y_pred = call_transform_model(
                model_name=model_name,
                model=model,
                enc=enc,
                dec=dec,
                enc_types=enc_types,
                dec_types=dec_types,
            )
            if y_pred.device != y_true.device:
                y_pred = y_pred.to(y_true.device)
            loss = mse(y_pred, y_true)
            total += float(loss.detach().cpu()) * int(y_true.shape[0])
            n_total += int(y_true.shape[0])
    return float(total / max(1, n_total))


def build_checkpoint_dir(model_tag: str) -> str:
    path = os.path.join(CKPT_DIR, model_tag)
    os.makedirs(path, exist_ok=True)
    return path


def save_model_checkpoint(
    *, model_tag: str, model: nn.Module, metadata: Dict[str, Any], logger: logging.Logger
) -> str:
    metadata = dict(metadata)
    metadata["training_implementation"] = "bloom.training.trainer"
    metadata["model_family"] = MODEL_FAMILY
    ckpt_dir = build_checkpoint_dir(model_tag)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    meta_path = os.path.join(ckpt_dir, "metadata.json")
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[{model_tag}] checkpoint saved to {ckpt_path}")
    return ckpt_path


def load_saved_checkpoint_for_eval(
    *, model_tag: str, model: nn.Module
) -> Tuple[str, Dict[str, Any]]:
    if ARGS.EXISTING_CHECKPOINT_ROOT is not None:
        ckpt_dir = os.path.join(str(ARGS.EXISTING_CHECKPOINT_ROOT), str(model_tag))
    else:
        ckpt_dir = build_checkpoint_dir(model_tag)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    meta_path = os.path.join(ckpt_dir, "metadata.json")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Saved checkpoint not found: {ckpt_path}")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=False)
    return (ckpt_path, metadata)


def should_skip_transform_spec_before_building(spec: TransformRunSpec) -> Optional[str]:
    """
    Return a human-readable skip reason, or None if we should try the model.

    These are safety skips.  They keep the whole benchmark running even when a
    particular upstream foundation wrapper is not suitable for the current mode.
    """
    if str(spec.model_name) in set((str(x) for x in ARGS.TRANS_SKIP_MODELS)):
        return "model explicitly listed in --TRANS_SKIP_MODELS"
    if (
        str(spec.model_name) in TRANS_FOUNDATION_MODEL_SET
        and str(spec.run_mode) == "scratch"
        and (not bool(ARGS.FOUNDATION_ENABLE_SCRATCH))
    ):
        return "foundation scratch mode disabled by default because several upstream wrappers return None or expect pretrained objects; pass --FOUNDATION_ENABLE_SCRATCH to force it"
    if (
        str(spec.model_name) in TRANS_FOUNDATION_MODEL_SET
        and str(spec.run_mode) == "zero_shot_no_ckpt"
        and (not bool(ARGS.FOUNDATION_ENABLE_ZERO_SHOT_NO_CKPT))
    ):
        return "foundation zero_shot_no_ckpt disabled by --no-FOUNDATION_ENABLE_ZERO_SHOT_NO_CKPT"
    if str(spec.model_name) == "SCINet" and int(ARGS.SEQ_LEN) < 8:
        return "SCINet skipped for SEQ_LEN < 8 because it commonly creates zero-width tensors on short histories"
    return None


def train_one_transform_spec(
    *, spec: TransformRunSpec, device: torch.device, fp16: bool, logger: logging.Logger
) -> Dict[str, Any]:
    if TRANS_DATA is None:
        return {
            "family": "TRANS",
            "model": spec.model_name,
            "run_mode": spec.run_mode,
            "mean_cell_pcc": float("nan"),
            "mean_gene_pcc": float("nan"),
            "seconds": 0.0,
            "skipped": True,
            "skip_reason": "TRANS data unavailable for this run",
        }
    if int(ARGS.SEQ_LEN) <= 0:
        return {
            "family": "TRANS",
            "model": spec.model_name,
            "run_mode": spec.run_mode,
            "mean_cell_pcc": float("nan"),
            "mean_gene_pcc": float("nan"),
            "seconds": 0.0,
            "skipped": True,
            "skip_reason": "TRANS does not support SEQ_LEN=0",
        }
    safety_skip_reason = should_skip_transform_spec_before_building(spec)
    if safety_skip_reason is not None:
        return {
            "family": "TRANS",
            "model": spec.model_name,
            "run_mode": spec.run_mode,
            "checkpoint_label": spec.checkpoint_label,
            "checkpoint_source": spec.checkpoint_source,
            "mean_cell_pcc": float("nan"),
            "mean_gene_pcc": float("nan"),
            "seconds": 0.0,
            "skipped": True,
            "skip_reason": safety_skip_reason,
        }
    if spec.run_mode in {"zero_shot_ckpt", "finetune_ckpt"} and spec.checkpoint_source is None:
        return {
            "family": "TRANS",
            "model": spec.model_name,
            "run_mode": spec.run_mode,
            "checkpoint_label": spec.checkpoint_label,
            "mean_cell_pcc": float("nan"),
            "mean_gene_pcc": float("nan"),
            "seconds": 0.0,
            "skipped": True,
            "skip_reason": "no checkpoint spec supplied for requested checkpoint run mode",
        }
    attempted_batch = int(ARGS.TRANS_BATCH)
    attempt_index = 0
    t_start = time.time()
    while attempted_batch >= 1:
        attempt_index += 1
        logger.info(
            f"[{spec.tag}] attempt {attempt_index}: trying TRANS batch_size={attempted_batch}"
        )
        tr_loader, va_loader, te_loader = TRANS_DATA.make_loaders(batch_size=attempted_batch)
        eval_loader = va_loader if str(ARGS.EVAL_SPLIT) == "val" else te_loader
        resolved_checkpoint_source = resolve_foundation_checkpoint_source(
            source=spec.checkpoint_source, logger=logger, model_tag=spec.tag
        )
        model = build_trans_base_model(
            model_name=spec.model_name,
            run_mode=spec.run_mode,
            n_genes=int(N_GENES),
            seq_len=int(ARGS.SEQ_LEN),
            label_len=effective_trans_label_len(),
            pred_len=int(ARGS.PRED_LEN),
            d_model=int(ARGS.TRANS_D_MODEL),
            n_heads=int(ARGS.TRANS_N_HEADS),
            e_layers=int(ARGS.TRANS_E_LAYERS),
            d_layers=int(ARGS.TRANS_D_LAYERS),
            d_ff=int(ARGS.TRANS_D_FF),
            device=device,
            foundation_checkpoint_source=resolved_checkpoint_source,
            foundation_checkpoint_label=spec.checkpoint_label,
        )
        model = maybe_wrap_celltype_conditioning(model=model, n_genes=int(N_GENES), device=device)
        try:
            model = model.to(device)
        except Exception:
            pass
        nparams = sum((p.numel() for p in model.parameters() if p.requires_grad))
        logger.info(f"[{spec.tag}] trainable params = {nparams / 1000000.0:.2f}M")
        try:
            if spec.run_mode in {"zero_shot_ckpt", "finetune_ckpt"}:
                source_for_generic = (
                    str(resolved_checkpoint_source)
                    if resolved_checkpoint_source is not None
                    else ""
                )
                is_local_single_weight_file = os.path.isfile(source_for_generic)
                if bool(ARGS.FOUNDATION_FORCE_GENERIC_LOAD) or is_local_single_weight_file:
                    generic_load_checkpoint_into_model(
                        model=model,
                        checkpoint_source=source_for_generic,
                        model_name=spec.tag,
                        logger=logger,
                    )
                else:
                    logger.info(
                        f"[{spec.tag}] checkpoint source was passed into the model config; generic post-hoc loading is skipped. source={source_for_generic!r}. Use --FOUNDATION_FORCE_GENERIC_LOAD only for compatible local state_dict checkpoints."
                    )
            if str(ARGS.EXECUTION_MODE) == "eval_only":
                ckpt_path, meta = load_saved_checkpoint_for_eval(model_tag=spec.tag, model=model)
                logger.info(f"[{spec.tag}] eval_only loaded saved checkpoint {ckpt_path}")
                set_global_seed(
                    seed=int(ARGS.SEED), deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN)
                )
                if bool(ARGS.TRANS_EVAL_BOTH_WEIGHTINGS):
                    eval_res, eval_res_unique, _ = evaluate_transform_loader_dual_weighting(
                        model_name=spec.model_name,
                        model=model,
                        loader=eval_loader,
                        device=device,
                        fp16=fp16,
                        apply_retrieval=False,
                    )
                else:
                    y_pred, y_true, day_idx = predict_all_transform(
                        model_name=spec.model_name,
                        model=model,
                        loader=eval_loader,
                        device=device,
                        fp16=fp16,
                        apply_retrieval=False,
                    )
                    eval_res = evaluate_predictions_unified(
                        y_pred=y_pred, y_true=y_true, time_idx=day_idx
                    )
                    eval_res_unique = None
                out = {
                    "family": "TRANS",
                    "model": spec.model_name,
                    "run_mode": spec.run_mode,
                    "checkpoint_label": spec.checkpoint_label,
                    "checkpoint_source": (
                        resolved_checkpoint_source
                        if "resolved_checkpoint_source" in locals()
                        else spec.checkpoint_source
                    ),
                    "batch_size_used": int(attempted_batch),
                    "batch_size_attempts": int(attempt_index),
                    "seconds": float(time.time() - t_start),
                    "mean_cell_pcc": float(eval_res.mean_cell_pcc),
                    "mean_gene_pcc": float(eval_res.mean_gene_pcc),
                    "best_epoch": meta.get("best_epoch", None),
                    "best_stop": meta.get("best_stop", float("nan")),
                    "skipped": False,
                }
                out.update(eval_res.day_metrics)
                if eval_res_unique is not None:
                    out["mean_cell_pcc_unique"] = float(eval_res_unique.mean_cell_pcc)
                    out["mean_gene_pcc_unique"] = float(eval_res_unique.mean_gene_pcc)
                    out["n_cells_occurrence"] = int(eval_res.n_cells)
                    out["n_cells_unique"] = int(eval_res_unique.n_cells)
                    out["reported_lower_cell_pcc"] = float(
                        min(eval_res.mean_cell_pcc, eval_res_unique.mean_cell_pcc)
                    )
                    out["reported_lower_gene_pcc"] = float(
                        min(eval_res.mean_gene_pcc, eval_res_unique.mean_gene_pcc)
                    )
                    out.update(
                        {f"{k}_unique": float(v) for k, v in eval_res_unique.day_metrics.items()}
                    )
                return out
            trainable = True
            if spec.run_mode in {"zero_shot_no_ckpt", "zero_shot_ckpt"}:
                trainable = False
            elif not any((p.requires_grad for p in model.parameters())):
                trainable = False
            best_stop = math.inf if str(ARGS.TRANS_STOP_METRIC) == "val_loss" else -math.inf
            best_epoch = None
            best_state = None
            no_improve = 0
            epoch_rows: List[Dict[str, Any]] = []

            def eval_val_metrics(train_loss: Optional[float]) -> Dict[str, float]:
                set_global_seed(
                    seed=int(ARGS.SEED), deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN)
                )
                if bool(ARGS.TRANS_EVAL_BOTH_WEIGHTINGS):
                    val_eval, val_eval_unique, val_loss = evaluate_transform_loader_dual_weighting(
                        model_name=spec.model_name,
                        model=model,
                        loader=va_loader,
                        device=device,
                        fp16=fp16,
                        apply_retrieval=False,
                    )
                    metrics = {
                        "val_loss": float(val_loss),
                        "val_pcc_cell": float(val_eval.mean_cell_pcc),
                        "val_pcc_gene": float(val_eval.mean_gene_pcc),
                        "val_pcc_cell_unique": float(val_eval_unique.mean_cell_pcc),
                        "val_pcc_gene_unique": float(val_eval_unique.mean_gene_pcc),
                        "val_pcc_cell_lower": float(
                            min(val_eval.mean_cell_pcc, val_eval_unique.mean_cell_pcc)
                        ),
                        "val_pcc_gene_lower": float(
                            min(val_eval.mean_gene_pcc, val_eval_unique.mean_gene_pcc)
                        ),
                        "val_n_cells_occurrence": int(val_eval.n_cells),
                        "val_n_cells_unique": int(val_eval_unique.n_cells),
                    }
                else:
                    val_loss = mse_over_transform(
                        model_name=spec.model_name,
                        model=model,
                        loader=va_loader,
                        device=device,
                        fp16=fp16,
                    )
                    y_pred_val, y_true_val, day_idx_val = predict_all_transform(
                        model_name=spec.model_name,
                        model=model,
                        loader=va_loader,
                        device=device,
                        fp16=fp16,
                        apply_retrieval=False,
                    )
                    val_eval = evaluate_predictions_unified(
                        y_pred=y_pred_val, y_true=y_true_val, time_idx=day_idx_val
                    )
                    metrics = {
                        "val_loss": float(val_loss),
                        "val_pcc_cell": float(val_eval.mean_cell_pcc),
                        "val_pcc_gene": float(val_eval.mean_gene_pcc),
                    }
                if train_loss is not None:
                    metrics["train_loss"] = float(train_loss)
                return metrics

            if trainable:
                scaler = GradScaler(enabled=bool(fp16))
                opt = torch.optim.Adam(
                    params=[p for p in model.parameters() if p.requires_grad],
                    lr=float(ARGS.TRANS_LR),
                )
                for ep in range(1, int(ARGS.TRANS_EPOCHS) + 1):
                    set_global_seed(
                        seed=int(ARGS.SEED) + int(ep),
                        deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN),
                    )
                    model.train()
                    running = 0.0
                    n_seen = 0
                    for batch in tqdm(tr_loader, desc=f"[{spec.tag}] ep{ep:02d}", leave=False):
                        enc = batch["encoder_tensor"].to(device)
                        dec = batch["decoder_tensor"].to(device)
                        enc_types = batch["encoder_type_ids"].to(device)
                        dec_types = batch["decoder_type_ids"].to(device)
                        y_true = batch["target_tensor"].to(device)[:, 0, :]
                        opt.zero_grad(set_to_none=True)
                        with autocast(enabled=bool(fp16)):
                            y_pred = call_transform_model(
                                model_name=spec.model_name,
                                model=model,
                                enc=enc,
                                dec=dec,
                                enc_types=enc_types,
                                dec_types=dec_types,
                            )
                            if y_pred.device != y_true.device:
                                y_pred = y_pred.to(y_true.device)
                            loss = torch.nn.functional.mse_loss(y_pred, y_true, reduction="mean")
                        if not torch.isfinite(loss):
                            raise RuntimeError(
                                f"[{spec.tag}] non-finite training loss: {float(loss.detach().cpu())}"
                            )
                        scaler.scale(loss).backward()
                        scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_(model.parameters(), float(ARGS.TRANS_GRAD_CLIP))
                        scaler.step(opt)
                        scaler.update()
                        bsz = int(y_true.shape[0])
                        running += float(loss.detach().cpu()) * bsz
                        n_seen += bsz
                    train_loss = float(running / max(1, n_seen))
                    metrics = eval_val_metrics(train_loss=train_loss)
                    stop_metric = str(ARGS.TRANS_STOP_METRIC)
                    stop_value = float(metrics[stop_metric])
                    stop_mode = "min" if stop_metric == "val_loss" else "max"
                    improved = (
                        stop_value < best_stop - float(ARGS.TRANS_EARLY_STOPPING_MIN_DELTA)
                        if stop_mode == "min"
                        else stop_value > best_stop + float(ARGS.TRANS_EARLY_STOPPING_MIN_DELTA)
                    )
                    if improved:
                        best_stop = float(stop_value)
                        best_epoch = int(ep)
                        best_state = clone_state_dict_to_cpu(model)
                        no_improve = 0
                    else:
                        no_improve += 1
                    row = {
                        "epoch": int(ep),
                        "train_loss": float(train_loss),
                        "val_loss": float(metrics["val_loss"]),
                        "val_pcc_cell": float(metrics["val_pcc_cell"]),
                        "val_pcc_gene": float(metrics["val_pcc_gene"]),
                        "val_pcc_cell_unique": float(metrics.get("val_pcc_cell_unique", np.nan)),
                        "val_pcc_gene_unique": float(metrics.get("val_pcc_gene_unique", np.nan)),
                        "val_pcc_cell_lower": float(metrics.get("val_pcc_cell_lower", np.nan)),
                        "val_pcc_gene_lower": float(metrics.get("val_pcc_gene_lower", np.nan)),
                        "val_n_cells_occurrence": int(metrics.get("val_n_cells_occurrence", 0)),
                        "val_n_cells_unique": int(metrics.get("val_n_cells_unique", 0)),
                        "stop_value": float(stop_value),
                        "no_improve": int(no_improve),
                    }
                    epoch_rows.append(row)
                    unique_log = ""
                    if "val_pcc_gene_unique" in metrics:
                        unique_log = f" val_pcc_cell_unique={metrics['val_pcc_cell_unique']:.4f} val_pcc_gene_unique={metrics['val_pcc_gene_unique']:.4f} val_pcc_cell_lower={metrics['val_pcc_cell_lower']:.4f} val_pcc_gene_lower={metrics['val_pcc_gene_lower']:.4f} val_n_occurrence={metrics['val_n_cells_occurrence']} val_n_unique={metrics['val_n_cells_unique']}"
                    logger.info(
                        f"[{spec.tag}] epoch {ep:03d} train_loss={train_loss:.6f} val_loss={metrics['val_loss']:.6f} val_pcc_cell={metrics['val_pcc_cell']:.4f} val_pcc_gene={metrics['val_pcc_gene']:.4f}{unique_log} {stop_metric}={stop_value:.6g} best={best_stop:.6g}@ep{best_epoch} no_improve={no_improve}/{ARGS.TRANS_EARLY_STOPPING_PATIENCE}"
                    )
                    if no_improve >= int(ARGS.TRANS_EARLY_STOPPING_PATIENCE):
                        break
                if bool(ARGS.TRANS_LOAD_BEST_AT_END) and best_state is not None:
                    model.load_state_dict(best_state, strict=False)
            else:
                logger.info(f"[{spec.tag}] no training for run_mode={spec.run_mode}")
                metrics0 = eval_val_metrics(train_loss=None)
                best_epoch = 0
                best_stop = float(metrics0[str(ARGS.TRANS_STOP_METRIC)])
                epoch_rows.append(
                    {
                        "epoch": 0,
                        "train_loss": np.nan,
                        "val_loss": float(metrics0["val_loss"]),
                        "val_pcc_cell": float(metrics0["val_pcc_cell"]),
                        "val_pcc_gene": float(metrics0["val_pcc_gene"]),
                        "val_pcc_cell_unique": float(metrics0.get("val_pcc_cell_unique", np.nan)),
                        "val_pcc_gene_unique": float(metrics0.get("val_pcc_gene_unique", np.nan)),
                        "val_pcc_cell_lower": float(metrics0.get("val_pcc_cell_lower", np.nan)),
                        "val_pcc_gene_lower": float(metrics0.get("val_pcc_gene_lower", np.nan)),
                        "val_n_cells_occurrence": int(metrics0.get("val_n_cells_occurrence", 0)),
                        "val_n_cells_unique": int(metrics0.get("val_n_cells_unique", 0)),
                        "stop_value": float(best_stop),
                        "no_improve": 0,
                    }
                )
            metadata = {
                "family": "TRANS",
                "model": spec.model_name,
                "run_mode": spec.run_mode,
                "checkpoint_label": spec.checkpoint_label,
                "checkpoint_source": (
                    resolved_checkpoint_source
                    if "resolved_checkpoint_source" in locals()
                    else spec.checkpoint_source
                ),
                "batch_size_used": int(attempted_batch),
                "best_epoch": best_epoch,
                "best_stop": best_stop,
                "seq_len": int(ARGS.SEQ_LEN),
                "label_len": effective_trans_label_len(),
                "pred_len": int(ARGS.PRED_LEN),
                "epoch_history": epoch_rows,
            }
            ckpt_path = None
            if bool(ARGS.SAVE_CHECKPOINTS):
                ckpt_path = save_model_checkpoint(
                    model_tag=spec.tag, model=model, metadata=metadata, logger=logger
                )
            if str(ARGS.EXECUTION_MODE) == "train_only":
                return {
                    "family": "TRANS",
                    "model": spec.model_name,
                    "run_mode": spec.run_mode,
                    "checkpoint_label": spec.checkpoint_label,
                    "checkpoint_source": (
                        resolved_checkpoint_source
                        if "resolved_checkpoint_source" in locals()
                        else spec.checkpoint_source
                    ),
                    "checkpoint_path": ckpt_path,
                    "batch_size_used": int(attempted_batch),
                    "batch_size_attempts": int(attempt_index),
                    "seconds": float(time.time() - t_start),
                    "val_loss": (
                        float(epoch_rows[-1]["val_loss"]) if len(epoch_rows) > 0 else float("nan")
                    ),
                    "best_epoch": best_epoch,
                    "best_stop": float(best_stop),
                    "skipped": False,
                }
            set_global_seed(seed=int(ARGS.SEED), deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN))
            if bool(ARGS.TRANS_EVAL_BOTH_WEIGHTINGS):
                eval_res, eval_res_unique, _ = evaluate_transform_loader_dual_weighting(
                    model_name=spec.model_name,
                    model=model,
                    loader=eval_loader,
                    device=device,
                    fp16=fp16,
                    apply_retrieval=False,
                )
            else:
                y_pred, y_true, day_idx = predict_all_transform(
                    model_name=spec.model_name,
                    model=model,
                    loader=eval_loader,
                    device=device,
                    fp16=fp16,
                    apply_retrieval=False,
                )
                eval_res = evaluate_predictions_unified(
                    y_pred=y_pred, y_true=y_true, time_idx=day_idx
                )
                eval_res_unique = None
            out = {
                "family": "TRANS",
                "model": spec.model_name,
                "run_mode": spec.run_mode,
                "checkpoint_label": spec.checkpoint_label,
                "checkpoint_source": (
                    resolved_checkpoint_source
                    if "resolved_checkpoint_source" in locals()
                    else spec.checkpoint_source
                ),
                "checkpoint_path": ckpt_path,
                "batch_size_used": int(attempted_batch),
                "batch_size_attempts": int(attempt_index),
                "seconds": float(time.time() - t_start),
                "mean_cell_pcc": float(eval_res.mean_cell_pcc),
                "mean_gene_pcc": float(eval_res.mean_gene_pcc),
                "val_loss": (
                    float(epoch_rows[-1]["val_loss"]) if len(epoch_rows) > 0 else float("nan")
                ),
                "best_epoch": best_epoch,
                "best_stop": float(best_stop),
                "skipped": False,
            }
            out.update(eval_res.day_metrics)
            if eval_res_unique is not None:
                out["mean_cell_pcc_unique"] = float(eval_res_unique.mean_cell_pcc)
                out["mean_gene_pcc_unique"] = float(eval_res_unique.mean_gene_pcc)
                out["n_cells_occurrence"] = int(eval_res.n_cells)
                out["n_cells_unique"] = int(eval_res_unique.n_cells)
                out["reported_lower_cell_pcc"] = float(
                    min(eval_res.mean_cell_pcc, eval_res_unique.mean_cell_pcc)
                )
                out["reported_lower_gene_pcc"] = float(
                    min(eval_res.mean_gene_pcc, eval_res_unique.mean_gene_pcc)
                )
                out.update(
                    {f"{k}_unique": float(v) for k, v in eval_res_unique.day_metrics.items()}
                )
            return out
        except Exception as exc:
            if is_memory_related_runtime_error(exc):
                logger.exception(
                    f"[{spec.tag}] memory-related failure at batch_size={attempted_batch}"
                )
                release_memory_after_failure()
                if attempted_batch == 1:
                    raise
                attempted_batch = max(1, attempted_batch // 2)
                logger.info(f"[{spec.tag}] retry with smaller batch_size={attempted_batch}")
                continue
            raise
    raise RuntimeError(f"[{spec.tag}] exhausted adaptive batch attempts")


@dataclass(frozen=True)
# BLOOM: model configuration, direction-aware population context and training.
class FMVariantSpec:
    base_mode: Literal["standard", "context_gaussian"]
    use_history: bool
    celltype_mode: Literal["none", "soft", "hard"]
    use_traj_soft: bool
    bio_ablation: str = "bio0_baseline"

    @property
    def history_days_used(self) -> int:
        if not bool(self.use_history):
            return 1
        return int(ARGS.SEQ_LEN)

    @property
    def is_valid_for_current_seq(self) -> bool:
        if bool(self.use_history) and int(ARGS.SEQ_LEN) <= 0:
            return False
        return True

    @property
    def bio_level(self) -> int:
        return int(BIO_ABLATION_LEVEL[str(self.bio_ablation)])

    def uses_bio_component(self, component: str) -> bool:
        return str(component) in BIO_ABLATION_COMPONENTS[str(self.bio_ablation)]

    @property
    def tag(self) -> str:
        base = f"FM_{self.base_mode}_hist{int(self.use_history)}_ctype_{self.celltype_mode}_traj{int(self.use_traj_soft)}"
        suffix_parts: List[str] = []
        try:
            if str(ARGS.FM_RESIDUAL_MODE) != "none":
                suffix_parts.append(f"resid_{ARGS.FM_RESIDUAL_MODE}")
            if str(ARGS.FM_PREDICTION_MODE) != "flow_sample":
                suffix_parts.append(f"pred_{ARGS.FM_PREDICTION_MODE}")
            if str(ARGS.FM_TEMPORAL_DIRECTION) != "forward":
                suffix_parts.append(f"dir_{ARGS.FM_TEMPORAL_DIRECTION}")
            if float(ARGS.FM_CONTINUOUS_TIME_WEIGHT) > 0.0:
                ctime_tag = format(float(ARGS.FM_CONTINUOUS_TIME_WEIGHT), ".6g").replace(".", "p")
                suffix_parts.append(f"ctime{ctime_tag}")
        except NameError:
            pass
        if len(suffix_parts) > 0:
            base = base + "_" + "_".join(suffix_parts)
        if str(self.bio_ablation) != "bio0_baseline":
            base = base + "__" + str(self.bio_ablation)
        return base


def fm_bio_unavailable_reason(variant: FMVariantSpec) -> Optional[str]:
    """Return a precise skip reason instead of running an invalid duplicate ablation."""
    if BIO_PRIOR_DATA is None or variant.bio_level == 0:
        return None
    for component in ("actual_time", "hierarchy", "graph", "anatomy", "drivers", "grn"):
        if variant.uses_bio_component(component) and (
            not bool(BIO_PRIOR_DATA.available.get(component, False))
        ):
            if bool(ARGS.BIO_ALLOW_MISSING_COMPONENTS):
                continue
            return f"{variant.bio_ablation} requires unavailable biological component: {component}"
    return None


def estimate_fm_memory_priority(variant: FMVariantSpec) -> Tuple[int, int, int, int, int, int]:
    """
    Legacy helper retained for backward compatibility.

    In the earlier version, FM variants were generated by a Cartesian product and
    then sorted by this memory-oriented priority.  The new default behavior uses
    the explicit user-editable DEFAULT_FM_VARIANT_SPECS list above, so this
    function is only used when ARGS.FM_VARIANTS is intentionally empty.
    """
    celltype_priority = {"none": 2, "soft": 1, "hard": 0}
    base_priority = {"context_gaussian": 1, "standard": 0}
    return (
        int(variant.bio_level),
        int(bool(variant.use_history)),
        int(bool(variant.use_traj_soft)),
        int(celltype_priority.get(str(variant.celltype_mode), 0)),
        int(base_priority.get(str(variant.base_mode), 0)),
        int(variant.history_days_used),
    )


def parse_one_fm_variant_spec(spec: str) -> FMVariantSpec:
    """
    Parse one FM variant string into an FMVariantSpec.

    Accepted examples:
        FM_context_gaussian_hist1_ctype_soft_traj1
        FM_standard_hist0_ctype_hard_traj0

    For convenience, the leading "FM_" prefix is optional, so this also works:
        context_gaussian_hist1_ctype_soft_traj1

    The parser is intentionally strict after removing the optional prefix.  This
    prevents silent mistakes such as "hist2", "ctype_softt", or "trajTrue".
    """
    raw_spec = str(spec).strip()
    if raw_spec == "":
        raise ValueError("Empty FM variant spec is not valid.")
    if raw_spec.startswith("#"):
        raise ValueError(
            f"Commented FM variant spec should have been removed before parsing: {raw_spec!r}"
        )
    if raw_spec.startswith("FM_"):
        raw_spec = raw_spec[len("FM_") :]
    if raw_spec.startswith("context_gaussian_"):
        base_mode = "context_gaussian"
        rest = raw_spec[len("context_gaussian_") :]
    elif raw_spec.startswith("standard_"):
        base_mode = "standard"
        rest = raw_spec[len("standard_") :]
    else:
        raise ValueError(
            f"Invalid FM base mode in spec={spec!r}. Expected prefix 'FM_context_gaussian_' or 'FM_standard_'."
        )
    parts = rest.split("_")
    if len(parts) != 4 or parts[1] != "ctype":
        raise ValueError(
            f"Invalid FM variant spec={spec!r}. Expected format: FM_<standard|context_gaussian>_hist<0|1>_ctype_<none|soft|hard>_traj<0|1>"
        )
    hist_part = parts[0]
    celltype_mode = parts[2]
    traj_part = parts[3]
    if hist_part not in {"hist0", "hist1"}:
        raise ValueError(f"Invalid history part in FM spec={spec!r}. Expected hist0 or hist1.")
    if celltype_mode not in {"none", "soft", "hard"}:
        raise ValueError(
            f"Invalid celltype mode in FM spec={spec!r}. Expected none, soft, or hard."
        )
    if traj_part not in {"traj0", "traj1"}:
        raise ValueError(f"Invalid trajectory part in FM spec={spec!r}. Expected traj0 or traj1.")
    return FMVariantSpec(
        base_mode=base_mode,
        use_history=bool(int(hist_part.replace("hist", ""))),
        celltype_mode=celltype_mode,
        use_traj_soft=bool(int(traj_part.replace("traj", ""))),
    )


def build_fm_variants_from_explicit_list(fm_variant_specs: Sequence[str]) -> List[FMVariantSpec]:
    """
    Build FM variants from the explicit user-editable list.

    Duplicate model strings are ignored after the first occurrence.  This keeps
    accidental duplicate entries from running the same expensive model twice.
    """
    variants: List[FMVariantSpec] = []
    seen_tags: Set[str] = set()
    for raw_spec in _drop_commented_or_empty_items(fm_variant_specs):
        variant = parse_one_fm_variant_spec(raw_spec)
        if variant.tag in seen_tags:
            continue
        variants.append(variant)
        seen_tags.add(variant.tag)
    return variants


def build_fm_variants_from_legacy_cartesian_product() -> List[FMVariantSpec]:
    """
    Original FM variant construction path.

    This is kept as a fallback so old command-line workflows can still generate
    variants from FLOW_BASE_MODES, FM_USE_HISTORY_OPTIONS, FM_CELLTYPE_MODES, and
    FM_USE_TRAJ_SOFT_OPTIONS if ARGS.FM_VARIANTS is empty.
    """
    variants: List[FMVariantSpec] = []
    for base_mode in ARGS.FLOW_BASE_MODES:
        for use_history in ARGS.FM_USE_HISTORY_OPTIONS:
            for celltype_mode in ARGS.FM_CELLTYPE_MODES:
                for use_traj_soft in ARGS.FM_USE_TRAJ_SOFT_OPTIONS:
                    variants.append(
                        FMVariantSpec(
                            base_mode=str(base_mode),
                            use_history=bool(int(use_history)),
                            celltype_mode=str(celltype_mode),
                            use_traj_soft=bool(int(use_traj_soft)),
                        )
                    )
    variants = sorted(variants, key=lambda x: estimate_fm_memory_priority(x), reverse=True)
    return variants


def build_fm_variants() -> List[FMVariantSpec]:
    """
    Build the FM model list used by the benchmark.

    New preferred behavior:
        Use ARGS.FM_VARIANTS exactly in the provided order.

    Legacy fallback:
        If ARGS.FM_VARIANTS is empty, generate the old Cartesian product and
        sort it using estimate_fm_memory_priority(...).
    """
    explicit_specs = _drop_commented_or_empty_items(ARGS.FM_VARIANTS)
    if len(explicit_specs) > 0:
        base_variants = build_fm_variants_from_explicit_list(explicit_specs)
    else:
        base_variants = build_fm_variants_from_legacy_cartesian_product()
    bio_tags = list(ARGS.FM_BIO_ABLATIONS) if len(ARGS.FM_BIO_ABLATIONS) > 0 else ["bio0_baseline"]
    expanded: List[FMVariantSpec] = []
    seen: Set[str] = set()
    for bio_tag in bio_tags:
        if bio_tag not in BIO_ABLATION_LEVEL:
            raise ValueError(f"Unknown FM biological ablation: {bio_tag!r}")
        for base_variant in base_variants:
            variant = FMVariantSpec(
                base_mode=base_variant.base_mode,
                use_history=base_variant.use_history,
                celltype_mode=base_variant.celltype_mode,
                use_traj_soft=base_variant.use_traj_soft,
                bio_ablation=str(bio_tag),
            )
            if variant.tag not in seen:
                expanded.append(variant)
                seen.add(variant.tag)
    return expanded


class RawTimestampTargetCellDataset(Dataset):

    def __init__(
        self,
        *,
        adata_X: sp.csr_matrix,
        ordinal_time: np.ndarray,
        celltype_codes: np.ndarray,
        split_name: Literal["train", "val", "test"],
        train_timestamps: Set[int],
        val_timestamps: Set[int],
        test_timestamps: Set[int],
        min_target_ts: int,
        target_cell_mask: Optional[np.ndarray] = None,
        biological_ordinal_time: Optional[np.ndarray] = None,
        direction_id: int = 0,
        allow_empty: bool = False,
    ):
        super().__init__()
        self.adata_X = adata_X
        self.ordinal_time = np.asarray(ordinal_time, dtype=np.int32)
        self.biological_ordinal_time = (
            self.ordinal_time
            if biological_ordinal_time is None
            else np.asarray(biological_ordinal_time, dtype=np.int32)
        )
        self.celltype_codes = np.asarray(celltype_codes, dtype=np.int32)
        self.direction_id = int(direction_id)
        if self.direction_id not in (0, 1):
            raise ValueError(
                f"direction_id must be 0 (forward) or 1 (reverse), got {self.direction_id}"
            )
        self.min_target_ts = int(min_target_ts)
        self.target_cell_mask = target_cell_mask
        if split_name == "train":
            allowed = train_timestamps
        elif split_name == "val":
            allowed = val_timestamps
        elif split_name == "test":
            allowed = test_timestamps
        else:
            raise ValueError(split_name)
        self.allowed_target_timestamps = set((int(x) for x in allowed))
        target_mask = np.isin(self.ordinal_time, list(self.allowed_target_timestamps)) & (
            self.ordinal_time >= int(self.min_target_ts)
        )
        if target_cell_mask is not None:
            target_mask = target_mask & np.asarray(target_cell_mask, dtype=bool)
        self.target_cell_indices = np.flatnonzero(target_mask).astype(np.int64)
        if self.target_cell_indices.size == 0 and (not bool(allow_empty)):
            raise RuntimeError(f"No target cells found for FM split={split_name}")

    def __len__(self) -> int:
        return int(self.target_cell_indices.size)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        cell_idx = int(self.target_cell_indices[int(index)])
        t_idx = int(self.ordinal_time[cell_idx])
        y = torch.tensor(self.adata_X[cell_idx, :].toarray().reshape(-1), dtype=torch.float32)
        target_type = int(self.celltype_codes[cell_idx])
        return {
            "target_cell_index": cell_idx,
            "target_timestamp_index": t_idx,
            "biological_timestamp_index": int(self.biological_ordinal_time[cell_idx]),
            "direction_id": self.direction_id,
            "target_type_id": target_type,
            "target_x": y,
        }


class FMBidirectionalTargetDataset(Dataset):
    """Concatenate forward and reverse examples while retaining their direction."""

    def __init__(
        self,
        forward_dataset: RawTimestampTargetCellDataset,
        reverse_dataset: RawTimestampTargetCellDataset,
    ):
        super().__init__()
        self.forward_dataset = forward_dataset
        self.reverse_dataset = reverse_dataset
        self._forward_len = len(forward_dataset)
        self.target_cell_indices = np.concatenate(
            [forward_dataset.target_cell_indices, reverse_dataset.target_cell_indices]
        ).astype(np.int64, copy=False)
        self.direction_ids = np.concatenate(
            [
                np.zeros(len(forward_dataset), dtype=np.int64),
                np.ones(len(reverse_dataset), dtype=np.int64),
            ]
        )

    def __len__(self) -> int:
        return int(self._forward_len + len(self.reverse_dataset))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        if index < self._forward_len:
            return self.forward_dataset[index]
        return self.reverse_dataset[index - self._forward_len]

    def direction_ids_for_positions(self, positions: Optional[np.ndarray] = None) -> np.ndarray:
        if positions is None:
            return self.direction_ids
        return self.direction_ids[np.asarray(positions, dtype=np.int64)]


class FMDirectionBalancedSampler(torch.utils.data.Sampler):
    """Exact-ratio sampler that avoids replacement until a direction is exhausted."""

    def __init__(
        self, *, direction_ids: np.ndarray, forward_fraction: float, generator: torch.Generator
    ):
        self.direction_ids = np.asarray(direction_ids, dtype=np.int64)
        self.forward_fraction = float(forward_fraction)
        self.generator = generator
        self.forward_positions = np.flatnonzero(self.direction_ids == 0).astype(np.int64)
        self.reverse_positions = np.flatnonzero(self.direction_ids == 1).astype(np.int64)
        if self.forward_positions.size == 0 or self.reverse_positions.size == 0:
            raise RuntimeError(
                "FMDirectionBalancedSampler requires at least one example from each direction"
            )
        self.num_samples = int(self.direction_ids.size)

    def __len__(self) -> int:
        return self.num_samples

    def _draw(self, pool: np.ndarray, count: int) -> List[int]:
        drawn: List[int] = []
        remaining = int(count)
        while remaining > 0:
            order = torch.randperm(int(pool.size), generator=self.generator).cpu().numpy()
            take = min(remaining, int(pool.size))
            drawn.extend((int(x) for x in pool[order[:take]]))
            remaining -= take
        return drawn

    def __iter__(self):
        n_forward = int(round(self.num_samples * self.forward_fraction))
        n_forward = max(1, min(self.num_samples - 1, n_forward))
        n_reverse = self.num_samples - n_forward
        positions = self._draw(self.forward_positions, n_forward)
        positions.extend(self._draw(self.reverse_positions, n_reverse))
        mixed_order = torch.randperm(len(positions), generator=self.generator).cpu().tolist()
        return iter((positions[int(i)] for i in mixed_order))


def build_traj_prev_map_if_possible() -> Optional[Dict[int, np.ndarray]]:
    if TRAJECTORIES is None:
        return None
    prev_map_list: Dict[int, List[int]] = defaultdict(list)
    for traj in tqdm(TRAJECTORIES, desc="build_fm_traj_hint_map", leave=False):
        traj_arr = np.asarray(traj, dtype=np.int64)
        if traj_arr.size < 2:
            continue
        for pos in range(1, traj_arr.size):
            cur_cell = int(traj_arr[pos])
            prev_cell = int(traj_arr[pos - 1])
            prev_map_list[cur_cell].append(prev_cell)
    return {k: np.asarray(v, dtype=np.int64) for k, v in prev_map_list.items()}


def build_traj_next_map_if_possible() -> Optional[Dict[int, np.ndarray]]:
    """Return current-cell -> next-cell candidates for a reverse-time FM."""
    if TRAJECTORIES is None:
        return None
    next_map_list: Dict[int, List[int]] = defaultdict(list)
    for traj in tqdm(TRAJECTORIES, desc="build_fm_traj_next_map", leave=False):
        traj_arr = np.asarray(traj, dtype=np.int64)
        if traj_arr.size < 2:
            continue
        for pos in range(0, traj_arr.size - 1):
            cur_cell = int(traj_arr[pos])
            next_cell = int(traj_arr[pos + 1])
            next_map_list[cur_cell].append(next_cell)
    return {k: np.asarray(v, dtype=np.int64) for k, v in next_map_list.items()}


class FMContextBank:

    def __init__(
        self,
        *,
        adata_X: sp.csr_matrix,
        cells_by_time: List[np.ndarray],
        cells_by_time_type: List[Dict[int, np.ndarray]],
        time_type_counts: np.ndarray,
        celltype_codes: np.ndarray,
        traj_prev_map: Optional[Dict[int, np.ndarray]],
        max_cells_per_day: int,
        dataset_random_seed: int,
        n_genes: int,
        actual_time_by_ordinal: Optional[np.ndarray] = None,
        direction_id: int = 0,
    ):
        self.adata_X = adata_X
        self.cells_by_time = cells_by_time
        self.cells_by_time_type = cells_by_time_type
        self.time_type_counts = time_type_counts
        self.celltype_codes = celltype_codes
        self.traj_prev_map = traj_prev_map
        self.max_cells_per_day = int(max_cells_per_day)
        self.dataset_random_seed = int(dataset_random_seed)
        self.n_genes = int(n_genes)
        self.direction_id = int(direction_id)
        if self.direction_id not in (0, 1):
            raise ValueError(
                f"direction_id must be 0 (forward) or 1 (reverse), got {self.direction_id}"
            )
        self.actual_time_by_ordinal = (
            np.arange(len(self.cells_by_time), dtype=np.float32)
            if actual_time_by_ordinal is None
            else np.asarray(actual_time_by_ordinal, dtype=np.float32)
        )
        self._context_cache: Dict[Tuple[int, int, str, int], Dict[str, np.ndarray]] = {}
        self._traj_hint_cache: Dict[int, Tuple[np.ndarray, float]] = {}

    def _build_one_context(
        self,
        *,
        target_timestamp: int,
        target_type_id: int,
        celltype_mode: str,
        history_days_used: int,
    ) -> Dict[str, np.ndarray]:
        key = (
            int(target_timestamp),
            int(target_type_id),
            str(celltype_mode),
            int(history_days_used),
        )
        if key in self._context_cache:
            return self._context_cache[key]
        d = int(history_days_used)
        k = int(self.max_cells_per_day)
        g = int(self.n_genes)
        context_x = np.zeros((d, k, g), dtype=np.float32)
        context_mask = np.zeros((d, k), dtype=np.float32)
        day_valid = np.zeros((d,), dtype=np.float32)
        target_type_prop = np.zeros((d,), dtype=np.float32)
        for out_pos, lag in enumerate(range(d, 0, -1)):
            src_t = int(target_timestamp) - int(lag)
            if src_t < 0:
                continue
            if str(celltype_mode) == "hard":
                source_pool = self.cells_by_time_type[src_t].get(
                    int(target_type_id), np.asarray([], dtype=np.int64)
                )
            else:
                source_pool = self.cells_by_time[src_t]
            if source_pool.size == 0:
                continue
            seed = (
                int(self.dataset_random_seed) * 1000003
                + int(target_timestamp) * 10007
                + int(lag) * 101
                + int(target_type_id) * 17
            )
            sampled = deterministic_sample(source_pool, int(k), seed)
            x_day = self.adata_X[sampled, :].toarray().astype(np.float32, copy=False)
            n_keep = int(x_day.shape[0])
            context_x[out_pos, :n_keep, :] = x_day
            context_mask[out_pos, :n_keep] = 1.0
            day_valid[out_pos] = 1.0
            full_day_n = max(1, int(self.cells_by_time[src_t].size))
            type_n = int(self.time_type_counts[src_t, int(target_type_id)])
            target_type_prop[out_pos] = float(type_n / full_day_n)
        out = {
            "context_x": context_x,
            "context_mask": context_mask,
            "day_valid": day_valid,
            "target_type_prop": target_type_prop,
        }
        self._context_cache[key] = out
        return out

    def _get_traj_hint(self, target_cell_index: int) -> Tuple[np.ndarray, float]:
        if target_cell_index in self._traj_hint_cache:
            return self._traj_hint_cache[target_cell_index]
        g = int(self.n_genes)
        if self.traj_prev_map is None:
            hint = np.zeros((g,), dtype=np.float32)
            out = (hint, 0.0)
            self._traj_hint_cache[target_cell_index] = out
            return out
        prev_cells = self.traj_prev_map.get(int(target_cell_index), None)
        if prev_cells is None or len(prev_cells) == 0:
            hint = np.zeros((g,), dtype=np.float32)
            out = (hint, 0.0)
            self._traj_hint_cache[target_cell_index] = out
            return out
        x_prev = (
            self.adata_X[np.asarray(prev_cells, dtype=np.int64), :]
            .toarray()
            .astype(np.float32, copy=False)
        )
        hint = np.mean(x_prev, axis=0).astype(np.float32, copy=False)
        out = (hint, 1.0)
        self._traj_hint_cache[target_cell_index] = out
        return out

    def build_batch(
        self,
        *,
        target_timestamp_index: torch.Tensor,
        target_type_id: torch.Tensor,
        target_cell_index: torch.Tensor,
        variant: FMVariantSpec,
        device: torch.device,
        direction_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b = int(target_timestamp_index.shape[0])
        d = int(variant.history_days_used)
        k = int(self.max_cells_per_day)
        g = int(self.n_genes)
        batch_context_x = np.zeros((b, d, k, g), dtype=np.float32)
        batch_context_mask = np.zeros((b, d, k), dtype=np.float32)
        batch_day_valid = np.zeros((b, d), dtype=np.float32)
        batch_target_type_prop = np.zeros((b, d), dtype=np.float32)
        batch_traj_hint = np.zeros((b, g), dtype=np.float32)
        batch_traj_hint_mask = np.zeros((b,), dtype=np.float32)
        t_np = target_timestamp_index.detach().cpu().numpy().astype(np.int64)
        ty_np = target_type_id.detach().cpu().numpy().astype(np.int64)
        cell_np = target_cell_index.detach().cpu().numpy().astype(np.int64)
        for i in range(b):
            one = self._build_one_context(
                target_timestamp=int(t_np[i]),
                target_type_id=int(ty_np[i]),
                celltype_mode=str(variant.celltype_mode),
                history_days_used=int(variant.history_days_used),
            )
            batch_context_x[i] = one["context_x"]
            batch_context_mask[i] = one["context_mask"]
            batch_day_valid[i] = one["day_valid"]
            batch_target_type_prop[i] = one["target_type_prop"]
            need_traj_hint = (
                bool(variant.use_traj_soft)
                or str(ARGS.FM_RESIDUAL_MODE) == "trajectory"
                or str(ARGS.FM_PREDICTION_MODE) in {"residual_point", "blend"}
            )
            if need_traj_hint:
                hint, hint_mask = self._get_traj_hint(int(cell_np[i]))
                batch_traj_hint[i] = hint
                batch_traj_hint_mask[i] = hint_mask
        out = {
            "context_x": torch.tensor(batch_context_x, dtype=torch.float32, device=device),
            "context_mask": torch.tensor(batch_context_mask, dtype=torch.float32, device=device),
            "day_valid": torch.tensor(batch_day_valid, dtype=torch.float32, device=device),
            "target_type_prop": torch.tensor(
                batch_target_type_prop, dtype=torch.float32, device=device
            ),
            "traj_hint": torch.tensor(batch_traj_hint, dtype=torch.float32, device=device),
            "traj_hint_mask": torch.tensor(
                batch_traj_hint_mask, dtype=torch.float32, device=device
            ),
            "direction_id": torch.full((b,), self.direction_id, dtype=torch.long, device=device),
        }
        if variant.bio_level == 0 or BIO_PRIOR_DATA is None:
            return out
        if variant.uses_bio_component("actual_time"):
            time_features = np.zeros((b, d, 3), dtype=np.float32)
            for i in range(b):
                target_t = int(t_np[i])
                target_value = float(self.actual_time_by_ordinal[target_t])
                for out_pos, lag in enumerate(range(d, 0, -1)):
                    src_t = target_t - int(lag)
                    if src_t < 0:
                        continue
                    source_value = float(self.actual_time_by_ordinal[src_t])
                    time_features[i, out_pos] = (
                        source_value,
                        target_value,
                        target_value - source_value,
                    )
            out["actual_time_features"] = torch.tensor(
                time_features, dtype=torch.float32, device=device
            )
        if variant.uses_bio_component("hierarchy"):
            if BIO_PRIOR_DATA.hierarchy_codes:
                hierarchy = np.stack(
                    [codes[cell_np] for codes in BIO_PRIOR_DATA.hierarchy_codes], axis=1
                )
            else:
                hierarchy = np.zeros((b, 0), dtype=np.int32)
            out["hierarchy_codes"] = torch.tensor(hierarchy, dtype=torch.long, device=device)
        if variant.uses_bio_component("graph"):
            graph_codes = BIO_PRIOR_DATA.graph_cell_codes[cell_np]
            graph_features = BIO_PRIOR_DATA.graph_node_features[graph_codes]
            out["graph_node_codes"] = torch.tensor(graph_codes, dtype=torch.long, device=device)
            out["graph_node_features"] = torch.tensor(
                graph_features, dtype=torch.float32, device=device
            )
        if variant.uses_bio_component("anatomy"):
            if BIO_PRIOR_DATA.anatomy_codes:
                anatomy = np.stack(
                    [codes[cell_np] for codes in BIO_PRIOR_DATA.anatomy_codes], axis=1
                )
            else:
                anatomy = np.zeros((b, 0), dtype=np.int32)
            out["anatomy_codes"] = torch.tensor(anatomy, dtype=torch.long, device=device)
            out["numeric_context"] = torch.tensor(
                BIO_PRIOR_DATA.numeric_context[cell_np], dtype=torch.float32, device=device
            )
        if variant.uses_bio_component("drivers"):
            graph_codes = BIO_PRIOR_DATA.graph_cell_codes[cell_np]
            driver = BIO_PRIOR_DATA.driver_program_by_node[graph_codes]
            out["driver_program"] = torch.tensor(driver, dtype=torch.float32, device=device)
        return out

    def build_population_batch(
        self,
        *,
        target_timestamp_index: torch.Tensor,
        variant: FMVariantSpec,
        device: torch.device,
        direction_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build target-label-free history context for the composition head.

        Sampling uses the complete source-stage population and a fixed neutral
        cache key, so neither the target cell type nor target cell identity can
        leak into this context. Target-specific hierarchy/graph/anatomy/driver
        tensors are intentionally absent and ignored by the population encoder.
        """
        b = int(target_timestamp_index.shape[0])
        d = int(variant.history_days_used)
        k = int(self.max_cells_per_day)
        g = int(self.n_genes)
        batch_context_x = np.zeros((b, d, k, g), dtype=np.float32)
        batch_context_mask = np.zeros((b, d, k), dtype=np.float32)
        batch_day_valid = np.zeros((b, d), dtype=np.float32)
        t_np = target_timestamp_index.detach().cpu().numpy().astype(np.int64)
        for i in range(b):
            one = self._build_one_context(
                target_timestamp=int(t_np[i]),
                target_type_id=0,
                celltype_mode="soft",
                history_days_used=d,
            )
            batch_context_x[i] = one["context_x"]
            batch_context_mask[i] = one["context_mask"]
            batch_day_valid[i] = one["day_valid"]
        out: Dict[str, torch.Tensor] = {
            "context_x": torch.tensor(batch_context_x, dtype=torch.float32, device=device),
            "context_mask": torch.tensor(batch_context_mask, dtype=torch.float32, device=device),
            "day_valid": torch.tensor(batch_day_valid, dtype=torch.float32, device=device),
            "target_type_prop": torch.zeros((b, d), dtype=torch.float32, device=device),
            "traj_hint": torch.zeros((b, g), dtype=torch.float32, device=device),
            "traj_hint_mask": torch.zeros((b,), dtype=torch.float32, device=device),
            "direction_id": torch.full((b,), self.direction_id, dtype=torch.long, device=device),
        }
        if variant.uses_bio_component("actual_time"):
            time_features = np.zeros((b, d, 3), dtype=np.float32)
            for i in range(b):
                target_t = int(t_np[i])
                target_value = float(self.actual_time_by_ordinal[target_t])
                for out_pos, lag in enumerate(range(d, 0, -1)):
                    src_t = target_t - int(lag)
                    if src_t < 0:
                        continue
                    source_value = float(self.actual_time_by_ordinal[src_t])
                    time_features[i, out_pos] = (
                        source_value,
                        target_value,
                        target_value - source_value,
                    )
            out["actual_time_features"] = torch.tensor(
                time_features, dtype=torch.float32, device=device
            )
        return out


class FMBidirectionalContextBank:
    """Dispatch mixed batches to direction-specific causal context banks."""

    def __init__(self, *, forward_bank: FMContextBank, reverse_bank: FMContextBank):
        self.forward_bank = forward_bank
        self.reverse_bank = reverse_bank

    @staticmethod
    def _merge_parts(
        parts: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]], batch_size: int
    ) -> Dict[str, torch.Tensor]:
        merged: Dict[str, torch.Tensor] = {}
        for positions, part in parts:
            for key, value in part.items():
                if key not in merged:
                    out_shape = (int(batch_size),) + tuple(value.shape[1:])
                    merged[key] = torch.zeros(out_shape, dtype=value.dtype, device=value.device)
                merged[key].index_copy_(0, positions.to(value.device), value)
        return merged

    def build_batch(
        self,
        *,
        target_timestamp_index: torch.Tensor,
        target_type_id: torch.Tensor,
        target_cell_index: torch.Tensor,
        variant: FMVariantSpec,
        device: torch.device,
        direction_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if direction_id is None:
            raise RuntimeError("A bidirectional context batch requires direction_id")
        direction_cpu = direction_id.detach().cpu().long()
        parts: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = []
        for one_direction, bank in ((0, self.forward_bank), (1, self.reverse_bank)):
            positions = torch.nonzero(direction_cpu == one_direction, as_tuple=False).reshape(-1)
            if positions.numel() == 0:
                continue
            part = bank.build_batch(
                target_timestamp_index=target_timestamp_index.index_select(
                    0, positions.to(target_timestamp_index.device)
                ),
                target_type_id=target_type_id.index_select(0, positions.to(target_type_id.device)),
                target_cell_index=target_cell_index.index_select(
                    0, positions.to(target_cell_index.device)
                ),
                variant=variant,
                device=device,
            )
            parts.append((positions, part))
        if not parts:
            raise RuntimeError("Bidirectional context dispatch received an empty batch")
        return self._merge_parts(parts, int(direction_cpu.shape[0]))

    def build_population_batch(
        self,
        *,
        target_timestamp_index: torch.Tensor,
        variant: FMVariantSpec,
        device: torch.device,
        direction_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if direction_id is None:
            raise RuntimeError("A bidirectional population context requires direction_id")
        direction_cpu = direction_id.detach().cpu().long()
        parts: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = []
        for one_direction, bank in ((0, self.forward_bank), (1, self.reverse_bank)):
            positions = torch.nonzero(direction_cpu == one_direction, as_tuple=False).reshape(-1)
            if positions.numel() == 0:
                continue
            part = bank.build_population_batch(
                target_timestamp_index=target_timestamp_index.index_select(
                    0, positions.to(target_timestamp_index.device)
                ),
                variant=variant,
                device=device,
            )
            parts.append((positions, part))
        if not parts:
            raise RuntimeError("Bidirectional population dispatch received an empty batch")
        return self._merge_parts(parts, int(direction_cpu.shape[0]))


class FMDataBundle:

    def __init__(self):
        self.temporal_direction = str(ARGS.FM_TEMPORAL_DIRECTION)
        base_actual_time = (
            BIO_PRIOR_DATA.actual_time_by_ordinal
            if BIO_PRIOR_DATA is not None
            else np.arange(TOTAL_TIMESTAMP, dtype=np.float32)
        )
        reverse_ordinal_time = (int(TOTAL_TIMESTAMP) - 1 - ORDINAL_TIME).astype(np.int32)
        reverse_cells_by_time = list(reversed(CELLS_BY_TIME))
        reverse_cells_by_time_type = list(reversed(CELLS_BY_TIME_TYPE))
        reverse_time_type_counts = TIME_TYPE_COUNTS[::-1].copy()
        reverse_actual_time = np.asarray(base_actual_time, dtype=np.float32)[::-1].copy()
        if self.temporal_direction == "forward":
            self.model_ordinal_time = ORDINAL_TIME
            self.model_cells_by_time = CELLS_BY_TIME
            self.model_cells_by_time_type = CELLS_BY_TIME_TYPE
            self.model_time_type_counts = TIME_TYPE_COUNTS
            self.model_traj_hint_map = FM_TRAJ_PREV_MAP
            self.model_actual_time = np.asarray(base_actual_time, dtype=np.float32)
        elif self.temporal_direction == "reverse":
            self.model_ordinal_time = reverse_ordinal_time
            self.model_cells_by_time = reverse_cells_by_time
            self.model_cells_by_time_type = reverse_cells_by_time_type
            self.model_time_type_counts = reverse_time_type_counts
            self.model_traj_hint_map = FM_TRAJ_NEXT_MAP
            self.model_actual_time = reverse_actual_time
        elif self.temporal_direction == "bidirectional":
            self.model_ordinal_time = ORDINAL_TIME
            self.model_cells_by_time = CELLS_BY_TIME
            self.model_cells_by_time_type = CELLS_BY_TIME_TYPE
            self.model_time_type_counts = TIME_TYPE_COUNTS
            self.model_traj_hint_map = FM_TRAJ_PREV_MAP
            self.model_actual_time = np.asarray(base_actual_time, dtype=np.float32)
            self.reverse_model_ordinal_time = reverse_ordinal_time
            self.reverse_model_cells_by_time = reverse_cells_by_time
            self.reverse_model_cells_by_time_type = reverse_cells_by_time_type
            self.reverse_model_time_type_counts = reverse_time_type_counts
            self.reverse_model_traj_hint_map = FM_TRAJ_NEXT_MAP
            self.reverse_model_actual_time = reverse_actual_time
        else:
            raise ValueError(f"Unsupported FM_TEMPORAL_DIRECTION={self.temporal_direction!r}")
        self.has_required_traj_hints = (
            FM_TRAJ_PREV_MAP is not None and FM_TRAJ_NEXT_MAP is not None
            if self.temporal_direction == "bidirectional"
            else self.model_traj_hint_map is not None
        )
        min_target_ts = max(1, int(ARGS.SEQ_LEN))
        split_min_target_ts = 0 if self.temporal_direction == "bidirectional" else min_target_ts
        self.train_timestamps, self.val_timestamps, self.test_timestamps = make_timestamp_splits(
            total_timestamp=TOTAL_TIMESTAMP,
            min_target_ts=split_min_target_ts,
            split_method=str(ARGS.SPLIT_METHOD),
            train_ratio=float(ARGS.TRAIN_RATIO),
            valid_ratio=float(ARGS.VALID_RATIO),
            prespecified_test_timestamps=ARGS.PRESPECIFIED_TEST_TIMESTAMPS,
            dataset_random_seed=int(ARGS.DATASET_RANDOM_SEED),
        )
        self.bidirectional_split_repair: Dict[str, Any] = {}
        if self.temporal_direction == "bidirectional":
            if str(ARGS.SPLIT_METHOD) == "random_proportional_split":
                (
                    self.train_timestamps,
                    self.val_timestamps,
                    self.test_timestamps,
                    self.bidirectional_split_repair,
                ) = repair_bidirectional_stage_split_coverage(
                    train_timestamps=self.train_timestamps,
                    val_timestamps=self.val_timestamps,
                    test_timestamps=self.test_timestamps,
                    total_timestamp=int(TOTAL_TIMESTAMP),
                    history_days=int(min_target_ts),
                )
            else:
                self.bidirectional_split_repair = {
                    "applied": False,
                    "reason": "coverage repair is disabled for an explicitly ordered/prespecified split",
                    "split_method": str(ARGS.SPLIT_METHOD),
                }
            LOGGER.info(
                "[BLOOM] temporal_direction=bidirectional; direction_id=0 uses earlier context and direction_id=1 uses later context. Both share one biological-stage split."
            )
            if bool(self.bidirectional_split_repair.get("applied", False)):
                LOGGER.info(
                    "[BLOOM] deterministic direction-coverage repair was applied: "
                    + json.dumps(
                        {
                            "moved_in": self.bidirectional_split_repair.get("moved_in", {}),
                            "moved_out": self.bidirectional_split_repair.get("moved_out", {}),
                        },
                        sort_keys=True,
                    )
                )
            LOGGER.info(
                f"[BLOOM] leakage-safe biological target split\n  train biological timestamps: {sorted(self.train_timestamps)}\n  val   biological timestamps: {sorted(self.val_timestamps)}\n  test  biological timestamps: {sorted(self.test_timestamps)}"
            )
        else:
            LOGGER.info(
                f"[FM] temporal_direction={self.temporal_direction}; model-facing timestamp 0 is "
                + (
                    "the earliest measured stage"
                    if self.temporal_direction == "forward"
                    else "the latest measured stage"
                )
            )
            LOGGER.info(
                f"[FM] split\n  train model timestamps: {sorted(self.train_timestamps)}\n  val   model timestamps: {sorted(self.val_timestamps)}\n  test  model timestamps: {sorted(self.test_timestamps)}"
            )
        self.scale_experiment = str(ARGS.FM_SCALE_EXPERIMENT)
        self.scale_fraction = float(ARGS.FM_SCALE_FRACTION)
        self.scale_seed = int(ARGS.FM_SCALE_SEED)
        self.fixed_scale_row_indices: Optional[np.ndarray] = None
        self.fixed_scale_cell_mask: Optional[np.ndarray] = None
        legacy_all_split_mask = None
        if SUBSAMPLED_TRAJ_CELL_MASK is not None and bool(
            ARGS.TRAJECTORY_SUBSAMPLE_RESTRICT_FM_TARGETS
        ):
            legacy_all_split_mask = SUBSAMPLED_TRAJ_CELL_MASK
            LOGGER.info(
                "[FM] restricting train/val/test target cells to legacy deterministic trajectory subsample mask"
            )
        train_target_mask = legacy_all_split_mask
        val_target_mask = legacy_all_split_mask
        test_target_mask = legacy_all_split_mask
        if self.scale_experiment in {"B", "E"}:
            row_indices, cell_mask = sample_full_trajectory_rows_and_cell_mask(
                fraction=self.scale_fraction,
                seed=self.scale_seed,
                min_rows=int(ARGS.TRAJECTORY_SUBSAMPLE_MIN_ROWS),
            )
            self.fixed_scale_row_indices = row_indices
            self.fixed_scale_cell_mask = cell_mask
            train_target_mask = cell_mask
            val_target_mask = None
            test_target_mask = None
            LOGGER.info(
                f"[FM scale {self.scale_experiment}] fixed training trajectory rows={row_indices.size}/{(TRAJECTORIES_FULL_ARRAY.shape[0] if TRAJECTORIES_FULL_ARRAY is not None else 0)} fraction={self.scale_fraction} seed={self.scale_seed}; validation/test targets remain full"
            )
            try:
                np.save(
                    os.path.join(
                        SUMMARY_DIR, f"{ARGS.MSG}_fm_scale_{self.scale_experiment}_train_rows.npy"
                    ),
                    row_indices,
                )
            except Exception as exc:
                LOGGER.info(
                    f"[FM scale {self.scale_experiment}] could not save fixed row indices: {exc}"
                )
        if self.scale_experiment == "C":
            train_target_mask = None
            val_target_mask = None
            test_target_mask = None
            LOGGER.info(
                f"[FM scale C] rotating trajectory-row training subset each epoch: fraction={self.scale_fraction} base_seed={self.scale_seed}; validation/test targets remain full"
            )
        if self.scale_experiment == "D":
            train_target_mask = None
            val_target_mask = None
            test_target_mask = None
            LOGGER.info(
                f"[FM scale D] full training pool with fixed max steps/epoch={int(ARGS.FM_MAX_TRAIN_STEPS_PER_EPOCH)}; validation/test targets remain full"
            )
        if self.temporal_direction == "bidirectional":
            reverse_split_sets = {
                "train": {int(TOTAL_TIMESTAMP) - 1 - int(t) for t in self.train_timestamps},
                "val": {int(TOTAL_TIMESTAMP) - 1 - int(t) for t in self.val_timestamps},
                "test": {int(TOTAL_TIMESTAMP) - 1 - int(t) for t in self.test_timestamps},
            }
            forward_common = dict(
                adata_X=ADATA_X,
                ordinal_time=ORDINAL_TIME,
                biological_ordinal_time=ORDINAL_TIME,
                celltype_codes=CELLTYPE_CODES,
                train_timestamps=self.train_timestamps,
                val_timestamps=self.val_timestamps,
                test_timestamps=self.test_timestamps,
                min_target_ts=min_target_ts,
                direction_id=0,
                allow_empty=True,
            )
            reverse_common = dict(
                adata_X=ADATA_X,
                ordinal_time=self.reverse_model_ordinal_time,
                biological_ordinal_time=ORDINAL_TIME,
                celltype_codes=CELLTYPE_CODES,
                train_timestamps=reverse_split_sets["train"],
                val_timestamps=reverse_split_sets["val"],
                test_timestamps=reverse_split_sets["test"],
                min_target_ts=min_target_ts,
                direction_id=1,
                allow_empty=True,
            )
            direction_sizes: Dict[str, Dict[str, int]] = {}
            combined: Dict[str, FMBidirectionalTargetDataset] = {}
            for split_name, target_mask in (
                ("train", train_target_mask),
                ("val", val_target_mask),
                ("test", test_target_mask),
            ):
                forward_dataset = RawTimestampTargetCellDataset(
                    split_name=split_name, target_cell_mask=target_mask, **forward_common
                )
                reverse_dataset = RawTimestampTargetCellDataset(
                    split_name=split_name, target_cell_mask=target_mask, **reverse_common
                )
                direction_sizes[split_name] = {
                    "forward": len(forward_dataset),
                    "reverse": len(reverse_dataset),
                }
                combined[split_name] = FMBidirectionalTargetDataset(
                    forward_dataset, reverse_dataset
                )
                if len(combined[split_name]) == 0:
                    raise RuntimeError(
                        f"Bidirectional FM split={split_name} has no eligible target cells"
                    )
            if direction_sizes["train"]["forward"] == 0 or direction_sizes["train"]["reverse"] == 0:
                raise RuntimeError(
                    "Bidirectional FM requires at least one forward and one reverse TRAIN target. Reduce SEQ_LEN or change the stage split."
                )
            self.bidirectional_direction_sizes = direction_sizes
            self.train_dataset = combined["train"]
            self.val_dataset = combined["val"]
            self.test_dataset = combined["test"]
            LOGGER.info(
                f"[BLOOM] per-direction dataset sizes: {json.dumps(direction_sizes, sort_keys=True)}"
            )
            for split_name in ("train", "val", "test"):
                if min(direction_sizes[split_name].values()) == 0:
                    raise RuntimeError(
                        f"Bidirectional FM split={split_name} must contain both directions, got {direction_sizes[split_name]}. Reduce SEQ_LEN or use a stage split with forward- and reverse-eligible targets."
                    )
        else:
            common_base = dict(
                adata_X=ADATA_X,
                ordinal_time=self.model_ordinal_time,
                biological_ordinal_time=ORDINAL_TIME,
                celltype_codes=CELLTYPE_CODES,
                train_timestamps=self.train_timestamps,
                val_timestamps=self.val_timestamps,
                test_timestamps=self.test_timestamps,
                min_target_ts=min_target_ts,
                direction_id=1 if self.temporal_direction == "reverse" else 0,
            )
            self.train_dataset = RawTimestampTargetCellDataset(
                split_name="train", target_cell_mask=train_target_mask, **common_base
            )
            self.val_dataset = RawTimestampTargetCellDataset(
                split_name="val", target_cell_mask=val_target_mask, **common_base
            )
            self.test_dataset = RawTimestampTargetCellDataset(
                split_name="test", target_cell_mask=test_target_mask, **common_base
            )
        LOGGER.info(
            f"[FM] dataset sizes train={len(self.train_dataset)} val={len(self.val_dataset)} test={len(self.test_dataset)}"
        )
        self._persistent_train_generators: Dict[int, torch.Generator] = {}
        context_common = dict(
            adata_X=ADATA_X,
            celltype_codes=CELLTYPE_CODES,
            max_cells_per_day=int(ARGS.MAX_SOURCE_CELLS_PER_DAY),
            dataset_random_seed=int(ARGS.DATASET_RANDOM_SEED),
            n_genes=int(N_GENES),
        )
        if self.temporal_direction == "bidirectional":
            forward_eval_bank = FMContextBank(
                cells_by_time=CELLS_BY_TIME,
                cells_by_time_type=CELLS_BY_TIME_TYPE,
                time_type_counts=TIME_TYPE_COUNTS,
                traj_prev_map=FM_TRAJ_PREV_MAP,
                actual_time_by_ordinal=np.asarray(base_actual_time, dtype=np.float32),
                direction_id=0,
                **context_common,
            )
            reverse_eval_bank = FMContextBank(
                cells_by_time=reverse_cells_by_time,
                cells_by_time_type=reverse_cells_by_time_type,
                time_type_counts=reverse_time_type_counts,
                traj_prev_map=FM_TRAJ_NEXT_MAP,
                actual_time_by_ordinal=reverse_actual_time,
                direction_id=1,
                **context_common,
            )
            self.eval_context_bank = FMBidirectionalContextBank(
                forward_bank=forward_eval_bank, reverse_bank=reverse_eval_bank
            )
        else:
            self.eval_context_bank = FMContextBank(
                cells_by_time=self.model_cells_by_time,
                cells_by_time_type=self.model_cells_by_time_type,
                time_type_counts=self.model_time_type_counts,
                traj_prev_map=self.model_traj_hint_map,
                actual_time_by_ordinal=self.model_actual_time,
                direction_id=1 if self.temporal_direction == "reverse" else 0,
                **context_common,
            )
        self.train_context_bank = self.eval_context_bank
        if (
            self.scale_experiment in {"B", "E"}
            and self.fixed_scale_cell_mask is not None
            and (self.fixed_scale_row_indices is not None)
        ):
            sampled_cells_by_time, sampled_cells_by_time_type, sampled_counts = (
                build_context_pools_from_cell_mask(self.fixed_scale_cell_mask)
            )
            sampled_traj_array = TRAJECTORIES_FULL_ARRAY[self.fixed_scale_row_indices, :]
            if self.temporal_direction == "bidirectional":
                sampled_forward_bank = FMContextBank(
                    cells_by_time=sampled_cells_by_time,
                    cells_by_time_type=sampled_cells_by_time_type,
                    time_type_counts=sampled_counts,
                    traj_prev_map=build_traj_prev_map_from_array(sampled_traj_array),
                    actual_time_by_ordinal=np.asarray(base_actual_time, dtype=np.float32),
                    direction_id=0,
                    **context_common,
                )
                sampled_reverse_bank = FMContextBank(
                    cells_by_time=list(reversed(sampled_cells_by_time)),
                    cells_by_time_type=list(reversed(sampled_cells_by_time_type)),
                    time_type_counts=sampled_counts[::-1].copy(),
                    traj_prev_map=build_traj_prev_map_from_array(sampled_traj_array[:, ::-1]),
                    actual_time_by_ordinal=reverse_actual_time,
                    direction_id=1,
                    **context_common,
                )
                self.train_context_bank = FMBidirectionalContextBank(
                    forward_bank=sampled_forward_bank, reverse_bank=sampled_reverse_bank
                )
            else:
                if self.temporal_direction == "reverse":
                    sampled_cells_by_time = list(reversed(sampled_cells_by_time))
                    sampled_cells_by_time_type = list(reversed(sampled_cells_by_time_type))
                    sampled_counts = sampled_counts[::-1].copy()
                    sampled_traj_array = sampled_traj_array[:, ::-1]
                sampled_traj_prev_map = build_traj_prev_map_from_array(sampled_traj_array)
                self.train_context_bank = FMContextBank(
                    cells_by_time=sampled_cells_by_time,
                    cells_by_time_type=sampled_cells_by_time_type,
                    time_type_counts=sampled_counts,
                    traj_prev_map=sampled_traj_prev_map,
                    actual_time_by_ordinal=self.model_actual_time,
                    direction_id=1 if self.temporal_direction == "reverse" else 0,
                    **context_common,
                )
            LOGGER.info(
                f"[FM scale {self.scale_experiment}] training context/hint pools restricted to the fixed sampled trajectories; validation/test context/hints remain full"
            )
        self.context_bank = self.eval_context_bank

    def _loader_kwargs(self, batch_size: int) -> Dict[str, Any]:
        return dict(
            batch_size=int(batch_size),
            num_workers=int(ARGS.NUM_WORKERS),
            pin_memory=True,
            worker_init_fn=make_worker_init_fn(int(ARGS.SEED) + 3000),
        )

    def _make_direction_balanced_train_loader(
        self,
        *,
        dataset: Dataset,
        direction_ids: np.ndarray,
        batch_size: int,
        generator: torch.Generator,
    ) -> DataLoader:
        """Sample the two directions at a user-controlled per-epoch ratio."""
        direction_ids = np.asarray(direction_ids, dtype=np.int64)
        n_forward = int(np.sum(direction_ids == 0))
        n_reverse = int(np.sum(direction_ids == 1))
        if n_forward == 0 or n_reverse == 0:
            raise RuntimeError(
                f"Direction-balanced loader needs both directions, got forward={n_forward}, reverse={n_reverse}"
            )
        forward_fraction = float(ARGS.FM_BIDIRECTIONAL_FORWARD_FRACTION)
        sampler = FMDirectionBalancedSampler(
            direction_ids=direction_ids, forward_fraction=forward_fraction, generator=generator
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            generator=generator,
            **self._loader_kwargs(batch_size),
        )

    def make_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        dl_kw = self._loader_kwargs(batch_size)
        generator = make_torch_generator(int(ARGS.SEED) + 4000 + int(batch_size))
        if self.temporal_direction == "bidirectional":
            tr = self._make_direction_balanced_train_loader(
                dataset=self.train_dataset,
                direction_ids=self.train_dataset.direction_ids,
                batch_size=batch_size,
                generator=generator,
            )
        else:
            tr = DataLoader(
                self.train_dataset, shuffle=True, drop_last=True, generator=generator, **dl_kw
            )
        va = DataLoader(self.val_dataset, shuffle=False, drop_last=False, **dl_kw)
        te = DataLoader(self.test_dataset, shuffle=False, drop_last=False, **dl_kw)
        return (tr, va, te)

    def make_train_loader_for_epoch(
        self, *, batch_size: int, epoch: int
    ) -> Tuple[DataLoader, Dict[str, Any]]:
        """
        Return the FM training loader for one epoch.

        For experiment C, a new deterministic trajectory-row subset is sampled
        using seed = FM_SCALE_SEED + epoch - 1. The underlying full train dataset
        is never mutated. All other modes use the ordinary train dataset.
        """
        dl_kw = self._loader_kwargs(batch_size)
        epoch_seed = int(ARGS.SEED) + 4000 + int(batch_size) + int(epoch) * 100003
        if self.scale_experiment != "C":
            if int(batch_size) not in self._persistent_train_generators:
                self._persistent_train_generators[int(batch_size)] = make_torch_generator(
                    int(ARGS.SEED) + 4000 + int(batch_size)
                )
            generator = self._persistent_train_generators[int(batch_size)]
            if self.temporal_direction == "bidirectional":
                loader = self._make_direction_balanced_train_loader(
                    dataset=self.train_dataset,
                    direction_ids=self.train_dataset.direction_ids,
                    batch_size=batch_size,
                    generator=generator,
                )
            else:
                loader = DataLoader(
                    self.train_dataset, shuffle=True, drop_last=True, generator=generator, **dl_kw
                )
            return (
                loader,
                {
                    "mode": self.scale_experiment,
                    "epoch_seed": None,
                    "n_targets": len(self.train_dataset),
                },
            )
        row_seed = int(self.scale_seed) + int(epoch) - 1
        row_indices, cell_mask = sample_full_trajectory_rows_and_cell_mask(
            fraction=self.scale_fraction,
            seed=row_seed,
            min_rows=int(ARGS.TRAJECTORY_SUBSAMPLE_MIN_ROWS),
        )
        full_target_cells = self.train_dataset.target_cell_indices
        selected_positions = np.flatnonzero(cell_mask[full_target_cells]).astype(np.int64)
        if selected_positions.size == 0:
            raise RuntimeError(f"[FM scale C] epoch={epoch} produced zero training targets")
        subset = torch.utils.data.Subset(self.train_dataset, selected_positions.tolist())
        generator = make_torch_generator(epoch_seed)
        if self.temporal_direction == "bidirectional":
            subset_direction_ids = self.train_dataset.direction_ids_for_positions(
                selected_positions
            )
            loader = self._make_direction_balanced_train_loader(
                dataset=subset,
                direction_ids=subset_direction_ids,
                batch_size=batch_size,
                generator=generator,
            )
        else:
            loader = DataLoader(subset, shuffle=True, drop_last=True, generator=generator, **dl_kw)
        if bool(ARGS.FM_SAVE_EPOCH_SUBSAMPLE_INDICES):
            try:
                np.save(
                    os.path.join(
                        SUMMARY_DIR,
                        f"{ARGS.MSG}_fm_scale_C_epoch{int(epoch):03d}_seed{row_seed}_trajectory_rows.npy",
                    ),
                    row_indices,
                )
            except Exception as exc:
                LOGGER.info(f"[FM scale C] could not save epoch {epoch} row indices: {exc}")
        info = {
            "mode": "C",
            "epoch_seed": epoch_seed,
            "trajectory_seed": row_seed,
            "n_trajectory_rows": int(row_indices.size),
            "n_targets": int(selected_positions.size),
            "target_fraction_of_full_train": float(
                selected_positions.size / max(1, len(self.train_dataset))
            ),
        }
        LOGGER.info(
            f"[FM scale C] epoch={epoch} trajectory_seed={row_seed} rows={row_indices.size} targets={selected_positions.size}/{len(self.train_dataset)} ({info['target_fraction_of_full_train']:.4f})"
        )
        return (loader, info)


@dataclass(frozen=True)
class FMReadoutSpec:
    key: str
    kind: str
    blend_weight: Optional[float] = None


def _canonical_blend_key(weight: float) -> str:
    return f"blend_{float(weight):.2f}"


def build_active_fm_readouts() -> List[FMReadoutSpec]:
    """
    Expand the user-editable FM readout list.

    Examples:
        FM_EVAL_READOUTS = [flow_sample, point_head, blend]
        FM_BLEND_WEIGHTS = [0.50, 0.70, 0.85, 1.00]

    becomes:
        flow_sample, point_head, blend_0.50, blend_0.70, blend_0.85, blend_1.00

    If FM_EVAL_READOUTS is empty, the old single FM_PREDICTION_MODE behavior is
    preserved exactly.
    """
    requested = _drop_commented_or_empty_items(ARGS.FM_EVAL_READOUTS)
    out: List[FMReadoutSpec] = []
    if len(requested) == 0:
        mode = str(ARGS.FM_PREDICTION_MODE)
        if mode == "blend":
            w = min(1.0, max(0.0, float(ARGS.FM_POINT_BLEND_WEIGHT)))
            return [FMReadoutSpec(key=_canonical_blend_key(w), kind="blend", blend_weight=w)]
        if mode in {"point_head", "residual_point"}:
            return [FMReadoutSpec(key=mode, kind="point_head", blend_weight=None)]
        return [FMReadoutSpec(key="flow_sample", kind="flow_sample", blend_weight=None)]
    for item in requested:
        item = str(item).strip()
        if item == "blend":
            for raw_w in list(ARGS.FM_BLEND_WEIGHTS):
                w = min(1.0, max(0.0, float(raw_w)))
                out.append(FMReadoutSpec(key=_canonical_blend_key(w), kind="blend", blend_weight=w))
            continue
        if item.startswith("blend_"):
            try:
                w = min(1.0, max(0.0, float(item.split("_", 1)[1])))
            except Exception as exc:
                raise ValueError(
                    f"Invalid explicit FM readout {item!r}; expected blend_0.75 style"
                ) from exc
            out.append(FMReadoutSpec(key=_canonical_blend_key(w), kind="blend", blend_weight=w))
            continue
        if item == "flow_sample":
            out.append(FMReadoutSpec(key="flow_sample", kind="flow_sample", blend_weight=None))
            continue
        if item in {"point_head", "residual_point"}:
            out.append(FMReadoutSpec(key=item, kind="point_head", blend_weight=None))
            continue
        raise ValueError(f"Unsupported FM readout {item!r}")
    deduped: List[FMReadoutSpec] = []
    seen: Set[str] = set()
    for spec in out:
        if spec.key in seen:
            continue
        deduped.append(spec)
        seen.add(spec.key)
    return deduped


def resolve_primary_fm_readout(readouts: Sequence[FMReadoutSpec]) -> str:
    available = [x.key for x in readouts]
    requested = str(ARGS.FM_PRIMARY_READOUT).strip()
    if requested.startswith("blend_"):
        try:
            requested = _canonical_blend_key(float(requested.split("_", 1)[1]))
        except Exception:
            pass
    if requested in available:
        return requested
    if str(ARGS.FM_PREDICTION_MODE) == "blend":
        fallback = _canonical_blend_key(float(ARGS.FM_POINT_BLEND_WEIGHT))
        if fallback in available:
            return fallback
    if "point_head" in available:
        return "point_head"
    return available[0]


def fm_point_and_flow_predictions_from_one_context(
    *,
    model: RawConditionalFlowMatching,
    batch_context: Dict[str, torch.Tensor],
    target_type_id: torch.Tensor,
    variant: FMVariantSpec,
    need_point: bool,
    need_flow: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute context once, then reuse it for point and stochastic-flow readouts."""
    context = model.encode_context(
        batch_context=batch_context, target_type_id=target_type_id, variant=variant
    )
    reference_x = torch.zeros(
        (int(context.shape[0]), int(model.n_genes)), device=context.device, dtype=context.dtype
    )
    point_pred: Optional[torch.Tensor] = None
    if need_point:
        point_pred = model.point_predict_expression(
            context=context, batch_context=batch_context, reference_x=reference_x
        )
    flow_pred: Optional[torch.Tensor] = None
    if need_flow:
        b = int(context.shape[0])
        dt = 1.0 / float(ARGS.FLOW_NUM_STEPS)
        samples: List[torch.Tensor] = []
        for _ in range(int(ARGS.FLOW_NUM_SAMPLES)):
            x = model.draw_x0(context=context, variant=variant, dtype=context.dtype)
            for step in range(int(ARGS.FLOW_NUM_STEPS)):
                tau_value = (step + 0.5) / float(ARGS.FLOW_NUM_STEPS)
                tau = torch.full(
                    (b, 1), fill_value=float(tau_value), device=context.device, dtype=context.dtype
                )
                v = model.velocity_field(x_tau=x, tau=tau, context=context)
                x = x + dt * v
            samples.append(x)
        flow_target_prediction = torch.stack(samples, dim=0).mean(dim=0)
        anchor, _ = model._trajectory_anchor(
            batch_context=batch_context, reference_x=flow_target_prediction
        )
        flow_pred = model._expression_from_flow_target(
            flow_target_prediction=flow_target_prediction, anchor=anchor
        )
    return (point_pred, flow_pred)


def evaluate_fm_readouts_streaming(
    *,
    model: RawConditionalFlowMatching,
    loader: DataLoader,
    variant: FMVariantSpec,
    readouts: Sequence[FMReadoutSpec],
    device: torch.device,
    fp16: bool,
    context_bank: FMContextBank,
) -> Dict[str, Dict[str, float]]:
    """Evaluate many FM readouts in one dataloader pass and one flow sampling pass per batch."""
    readouts = list(readouts)
    accumulators = {spec.key: StreamingDayPCCAccumulator(n_genes=int(N_GENES)) for spec in readouts}
    directional_accumulators: Dict[str, Dict[int, StreamingDayPCCAccumulator]] = {}
    if str(ARGS.FM_TEMPORAL_DIRECTION) == "bidirectional":
        directional_accumulators = {
            spec.key: {
                0: StreamingDayPCCAccumulator(n_genes=int(N_GENES)),
                1: StreamingDayPCCAccumulator(n_genes=int(N_GENES)),
            }
            for spec in readouts
        }
    need_point = any((spec.kind in {"point_head", "blend"} for spec in readouts))
    need_flow = any((spec.kind in {"flow_sample", "blend"} for spec in readouts))
    gate_weighted_sums: Dict[str, float] = defaultdict(float)
    gate_row_count = 0
    gate_day_sums: Dict[Tuple[int, str], float] = defaultdict(float)
    gate_day_counts: Dict[Tuple[int, str], int] = defaultdict(int)
    gate_direction_sums: Dict[Tuple[int, str], float] = defaultdict(float)
    gate_direction_counts: Dict[Tuple[int, str], int] = defaultdict(int)
    gate_direction_day_sums: Dict[Tuple[int, int, str], float] = defaultdict(float)
    gate_direction_day_counts: Dict[Tuple[int, int, str], int] = defaultdict(int)
    composition_true_counts: Dict[int, np.ndarray] = {}
    composition_pred_sums: Dict[int, np.ndarray] = {}
    composition_pred_counts: Dict[int, int] = defaultdict(int)
    model.eval()
    set_global_seed(seed=int(ARGS.SEED), deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN))
    with torch.no_grad(), autocast(enabled=bool(fp16)):
        for batch in loader:
            target_x = batch["target_x"].to(device)
            target_type_id = batch["target_type_id"].to(device)
            day_idx = batch.get("biological_timestamp_index", batch["target_timestamp_index"]).to(
                device
            )
            batch_context = context_bank.build_batch(
                target_timestamp_index=batch["target_timestamp_index"],
                target_type_id=target_type_id,
                target_cell_index=batch["target_cell_index"],
                variant=variant,
                device=device,
                direction_id=batch.get("direction_id"),
            )
            point_pred, flow_pred = fm_point_and_flow_predictions_from_one_context(
                model=model,
                batch_context=batch_context,
                target_type_id=target_type_id,
                variant=variant,
                need_point=need_point,
                need_flow=need_flow,
            )
            one_gate_summary = model.encoder.prior_gate_summary()
            batch_rows = int(target_x.shape[0])
            for gate_name, gate_value in one_gate_summary.items():
                gate_weighted_sums[gate_name] += float(gate_value) * batch_rows
            if one_gate_summary:
                gate_row_count += batch_rows
                day_np_for_gates = day_idx.detach().cpu().numpy().astype(np.int64, copy=False)
                direction_np_for_gates = (
                    batch["direction_id"].detach().cpu().numpy().astype(np.int64, copy=False)
                    if "direction_id" in batch
                    else np.zeros((batch_rows,), dtype=np.int64)
                )
                for gate_name, gate_tensor in model.encoder._last_prior_gate_tensors.items():
                    gate_np = (
                        gate_tensor.detach()
                        .reshape(-1)
                        .cpu()
                        .numpy()
                        .astype(np.float64, copy=False)
                    )
                    for biological_day in np.unique(day_np_for_gates):
                        day_mask = day_np_for_gates == int(biological_day)
                        key = (int(biological_day), str(gate_name))
                        gate_day_sums[key] += float(np.sum(gate_np[day_mask]))
                        gate_day_counts[key] += int(np.sum(day_mask))
                    if str(ARGS.FM_TEMPORAL_DIRECTION) == "bidirectional":
                        for one_direction in (0, 1):
                            direction_mask = direction_np_for_gates == one_direction
                            direction_key = (one_direction, str(gate_name))
                            gate_direction_sums[direction_key] += float(
                                np.sum(gate_np[direction_mask])
                            )
                            gate_direction_counts[direction_key] += int(np.sum(direction_mask))
                            for biological_day in np.unique(day_np_for_gates[direction_mask]):
                                direction_day_mask = direction_mask & (
                                    day_np_for_gates == int(biological_day)
                                )
                                direction_day_key = (
                                    one_direction,
                                    int(biological_day),
                                    str(gate_name),
                                )
                                gate_direction_day_sums[direction_day_key] += float(
                                    np.sum(gate_np[direction_day_mask])
                                )
                                gate_direction_day_counts[direction_day_key] += int(
                                    np.sum(direction_day_mask)
                                )
            if float(ARGS.FM_COMPOSITION_LOSS_WEIGHT) > 0.0:
                population_context = context_bank.build_population_batch(
                    target_timestamp_index=batch["target_timestamp_index"],
                    variant=variant,
                    device=device,
                    direction_id=batch.get("direction_id"),
                )
                composition_prob = (
                    model.predict_celltype_proportions(population_context=population_context)
                    .detach()
                    .cpu()
                    .numpy()
                )
                type_np = target_type_id.detach().cpu().numpy().astype(np.int64, copy=False)
                day_np = day_idx.detach().cpu().numpy().astype(np.int64, copy=False)
                for biological_day in np.unique(day_np):
                    day_mask = day_np == int(biological_day)
                    if int(biological_day) not in composition_true_counts:
                        composition_true_counts[int(biological_day)] = np.zeros(
                            (N_CELLTYPES,), dtype=np.float64
                        )
                        composition_pred_sums[int(biological_day)] = np.zeros(
                            (N_CELLTYPES,), dtype=np.float64
                        )
                    composition_true_counts[int(biological_day)] += np.bincount(
                        type_np[day_mask], minlength=N_CELLTYPES
                    ).astype(np.float64)
                    composition_pred_sums[int(biological_day)] += composition_prob[day_mask].sum(
                        axis=0
                    )
                    composition_pred_counts[int(biological_day)] += int(np.sum(day_mask))
            for spec in readouts:
                if spec.kind == "point_head":
                    if point_pred is None:
                        raise RuntimeError(
                            "point readout requested but point prediction is unavailable"
                        )
                    pred = point_pred
                elif spec.kind == "flow_sample":
                    if flow_pred is None:
                        raise RuntimeError(
                            "flow readout requested but flow prediction is unavailable"
                        )
                    pred = flow_pred
                elif spec.kind == "blend":
                    if point_pred is None or flow_pred is None:
                        raise RuntimeError(
                            "blend readout requested but point/flow prediction is unavailable"
                        )
                    w = float(spec.blend_weight)
                    pred = w * point_pred + (1.0 - w) * flow_pred
                else:
                    raise ValueError(spec.kind)
                accumulators[spec.key].update(y_pred=pred, y_true=target_x, time_idx=day_idx)
                if directional_accumulators:
                    direction_id = batch["direction_id"].long()
                    for one_direction in (0, 1):
                        positions = torch.nonzero(
                            direction_id == one_direction, as_tuple=False
                        ).reshape(-1)
                        if positions.numel() == 0:
                            continue
                        pred_positions = positions.to(pred.device)
                        target_positions = positions.to(target_x.device)
                        day_positions = positions.to(day_idx.device)
                        directional_accumulators[spec.key][one_direction].update(
                            y_pred=pred.index_select(0, pred_positions),
                            y_true=target_x.index_select(0, target_positions),
                            time_idx=day_idx.index_select(0, day_positions),
                        )
    aggregate_gate_summary = {
        name: value / max(1, gate_row_count) for name, value in gate_weighted_sums.items()
    }
    if aggregate_gate_summary:
        model.encoder._last_prior_gate_means = dict(aggregate_gate_summary)
    gate_metrics: Dict[str, float] = {
        f"day{day}_prior_gate_{name}": float(total / max(1, gate_day_counts[day, name]))
        for (day, name), total in sorted(gate_day_sums.items())
    }
    for (one_direction, gate_name), total in sorted(gate_direction_sums.items()):
        direction_name = "forward" if one_direction == 0 else "reverse"
        key = (one_direction, gate_name)
        gate_metrics[f"{direction_name}_prior_gate_{gate_name}"] = float(
            total / max(1, gate_direction_counts[key])
        )
    for (one_direction, day, gate_name), total in sorted(gate_direction_day_sums.items()):
        direction_name = "forward" if one_direction == 0 else "reverse"
        key = (one_direction, day, gate_name)
        gate_metrics[f"{direction_name}_day{day}_prior_gate_{gate_name}"] = float(
            total / max(1, gate_direction_day_counts[key])
        )
    composition_metrics: Dict[str, float] = {}
    composition_jsd: List[float] = []
    composition_tv: List[float] = []
    composition_hellinger: List[float] = []
    for biological_day in sorted(composition_true_counts):
        p = composition_true_counts[biological_day]
        q = composition_pred_sums[biological_day] / max(1, composition_pred_counts[biological_day])
        p = p / max(1e-12, float(np.sum(p)))
        q = q / max(1e-12, float(np.sum(q)))
        midpoint = 0.5 * (p + q)
        eps = 1e-12
        jsd = 0.5 * float(np.sum(p * np.log((p + eps) / (midpoint + eps))))
        jsd += 0.5 * float(np.sum(q * np.log((q + eps) / (midpoint + eps))))
        tv = 0.5 * float(np.sum(np.abs(p - q)))
        hellinger = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
        composition_jsd.append(jsd)
        composition_tv.append(tv)
        composition_hellinger.append(hellinger)
        composition_metrics[f"day{biological_day}_composition_jsd"] = jsd
        composition_metrics[f"day{biological_day}_composition_tv"] = tv
        composition_metrics[f"day{biological_day}_composition_hellinger"] = hellinger
    if composition_jsd:
        composition_metrics["composition_mean_jsd"] = float(np.mean(composition_jsd))
        composition_metrics["composition_mean_tv"] = float(np.mean(composition_tv))
        composition_metrics["composition_mean_hellinger"] = float(np.mean(composition_hellinger))
    results: Dict[str, Dict[str, float]] = {}
    for spec in readouts:
        eval_result, mse = accumulators[spec.key].finalize()
        one: Dict[str, float]
        if directional_accumulators:
            one = {
                "pooled_val_loss": float(mse),
                "pooled_val_pcc_cell": float(eval_result.mean_cell_pcc),
                "pooled_val_pcc_gene": float(eval_result.mean_gene_pcc),
                "n_cells": float(eval_result.n_cells),
            }
            one.update({f"pooled_{k}": float(v) for k, v in eval_result.day_metrics.items()})
            direction_loss: List[float] = []
            direction_cell: List[float] = []
            direction_gene: List[float] = []
            for one_direction, direction_name in ((0, "forward"), (1, "reverse")):
                direction_accumulator = directional_accumulators[spec.key][one_direction]
                if not direction_accumulator.stats:
                    one[f"{direction_name}_n_cells"] = 0.0
                    one[f"{direction_name}_val_loss"] = float("nan")
                    one[f"{direction_name}_val_pcc_cell"] = float("nan")
                    one[f"{direction_name}_val_pcc_gene"] = float("nan")
                    continue
                direction_eval, direction_mse = direction_accumulator.finalize()
                direction_loss.append(float(direction_mse))
                direction_cell.append(float(direction_eval.mean_cell_pcc))
                direction_gene.append(float(direction_eval.mean_gene_pcc))
                one[f"{direction_name}_n_cells"] = float(direction_eval.n_cells)
                one[f"{direction_name}_val_loss"] = float(direction_mse)
                one[f"{direction_name}_val_pcc_cell"] = float(direction_eval.mean_cell_pcc)
                one[f"{direction_name}_val_pcc_gene"] = float(direction_eval.mean_gene_pcc)
                one.update(
                    {
                        f"{direction_name}_{k}": float(v)
                        for k, v in direction_eval.day_metrics.items()
                    }
                )
            one["val_loss"] = float(np.mean(direction_loss)) if direction_loss else float("nan")
            one["val_pcc_cell"] = float(np.mean(direction_cell)) if direction_cell else float("nan")
            one["val_pcc_gene"] = float(np.mean(direction_gene)) if direction_gene else float("nan")
        else:
            one = {
                "val_loss": float(mse),
                "val_pcc_cell": float(eval_result.mean_cell_pcc),
                "val_pcc_gene": float(eval_result.mean_gene_pcc),
                "n_cells": float(eval_result.n_cells),
            }
            one.update({str(k): float(v) for k, v in eval_result.day_metrics.items()})
        one.update({f"prior_gate_{k}": float(v) for k, v in aggregate_gate_summary.items()})
        one.update(gate_metrics)
        one.update(composition_metrics)
        results[spec.key] = one
    return results


def predict_all_fm(
    *,
    model: RawConditionalFlowMatching,
    loader: DataLoader,
    variant: FMVariantSpec,
    device: torch.device,
    fp16: bool,
    context_bank: FMContextBank,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    preds, truths, day_ids = ([], [], [])
    model.eval()
    set_global_seed(seed=int(ARGS.SEED), deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN))
    with torch.no_grad(), autocast(enabled=bool(fp16)):
        for batch in loader:
            target_x = batch["target_x"].to(device)
            target_type_id = batch["target_type_id"].to(device)
            batch_context = context_bank.build_batch(
                target_timestamp_index=batch["target_timestamp_index"],
                target_type_id=target_type_id,
                target_cell_index=batch["target_cell_index"],
                variant=variant,
                device=device,
                direction_id=batch.get("direction_id"),
            )
            pred = model.predict_expression(
                batch_context=batch_context,
                target_type_id=target_type_id,
                variant=variant,
                num_steps=int(ARGS.FLOW_NUM_STEPS),
                num_samples=int(ARGS.FLOW_NUM_SAMPLES),
            )
            preds.append(pred.detach().cpu())
            truths.append(target_x.detach().cpu())
            day_ids.append(
                batch.get("biological_timestamp_index", batch["target_timestamp_index"]).cpu()
            )
    return (torch.cat(preds, dim=0), torch.cat(truths, dim=0), torch.cat(day_ids, dim=0).long())


def mse_over_fm(
    *,
    model: RawConditionalFlowMatching,
    loader: DataLoader,
    variant: FMVariantSpec,
    device: torch.device,
    fp16: bool,
    context_bank: FMContextBank,
) -> float:
    total, n_total = (0.0, 0)
    model.eval()
    set_global_seed(seed=int(ARGS.SEED), deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN))
    with torch.no_grad(), autocast(enabled=bool(fp16)):
        for batch in loader:
            target_x = batch["target_x"].to(device)
            target_type_id = batch["target_type_id"].to(device)
            batch_context = context_bank.build_batch(
                target_timestamp_index=batch["target_timestamp_index"],
                target_type_id=target_type_id,
                target_cell_index=batch["target_cell_index"],
                variant=variant,
                device=device,
                direction_id=batch.get("direction_id"),
            )
            pred = model.predict_expression(
                batch_context=batch_context,
                target_type_id=target_type_id,
                variant=variant,
                num_steps=int(ARGS.FLOW_NUM_STEPS),
                num_samples=int(ARGS.FLOW_NUM_SAMPLES),
            )
            loss = torch.mean((pred - target_x) ** 2)
            total += float(loss.detach().cpu()) * int(target_x.shape[0])
            n_total += int(target_x.shape[0])
    return float(total / max(1, n_total))


def train_single_readout(
    *, variant: FMVariantSpec, device: torch.device, fp16: bool, logger: logging.Logger
) -> Dict[str, Any]:
    if FM_DATA is None:
        return {
            "family": "FM",
            "model": variant.tag,
            "mean_cell_pcc": float("nan"),
            "mean_gene_pcc": float("nan"),
            "seconds": 0.0,
            "skipped": True,
            "skip_reason": "FM data unavailable",
        }
    if not variant.is_valid_for_current_seq:
        return {
            "family": "FM",
            "model": variant.tag,
            "mean_cell_pcc": float("nan"),
            "mean_gene_pcc": float("nan"),
            "seconds": 0.0,
            "skipped": True,
            "skip_reason": "hist1 FM variant is incompatible with SEQ_LEN=0",
        }
    if bool(variant.use_traj_soft) and (not bool(FM_DATA.has_required_traj_hints)):
        return {
            "family": "FM",
            "model": variant.tag,
            "mean_cell_pcc": float("nan"),
            "mean_gene_pcc": float("nan"),
            "seconds": 0.0,
            "skipped": True,
            "skip_reason": "traj-soft requested but trajectory file unavailable",
        }
    bio_reason = fm_bio_unavailable_reason(variant)
    if bio_reason is not None:
        return {
            "family": "FM",
            "model": variant.tag,
            "mean_cell_pcc": float("nan"),
            "mean_gene_pcc": float("nan"),
            "seconds": 0.0,
            "skipped": True,
            "skip_reason": bio_reason,
        }
    attempted_batch = int(ARGS.FM_BATCH)
    attempt_index = 0
    t_start = time.time()
    while attempted_batch >= 1:
        attempt_index += 1
        logger.info(
            f"[{variant.tag}] attempt {attempt_index}: trying FM batch_size={attempted_batch}"
        )
        tr_loader, va_loader, te_loader = FM_DATA.make_loaders(batch_size=attempted_batch)
        eval_loader = va_loader if str(ARGS.EVAL_SPLIT) == "val" else te_loader
        model = RawConditionalFlowMatching(
            n_genes=int(N_GENES), n_celltypes=int(N_CELLTYPES), variant=variant
        ).to(device)
        nparams = sum((p.numel() for p in model.parameters()))
        logger.info(f"[{variant.tag}] params={nparams / 1000000.0:.2f}M")
        try:
            if str(ARGS.EXECUTION_MODE) == "eval_only":
                ckpt_path, meta = load_saved_checkpoint_for_eval(model_tag=variant.tag, model=model)
                logger.info(f"[{variant.tag}] eval_only loaded saved checkpoint {ckpt_path}")
                y_pred, y_true, day_idx = predict_all_fm(
                    model=model,
                    loader=eval_loader,
                    variant=variant,
                    device=device,
                    fp16=fp16,
                    context_bank=FM_DATA.context_bank,
                )
                eval_res = evaluate_predictions_unified(
                    y_pred=y_pred, y_true=y_true, time_idx=day_idx
                )
                out = {
                    "family": "FM",
                    "model": variant.tag,
                    "batch_size_used": int(attempted_batch),
                    "batch_size_attempts": int(attempt_index),
                    "seconds": float(time.time() - t_start),
                    "mean_cell_pcc": float(eval_res.mean_cell_pcc),
                    "mean_gene_pcc": float(eval_res.mean_gene_pcc),
                    "best_epoch": meta.get("best_epoch", None),
                    "best_stop": meta.get("best_stop", float("nan")),
                    "skipped": False,
                }
                out.update(eval_res.day_metrics)
                return out
            scaler = GradScaler(enabled=bool(fp16))
            opt = torch.optim.AdamW(
                params=model.parameters(),
                lr=float(ARGS.FM_LR),
                weight_decay=float(ARGS.FM_WEIGHT_DECAY),
            )
            stop_metric = str(ARGS.FM_STOP_METRIC)
            stop_mode = "min" if stop_metric == "val_loss" else "max"
            best_stop = math.inf if stop_mode == "min" else -math.inf
            best_epoch = None
            best_state = None
            no_improve = 0
            epoch_rows: List[Dict[str, Any]] = []

            def eval_val_metrics(train_loss: Optional[float]) -> Dict[str, float]:
                val_loss = mse_over_fm(
                    model=model,
                    loader=va_loader,
                    variant=variant,
                    device=device,
                    fp16=fp16,
                    context_bank=FM_DATA.context_bank,
                )
                y_pred_val, y_true_val, day_idx_val = predict_all_fm(
                    model=model,
                    loader=va_loader,
                    variant=variant,
                    device=device,
                    fp16=fp16,
                    context_bank=FM_DATA.context_bank,
                )
                val_eval = evaluate_predictions_unified(
                    y_pred=y_pred_val, y_true=y_true_val, time_idx=day_idx_val
                )
                metrics = {
                    "val_loss": float(val_loss),
                    "val_pcc_cell": float(val_eval.mean_cell_pcc),
                    "val_pcc_gene": float(val_eval.mean_gene_pcc),
                }
                if train_loss is not None:
                    metrics["train_loss"] = float(train_loss)
                return metrics

            for ep in range(1, int(ARGS.FM_EPOCHS) + 1):
                set_global_seed(
                    seed=int(ARGS.SEED) + int(ep),
                    deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN),
                )
                model.train()
                running = 0.0
                running_flow = 0.0
                running_aux = 0.0
                running_traj = 0.0
                running_graph = 0.0
                running_anchor = 0.0
                n_seen = 0
                for batch in tqdm(tr_loader, desc=f"[{variant.tag}] ep{ep:02d}", leave=False):
                    target_x = batch["target_x"].to(device)
                    target_type_id = batch["target_type_id"].to(device)
                    batch_context = FM_DATA.context_bank.build_batch(
                        target_timestamp_index=batch["target_timestamp_index"],
                        target_type_id=target_type_id,
                        target_cell_index=batch["target_cell_index"],
                        variant=variant,
                        device=device,
                        direction_id=batch.get("direction_id"),
                    )
                    opt.zero_grad(set_to_none=True)
                    with autocast(enabled=bool(fp16)):
                        loss, loss_parts = model.loss(
                            batch_context=batch_context,
                            target_x=target_x,
                            target_type_id=target_type_id,
                            variant=variant,
                        )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"[{variant.tag}] non-finite training loss: {float(loss.detach().cpu())}"
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), float(ARGS.FM_GRAD_CLIP))
                    scaler.step(opt)
                    scaler.update()
                    bsz = int(target_x.shape[0])
                    running += float(loss.detach().cpu()) * bsz
                    running_flow += float(loss_parts["flow_loss"]) * bsz
                    running_aux += float(loss_parts["aux_point_loss"]) * bsz
                    running_traj += float(loss_parts["traj_soft_loss"]) * bsz
                    running_graph += float(loss_parts.get("graph_reg_loss", 0.0)) * bsz
                    running_anchor += float(loss_parts.get("anchor_fraction", 0.0)) * bsz
                    n_seen += bsz
                train_loss = float(running / max(1, n_seen))
                flow_loss = float(running_flow / max(1, n_seen))
                aux_loss = float(running_aux / max(1, n_seen))
                traj_loss = float(running_traj / max(1, n_seen))
                graph_loss = float(running_graph / max(1, n_seen))
                anchor_fraction = float(running_anchor / max(1, n_seen))
                metrics = eval_val_metrics(train_loss=train_loss)
                stop_value = float(metrics[stop_metric])
                improved = (
                    stop_value < best_stop - float(ARGS.FM_EARLY_STOPPING_MIN_DELTA)
                    if stop_mode == "min"
                    else stop_value > best_stop + float(ARGS.FM_EARLY_STOPPING_MIN_DELTA)
                )
                if improved:
                    best_stop = float(stop_value)
                    best_epoch = int(ep)
                    best_state = copy.deepcopy(model.state_dict())
                    no_improve = 0
                elif int(ep) >= int(ARGS.FM_EARLY_STOPPING_START_EPOCH):
                    no_improve += 1
                else:
                    no_improve = 0
                early_stopping_active = int(ep) >= int(ARGS.FM_EARLY_STOPPING_START_EPOCH)
                epoch_rows.append(
                    {
                        "epoch": int(ep),
                        "train_loss": float(train_loss),
                        "train_flow_loss": float(flow_loss),
                        "train_aux_point_loss": float(aux_loss),
                        "train_traj_soft_loss": float(traj_loss),
                        "train_graph_reg_loss": float(graph_loss),
                        "train_anchor_fraction": float(anchor_fraction),
                        "val_loss": float(metrics["val_loss"]),
                        "val_pcc_cell": float(metrics["val_pcc_cell"]),
                        "val_pcc_gene": float(metrics["val_pcc_gene"]),
                        "stop_value": float(stop_value),
                        "no_improve": int(no_improve),
                        "early_stopping_active": bool(early_stopping_active),
                        "early_stopping_start_epoch": int(ARGS.FM_EARLY_STOPPING_START_EPOCH),
                    }
                )
                logger.info(
                    f"[{variant.tag}] epoch {ep:03d} train_loss={train_loss:.6f} flow={flow_loss:.6f} aux={aux_loss:.6f} traj={traj_loss:.6f} graph={graph_loss:.6f} anchor={anchor_fraction:.3f} val_loss={metrics['val_loss']:.6f} val_pcc_cell={metrics['val_pcc_cell']:.4f} val_pcc_gene={metrics['val_pcc_gene']:.4f} {stop_metric}={stop_value:.6g} best={best_stop:.6g}@ep{best_epoch} early_stop_active={early_stopping_active} start_epoch={ARGS.FM_EARLY_STOPPING_START_EPOCH} no_improve={no_improve}/{ARGS.FM_EARLY_STOPPING_PATIENCE}"
                )
                if early_stopping_active and no_improve >= int(ARGS.FM_EARLY_STOPPING_PATIENCE):
                    break
            if bool(ARGS.FM_LOAD_BEST_AT_END) and best_state is not None:
                model.load_state_dict(best_state, strict=False)
            metadata = {
                "family": "FM",
                "variant_tag": variant.tag,
                "base_variant_without_runtime_suffix": f"FM_{variant.base_mode}_hist{int(variant.use_history)}_ctype_{variant.celltype_mode}_traj{int(variant.use_traj_soft)}",
                "fm_bio_ablation": str(variant.bio_ablation),
                "fm_residual_mode": str(ARGS.FM_RESIDUAL_MODE),
                "fm_prediction_mode": str(ARGS.FM_PREDICTION_MODE),
                "fm_point_blend_weight": float(ARGS.FM_POINT_BLEND_WEIGHT),
                "fm_early_stopping_patience": int(ARGS.FM_EARLY_STOPPING_PATIENCE),
                "fm_early_stopping_min_delta": float(ARGS.FM_EARLY_STOPPING_MIN_DELTA),
                "fm_early_stopping_start_epoch": int(ARGS.FM_EARLY_STOPPING_START_EPOCH),
                "batch_size_used": int(attempted_batch),
                "best_epoch": best_epoch,
                "best_stop": best_stop,
                "seq_len": int(ARGS.SEQ_LEN),
                "history_days_used": int(variant.history_days_used),
                "epoch_history": epoch_rows,
            }
            ckpt_path = None
            if bool(ARGS.SAVE_CHECKPOINTS):
                ckpt_path = save_model_checkpoint(
                    model_tag=variant.tag, model=model, metadata=metadata, logger=logger
                )
            if str(ARGS.EXECUTION_MODE) == "train_only":
                return {
                    "family": "FM",
                    "model": variant.tag,
                    "checkpoint_path": ckpt_path,
                    "batch_size_used": int(attempted_batch),
                    "batch_size_attempts": int(attempt_index),
                    "seconds": float(time.time() - t_start),
                    "val_loss": (
                        float(epoch_rows[-1]["val_loss"]) if len(epoch_rows) > 0 else float("nan")
                    ),
                    "best_epoch": best_epoch,
                    "best_stop": float(best_stop),
                    "skipped": False,
                }
            y_pred, y_true, day_idx = predict_all_fm(
                model=model,
                loader=eval_loader,
                variant=variant,
                device=device,
                fp16=fp16,
                context_bank=FM_DATA.context_bank,
            )
            eval_res = evaluate_predictions_unified(y_pred=y_pred, y_true=y_true, time_idx=day_idx)
            out = {
                "family": "FM",
                "model": variant.tag,
                "checkpoint_path": ckpt_path,
                "batch_size_used": int(attempted_batch),
                "batch_size_attempts": int(attempt_index),
                "seconds": float(time.time() - t_start),
                "mean_cell_pcc": float(eval_res.mean_cell_pcc),
                "mean_gene_pcc": float(eval_res.mean_gene_pcc),
                "val_loss": (
                    float(epoch_rows[-1]["val_loss"]) if len(epoch_rows) > 0 else float("nan")
                ),
                "best_epoch": best_epoch,
                "best_stop": float(best_stop),
                "skipped": False,
            }
            out.update(eval_res.day_metrics)
            return out
        except Exception as exc:
            if is_memory_related_runtime_error(exc):
                logger.exception(
                    f"[{variant.tag}] memory-related failure at batch_size={attempted_batch}"
                )
                release_memory_after_failure()
                if attempted_batch == 1:
                    raise
                attempted_batch = max(1, attempted_batch // 2)
                logger.info(f"[{variant.tag}] retry with smaller batch_size={attempted_batch}")
                continue
            raise
    raise RuntimeError(f"[{variant.tag}] exhausted adaptive batch attempts")


def save_state_dict_checkpoint(
    *,
    model_tag: str,
    state_dict: Dict[str, torch.Tensor],
    metadata: Dict[str, Any],
    logger: logging.Logger,
) -> str:
    metadata = dict(metadata)
    metadata["training_implementation"] = "bloom.training.trainer"
    metadata["model_family"] = MODEL_FAMILY
    ckpt_dir = build_checkpoint_dir(model_tag)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    meta_path = os.path.join(ckpt_dir, "metadata.json")
    torch.save({"state_dict": state_dict}, ckpt_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[{model_tag}] checkpoint saved to {ckpt_path}")
    return ckpt_path


def _fm_readout_metric_value(metrics: Dict[str, float], stop_metric: str) -> float:
    if stop_metric not in metrics:
        raise KeyError(
            f"FM stop metric {stop_metric!r} not found in readout metrics: {sorted(metrics.keys())}"
        )
    return float(metrics[stop_metric])


def initialize_fm_from_checkpoint(
    *, model: nn.Module, checkpoint_path: str, logger: logging.Logger
) -> Dict[str, Any]:
    """Load only name/shape-compatible tensors, while retaining initialized values for unmatched parameters."""
    path = str(checkpoint_path).strip()
    if not path:
        return {"enabled": False}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FM_INIT_CHECKPOINT not found: {path}")
    payload = torch.load(path, map_location="cpu")
    source_state = payload.get("state_dict", payload)
    if not isinstance(source_state, dict):
        raise TypeError(f"FM_INIT_CHECKPOINT does not contain a state_dict mapping: {path}")
    destination_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped_shape: List[str] = []
    unexpected: List[str] = []
    for key, value in source_state.items():
        if key not in destination_state:
            unexpected.append(str(key))
            continue
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(destination_state[key].shape):
            skipped_shape.append(str(key))
            continue
        compatible[str(key)] = value
    missing, load_unexpected = model.load_state_dict(compatible, strict=False)
    report = {
        "enabled": True,
        "path": path,
        "loaded_tensor_count": int(len(compatible)),
        "source_tensor_count": int(len(source_state)),
        "missing_tensor_count": int(len(missing)),
        "shape_skipped_count": int(len(skipped_shape)),
        "unexpected_tensor_count": int(len(unexpected) + len(load_unexpected)),
        "shape_skipped_head": skipped_shape[:20],
        "unexpected_head": (unexpected + list(load_unexpected))[:20],
    }
    logger.info(
        f"[FM warm-start] loaded={report['loaded_tensor_count']}/{report['source_tensor_count']} shape_skipped={report['shape_skipped_count']} missing={report['missing_tensor_count']} unexpected={report['unexpected_tensor_count']} from {path}"
    )
    return report


def _slice_batch_context(
    batch_context: Dict[str, torch.Tensor], row_mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Slice only tensors whose first dimension is the batch dimension."""
    row_mask = row_mask.to(dtype=torch.bool)
    batch_size = int(row_mask.shape[0])
    out: Dict[str, torch.Tensor] = {}
    for key, value in batch_context.items():
        if torch.is_tensor(value) and value.ndim > 0 and (int(value.shape[0]) == batch_size):
            out[key] = value[row_mask]
        else:
            out[key] = value
    return out


def continuous_developmental_time_loss(
    *,
    model: RawConditionalFlowMatching,
    batch_context: Dict[str, torch.Tensor],
    target_x: torch.Tensor,
    target_type_id: torch.Tensor,
    variant: FMVariantSpec,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Supervise arbitrary positions between a trajectory-linked source cell and
    its measured endpoint. The intermediate expression is a declared linear
    proxy, not unmeasured biological ground truth.
    """
    zero = target_x.new_zeros(())
    valid = batch_context["traj_hint_mask"] > 0
    if not torch.any(valid):
        return (zero, {"continuous_rows": 0.0, "continuous_fraction_mean": 0.0})
    lo = float(ARGS.FM_CONTINUOUS_TIME_MIN_FRACTION)
    hi = float(ARGS.FM_CONTINUOUS_TIME_MAX_FRACTION)
    fractions_all = lo + (hi - lo) * torch.rand(
        int(target_x.shape[0]), device=target_x.device, dtype=target_x.dtype
    )
    fraction_context_all = model.context_at_developmental_fraction(batch_context, fractions_all)
    fraction_context = _slice_batch_context(fraction_context_all, valid)
    fractions = fractions_all[valid].unsqueeze(-1)
    source_expression = batch_context["traj_hint"][valid].to(target_x.dtype)
    endpoint_expression = target_x[valid]
    proxy_target = (1.0 - fractions) * source_expression + fractions * endpoint_expression
    proxy_loss, proxy_parts = model.loss(
        batch_context=fraction_context,
        target_x=proxy_target,
        target_type_id=target_type_id[valid],
        variant=variant,
        include_regularization=False,
    )
    metrics = {
        "continuous_rows": float(torch.sum(valid).detach().cpu()),
        "continuous_fraction_mean": float(fractions.detach().mean().cpu()),
        "continuous_flow_loss": float(proxy_parts["flow_loss"]),
        "continuous_aux_point_loss": float(proxy_parts["aux_point_loss"]),
    }
    return (proxy_loss, metrics)


def train_multiple_readouts(
    *, variant: FMVariantSpec, device: torch.device, fp16: bool, logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Train BLOOM with separate validation checkpoints for each requested readout."""
    if FM_DATA is None:
        return [
            {
                "family": "FM",
                "model": variant.tag,
                "mean_cell_pcc": float("nan"),
                "mean_gene_pcc": float("nan"),
                "seconds": 0.0,
                "skipped": True,
                "skip_reason": "FM data unavailable",
            }
        ]
    if not variant.is_valid_for_current_seq:
        return [
            {
                "family": "FM",
                "model": variant.tag,
                "mean_cell_pcc": float("nan"),
                "mean_gene_pcc": float("nan"),
                "seconds": 0.0,
                "skipped": True,
                "skip_reason": "hist1 FM variant is incompatible with SEQ_LEN=0",
            }
        ]
    if bool(variant.use_traj_soft) and (not bool(FM_DATA.has_required_traj_hints)):
        return [
            {
                "family": "FM",
                "model": variant.tag,
                "mean_cell_pcc": float("nan"),
                "mean_gene_pcc": float("nan"),
                "seconds": 0.0,
                "skipped": True,
                "skip_reason": "traj-soft requested but trajectory file unavailable",
            }
        ]
    bio_reason = fm_bio_unavailable_reason(variant)
    if bio_reason is not None:
        return [
            {
                "family": "FM",
                "model": variant.tag,
                "mean_cell_pcc": float("nan"),
                "mean_gene_pcc": float("nan"),
                "seconds": 0.0,
                "skipped": True,
                "skip_reason": bio_reason,
            }
        ]
    if float(ARGS.FM_CONTINUOUS_TIME_WEIGHT) > 0.0:
        if not variant.uses_bio_component("actual_time"):
            raise ValueError(
                "FM_CONTINUOUS_TIME_WEIGHT > 0 requires bio1 or a later actual-time biological mode"
            )
        if not bool(variant.use_traj_soft):
            raise ValueError("FM_CONTINUOUS_TIME_WEIGHT > 0 requires a traj1 FM variant")
    readouts = build_active_fm_readouts()
    primary_readout = resolve_primary_fm_readout(readouts)
    logger.info(f"[{variant.tag}] FM multi-readouts: {', '.join((x.key for x in readouts))}")
    logger.info(
        f"[{variant.tag}] FM primary readout for backward-compatible primary checkpoint/reporting: {primary_readout}"
    )
    logger.info(
        f"[{variant.tag}] FM early stopping strategy: ALL_READOUT_PLATEAU; patience resets when ANY requested readout improves; training stops only after NO requested readout improves for {ARGS.FM_EARLY_STOPPING_PATIENCE} consecutive validation epochs; monitoring starts at epoch {ARGS.FM_EARLY_STOPPING_START_EPOCH}; min_delta={ARGS.FM_EARLY_STOPPING_MIN_DELTA}"
    )
    attempted_batch = int(ARGS.FM_BATCH)
    attempt_index = 0
    t_start = time.time()
    while attempted_batch >= 1:
        attempt_index += 1
        logger.info(
            f"[{variant.tag}] attempt {attempt_index}: trying FM batch_size={attempted_batch}"
        )
        _, va_loader, te_loader = FM_DATA.make_loaders(batch_size=attempted_batch)
        eval_loader = va_loader if str(ARGS.EVAL_SPLIT) == "val" else te_loader
        model = RawConditionalFlowMatching(
            n_genes=int(N_GENES), n_celltypes=int(N_CELLTYPES), variant=variant
        ).to(device)
        nparams = sum((p.numel() for p in model.parameters()))
        logger.info(f"[{variant.tag}] params={nparams / 1000000.0:.2f}M")
        warm_start_report: Dict[str, Any] = {"enabled": False}
        try:
            if str(ARGS.EXECUTION_MODE) == "eval_only":
                ckpt_path, meta = load_saved_checkpoint_for_eval(model_tag=variant.tag, model=model)
                logger.info(f"[{variant.tag}] eval_only loaded saved checkpoint {ckpt_path}")
                eval_results = evaluate_fm_readouts_streaming(
                    model=model,
                    loader=eval_loader,
                    variant=variant,
                    readouts=readouts,
                    device=device,
                    fp16=fp16,
                    context_bank=FM_DATA.eval_context_bank,
                )
                rows: List[Dict[str, Any]] = []
                for spec in readouts:
                    m = eval_results[spec.key]
                    row = {
                        "family": "FM",
                        "model": f"{variant.tag}__readout_{spec.key}",
                        "base_model": variant.tag,
                        "fm_bio_ablation": str(variant.bio_ablation),
                        "fm_readout": spec.key,
                        "checkpoint_path": ckpt_path,
                        "batch_size_used": int(attempted_batch),
                        "batch_size_attempts": int(attempt_index),
                        "seconds": float(time.time() - t_start),
                        "mean_cell_pcc": float(m["val_pcc_cell"]),
                        "mean_gene_pcc": float(m["val_pcc_gene"]),
                        "test_mse": float(m["val_loss"]),
                        "best_epoch": meta.get("best_epoch", None),
                        "best_stop": meta.get("best_stop", float("nan")),
                        "skipped": False,
                    }
                    row.update(
                        {
                            k: v
                            for k, v in m.items()
                            if k.startswith("day")
                            or k.startswith("forward_")
                            or k.startswith("reverse_")
                            or k.startswith("pooled_")
                            or k.startswith("prior_gate_")
                            or k.startswith("composition_")
                        }
                    )
                    rows.append(row)
                return rows
            warm_start_report = initialize_fm_from_checkpoint(
                model=model, checkpoint_path=str(ARGS.FM_INIT_CHECKPOINT), logger=logger
            )
            scaler = GradScaler(enabled=bool(fp16))
            opt = torch.optim.AdamW(
                params=model.parameters(),
                lr=float(ARGS.FM_LR),
                weight_decay=float(ARGS.FM_WEIGHT_DECAY),
            )
            stop_metric = str(ARGS.FM_STOP_METRIC)
            stop_mode = "min" if stop_metric == "val_loss" else "max"
            best_value_by_readout: Dict[str, float] = {
                spec.key: math.inf if stop_mode == "min" else -math.inf for spec in readouts
            }
            best_epoch_by_readout: Dict[str, Optional[int]] = {spec.key: None for spec in readouts}
            best_state_by_readout: Dict[str, Optional[Dict[str, torch.Tensor]]] = {
                spec.key: None for spec in readouts
            }
            best_val_metrics_by_readout: Dict[str, Dict[str, float]] = {
                spec.key: {} for spec in readouts
            }
            no_improve_all_readouts = 0
            no_improve_by_readout: Dict[str, int] = {spec.key: 0 for spec in readouts}
            epoch_rows: List[Dict[str, Any]] = []
            for ep in range(1, int(ARGS.FM_EPOCHS) + 1):
                set_global_seed(
                    seed=int(ARGS.SEED) + int(ep),
                    deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN),
                )
                train_loader_ep, scale_info = FM_DATA.make_train_loader_for_epoch(
                    batch_size=attempted_batch, epoch=ep
                )
                prior_strength = model.set_prior_training_progress(ep)
                model.train()
                running = 0.0
                running_flow = 0.0
                running_aux = 0.0
                running_traj = 0.0
                running_graph = 0.0
                running_prior_gate_reg = 0.0
                running_continuous = 0.0
                running_continuous_rows = 0.0
                running_composition = 0.0
                running_anchor = 0.0
                running_direction_counts = np.zeros((2,), dtype=np.int64)
                n_seen = 0
                n_steps = 0
                for batch in tqdm(train_loader_ep, desc=f"[{variant.tag}] ep{ep:02d}", leave=False):
                    target_x = batch["target_x"].to(device)
                    target_type_id = batch["target_type_id"].to(device)
                    batch_context = FM_DATA.train_context_bank.build_batch(
                        target_timestamp_index=batch["target_timestamp_index"],
                        target_type_id=target_type_id,
                        target_cell_index=batch["target_cell_index"],
                        variant=variant,
                        device=device,
                        direction_id=batch.get("direction_id"),
                    )
                    population_context: Optional[Dict[str, torch.Tensor]] = None
                    if float(ARGS.FM_COMPOSITION_LOSS_WEIGHT) > 0.0:
                        population_context = FM_DATA.train_context_bank.build_population_batch(
                            target_timestamp_index=batch["target_timestamp_index"],
                            variant=variant,
                            device=device,
                            direction_id=batch.get("direction_id"),
                        )
                    opt.zero_grad(set_to_none=True)
                    with autocast(enabled=bool(fp16)):
                        loss, loss_parts = model.loss(
                            batch_context=batch_context,
                            target_x=target_x,
                            target_type_id=target_type_id,
                            variant=variant,
                        )
                        continuous_loss = target_x.new_zeros(())
                        continuous_parts: Dict[str, float] = {
                            "continuous_rows": 0.0,
                            "continuous_fraction_mean": 0.0,
                        }
                        if float(ARGS.FM_CONTINUOUS_TIME_WEIGHT) > 0.0:
                            continuous_loss, continuous_parts = continuous_developmental_time_loss(
                                model=model,
                                batch_context=batch_context,
                                target_x=target_x,
                                target_type_id=target_type_id,
                                variant=variant,
                            )
                            loss = loss + float(ARGS.FM_CONTINUOUS_TIME_WEIGHT) * continuous_loss
                        composition_loss = target_x.new_zeros(())
                        if population_context is not None:
                            population_encoded = model.encode_population_context(
                                batch_context=population_context
                            )
                            composition_logits = model.composition_head(population_encoded)
                            composition_loss = nn.functional.cross_entropy(
                                composition_logits, target_type_id
                            )
                            loss = loss + float(ARGS.FM_COMPOSITION_LOSS_WEIGHT) * composition_loss
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"[{variant.tag}] non-finite training loss: {float(loss.detach().cpu())}"
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), float(ARGS.FM_GRAD_CLIP))
                    scaler.step(opt)
                    scaler.update()
                    bsz = int(target_x.shape[0])
                    if "direction_id" in batch:
                        batch_direction = (
                            batch["direction_id"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.int64, copy=False)
                        )
                        running_direction_counts += np.bincount(batch_direction, minlength=2)[:2]
                    running += float(loss.detach().cpu()) * bsz
                    running_flow += float(loss_parts["flow_loss"]) * bsz
                    running_aux += float(loss_parts["aux_point_loss"]) * bsz
                    running_traj += float(loss_parts["traj_soft_loss"]) * bsz
                    running_graph += float(loss_parts.get("graph_reg_loss", 0.0)) * bsz
                    running_prior_gate_reg += (
                        float(loss_parts.get("prior_gate_reg_loss", 0.0)) * bsz
                    )
                    running_continuous += float(continuous_loss.detach().cpu()) * bsz
                    running_continuous_rows += float(continuous_parts.get("continuous_rows", 0.0))
                    running_composition += float(composition_loss.detach().cpu()) * bsz
                    running_anchor += float(loss_parts.get("anchor_fraction", 0.0)) * bsz
                    n_seen += bsz
                    n_steps += 1
                    max_steps = int(ARGS.FM_MAX_TRAIN_STEPS_PER_EPOCH)
                    if max_steps > 0 and n_steps >= max_steps:
                        break
                train_loss = float(running / max(1, n_seen))
                flow_loss = float(running_flow / max(1, n_seen))
                aux_loss = float(running_aux / max(1, n_seen))
                traj_loss = float(running_traj / max(1, n_seen))
                graph_loss = float(running_graph / max(1, n_seen))
                prior_gate_reg_loss = float(running_prior_gate_reg / max(1, n_seen))
                continuous_loss_mean = float(running_continuous / max(1, n_seen))
                composition_loss_mean = float(running_composition / max(1, n_seen))
                anchor_fraction = float(running_anchor / max(1, n_seen))
                model.set_prior_inference_strength()
                val_results = evaluate_fm_readouts_streaming(
                    model=model,
                    loader=va_loader,
                    variant=variant,
                    readouts=readouts,
                    device=device,
                    fp16=fp16,
                    context_bank=FM_DATA.eval_context_bank,
                )
                improved_keys: List[str] = []
                for spec in readouts:
                    value = _fm_readout_metric_value(val_results[spec.key], stop_metric)
                    old_best = float(best_value_by_readout[spec.key])
                    improved = (
                        value < old_best - float(ARGS.FM_EARLY_STOPPING_MIN_DELTA)
                        if stop_mode == "min"
                        else value > old_best + float(ARGS.FM_EARLY_STOPPING_MIN_DELTA)
                    )
                    if improved:
                        best_value_by_readout[spec.key] = float(value)
                        best_epoch_by_readout[spec.key] = int(ep)
                        best_val_metrics_by_readout[spec.key] = dict(val_results[spec.key])
                        improved_keys.append(spec.key)
                if len(improved_keys) > 0:
                    epoch_state = clone_state_dict_to_cpu(model)
                    for key in improved_keys:
                        best_state_by_readout[key] = epoch_state
                early_stopping_active = int(ep) >= int(ARGS.FM_EARLY_STOPPING_START_EPOCH)
                improved_key_set = set(improved_keys)
                if early_stopping_active:
                    for spec in readouts:
                        if spec.key in improved_key_set:
                            no_improve_by_readout[spec.key] = 0
                        else:
                            no_improve_by_readout[spec.key] += 1
                    if len(improved_keys) > 0:
                        no_improve_all_readouts = 0
                    else:
                        no_improve_all_readouts += 1
                else:
                    no_improve_all_readouts = 0
                    for spec in readouts:
                        no_improve_by_readout[spec.key] = 0
                primary_metrics = val_results[primary_readout]
                row: Dict[str, Any] = {
                    "epoch": int(ep),
                    "train_loss": float(train_loss),
                    "train_flow_loss": float(flow_loss),
                    "train_aux_point_loss": float(aux_loss),
                    "train_traj_soft_loss": float(traj_loss),
                    "train_graph_reg_loss": float(graph_loss),
                    "train_prior_gate_reg_loss": float(prior_gate_reg_loss),
                    "train_continuous_time_loss": float(continuous_loss_mean),
                    "train_continuous_rows": float(running_continuous_rows),
                    "train_composition_loss": float(composition_loss_mean),
                    "prior_strength_during_train": float(prior_strength),
                    "train_anchor_fraction": float(anchor_fraction),
                    "train_steps": int(n_steps),
                    "train_cells_seen": int(n_seen),
                    "train_forward_cells_seen": int(running_direction_counts[0]),
                    "train_reverse_cells_seen": int(running_direction_counts[1]),
                    "train_realized_forward_fraction": float(
                        running_direction_counts[0] / max(1, int(np.sum(running_direction_counts)))
                    ),
                    "scale_info": scale_info,
                    "primary_readout": primary_readout,
                    "val_loss": float(primary_metrics["val_loss"]),
                    "val_pcc_cell": float(primary_metrics["val_pcc_cell"]),
                    "val_pcc_gene": float(primary_metrics["val_pcc_gene"]),
                    "stop_value": float(_fm_readout_metric_value(primary_metrics, stop_metric)),
                    "no_improve": int(no_improve_all_readouts),
                    "no_improve_all_readouts": int(no_improve_all_readouts),
                    "early_stopping_active": bool(early_stopping_active),
                    "early_stopping_start_epoch": int(ARGS.FM_EARLY_STOPPING_START_EPOCH),
                    "improved_readouts": list(improved_keys),
                }
                for spec in readouts:
                    m = val_results[spec.key]
                    row[f"{spec.key}__val_loss"] = float(m["val_loss"])
                    row[f"{spec.key}__val_pcc_cell"] = float(m["val_pcc_cell"])
                    row[f"{spec.key}__val_pcc_gene"] = float(m["val_pcc_gene"])
                    row[f"{spec.key}__best_stop"] = float(best_value_by_readout[spec.key])
                    row[f"{spec.key}__best_epoch"] = best_epoch_by_readout[spec.key]
                    row[f"{spec.key}__no_improve"] = int(no_improve_by_readout[spec.key])
                    for direction_metric in (
                        "forward_val_loss",
                        "forward_val_pcc_cell",
                        "forward_val_pcc_gene",
                        "reverse_val_loss",
                        "reverse_val_pcc_cell",
                        "reverse_val_pcc_gene",
                        "pooled_val_loss",
                        "pooled_val_pcc_cell",
                        "pooled_val_pcc_gene",
                    ):
                        if direction_metric in m:
                            row[f"{spec.key}__{direction_metric}"] = float(m[direction_metric])
                gate_summary = model.encoder.prior_gate_summary()
                for gate_name, gate_value in gate_summary.items():
                    row[f"prior_gate_{gate_name}"] = float(gate_value)
                for metric_name, metric_value in primary_metrics.items():
                    if metric_name.startswith("composition_") or "_composition_" in metric_name:
                        row[metric_name] = float(metric_value)
                epoch_rows.append(row)
                readout_log = " | ".join(
                    (
                        f"{spec.key}:cell={val_results[spec.key]['val_pcc_cell']:.4f},gene={val_results[spec.key]['val_pcc_gene']:.4f},mse={val_results[spec.key]['val_loss']:.6f}"
                        for spec in readouts
                    )
                )
                improved_log = ",".join(improved_keys) if len(improved_keys) > 0 else "NONE"
                per_readout_patience_log = ", ".join(
                    (f"{spec.key}={no_improve_by_readout[spec.key]}" for spec in readouts)
                )
                best_readout_log = ", ".join(
                    (
                        f"{spec.key}={best_value_by_readout[spec.key]:.6g}@ep{best_epoch_by_readout[spec.key]}"
                        for spec in readouts
                    )
                )
                logger.info(
                    f"[{variant.tag}] epoch {ep:03d} train_loss={train_loss:.6f} flow={flow_loss:.6f} aux={aux_loss:.6f} traj={traj_loss:.6f} graph={graph_loss:.6f} gate_reg={prior_gate_reg_loss:.6f} ctime={continuous_loss_mean:.6f} composition={composition_loss_mean:.6f} prior_strength={prior_strength:.3f} anchor={anchor_fraction:.3f} steps={n_steps} cells_seen={n_seen} direction_seen=({running_direction_counts[0]},{running_direction_counts[1]}) primary={primary_readout} {stop_metric}={_fm_readout_metric_value(primary_metrics, stop_metric):.6g} improved_readouts={improved_log} early_stop_active={early_stopping_active} start_epoch={ARGS.FM_EARLY_STOPPING_START_EPOCH} no_improve_all={no_improve_all_readouts}/{ARGS.FM_EARLY_STOPPING_PATIENCE}"
                )
                logger.info(f"[{variant.tag}] readouts | {readout_log}")
                if str(ARGS.FM_TEMPORAL_DIRECTION) == "bidirectional":
                    logger.info(
                        f"[{variant.tag}] direction metrics ({primary_readout}) | forward: cell={primary_metrics.get('forward_val_pcc_cell', np.nan):.4f}, gene={primary_metrics.get('forward_val_pcc_gene', np.nan):.4f}; reverse: cell={primary_metrics.get('reverse_val_pcc_cell', np.nan):.4f}, gene={primary_metrics.get('reverse_val_pcc_gene', np.nan):.4f}; checkpoint score is the equal-direction macro average"
                    )
                logger.info(f"[{variant.tag}] best_by_readout | {best_readout_log}")
                logger.info(f"[{variant.tag}] no_improve_by_readout | {per_readout_patience_log}")
                if gate_summary:
                    logger.info(
                        f"[{variant.tag}] prior_gates | "
                        + ", ".join(
                            (f"{name}={value:.4f}" for name, value in sorted(gate_summary.items()))
                        )
                    )
                if "composition_mean_jsd" in primary_metrics:
                    logger.info(
                        f"[{variant.tag}] composition | JSD={primary_metrics['composition_mean_jsd']:.6f}, TV={primary_metrics['composition_mean_tv']:.6f}, Hellinger={primary_metrics['composition_mean_hellinger']:.6f}"
                    )
                if early_stopping_active and no_improve_all_readouts >= int(
                    ARGS.FM_EARLY_STOPPING_PATIENCE
                ):
                    logger.info(
                        f"[{variant.tag}] early stopping: no requested FM readout improved for {no_improve_all_readouts} consecutive validation epochs; all requested readouts are considered plateaued for the configured patience window"
                    )
                    break
            final_state = clone_state_dict_to_cpu(model)
            for spec in readouts:
                if best_state_by_readout[spec.key] is None:
                    best_state_by_readout[spec.key] = final_state
                    best_epoch_by_readout[spec.key] = (
                        int(epoch_rows[-1]["epoch"]) if epoch_rows else 0
                    )
                    best_val_metrics_by_readout[spec.key] = dict(val_results[spec.key])
                    best_value_by_readout[spec.key] = _fm_readout_metric_value(
                        val_results[spec.key], stop_metric
                    )
            primary_state = best_state_by_readout[primary_readout]
            model.load_state_dict(primary_state, strict=False)
            primary_metadata = {
                "family": "FM",
                "code_version": "v7" if MODEL_FAMILY == "bidirectional" else "v6",
                "variant_tag": variant.tag,
                "base_variant_without_runtime_suffix": f"FM_{variant.base_mode}_hist{int(variant.use_history)}_ctype_{variant.celltype_mode}_traj{int(variant.use_traj_soft)}",
                "fm_residual_mode": str(ARGS.FM_RESIDUAL_MODE),
                "fm_prediction_mode": str(ARGS.FM_PREDICTION_MODE),
                "fm_bio_ablation": str(variant.bio_ablation),
                "fm_bio_level": int(variant.bio_level),
                "fm_bio_available": (
                    dict(BIO_PRIOR_DATA.available) if BIO_PRIOR_DATA is not None else {}
                ),
                "fm_temporal_direction": str(ARGS.FM_TEMPORAL_DIRECTION),
                "fm_bidirectional_conditioning": str(ARGS.FM_BIDIRECTIONAL_CONDITIONING),
                "fm_bidirectional_adapter_hidden": int(ARGS.FM_BIDIRECTIONAL_ADAPTER_HIDDEN),
                "fm_bidirectional_forward_fraction": float(ARGS.FM_BIDIRECTIONAL_FORWARD_FRACTION),
                "fm_bidirectional_split_policy": str(ARGS.FM_BIDIRECTIONAL_SPLIT_POLICY),
                "fm_bidirectional_split_semantics": "one global biological-stage partition; for random_proportional_split only, deterministic size-preserving interior-stage coverage repair if needed; then apply direction-specific SEQ_LEN context eligibility",
                "fm_bidirectional_checkpoint_metric_semantics": "equal-direction macro average over forward and reverse validation metrics; both directions are required",
                "fm_reverse_prediction_claim": "conditional reconstruction from later context; not a literal inversion of embryogenesis",
                "fm_cycle_consistency_used": False,
                "fm_biological_train_timestamps": sorted(
                    (int(x) for x in FM_DATA.train_timestamps)
                ),
                "fm_biological_val_timestamps": sorted((int(x) for x in FM_DATA.val_timestamps)),
                "fm_biological_test_timestamps": sorted((int(x) for x in FM_DATA.test_timestamps)),
                "fm_bidirectional_split_repair": getattr(FM_DATA, "bidirectional_split_repair", {}),
                "fm_bidirectional_direction_sizes": getattr(
                    FM_DATA, "bidirectional_direction_sizes", {}
                ),
                "reported_metric_timestamp_semantics": "original_biological_ordinal_time",
                "fm_solver_time_semantics": "noise_to_data_tau_in_[0,1]",
                "developmental_time_semantics": "normalized_physical_time_from_BIO_ACTUAL_TIME_COL",
                "adaptive_prior_fusion": bool(model.encoder.adaptive_prior_fusion),
                "fm_prior_gate_mode": str(ARGS.FM_PRIOR_GATE_MODE),
                "fm_prior_gate_init": float(ARGS.FM_PRIOR_GATE_INIT),
                "fm_prior_dropout": float(ARGS.FM_PRIOR_DROPOUT),
                "fm_prior_gate_l1_weight": float(ARGS.FM_PRIOR_GATE_L1_WEIGHT),
                "fm_prior_warmup_epochs": int(ARGS.FM_PRIOR_WARMUP_EPOCHS),
                "fm_prior_ramp_epochs": int(ARGS.FM_PRIOR_RAMP_EPOCHS),
                "last_validation_prior_gate_summary": model.encoder.prior_gate_summary(),
                "fm_continuous_time_weight": float(ARGS.FM_CONTINUOUS_TIME_WEIGHT),
                "fm_continuous_time_min_fraction": float(ARGS.FM_CONTINUOUS_TIME_MIN_FRACTION),
                "fm_continuous_time_max_fraction": float(ARGS.FM_CONTINUOUS_TIME_MAX_FRACTION),
                "fm_continuous_target_semantics": "linear_expression_proxy_between_trajectory_hint_and_measured_endpoint",
                "fm_context_gene_dropout": float(ARGS.FM_CONTEXT_GENE_DROPOUT),
                "fm_context_cell_dropout": float(ARGS.FM_CONTEXT_CELL_DROPOUT),
                "fm_composition_loss_weight": float(ARGS.FM_COMPOSITION_LOSS_WEIGHT),
                "fm_composition_semantics": "unconditional_requested_target_stage_celltype_distribution_from_direction_specific_population_context",
                "row_aligned_expression_uses_oracle_target_celltype": bool(
                    str(variant.celltype_mode) in {"soft", "hard"}
                ),
                "row_aligned_expression_uses_oracle_target_count": True,
                "composition_head_uses_oracle_target_celltype": False,
                "fm_warm_start": warm_start_report,
                "bio_hierarchy_columns": list(ARGS.BIO_HIERARCHY_COLS),
                "bio_anatomy_columns": list(ARGS.BIO_ANATOMY_COLS),
                "bio_numeric_context_columns": list(ARGS.BIO_NUMERIC_CONTEXT_COLS),
                "curated_graph_path": str(ARGS.CURATED_GRAPH_PATH),
                "driver_program_path": str(ARGS.DRIVER_PROGRAM_PATH),
                "grn_path": str(ARGS.GRN_PATH),
                "curated_graph_reg_weight": float(ARGS.CURATED_GRAPH_REG_WEIGHT),
                "fm_eval_readouts": [x.key for x in readouts],
                "fm_blend_weights": [float(x) for x in ARGS.FM_BLEND_WEIGHTS],
                "fm_primary_readout": primary_readout,
                "fm_early_stopping_strategy": "all_readout_plateau",
                "fm_early_stopping_patience": int(ARGS.FM_EARLY_STOPPING_PATIENCE),
                "fm_early_stopping_min_delta": float(ARGS.FM_EARLY_STOPPING_MIN_DELTA),
                "fm_early_stopping_start_epoch": int(ARGS.FM_EARLY_STOPPING_START_EPOCH),
                "fm_early_stopping_rule": "track best checkpoints from epoch 1; begin patience at the configured start epoch; reset patience if any requested readout improves; stop after no requested readout improves for patience consecutive active validation epochs",
                "final_no_improve_all_readouts": int(no_improve_all_readouts),
                "final_no_improve_by_readout": {
                    k: int(v) for k, v in no_improve_by_readout.items()
                },
                "batch_size_used": int(attempted_batch),
                "best_epoch": best_epoch_by_readout[primary_readout],
                "best_stop": best_value_by_readout[primary_readout],
                "best_epoch_by_readout": best_epoch_by_readout,
                "best_stop_by_readout": best_value_by_readout,
                "seq_len": int(ARGS.SEQ_LEN),
                "history_days_used": int(variant.history_days_used),
                "fm_scale_experiment": str(ARGS.FM_SCALE_EXPERIMENT),
                "fm_scale_fraction": float(ARGS.FM_SCALE_FRACTION),
                "fm_scale_seed": int(ARGS.FM_SCALE_SEED),
                "fm_max_train_steps_per_epoch": int(ARGS.FM_MAX_TRAIN_STEPS_PER_EPOCH),
                "epoch_history": epoch_rows,
            }
            primary_metadata = checkpoint_metadata_for_model(primary_metadata)
            primary_ckpt_path = None
            if bool(ARGS.SAVE_CHECKPOINTS):
                primary_ckpt_path = save_state_dict_checkpoint(
                    model_tag=variant.tag,
                    state_dict=primary_state,
                    metadata=primary_metadata,
                    logger=logger,
                )
            checkpoint_path_by_readout: Dict[str, Optional[str]] = {
                spec.key: primary_ckpt_path for spec in readouts
            }
            if bool(ARGS.SAVE_CHECKPOINTS) and bool(ARGS.FM_SAVE_BEST_PER_READOUT):
                for spec in readouts:
                    readout_tag = f"{variant.tag}__readout_{spec.key}"
                    readout_metadata = dict(primary_metadata)
                    readout_metadata["fm_readout"] = spec.key
                    readout_metadata["best_epoch"] = best_epoch_by_readout[spec.key]
                    readout_metadata["best_stop"] = best_value_by_readout[spec.key]
                    checkpoint_path_by_readout[spec.key] = save_state_dict_checkpoint(
                        model_tag=readout_tag,
                        state_dict=best_state_by_readout[spec.key],
                        metadata=readout_metadata,
                        logger=logger,
                    )
            if str(ARGS.EXECUTION_MODE) == "train_only":
                rows: List[Dict[str, Any]] = []
                for spec in readouts:
                    rows.append(
                        {
                            "family": "FM",
                            "model": f"{variant.tag}__readout_{spec.key}",
                            "base_model": variant.tag,
                            "fm_bio_ablation": str(variant.bio_ablation),
                            "fm_readout": spec.key,
                            "checkpoint_path": checkpoint_path_by_readout[spec.key],
                            "batch_size_used": int(attempted_batch),
                            "batch_size_attempts": int(attempt_index),
                            "seconds": float(time.time() - t_start),
                            "best_epoch": best_epoch_by_readout[spec.key],
                            "best_stop": best_value_by_readout[spec.key],
                            "skipped": False,
                        }
                    )
                return rows
            specs_by_epoch: Dict[int, List[FMReadoutSpec]] = defaultdict(list)
            for spec in readouts:
                specs_by_epoch[int(best_epoch_by_readout[spec.key])].append(spec)
            final_metrics_by_readout: Dict[str, Dict[str, float]] = {}
            for epoch_id, epoch_specs in sorted(specs_by_epoch.items()):
                representative_key = epoch_specs[0].key
                model.load_state_dict(best_state_by_readout[representative_key], strict=False)
                one_group = evaluate_fm_readouts_streaming(
                    model=model,
                    loader=eval_loader,
                    variant=variant,
                    readouts=epoch_specs,
                    device=device,
                    fp16=fp16,
                    context_bank=FM_DATA.eval_context_bank,
                )
                final_metrics_by_readout.update(one_group)
            rows: List[Dict[str, Any]] = []
            elapsed = float(time.time() - t_start)
            for spec in readouts:
                m = final_metrics_by_readout[spec.key]
                row = {
                    "family": "FM",
                    "model": f"{variant.tag}__readout_{spec.key}",
                    "base_model": variant.tag,
                    "fm_bio_ablation": str(variant.bio_ablation),
                    "fm_readout": spec.key,
                    "checkpoint_path": checkpoint_path_by_readout[spec.key],
                    "batch_size_used": int(attempted_batch),
                    "batch_size_attempts": int(attempt_index),
                    "seconds": elapsed,
                    "mean_cell_pcc": float(m["val_pcc_cell"]),
                    "mean_gene_pcc": float(m["val_pcc_gene"]),
                    "test_mse": float(m["val_loss"]),
                    "val_loss": float(
                        best_val_metrics_by_readout[spec.key].get("val_loss", np.nan)
                    ),
                    "best_val_pcc_cell": float(
                        best_val_metrics_by_readout[spec.key].get("val_pcc_cell", np.nan)
                    ),
                    "best_val_pcc_gene": float(
                        best_val_metrics_by_readout[spec.key].get("val_pcc_gene", np.nan)
                    ),
                    "best_epoch": best_epoch_by_readout[spec.key],
                    "best_stop": float(best_value_by_readout[spec.key]),
                    "fm_scale_experiment": str(ARGS.FM_SCALE_EXPERIMENT),
                    "skipped": False,
                }
                row.update(
                    {
                        k: v
                        for k, v in m.items()
                        if k.startswith("day")
                        or k.startswith("forward_")
                        or k.startswith("reverse_")
                        or k.startswith("pooled_")
                        or k.startswith("prior_gate_")
                        or k.startswith("composition_")
                    }
                )
                rows.append(row)
            return rows
        except Exception as exc:
            if is_memory_related_runtime_error(exc):
                logger.exception(
                    f"[{variant.tag}] memory-related failure at batch_size={attempted_batch}"
                )
                release_memory_after_failure()
                if attempted_batch == 1:
                    raise
                attempted_batch = max(1, attempted_batch // 2)
                logger.info(f"[{variant.tag}] retry with smaller batch_size={attempted_batch}")
                continue
            raise
    raise RuntimeError(f"[{variant.tag}] exhausted adaptive batch attempts")


def train_one_fm_variant(
    *, variant: FMVariantSpec, device: torch.device, fp16: bool, logger: logging.Logger
):
    """Select single- or multiple-readout optimization from the requested training features."""
    use_new_path = (
        len(_drop_commented_or_empty_items(ARGS.FM_EVAL_READOUTS)) > 0
        or str(ARGS.FM_SCALE_EXPERIMENT) != "none"
        or int(ARGS.FM_MAX_TRAIN_STEPS_PER_EPOCH) > 0
        or (str(variant.bio_ablation) == "bio7_adaptive_full")
        or (str(ARGS.FM_TEMPORAL_DIRECTION) != "forward")
        or (float(ARGS.FM_CONTINUOUS_TIME_WEIGHT) > 0.0)
        or (float(ARGS.FM_CONTEXT_GENE_DROPOUT) > 0.0)
        or (float(ARGS.FM_CONTEXT_CELL_DROPOUT) > 0.0)
        or (float(ARGS.FM_COMPOSITION_LOSS_WEIGHT) > 0.0)
        or bool(str(ARGS.FM_INIT_CHECKPOINT).strip())
    )
    if use_new_path:
        return train_multiple_readouts(variant=variant, device=device, fp16=fp16, logger=logger)
    return train_single_readout(variant=variant, device=device, fp16=fp16, logger=logger)


def append_result_row(row: Dict[str, Any]) -> None:
    """
    Append one result row and print all aggregate evaluation metrics explicitly.

    Important for TRANS:
        - mean_*                  = original trajectory-occurrence weighting
        - mean_*_unique           = unique-target-cell weighting
        - reported_lower_*        = conservative minimum of the two
        - n_cells_occurrence      = number of trajectory-window occurrences
        - n_cells_unique          = number of unique biological target cells

    The final summary table below still prints every saved day-level metric.
    This RESULT line is intentionally verbose so a user reading only the .log
    can immediately see whether aligned evaluation changed the result.
    """
    RESULT_ROWS.append(row)
    family = str(row.get("family", "NA"))
    model_name = str(row.get("model", row.get("run_mode", "NA")))
    if family == "TRANS":
        message = f"RESULT {family:<6} {model_name:<50} | occurrence_cell_pcc={row.get('mean_cell_pcc', float('nan')):.4f} occurrence_gene_pcc={row.get('mean_gene_pcc', float('nan')):.4f}"
        if "mean_cell_pcc_unique" in row or "mean_gene_pcc_unique" in row:
            message += f" | unique_cell_pcc={row.get('mean_cell_pcc_unique', float('nan')):.4f} unique_gene_pcc={row.get('mean_gene_pcc_unique', float('nan')):.4f} | lower_cell_pcc={row.get('reported_lower_cell_pcc', float('nan')):.4f} lower_gene_pcc={row.get('reported_lower_gene_pcc', float('nan')):.4f} | n_occurrence={row.get('n_cells_occurrence', 'NA')} n_unique={row.get('n_cells_unique', 'NA')}"
        if "test_mse" in row:
            message += f" | test_mse={row.get('test_mse', float('nan')):.6f}"
        if "val_loss" in row:
            message += f" val_loss={row.get('val_loss', float('nan')):.6f}"
        if "best_epoch" in row:
            message += f" best_epoch={row.get('best_epoch', 'NA')}"
        if "best_stop" in row:
            try:
                message += f" best_stop={float(row.get('best_stop')):.6g}"
            except Exception:
                message += f" best_stop={row.get('best_stop')}"
        message += f" | batch={row.get('batch_size_used', float('nan'))} attempts={row.get('batch_size_attempts', float('nan'))} time={row.get('seconds', float('nan')):.1f}s"
        LOGGER.info(message)
        day_keys = sorted(
            (
                key
                for key in row.keys()
                if str(key).startswith("day")
                and (
                    str(key).endswith("_cell_pcc")
                    or str(key).endswith("_gene_pcc")
                    or str(key).endswith("_cell_pcc_unique")
                    or str(key).endswith("_gene_pcc_unique")
                )
            )
        )
        if len(day_keys) > 0:
            LOGGER.info(
                f"RESULT_DAY_METRICS {model_name} | "
                + " ".join((f"{key}={float(row[key]):.4f}" for key in day_keys))
            )
        return
    message = f"RESULT {family:<6} {model_name:<50} | mean_cell_pcc={row.get('mean_cell_pcc', float('nan')):.4f} mean_gene_pcc={row.get('mean_gene_pcc', float('nan')):.4f}"
    if "fm_readout" in row:
        message += f" readout={row.get('fm_readout')}"
    if "val_loss" in row:
        message += f" val_loss={row.get('val_loss', float('nan')):.6f}"
    if "best_val_pcc_cell" in row:
        message += f" best_val_cell={row.get('best_val_pcc_cell', float('nan')):.4f}"
    if "best_val_pcc_gene" in row:
        message += f" best_val_gene={row.get('best_val_pcc_gene', float('nan')):.4f}"
    if "best_epoch" in row:
        message += f" best_epoch={row.get('best_epoch', 'NA')}"
    if "best_stop" in row:
        try:
            message += f" best_stop={float(row.get('best_stop')):.6g}"
        except Exception:
            message += f" best_stop={row.get('best_stop')}"
    if "forward_val_pcc_gene" in row or "reverse_val_pcc_gene" in row:
        message += f" | forward_cell={row.get('forward_val_pcc_cell', float('nan')):.4f} forward_gene={row.get('forward_val_pcc_gene', float('nan')):.4f} reverse_cell={row.get('reverse_val_pcc_cell', float('nan')):.4f} reverse_gene={row.get('reverse_val_pcc_gene', float('nan')):.4f}"
    message += f" | batch={row.get('batch_size_used', float('nan'))} attempts={row.get('batch_size_attempts', float('nan'))} time={row.get('seconds', float('nan')):.1f}s"
    LOGGER.info(message)
    day_keys = sorted(
        (
            key
            for key in row.keys()
            if (
                str(key).startswith("day")
                or str(key).startswith("forward_day")
                or str(key).startswith("reverse_day")
                or str(key).startswith("pooled_day")
            )
            and (str(key).endswith("_cell_pcc") or str(key).endswith("_gene_pcc"))
        )
    )
    if len(day_keys) > 0:
        LOGGER.info(
            f"RESULT_DAY_METRICS {model_name} | "
            + " ".join((f"{key}={float(row[key]):.4f}" for key in day_keys))
        )


# Public training lifecycle and isolated contexts for checkpoint evaluation.
def checkpoint_metadata_for_model(metadata: dict) -> dict:
    """Retain historical schema tags while identifying the actual architecture."""
    metadata = dict(metadata)
    metadata["training_implementation"] = "bloom.training.trainer"
    metadata["model_family"] = MODEL_FAMILY
    if MODEL_FAMILY != "bidirectional":
        metadata = {
            key: value
            for key, value in metadata.items()
            if not key.startswith("fm_bidirectional_")
            and not key.startswith("fm_biological_")
            and key not in {"fm_reverse_prediction_claim", "fm_cycle_consistency_used"}
        }
        metadata["fm_composition_semantics"] = (
            "unconditional_requested_target_stage_celltype_distribution_from_population_history"
        )
    return metadata


def initialize(args: argparse.Namespace, *, build_data_bundle: bool = True) -> None:
    """Bind one run's arguments, aligned data, priors and model constructors."""
    global MODEL_FAMILY
    global ADATA_OBS, ADATA_VAR, ADATA_X, ADATA_X_ALL, ARGS, BIO_PRIOR_DATA, CELLS_BY_TIME, CELLS_BY_TIME_TYPE, CELLTYPE_CODES, CELLTYPE_COL, CELLTYPE_NAMES, CKPT_DIR, ConditionalVelocityField, DEVICE, FMCellEncoder, FMTemporalSetEncoder, FM_DATA, FM_TRAJ_NEXT_MAP, FM_TRAJ_PREV_MAP, FM_VARIANTS, FOUNDATION_CKPT_SPECS, FULL_CELLS_BY_TIME, FULL_CELLS_BY_TIME_TYPE, FULL_TIME_TYPE_COUNTS, HVG_MASK, LOGGER, LOG_DIR, LOG_PATH, N_CELLS, N_CELLTYPES, N_GENES, ORDINAL_TIME, RUN_ROOT, RawConditionalFlowMatching, SUBSAMPLED_TRAJECTORY_ROW_INDICES, SUBSAMPLED_TRAJ_CELL_INDICES_BY_TIME, SUBSAMPLED_TRAJ_CELL_MASK, SUMMARY_DIR, TIME_TYPE_COUNTS, TOTAL_TIMESTAMP, TRAJECTORIES, TRAJECTORIES_FULL_ARRAY, TRANS_DATA, TRANS_FOUNDATION_MODEL_SET, TRANS_RETRIEVER, TRANS_SPECS, _bound_models, _name, _only_tags, _value, by_type, cand, celltype_series, d, i, ids, idx, k, local_types, mapping, numeric_values, obs_time, spec, t, ts, ty, uniq, uniq_numeric, v, variant, x
    MODEL_FAMILY = str(args.model_family)
    if MODEL_FAMILY not in {"baseline", "unidirectional", "bidirectional"}:
        raise ValueError(f"Unknown model family: {MODEL_FAMILY}")
    if MODEL_FAMILY == "unidirectional" and args.FM_TEMPORAL_DIRECTION == "bidirectional":
        raise ValueError("Choose bloom-bidirectional for a shared bidirectional checkpoint")
    warnings.filterwarnings("ignore", category=UserWarning)
    ARGS = args
    ARGS.TRANS_BASE_MODELS = _drop_commented_or_empty_items(ARGS.TRANS_BASE_MODELS)
    ARGS.TRANS_FOUNDATION_MODELS = _drop_commented_or_empty_items(ARGS.TRANS_FOUNDATION_MODELS)
    ARGS.FOUNDATION_CHECKPOINT_SPECS = _drop_commented_or_empty_items(
        ARGS.FOUNDATION_CHECKPOINT_SPECS
    )
    ARGS.TRANS_SKIP_MODELS = _drop_commented_or_empty_items(ARGS.TRANS_SKIP_MODELS)
    ARGS.FM_VARIANTS = _drop_commented_or_empty_items(ARGS.FM_VARIANTS)
    ARGS.ONLY_MODEL_TAGS = _drop_commented_or_empty_items(ARGS.ONLY_MODEL_TAGS)
    ARGS.FM_BIO_ABLATIONS = _drop_commented_or_empty_items(ARGS.FM_BIO_ABLATIONS)
    ARGS.BIO_HIERARCHY_COLS = _drop_commented_or_empty_items(ARGS.BIO_HIERARCHY_COLS)
    ARGS.BIO_ANATOMY_COLS = _drop_commented_or_empty_items(ARGS.BIO_ANATOMY_COLS)
    ARGS.BIO_NUMERIC_CONTEXT_COLS = _drop_commented_or_empty_items(ARGS.BIO_NUMERIC_CONTEXT_COLS)
    ARGS.CURATED_GRAPH_NODE_OBS_COLS = _drop_commented_or_empty_items(
        ARGS.CURATED_GRAPH_NODE_OBS_COLS
    )
    if not 0.0 < float(ARGS.FM_PRIOR_GATE_INIT) < 1.0:
        raise ValueError("FM_PRIOR_GATE_INIT must lie strictly between 0 and 1")
    for _name in ("FM_PRIOR_DROPOUT", "FM_CONTEXT_GENE_DROPOUT", "FM_CONTEXT_CELL_DROPOUT"):
        _value = float(getattr(ARGS, _name))
        if not 0.0 <= _value < 1.0:
            raise ValueError(f"{_name} must be in [0, 1)")
    if float(ARGS.FM_PRIOR_GATE_L1_WEIGHT) < 0.0:
        raise ValueError("FM_PRIOR_GATE_L1_WEIGHT must be >= 0")
    if int(ARGS.FM_PRIOR_WARMUP_EPOCHS) < 0 or int(ARGS.FM_PRIOR_RAMP_EPOCHS) < 0:
        raise ValueError("FM_PRIOR_WARMUP_EPOCHS and FM_PRIOR_RAMP_EPOCHS must be >= 0")
    if int(ARGS.FM_EPOCHS) <= 0:
        raise ValueError("FM_EPOCHS must be > 0")
    if int(ARGS.FM_EARLY_STOPPING_PATIENCE) <= 0:
        raise ValueError("FM_EARLY_STOPPING_PATIENCE must be > 0")
    if float(ARGS.FM_EARLY_STOPPING_MIN_DELTA) < 0.0:
        raise ValueError("FM_EARLY_STOPPING_MIN_DELTA must be >= 0")
    if int(ARGS.FM_EARLY_STOPPING_START_EPOCH) < 1:
        raise ValueError("FM_EARLY_STOPPING_START_EPOCH must be >= 1")
    if not 0.0 < float(ARGS.FM_BIDIRECTIONAL_FORWARD_FRACTION) < 1.0:
        raise ValueError("FM_BIDIRECTIONAL_FORWARD_FRACTION must lie strictly between 0 and 1")
    if int(ARGS.FM_BIDIRECTIONAL_ADAPTER_HIDDEN) <= 0:
        raise ValueError("FM_BIDIRECTIONAL_ADAPTER_HIDDEN must be > 0")
    if float(ARGS.FM_CONTINUOUS_TIME_WEIGHT) < 0.0:
        raise ValueError("FM_CONTINUOUS_TIME_WEIGHT must be >= 0")
    if float(ARGS.FM_COMPOSITION_LOSS_WEIGHT) < 0.0:
        raise ValueError("FM_COMPOSITION_LOSS_WEIGHT must be >= 0")
    if (
        not 0.0
        < float(ARGS.FM_CONTINUOUS_TIME_MIN_FRACTION)
        < float(ARGS.FM_CONTINUOUS_TIME_MAX_FRACTION)
        < 1.0
    ):
        raise ValueError("continuous-time fractions must satisfy 0 < MIN < MAX < 1")
    for _name in (
        "TRANS_D_MODEL",
        "TRANS_N_HEADS",
        "TRANS_E_LAYERS",
        "TRANS_D_LAYERS",
        "TRANS_D_FF",
    ):
        if int(getattr(ARGS, _name)) <= 0:
            raise ValueError(f"{_name} must be > 0")
    if int(ARGS.TRANS_D_MODEL) % int(ARGS.TRANS_N_HEADS) != 0:
        raise ValueError("TRANS_D_MODEL must be divisible by TRANS_N_HEADS")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.CUDA_VISIBLE_DEVICES)
    if ARGS.HF_CACHE_DIR is not None:
        os.environ["HF_HOME"] = str(ARGS.HF_CACHE_DIR)
        os.environ["TRANSFORMERS_CACHE"] = str(ARGS.HF_CACHE_DIR)
    if bool(ARGS.FOUNDATION_LOCAL_FILES_ONLY):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    set_global_seed(seed=int(ARGS.SEED), deterministic_cudnn=bool(ARGS.DETERMINISTIC_CUDNN))
    torch.set_float32_matmul_precision(str(ARGS.TORCH_MATMUL_PRECISION))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUN_ROOT = str(ARGS.OUTPUT_DIR) if ARGS.OUTPUT_DIR is not None else _script_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = os.path.join(RUN_ROOT, f"{ARGS.MSG}_{ts}", str(ARGS.LOG_SUBDIR))
    CKPT_DIR = os.path.join(RUN_ROOT, f"{ARGS.MSG}_{ts}", str(ARGS.CHECKPOINT_SUBDIR))
    SUMMARY_DIR = os.path.join(RUN_ROOT, f"{ARGS.MSG}_{ts}", str(ARGS.SUMMARY_SUBDIR))
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    LOGGER, LOG_PATH = setup_logger()
    log_config(LOGGER)
    if int(ARGS.SEQ_LEN) > 0:
        LOGGER.info(
            f"Effective TRANS label length = {effective_trans_label_len()} (requested {ARGS.TRANS_LABEL_LEN}, SEQ_LEN {ARGS.SEQ_LEN})"
        )
    else:
        LOGGER.info(
            "SEQ_LEN=0: TRANS family will be skipped; FM hist0 uses one previous-day context; FM hist1 variants are skipped."
        )
    LOGGER.info("Loading sparse X / obs / var ...")
    ADATA_X_ALL = sp.load_npz(str(ARGS.ADATA_X_PATH)).tocsr()
    ADATA_OBS = read_obs_or_var_table(str(ARGS.ADATA_OBS_PATH))
    ADATA_VAR = read_obs_or_var_table(str(ARGS.ADATA_VAR_PATH))
    if "highly_variable" not in ADATA_VAR.columns:
        raise KeyError("ADATA_VAR_PATH must contain a 'highly_variable' column")
    if len(ADATA_VAR) != int(ADATA_X_ALL.shape[1]):
        raise ValueError(
            f"ADATA_VAR/X column mismatch: len(ADATA_VAR)={len(ADATA_VAR):,}, X.shape[1]={int(ADATA_X_ALL.shape[1]):,}. ADATA_VAR_PATH must be a full-length var table, not a selected-row subset."
        )
    HVG_MASK = ADATA_VAR["highly_variable"].fillna(False).to_numpy(dtype=bool)
    if not bool(HVG_MASK.any()):
        raise ValueError("ADATA_VAR_PATH selects zero genes via 'highly_variable'")
    ADATA_X = ADATA_X_ALL[:, HVG_MASK].tocsr()
    N_CELLS = int(ADATA_X.shape[0])
    N_GENES = int(ADATA_X.shape[1])
    LOGGER.info(f"Loaded X with N_CELLS={N_CELLS}, N_GENES={N_GENES}")
    if str(ARGS.TIME_COL) not in ADATA_OBS.columns:
        raise KeyError(f"TIME_COL={ARGS.TIME_COL!r} not found in obs")
    obs_time = ADATA_OBS[str(ARGS.TIME_COL)]
    if isinstance(obs_time.dtype, pd.CategoricalDtype):
        ORDINAL_TIME = obs_time.cat.codes.to_numpy(dtype=np.int32)
    elif pd.api.types.is_numeric_dtype(obs_time.dtype):
        numeric_values = pd.to_numeric(obs_time, errors="raise")
        uniq_numeric = np.sort(numeric_values.unique())
        mapping = {v: i for i, v in enumerate(uniq_numeric)}
        ORDINAL_TIME = numeric_values.map(mapping).to_numpy(dtype=np.int32)
    else:
        uniq = pd.Index(obs_time.astype(str).unique()).sort_values()
        mapping = {v: i for i, v in enumerate(uniq)}
        ORDINAL_TIME = obs_time.astype(str).map(mapping).to_numpy(dtype=np.int32)
    TOTAL_TIMESTAMP = int(ORDINAL_TIME.max() + 1)
    LOGGER.info(f"Total timestamps: {TOTAL_TIMESTAMP}")
    CELLTYPE_COL = str(ARGS.CELLTYPE_COL)
    if CELLTYPE_COL not in ADATA_OBS.columns:
        for cand in ("cell_type", "celltype", "CellType", "class_label", "type", "label"):
            if cand in ADATA_OBS.columns:
                CELLTYPE_COL = cand
                LOGGER.info(f"[celltype] fallback CELLTYPE_COL={CELLTYPE_COL}")
                break
        else:
            raise KeyError(
                f"CELLTYPE_COL={ARGS.CELLTYPE_COL!r} not found. Available columns head: {list(ADATA_OBS.columns)[:50]}"
            )
    celltype_series = ADATA_OBS[CELLTYPE_COL].astype("category")
    CELLTYPE_CODES = celltype_series.cat.codes.to_numpy(dtype=np.int32)
    CELLTYPE_NAMES = list(celltype_series.cat.categories)
    N_CELLTYPES = int(len(CELLTYPE_NAMES))
    LOGGER.info(f"[celltype] obs[{CELLTYPE_COL!r}] N_CELLTYPES={N_CELLTYPES}")
    BIO_PRIOR_DATA = build_biological_prior_data() if bool(ARGS.RUN_FM) else None
    if bool(ARGS.BIO_PRIOR_VALIDATE_ONLY):
        if not bool(ARGS.RUN_FM):
            raise ValueError("--BIO_PRIOR_VALIDATE_ONLY requires --RUN_FM")
        LOGGER.info(
            "[bio] validation-only mode completed; exiting before trajectory loading and model training"
        )
        raise SystemExit(0)
    CELLS_BY_TIME = []
    CELLS_BY_TIME_TYPE = []
    TIME_TYPE_COUNTS = np.zeros((TOTAL_TIMESTAMP, N_CELLTYPES), dtype=np.int64)
    for t in range(TOTAL_TIMESTAMP):
        idx = np.flatnonzero(ORDINAL_TIME == t).astype(np.int64)
        CELLS_BY_TIME.append(idx)
        by_type = {}
        if idx.size > 0:
            local_types = CELLTYPE_CODES[idx]
            for ty in np.unique(local_types):
                ty = int(ty)
                ids = idx[local_types == ty].astype(np.int64)
                by_type[ty] = ids
                TIME_TYPE_COUNTS[t, ty] = int(ids.size)
        CELLS_BY_TIME_TYPE.append(by_type)
    FULL_CELLS_BY_TIME = [x.copy() for x in CELLS_BY_TIME]
    FULL_CELLS_BY_TIME_TYPE = [{int(k): v.copy() for k, v in d.items()} for d in CELLS_BY_TIME_TYPE]
    FULL_TIME_TYPE_COUNTS = TIME_TYPE_COUNTS.copy()
    TRAJECTORIES = None
    TRAJECTORIES_FULL_ARRAY = None
    if ARGS.TRAJ_PATH is not None and os.path.exists(str(ARGS.TRAJ_PATH)):
        LOGGER.info("Loading trajectories ...")
        TRAJECTORIES = load_pickle_or_npy(str(ARGS.TRAJ_PATH))
        TRAJECTORIES_FULL_ARRAY = np.asarray(TRAJECTORIES, dtype=np.int64)
        LOGGER.info(
            f"Loaded trajectories: N_TRAJ={len(TRAJECTORIES)} N_TIME={(len(TRAJECTORIES[0]) if len(TRAJECTORIES) > 0 else 'NA')}"
        )
    else:
        LOGGER.info(
            "TRAJ_PATH missing or not provided. TRANS and FM traj-soft specs may be skipped."
        )
    SUBSAMPLED_TRAJECTORY_ROW_INDICES = None
    SUBSAMPLED_TRAJ_CELL_MASK = None
    SUBSAMPLED_TRAJ_CELL_INDICES_BY_TIME = None
    apply_deterministic_trajectory_subsampling_if_requested()
    if str(ARGS.REPO_DIR) not in sys.path:
        sys.path.append(str(ARGS.REPO_DIR))
    TRANS_FOUNDATION_MODEL_SET = set((str(x) for x in ARGS.TRANS_FOUNDATION_MODELS))
    FOUNDATION_CKPT_SPECS = parse_foundation_checkpoint_specs(ARGS.FOUNDATION_CHECKPOINT_SPECS)
    TRANS_SPECS = build_transform_specs()
    if len(ARGS.ONLY_MODEL_TAGS) > 0:
        _only_tags = set((str(x) for x in ARGS.ONLY_MODEL_TAGS))
        TRANS_SPECS = [spec for spec in TRANS_SPECS if spec.tag in _only_tags]
    TRANS_DATA = None
    if bool(ARGS.RUN_TRANS):
        if TRAJECTORIES is None:
            LOGGER.info("[TRANS] TRAJ_PATH unavailable -> TRANS family will be skipped.")
        elif int(ARGS.SEQ_LEN) <= 0:
            LOGGER.info("[TRANS] SEQ_LEN=0 -> TRANS family will be skipped for this run.")
        else:
            TRANS_DATA = TransformDataBundle()
    TRANS_RETRIEVER = None
    if bool(ARGS.RUN_TRANS) and bool(ARGS.TRANS_RETRIEVAL_ENABLED) and (TRANS_DATA is not None):
        TRANS_RETRIEVER = RetrievalPostProcessor(
            adata_X=ADATA_X,
            ordinal_time=ORDINAL_TIME,
            train_timestamps=TRANS_DATA.train_timestamps,
            pool_mode=str(ARGS.TRANS_RETRIEVAL_POOL_MODE),
            metric=str(ARGS.TRANS_RETRIEVAL_METRIC),
            topk=int(ARGS.TRANS_RETRIEVAL_TOPK),
            max_pool_per_ts=ARGS.TRANS_RETRIEVAL_MAX_POOL_PER_TS,
            seed=int(ARGS.SEED),
            logger=LOGGER,
        )
    FM_VARIANTS = build_fm_variants()
    if len(ARGS.ONLY_MODEL_TAGS) > 0:
        _only_tags = set((str(x) for x in ARGS.ONLY_MODEL_TAGS))
        FM_VARIANTS = [variant for variant in FM_VARIANTS if variant.tag in _only_tags]
    FM_TRAJ_PREV_MAP = (
        build_traj_prev_map_if_possible()
        if TRAJECTORIES is not None
        and str(ARGS.FM_TEMPORAL_DIRECTION) in {"forward", "bidirectional"}
        else None
    )
    FM_TRAJ_NEXT_MAP = (
        build_traj_next_map_if_possible()
        if TRAJECTORIES is not None
        and str(ARGS.FM_TEMPORAL_DIRECTION) in {"reverse", "bidirectional"}
        else None
    )
    FM_DATA = FMDataBundle() if bool(ARGS.RUN_FM) and build_data_bundle else None
    _bound_models = bind_native_models(
        ARGS,
        BIO_PRIOR_DATA,
        direction="bidirectional" if MODEL_FAMILY == "bidirectional" else "unidirectional",
    )
    FMCellEncoder = _bound_models["FMCellEncoder"]
    FMTemporalSetEncoder = _bound_models["FMTemporalSetEncoder"]
    ConditionalVelocityField = _bound_models["ConditionalVelocityField"]
    RawConditionalFlowMatching = _bound_models["RawConditionalFlowMatching"]
    del _bound_models


def _run_training() -> dict:
    """Run the requested optimization/evaluation jobs and save their summary."""
    global RESULT_ROWS, RUN_START, c, day_columns, front_columns, one_row, ordered_columns, remaining_columns, row, row_or_rows, spec, summary_csv_path, summary_df, v, variant
    RESULT_ROWS = []
    RUN_START = time.time()
    LOGGER.info("============================================================")
    LOGGER.info("TRANS specs:")
    LOGGER.info(", ".join((spec.tag for spec in TRANS_SPECS)) if len(TRANS_SPECS) > 0 else "(none)")
    LOGGER.info("FM variants:")
    LOGGER.info(", ".join((v.tag for v in FM_VARIANTS)) if len(FM_VARIANTS) > 0 else "(none)")
    LOGGER.info("============================================================")
    if bool(ARGS.RUN_TRANS):
        for spec in TRANS_SPECS:
            try:
                row = train_one_transform_spec(
                    spec=spec, device=DEVICE, fp16=bool(ARGS.FP16), logger=LOGGER
                )
            except Exception as exc:
                LOGGER.exception(f"[{spec.tag}] crashed; recording NaNs.")
                row = {
                    "family": "TRANS",
                    "model": spec.tag,
                    "mean_cell_pcc": float("nan"),
                    "mean_gene_pcc": float("nan"),
                    "batch_size_used": float("nan"),
                    "batch_size_attempts": float("nan"),
                    "seconds": float("nan"),
                    "crashed": True,
                    "crash_reason": str(exc),
                }
            append_result_row(row)
    if bool(ARGS.RUN_FM):
        for variant in FM_VARIANTS:
            try:
                row_or_rows = train_one_fm_variant(
                    variant=variant, device=DEVICE, fp16=bool(ARGS.FP16), logger=LOGGER
                )
            except Exception as exc:
                LOGGER.exception(f"[{variant.tag}] crashed; recording NaNs.")
                row_or_rows = {
                    "family": "FM",
                    "model": variant.tag,
                    "mean_cell_pcc": float("nan"),
                    "mean_gene_pcc": float("nan"),
                    "batch_size_used": float("nan"),
                    "batch_size_attempts": float("nan"),
                    "seconds": float("nan"),
                    "crashed": True,
                    "crash_reason": str(exc),
                }
            if isinstance(row_or_rows, list):
                for one_row in row_or_rows:
                    append_result_row(one_row)
            else:
                append_result_row(row_or_rows)
    summary_df = pd.DataFrame(RESULT_ROWS)
    front_columns = [
        "family",
        "model",
        "base_model",
        "fm_readout",
        "run_mode",
        "checkpoint_label",
        "mean_cell_pcc",
        "mean_gene_pcc",
        "mean_cell_pcc_unique",
        "mean_gene_pcc_unique",
        "reported_lower_cell_pcc",
        "reported_lower_gene_pcc",
        "n_cells_occurrence",
        "n_cells_unique",
        "test_mse",
        "val_loss",
        "val_pcc_cell_unique",
        "val_pcc_gene_unique",
        "val_pcc_cell_lower",
        "val_pcc_gene_lower",
        "val_n_cells_occurrence",
        "val_n_cells_unique",
        "best_val_pcc_cell",
        "best_val_pcc_gene",
        "best_epoch",
        "best_stop",
        "batch_size_used",
        "batch_size_attempts",
        "seconds",
        "fm_scale_experiment",
        "fm_bio_ablation",
        "skipped",
        "skip_reason",
        "crashed",
        "crash_reason",
        "checkpoint_path",
    ]
    day_columns = sorted(
        [
            c
            for c in summary_df.columns
            if c.startswith("day")
            and (
                c.endswith("_cell_pcc")
                or c.endswith("_gene_pcc")
                or c.endswith("_cell_pcc_unique")
                or c.endswith("_gene_pcc_unique")
            )
        ]
    )
    remaining_columns = [
        c for c in summary_df.columns if c not in front_columns and c not in day_columns
    ]
    ordered_columns = (
        [c for c in front_columns if c in summary_df.columns] + day_columns + remaining_columns
    )
    if "mean_gene_pcc" in summary_df.columns:
        summary_df = summary_df.sort_values(by="mean_gene_pcc", ascending=False, na_position="last")
    LOGGER.info("\n================ unified summary table ================")
    LOGGER.info(
        "\n"
        + pretty_md_table(
            rows=summary_df.to_dict(orient="records"), columns=ordered_columns, floatfmt=".4f"
        )
    )
    LOGGER.info("=======================================================")
    LOGGER.info(f"Benchmark finished in {time.time() - RUN_START:.2f} seconds.")
    LOGGER.info("=======================================================")
    if bool(ARGS.SAVE_CSV):
        summary_csv_path = os.path.join(
            SUMMARY_DIR, f"{ARGS.MSG}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        summary_df.loc[:, ordered_columns].to_csv(summary_csv_path, index=False)
        LOGGER.info(f"Saved summary CSV to {summary_csv_path}")
    return {
        "RESULT_ROWS": RESULT_ROWS,
        "summary": summary_df,
        "run_root": RUN_ROOT,
        "checkpoint_dir": CKPT_DIR,
        "summary_dir": SUMMARY_DIR,
    }


def load_runtime(model_family: str, argv=None, *, build_data_bundle: bool = True) -> ModuleType:
    """Create an independent, initialized training context without optimization.

    Checkpoint evaluation uses this same loader. Each returned context owns its
    arguments, data axes, priors, helpers and model bindings; another context
    cannot replace its global configuration.
    """
    name = f"bloom.training._context_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve())
    if spec is None or spec.loader is None:
        raise ImportError("Could not create a BLOOM training context")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        if isinstance(argv, argparse.Namespace):
            args = copy.deepcopy(argv)
            if getattr(args, "model_family", model_family) != model_family:
                raise ValueError("Requested context and argument model families disagree")
            args.model_family = model_family
        else:
            values = list(sys.argv[1:] if argv is None else argv)
            args = parse_args(values, model_family=model_family)
            if not any(value in {"--RUN_TRANS", "--no-RUN_TRANS"} for value in values):
                args.RUN_TRANS = model_family == "baseline"
            if not any(value in {"--RUN_FM", "--no-RUN_FM"} for value in values):
                args.RUN_FM = model_family != "baseline"
        module.initialize(args, build_data_bundle=build_data_bundle)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def run(args: argparse.Namespace) -> dict:
    """Train a prepared model configuration through the common implementation."""
    runtime = load_runtime(str(args.model_family), args)
    return runtime._run_training()


def main(argv=None) -> int:
    args = parse_args(argv)
    values = list(sys.argv[1:] if argv is None else argv)
    if not any(value in {"--RUN_TRANS", "--no-RUN_TRANS"} for value in values):
        args.RUN_TRANS = args.model_family == "baseline"
    if not any(value in {"--RUN_FM", "--no-RUN_FM"} for value in values):
        args.RUN_FM = args.model_family != "baseline"
    result = run(args)
    return int(any(row.get("crashed") or row.get("skipped") for row in result["RESULT_ROWS"]))


if __name__ == "__main__":
    raise SystemExit(main())
