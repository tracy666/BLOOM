#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Infer and export transcription-factor--target priors from aligned X/obs/var files.

This is the supplied pySCENIC pipeline: GRNBoost2 or GENIE3 adjacency inference,
optional cisTarget pruning and AUCell scoring, and model-panel GRN exports. The
broad inference-gene table and downstream model-gene table are separate inputs.
The model recipes use the weighted ``adjacencies.tsv`` associations; running
motif pruning is optional for that use case.

Use ``python -m bloom.preprocessing.grn --help`` for the complete source parser.
External TF lists, ranking databases and motif annotations are user resources.
"""

import argparse
import ast
import gc
import inspect
import json
import os
import pickle
import random
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from datetime import datetime

# =============================================================================
# Small general helpers
# =============================================================================


def str2bool(x: Any) -> bool:
    """Convert flexible CLI text into a boolean."""
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from {x!r}")


def now() -> str:
    """Human-readable timestamp for logs."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    """Print and immediately flush so logs appear in slurm output."""
    print(f"[{now()}] {message}", flush=True)


def ensure_dir(path: str | Path) -> Path:
    """Create an output directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    """Save a JSON file with pretty indentation."""
    path = Path(path)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def read_one_column_text(path: str | Path) -> List[str]:
    """Read a one-column text file, ignoring empty lines and comments."""
    out: List[str] = []
    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.split()[0])
    return out


def make_unique(names: Sequence[str]) -> List[str]:
    """
    Make gene names unique.

    pySCENIC expects unique gene names. In normal 10x/mouse data, gene_short_name is usually
    unique enough, but duplicates sometimes happen. We keep the first occurrence unchanged and
    append __dup1, __dup2, ... to repeated names.
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for raw in names:
        name = str(raw)
        if name not in seen:
            seen[name] = 0
            out.append(name)
        else:
            seen[name] += 1
            out.append(f"{name}__dup{seen[name]}")
    return out


def choose_gene_names(var_df: pd.DataFrame, requested_col: str) -> List[str]:
    """
    Choose gene names from var.

    For mouse pySCENIC, MGI gene symbols are preferred. The gene_short_name
    column is used when available, with explicit metadata-column fallbacks.
    """
    candidate_columns = [
        requested_col,
        "gene_short_name",
        "gene_name",
        "gene_symbol",
        "gene_symbols",
        "feature_name",
        "gene_id",
    ]

    for col in candidate_columns:
        if col is not None and col in var_df.columns:
            values = var_df[col].astype(str).tolist()
            log(f"[gene] using var[{col!r}] as gene names")
            return values

    index_values = var_df.index.astype(str).tolist()
    log("[gene] no requested/common gene column found; using var.index as gene names")
    return index_values


def check_required_file(path: str | Path, description: str) -> None:
    """Fail early with a readable message if a file is missing."""
    if path is None:
        raise FileNotFoundError(f"Missing required {description}: path is None")
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing required {description}: {path}")


# =============================================================================
# Data loading and selection
# =============================================================================


def load_project_data(
    *,
    adata_x_path: str,
    adata_obs_path: str,
    adata_var_path: str,
    model_var_path: Optional[str],
    gene_name_col: str,
    hvg_col: str,
    use_hvg_mask: bool,
    include_tfs_outside_hvg: bool,
    tf_names_path: Optional[str],
    min_cells_per_gene: int,
) -> Tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame, List[str], np.ndarray, np.ndarray]:
    """
    Load X/obs and construct TWO deliberately different gene sets:

    1. GRN-inference genes
       Selected from ``adata_var_path``.  For the FM project, this should usually
       be broader than the actual FM feature set (for example HVG2000 plus all
       available TFs), because inferring a GRN from only 200 model genes can remove
       many biologically important regulators before pySCENIC even starts.

    2. FM model genes
       Selected from ``model_var_path``.  This should be the exact var/HVG file
       used by the downstream FM model (for example HVG200).  Exported G x G
       matrices are aligned to this exact feature order.

    Backward compatibility
    ----------------------
    If ``model_var_path`` is omitted, the old behavior is preserved and
    ``adata_var_path`` is used for both gene sets.

    Returns
    -------
    X_grn:
        Sparse matrix [n_cells, n_grn_genes] used by GRNBoost2/GENIE3.
    obs:
        Cell metadata aligned to X rows.
    var:
        The GRN-selection var dataframe aligned to original X columns.
    grn_gene_names:
        Gene symbols used during pySCENIC GRN inference.
    grn_gene_original_indices:
        Original X-column indices used for GRN inference.
    model_gene_original_indices:
        Original X-column indices corresponding exactly to the FM model feature
        mask from ``model_var_path``.
    """
    log(f"[load] X          : {adata_x_path}")
    X_all = sp.load_npz(adata_x_path).tocsr()

    log(f"[load] obs        : {adata_obs_path}")
    obs = pd.read_parquet(adata_obs_path)

    log(f"[load] GRN var    : {adata_var_path}")
    var = pd.read_parquet(adata_var_path)

    resolved_model_var_path = model_var_path if model_var_path is not None else adata_var_path
    log(f"[load] model var  : {resolved_model_var_path}")
    model_var = pd.read_parquet(resolved_model_var_path)

    if X_all.shape[0] != obs.shape[0]:
        raise ValueError(
            f"X rows ({X_all.shape[0]}) != obs rows ({obs.shape[0]}). " "The files are not aligned."
        )
    if X_all.shape[1] != var.shape[0]:
        raise ValueError(
            f"X columns ({X_all.shape[1]}) != GRN-var rows ({var.shape[0]}). "
            "The var file must contain all genes with a highly_variable mask."
        )
    if X_all.shape[1] != model_var.shape[0]:
        raise ValueError(
            f"X columns ({X_all.shape[1]}) != model-var rows ({model_var.shape[0]}). "
            "The model var file must contain all genes with a highly_variable mask."
        )

    all_gene_names = np.asarray(
        choose_gene_names(var, requested_col=gene_name_col),
        dtype=object,
    )
    model_all_gene_names = np.asarray(
        choose_gene_names(model_var, requested_col=gene_name_col),
        dtype=object,
    )

    if all_gene_names.shape != model_all_gene_names.shape or not np.array_equal(
        all_gene_names, model_all_gene_names
    ):
        mismatch = np.flatnonzero(all_gene_names != model_all_gene_names)
        mismatch_preview = mismatch[:10].tolist()
        raise ValueError(
            "GRN-var and model-var gene orders/names are not identical. They must both align exactly to X. "
            f"First mismatch positions: {mismatch_preview}"
        )

    if use_hvg_mask:
        if hvg_col not in var.columns:
            raise KeyError(f"--use_hvg_mask True but GRN var does not contain column {hvg_col!r}")
        if hvg_col not in model_var.columns:
            raise KeyError(f"--use_hvg_mask True but model var does not contain column {hvg_col!r}")
        grn_gene_mask = var[hvg_col].fillna(False).to_numpy(dtype=bool)
        model_gene_mask = model_var[hvg_col].fillna(False).to_numpy(dtype=bool)
        log(
            f"[gene] GRN HVG mask n={int(grn_gene_mask.sum())}; "
            f"FM model HVG mask n={int(model_gene_mask.sum())}"
        )
    else:
        grn_gene_mask = np.ones(X_all.shape[1], dtype=bool)
        model_gene_mask = np.ones(X_all.shape[1], dtype=bool)
        log("[gene] --use_hvg_mask False; using all genes for both GRN and model masks")

    if int(grn_gene_mask.sum()) == 0:
        raise RuntimeError("No GRN genes selected. Check the GRN var highly_variable column.")
    if int(model_gene_mask.sum()) == 0:
        raise RuntimeError(
            "No FM model genes selected. Check the model var highly_variable column."
        )

    # Recommended for this project: infer the network on a broad HVG set while
    # retaining TFs that are not themselves HVGs.  The final G x G prior is still
    # projected back to the exact FM model genes later.
    if include_tfs_outside_hvg and tf_names_path is not None and Path(tf_names_path).exists():
        tf_names = set(read_one_column_text(tf_names_path))
        add_tf_mask = np.asarray([str(g) in tf_names for g in all_gene_names], dtype=bool)
        before = int(grn_gene_mask.sum())
        grn_gene_mask = grn_gene_mask | add_tf_mask
        after = int(grn_gene_mask.sum())
        log(
            f"[gene] added TFs outside GRN HVG: before={before}, after={after}, added={after - before}"
        )

    grn_gene_original_indices = np.flatnonzero(grn_gene_mask).astype(np.int64)
    X_tmp = X_all[:, grn_gene_original_indices].tocsr()

    if int(min_cells_per_gene) > 0:
        n_cells_by_gene = np.asarray((X_tmp > 0).sum(axis=0)).reshape(-1)
        expressed_mask = n_cells_by_gene >= int(min_cells_per_gene)
        log(
            f"[gene] global min_cells_per_gene={min_cells_per_gene}; "
            f"before={X_tmp.shape[1]}, after={int(expressed_mask.sum())}"
        )
        grn_gene_original_indices = grn_gene_original_indices[expressed_mask]
        X_tmp = X_all[:, grn_gene_original_indices].tocsr()

    # pySCENIC and cisTarget operate on gene symbols. If the underlying matrix
    # contains duplicate symbols, keep only the first occurrence for GRN inference
    # instead of inventing names such as Gene__dup1 that do not exist in the
    # motif/ranking databases.
    pre_dedup_indices = np.asarray(grn_gene_original_indices, dtype=np.int64)
    pre_dedup_names = [str(all_gene_names[i]) for i in pre_dedup_indices]
    seen_grn_names: set = set()
    dedup_positions: List[int] = []
    for pos, gene_name in enumerate(pre_dedup_names):
        if gene_name in seen_grn_names:
            continue
        seen_grn_names.add(gene_name)
        dedup_positions.append(pos)
    if len(dedup_positions) != len(pre_dedup_names):
        log(
            f"[gene] duplicate gene symbols in GRN set: keeping first occurrence; "
            f"before={len(pre_dedup_names)}, after={len(dedup_positions)}"
        )
        dedup_positions_array = np.asarray(dedup_positions, dtype=np.int64)
        grn_gene_original_indices = pre_dedup_indices[dedup_positions_array]
        X_tmp = X_all[:, grn_gene_original_indices].tocsr()

    model_gene_original_indices = np.flatnonzero(model_gene_mask).astype(np.int64)
    grn_gene_names = [str(all_gene_names[i]) for i in grn_gene_original_indices]

    # A gene-name prior cannot be mapped unambiguously if the FM feature set has
    # duplicate symbols. Fail early instead of silently mapping an edge to the
    # wrong duplicate position.
    model_gene_names = [str(all_gene_names[i]) for i in model_gene_original_indices]
    duplicated_model_names = (
        pd.Index(model_gene_names)[pd.Index(model_gene_names).duplicated()].unique().tolist()
    )
    if len(duplicated_model_names) > 0:
        raise ValueError(
            "The selected FM model genes contain duplicate gene symbols, which makes a gene-symbol GRN prior "
            "ambiguous. Resolve duplicates in the var file before training. Examples: "
            f"{duplicated_model_names[:20]}"
        )

    log(
        f"[gene] final pre-sampling sets: GRN genes={len(grn_gene_names)}, "
        f"FM model genes={len(model_gene_original_indices)}"
    )

    return X_tmp, obs, var, grn_gene_names, grn_gene_original_indices, model_gene_original_indices


def filter_grn_genes_after_cell_sampling(
    *,
    X_grn: sp.csr_matrix,
    selected_cells: np.ndarray,
    grn_gene_names: List[str],
    grn_gene_original_indices: np.ndarray,
    variance_epsilon: float,
) -> Tuple[sp.csr_matrix, List[str], np.ndarray]:
    """
    Remove genes that are constant in the ACTUAL sampled cells used by GRNBoost2.

    The older script filtered genes by global detection count before sampling,
    but a globally variable gene can still become constant after selecting a
    50k/100k-cell stratified subset.  Arboreto fits one regression per target
    gene, so removing those targets prevents useless/unstable regressions.

    This calculation remains sparse and therefore does not allocate the dense
    expression matrix twice.
    """
    selected_cells = np.asarray(selected_cells, dtype=np.int64)
    if selected_cells.size == 0:
        raise RuntimeError("Cannot variance-filter GRN genes because zero cells were selected.")

    X_sub = X_grn[selected_cells, :].tocsr()
    n = float(X_sub.shape[0])
    sums = np.asarray(X_sub.sum(axis=0)).reshape(-1).astype(np.float64, copy=False)
    squared_sums = (
        np.asarray(X_sub.multiply(X_sub).sum(axis=0)).reshape(-1).astype(np.float64, copy=False)
    )
    variances = np.maximum(0.0, squared_sums / n - (sums / n) ** 2)
    keep = np.isfinite(variances) & (variances > float(variance_epsilon))

    n_before = int(X_grn.shape[1])
    n_after = int(keep.sum())
    log(
        f"[gene] post-sampling variance filter epsilon={variance_epsilon}: "
        f"before={n_before}, after={n_after}, removed={n_before - n_after}"
    )
    if n_after < 2:
        raise RuntimeError(
            f"Too few GRN genes remain after sampled-cell variance filtering: {n_after}"
        )

    X_filtered = X_grn[:, keep].tocsr()
    names_filtered = [str(g) for g, use in zip(grn_gene_names, keep.tolist()) if use]
    original_indices_filtered = np.asarray(grn_gene_original_indices, dtype=np.int64)[keep]
    return X_filtered, names_filtered, original_indices_filtered


def select_cells(
    *,
    obs: pd.DataFrame,
    time_col: str,
    celltype_col: str,
    sample_strategy: str,
    max_cells_total: Optional[int],
    max_cells_per_time: Optional[int],
    max_cells_per_time_celltype: Optional[int],
    seed: int,
) -> np.ndarray:
    """
    Select cells for pySCENIC.

    pySCENIC can be expensive on very large atlases. For a first GRN prior, a stratified subset is often
    more practical and still preserves time/cell-type diversity.
    """
    rng = np.random.default_rng(int(seed))
    n_cells = int(obs.shape[0])
    all_indices = np.arange(n_cells, dtype=np.int64)

    if sample_strategy == "all":
        chosen = all_indices

    elif sample_strategy == "random":
        if max_cells_total is None or int(max_cells_total) <= 0 or int(max_cells_total) >= n_cells:
            chosen = all_indices
        else:
            chosen = np.sort(
                rng.choice(all_indices, size=int(max_cells_total), replace=False).astype(np.int64)
            )

    elif sample_strategy == "stratified_time":
        if time_col not in obs.columns:
            raise KeyError(f"time_col={time_col!r} not found in obs")
        if max_cells_per_time is None or int(max_cells_per_time) <= 0:
            raise ValueError("sample_strategy=stratified_time requires --max_cells_per_time > 0")
        pieces: List[np.ndarray] = []
        for _, idx in obs.groupby(time_col, observed=True).indices.items():
            idx_arr = np.asarray(idx, dtype=np.int64)
            if idx_arr.size > int(max_cells_per_time):
                idx_arr = rng.choice(idx_arr, size=int(max_cells_per_time), replace=False).astype(
                    np.int64
                )
            pieces.append(idx_arr)
        chosen = np.sort(np.concatenate(pieces)) if pieces else np.array([], dtype=np.int64)

    elif sample_strategy == "stratified_time_celltype":
        if time_col not in obs.columns:
            raise KeyError(f"time_col={time_col!r} not found in obs")
        if celltype_col not in obs.columns:
            raise KeyError(f"celltype_col={celltype_col!r} not found in obs")
        if max_cells_per_time_celltype is None or int(max_cells_per_time_celltype) <= 0:
            raise ValueError(
                "sample_strategy=stratified_time_celltype requires --max_cells_per_time_celltype > 0"
            )
        pieces = []
        grouped = obs.groupby([time_col, celltype_col], observed=True).indices
        for _, idx in grouped.items():
            idx_arr = np.asarray(idx, dtype=np.int64)
            if idx_arr.size > int(max_cells_per_time_celltype):
                idx_arr = rng.choice(
                    idx_arr, size=int(max_cells_per_time_celltype), replace=False
                ).astype(np.int64)
            pieces.append(idx_arr)
        chosen = np.sort(np.concatenate(pieces)) if pieces else np.array([], dtype=np.int64)

    else:
        raise ValueError(f"Unknown sample_strategy={sample_strategy!r}")

    # Optional global cap after stratified sampling.
    if (
        max_cells_total is not None
        and int(max_cells_total) > 0
        and chosen.size > int(max_cells_total)
    ):
        chosen = np.sort(
            rng.choice(chosen, size=int(max_cells_total), replace=False).astype(np.int64)
        )

    log(f"[cell] selected {chosen.size} / {n_cells} cells using strategy={sample_strategy}")
    return chosen.astype(np.int64)


def sparse_to_expression_dataframe(
    *,
    X: sp.csr_matrix,
    selected_cells: np.ndarray,
    gene_names: List[str],
    dtype: str,
) -> pd.DataFrame:
    """
    Convert selected sparse expression to a dense pandas DataFrame required by Arboreto/pySCENIC.

    Rows are cells, columns are genes.
    """
    log(
        f"[expr] converting sparse subset to dense DataFrame: cells={selected_cells.size}, genes={len(gene_names)}"
    )
    X_sub = X[selected_cells, :].toarray()
    if dtype == "float32":
        X_sub = X_sub.astype(np.float32, copy=False)
    elif dtype == "float64":
        X_sub = X_sub.astype(np.float64, copy=False)
    else:
        raise ValueError("dtype must be float32 or float64")
    X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
    X_sub[X_sub < 0] = 0.0
    cell_names = [f"cell_{int(i)}" for i in selected_cells.tolist()]
    expr_df = pd.DataFrame(X_sub, index=cell_names, columns=gene_names)
    return expr_df


# =============================================================================
# pySCENIC pipeline functions
# =============================================================================


def make_dask_client(
    *, num_workers: int, threads_per_worker: int, memory_limit: str, use_dask: bool
):
    """Create a local Dask client if requested; otherwise return 'local'."""
    if not use_dask:
        log("[dask] using Arboreto local mode")
        return "local", None

    try:
        from dask.distributed import Client, LocalCluster
    except Exception as exc:
        log(f"[dask] could not import dask.distributed ({exc}); falling back to local mode")
        return "local", None

    cluster = LocalCluster(
        n_workers=int(num_workers),
        threads_per_worker=int(threads_per_worker),
        processes=True,
        memory_limit=str(memory_limit),
        dashboard_address=None,
    )
    client = Client(cluster)
    log(
        f"[dask] local cluster started: workers={num_workers}, threads_per_worker={threads_per_worker}"
    )
    return client, cluster


def apply_arboreto_modern_dask_compatibility_patch() -> Dict[str, Any]:
    """
    Patch the released Arboreto 0.1.6 ``create_graph`` implementation for
    compatibility with modern Dask DataFrame.

    Why this is needed
    ------------------
    Arboreto 0.1.6 creates ``delayed_meta_dfs = []`` when the normal public
    GRNBoost2/GENIE3 path uses ``include_meta=False``.  Its old
    ``create_graph`` implementation nevertheless calls::

        from_delayed(delayed_meta_dfs, meta=_META_SCHEMA)

    Modern Dask raises ``TypeError: Must supply at least one delayed object``
    for that empty list.  The metadata dataframe is unused when
    ``include_meta=False``.  Arboreto's current source fixes the defect by only
    constructing ``all_meta_df`` inside ``if include_meta``.

    This runtime shim is deliberately narrow: it changes only Arboreto's
    module-local ``from_delayed`` reference, and only when Arboreto passes an
    empty list/tuple with an explicit ``meta`` dataframe.  All non-empty calls
    continue to use Dask's original implementation unchanged.  This avoids
    editing the Conda environment's site-packages and keeps the run
    reproducible from this script.
    """
    from importlib.metadata import version as package_version

    import arboreto.core as arboreto_core
    import dask.dataframe as dask_dataframe

    version_info: Dict[str, Any] = {}
    for package_name in ["arboreto", "dask", "distributed", "pandas"]:
        try:
            version_info[package_name] = package_version(package_name)
        except Exception:
            version_info[package_name] = None

    if bool(getattr(arboreto_core, "_fm_empty_delayed_patch_applied", False)):
        return version_info

    original_from_delayed = arboreto_core.from_delayed

    def arboreto_from_delayed_empty_safe(
        delayed_objects,
        *args,
        **kwargs,
    ):
        is_empty_sequence = isinstance(delayed_objects, (list, tuple)) and len(delayed_objects) == 0
        if not is_empty_sequence:
            return original_from_delayed(delayed_objects, *args, **kwargs)

        explicit_meta = kwargs.get("meta", None)
        if explicit_meta is None and len(args) >= 1:
            explicit_meta = args[0]
        if explicit_meta is None:
            raise TypeError(
                "Arboreto passed zero delayed objects without an explicit meta schema; "
                "the compatibility patch cannot construct a safe empty dataframe."
            )

        if isinstance(explicit_meta, pd.DataFrame):
            empty_meta_dataframe = explicit_meta.iloc[0:0].copy()
        else:
            from dask.dataframe.utils import make_meta

            empty_meta_dataframe = make_meta(explicit_meta).iloc[0:0].copy()

        return dask_dataframe.from_pandas(
            empty_meta_dataframe,
            npartitions=1,
        )

    arboreto_core.from_delayed = arboreto_from_delayed_empty_safe
    arboreto_core._fm_empty_delayed_patch_applied = True

    # Tiny preflight: exercise precisely the empty call that crashes in
    # Arboreto 0.1.6.  This finishes before any per-gene regressions are built.
    empty_meta_result = arboreto_core.from_delayed(
        [],
        meta=arboreto_core._META_SCHEMA,
    ).compute()
    expected_columns = list(arboreto_core._META_SCHEMA.columns)
    if empty_meta_result.shape[0] != 0 or list(empty_meta_result.columns) != expected_columns:
        raise RuntimeError(
            "Arboreto/Dask compatibility preflight returned an unexpected empty metadata dataframe: "
            f"shape={empty_meta_result.shape}, columns={list(empty_meta_result.columns)}"
        )

    log(
        "[compat] activated and validated Arboreto 0.1.6 / modern-Dask "
        f"empty-metadata patch; versions={version_info}"
    )
    return version_info


def _install_pyscenic_numpy_pandas_compatibility_aliases() -> Dict[str, Any]:
    """
    Install only the legacy aliases still used by the released pySCENIC 0.12.1.

    pySCENIC 0.12.1 predates NumPy 1.24 and pandas 2.0.  Its released source
    still references ``np.object``, ``np.float``, and ``Series.iteritems``.
    NumPy 2.x and pandas 2.x removed those aliases.  The corresponding builtin
    types/method are behavior-preserving replacements recommended by NumPy and
    pandas, respectively.

    This helper intentionally patches the imported modules in memory only.  It
    does not edit the Conda environment or site-packages.  It is top-level so
    Dask can serialize and execute it inside every worker process before ctx.
    """
    import numpy as compatibility_numpy
    import pandas as compatibility_pandas

    numpy_alias_targets = {
        "object": object,
        "float": float,
    }
    numpy_aliases_added: List[str] = []
    for alias_name, alias_target in numpy_alias_targets.items():
        # Inspect __dict__ directly.  hasattr(np, "object") itself emits a
        # FutureWarning on NumPy versions where the alias has been removed.
        if alias_name not in compatibility_numpy.__dict__:
            setattr(compatibility_numpy, alias_name, alias_target)
            numpy_aliases_added.append(alias_name)

    pandas_aliases_added: List[str] = []
    if "iteritems" not in compatibility_pandas.Series.__dict__:
        compatibility_pandas.Series.iteritems = compatibility_pandas.Series.items
        pandas_aliases_added.append("Series.iteritems")
    if "iteritems" not in compatibility_pandas.DataFrame.__dict__:
        compatibility_pandas.DataFrame.iteritems = compatibility_pandas.DataFrame.items
        pandas_aliases_added.append("DataFrame.iteritems")

    return {
        "numpy_version": str(compatibility_numpy.__version__),
        "pandas_version": str(compatibility_pandas.__version__),
        "numpy_aliases_added": numpy_aliases_added,
        "pandas_aliases_added": pandas_aliases_added,
    }


def _pyscenic_compatibility_worker_preflight() -> Dict[str, Any]:
    """Apply the compatibility aliases and verify pySCENIC imports in one worker."""
    compatibility_result = _install_pyscenic_numpy_pandas_compatibility_aliases()

    # Import the exact modules needed by motif pruning.  This catches the
    # np.object failure before Dask receives any expensive pruning tasks.
    import pyscenic.prune  # noqa: F401
    import pyscenic.transform  # noqa: F401

    compatibility_result["pyscenic_prune_import"] = "ok"
    compatibility_result["pyscenic_transform_import"] = "ok"
    return compatibility_result


def apply_pyscenic_numpy_pandas_compatibility_patch(
    client_or_address: Any = None,
) -> Dict[str, Any]:
    """
    Make pySCENIC 0.12.1 importable with NumPy 2.x and pandas 2.x.

    The parent process is always patched and preflight-tested.  If a live Dask
    Client is supplied, every already-started worker is patched and tested too;
    modern Dask workers are separate processes and do not necessarily inherit
    runtime changes made in the parent process.
    """
    parent_result = _pyscenic_compatibility_worker_preflight()
    worker_results: Dict[str, Any] = {}

    client_run = getattr(client_or_address, "run", None)
    if callable(client_run):
        worker_results = client_run(_pyscenic_compatibility_worker_preflight)
        bad_worker_results = {
            str(worker_address): worker_result
            for worker_address, worker_result in worker_results.items()
            if worker_result.get("pyscenic_prune_import") != "ok"
            or worker_result.get("pyscenic_transform_import") != "ok"
        }
        if len(bad_worker_results) > 0:
            raise RuntimeError(
                "pySCENIC compatibility preflight failed on one or more Dask workers: "
                f"{bad_worker_results}"
            )

    log(
        "[compat] activated and validated pySCENIC 0.12.1 compatibility for "
        f"NumPy/pandas; parent={parent_result}; dask_workers={len(worker_results)}"
    )
    return {
        "parent": parent_result,
        "workers": worker_results,
    }


def load_ranking_databases(ranking_db_paths: Sequence[str]):
    """Load cisTarget ranking databases for pySCENIC prune2df."""
    try:
        from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
    except Exception:
        from pyscenic.rnkdb import FeatherRankingDatabase as RankingDatabase

    dbs = []
    for db_path in ranking_db_paths:
        db_path = str(db_path)
        check_required_file(db_path, "ranking database .feather")
        dbs.append(RankingDatabase(fname=db_path, name=Path(db_path).stem))
    log(f"[ctx] loaded {len(dbs)} ranking database(s)")
    return dbs


def run_grnboost_or_genie3(
    *,
    expr_df: pd.DataFrame,
    tf_names_path: Optional[str],
    tf_fallback: str,
    method: str,
    seed: int,
    client_or_address: Any,
    arboreto_dask_compatibility_patch: bool,
    out_path: Path,
    resume: bool,
) -> pd.DataFrame:
    """Run Arboreto GRNBoost2 or GENIE3 and save TF-target adjacencies."""
    if resume and out_path.exists():
        log(f"[grn] resume=True and found {out_path}; loading")
        return pd.read_csv(out_path, sep="\t")

    if bool(arboreto_dask_compatibility_patch):
        apply_arboreto_modern_dask_compatibility_patch()
    else:
        log(
            "[compat] WARNING: Arboreto/Dask compatibility patch disabled. "
            "Arboreto 0.1.6 with modern Dask may fail while creating the GRN graph."
        )

    from arboreto.algo import genie3, grnboost2

    if tf_names_path is not None and Path(tf_names_path).exists():
        tf_names_raw = read_one_column_text(tf_names_path)
        tf_names = sorted(set(tf_names_raw).intersection(set(expr_df.columns)))
        log(f"[grn] TF list: raw={len(tf_names_raw)}, intersect_with_expr={len(tf_names)}")
    else:
        if tf_fallback == "all_genes":
            tf_names = list(expr_df.columns)
            log("[grn] WARNING: no TF list supplied; using all expression genes as candidate TFs")
        else:
            raise FileNotFoundError(
                "No valid --tf_names_path supplied. Provide a mouse TF list, or set --tf_fallback all_genes for a test run."
            )

    if len(tf_names) == 0:
        raise RuntimeError(
            "After intersecting TF list with expression genes, zero TFs remain. Check gene symbols and --gene_name_col."
        )

    log(
        f"[grn] running method={method}, cells={expr_df.shape[0]}, genes={expr_df.shape[1]}, TFs={len(tf_names)}"
    )
    t0 = time.time()
    if method == "grnboost2":

        # keep_gene_list = gene_std_series.index[
        #     np.asarray(gene_std_series.to_numpy(dtype=np.float64) > 0.0)
        # ].tolist()

        # expression_df = expression_df.loc[
        #     :,
        #     keep_gene_list,
        # ]

        # tf_names = [
        #     str(tf)
        #     for tf in tf_names
        #     if str(tf) in set(expression_df.columns.astype(str))
        # ]

        # log(
        #     f"[grn] after zero-variance filter: cells={expression_df.shape[0]}, "
        #     f"genes={expression_df.shape[1]}, TFs={len(tf_names)}"
        # )

        # if expression_df.shape[1] < 2:
        #     raise ValueError(
        #         f"Too few genes after zero-variance filtering: {expression_df.shape[1]}"
        #     )

        # if len(tf_names) < 2:
        #     raise ValueError(
        #         f"Too few TFs after intersecting with expression genes: {len(tf_names)}. "
        #         f"Please check whether gene names are MGI symbols, not Ensembl IDs."
        #     )

        adj = grnboost2(
            expression_data=expr_df,
            tf_names=tf_names,
            client_or_address=client_or_address,
            seed=int(seed),
            verbose=True,
        )
    elif method == "genie3":
        adj = genie3(
            expression_data=expr_df,
            tf_names=tf_names,
            client_or_address=client_or_address,
            seed=int(seed),
            verbose=True,
        )
    else:
        raise ValueError("method must be grnboost2 or genie3")

    adj.to_csv(out_path, sep="\t", index=False)
    log(f"[grn] saved adjacencies: {out_path}; shape={adj.shape}; seconds={time.time() - t0:.1f}")
    return adj


def run_ctx_pruning(
    *,
    expr_df: pd.DataFrame,
    adjacencies: pd.DataFrame,
    ranking_db_paths: Sequence[str],
    motif_annotations_path: str,
    client_or_address: Any,
    num_workers: int,
    out_motifs_path: Path,
    out_regulons_path: Path,
    out_modules_path: Path,
    resume: bool,
    pyscenic_numpy_pandas_compatibility_patch: bool,
) -> Tuple[pd.DataFrame, List[Any]]:
    """Create co-expression modules, prune by motif enrichment, and convert to regulons."""
    if resume and out_motifs_path.exists() and out_regulons_path.exists():
        log(f"[ctx] resume=True and found {out_motifs_path} and {out_regulons_path}; loading")
        motifs_df = pd.read_csv(out_motifs_path)
        with out_regulons_path.open("rb") as f:
            regulons = pickle.load(f)
        return motifs_df, regulons

    if bool(pyscenic_numpy_pandas_compatibility_patch):
        apply_pyscenic_numpy_pandas_compatibility_patch(
            client_or_address=client_or_address,
        )
    else:
        log(
            "[compat] WARNING: pySCENIC NumPy/pandas compatibility patch disabled. "
            "pySCENIC 0.12.1 cannot import with NumPy 2.x because it uses np.object."
        )

    # from pyscenic.prune import df2regulons, prune2df

    # ---------------------------------------------------------------------
    # Compatibility patch for pySCENIC prune2df + newer Dask.
    #
    # Problem:
    #   pySCENIC prune2df internally gives a generator to dask.dataframe.from_delayed.
    #   Newer Dask versions call len(dfs), but a Python generator has no len().
    #
    # Symptom:
    #   TypeError: object of type 'generator' has no len()
    #
    # Fix:
    #   Patch pyscenic.prune.from_delayed so that, whenever pySCENIC passes a
    #   generator, we first materialize it as a list.
    # ---------------------------------------------------------------------
    import dask.dataframe as dask_dataframe
    import pyscenic.prune as pyscenic_prune_module

    _dask_from_delayed_original = dask_dataframe.from_delayed

    def _from_delayed_generator_safe(
        dfs,
        *args,
        **kwargs,
    ):
        if not isinstance(dfs, list):
            dfs = list(dfs)

        return _dask_from_delayed_original(
            dfs,
            *args,
            **kwargs,
        )

    pyscenic_prune_module.from_delayed = _from_delayed_generator_safe

    from pyscenic.prune import df2regulons, prune2df

    from pyscenic.utils import modules_from_adjacencies

    check_required_file(motif_annotations_path, "motif annotations .tbl")
    dbs = load_ranking_databases(ranking_db_paths)

    log("[ctx] building modules from adjacencies")
    modules = list(modules_from_adjacencies(adjacencies, expr_df))
    with out_modules_path.open("wb") as f:
        pickle.dump(modules, f)
    log(f"[ctx] saved modules: {out_modules_path}; n_modules={len(modules)}")

    log("[ctx] running motif pruning with prune2df")
    t0 = time.time()
    try:
        motifs_df = prune2df(
            dbs,
            modules,
            motif_annotations_path,
            client_or_address=client_or_address,
            num_workers=int(num_workers),
        )
    except TypeError:
        # Some pySCENIC versions do not accept num_workers when a client is given.
        motifs_df = prune2df(
            dbs,
            modules,
            motif_annotations_path,
            client_or_address=client_or_address,
        )

    # motifs_df.to_csv(out_motifs_path, index=False)
    # log(f"[ctx] saved motifs/regulon table: {out_motifs_path}; shape={motifs_df.shape}; seconds={time.time() - t0:.1f}")

    # regulons = df2regulons(motifs_df)
    # with out_regulons_path.open("wb") as f:
    #     pickle.dump(regulons, f)
    # log(f"[ctx] saved regulons pickle: {out_regulons_path}; n_regulons={len(regulons)}")
    # return motifs_df, regulons

    # Save index=True because pySCENIC often stores TF/regulon identity in the index.
    motifs_df.to_csv(
        out_motifs_path,
        index=True,
    )

    log(
        f"[ctx] saved motifs/regulon table: {out_motifs_path}; "
        f"shape={motifs_df.shape}; seconds={time.time() - t0:.1f}"
    )

    # ------------------------------------------------------------
    # Parse motif-pruned TargetGenes directly. Some pySCENIC/ctxcore
    # combinations otherwise split serialized target strings into characters.
    # ------------------------------------------------------------
    candidate_tf_names = (
        set(adjacencies["TF"].astype(str).tolist()) if "TF" in adjacencies.columns else None
    )

    valid_gene_names = set(expr_df.columns.astype(str).tolist())

    reg_edges_from_motifs = motifs_df_to_regulon_edges_safe(
        motifs_df=motifs_df,
        candidate_tf_names=candidate_tf_names,
        valid_gene_names=valid_gene_names,
    )

    reg_edges_debug_path = out_regulons_path.with_name(
        "regulon_pruned_tf_target_edges_FROM_MOTIFS.tsv"
    )

    reg_edges_from_motifs.to_csv(
        reg_edges_debug_path,
        sep="\t",
        index=False,
    )

    log(
        f"[ctx] saved clean regulon edges parsed from motifs_df: "
        f"{reg_edges_debug_path}; shape={reg_edges_from_motifs.shape}"
    )

    if reg_edges_from_motifs.shape[0] == 0:
        raise RuntimeError(
            "Parsed zero clean regulon edges from motifs_df. "
            "Please inspect motifs.csv and check the TargetGenes column name/format."
        )

    bad_examples = _get_character_split_examples(
        edge_df=reg_edges_from_motifs,
        valid_gene_names=valid_gene_names,
        n=20,
    )

    if bad_examples.shape[0] > 0:
        raise RuntimeError(
            "The motif-parsed regulon edge table still contains character-split artifacts.\n"
            f"Examples:\n{bad_examples.to_string(index=False)}"
        )

    regulons = regulon_edges_to_clean_regulons(
        edge_df=reg_edges_from_motifs,
        valid_gene_names=valid_gene_names,
    )

    with out_regulons_path.open("wb") as f:
        pickle.dump(
            regulons,
            f,
        )

    log(f"[ctx] saved CLEAN regulons pickle: {out_regulons_path}; " f"n_regulons={len(regulons)}")

    return motifs_df, regulons


def run_aucell(
    *,
    expr_df: pd.DataFrame,
    regulons: List[Any],
    num_workers: int,
    out_auc_path: Path,
    resume: bool,
) -> pd.DataFrame:
    """Run AUCell regulon activity scoring."""
    if resume and out_auc_path.exists():
        log(f"[aucell] resume=True and found {out_auc_path}; loading")
        return pd.read_csv(out_auc_path, index_col=0)

    from pyscenic.aucell import aucell

    log(f"[aucell] running AUCell: cells={expr_df.shape[0]}, regulons={len(regulons)}")
    t0 = time.time()
    try:
        auc_mtx = aucell(expr_df, regulons, num_workers=int(num_workers))
    except TypeError:
        auc_mtx = aucell(expr_df, regulons)

    auc_mtx.to_csv(out_auc_path)
    log(
        f"[aucell] saved AUC matrix: {out_auc_path}; shape={auc_mtx.shape}; seconds={time.time() - t0:.1f}"
    )
    return auc_mtx


# =============================================================================
# Export model-aligned GRN resources
# =============================================================================


def regulon_name_to_tf(regulon: Any) -> str:
    """Robustly infer TF name from a pySCENIC regulon object."""
    tf = getattr(regulon, "transcription_factor", None)
    if tf is not None:
        return str(tf)
    name = str(getattr(regulon, "name", ""))
    if "(" in name:
        return name.split("(")[0]
    if "_" in name:
        return name.split("_")[0]
    return name


class SimpleGeneSignature:
    """
    Minimal AUCell-compatible gene signature.

    This is only a fallback. The code first tries to build official
    ctxcore/pyscenic GeneSignature or Regulon objects.
    """

    def __init__(
        self,
        name: str,
        gene2weight: Dict[str, float],
        transcription_factor: Optional[str] = None,
    ):
        self.name = str(name)
        self.transcription_factor = transcription_factor
        self.gene2weight = {str(gene): float(weight) for gene, weight in gene2weight.items()}
        self.genes = frozenset(self.gene2weight.keys())


def _flatten_column_name(
    column_name: Any,
) -> str:
    """Convert normal or MultiIndex column names into searchable strings."""
    if isinstance(column_name, tuple):
        return "__".join([str(x) for x in column_name if str(x) != ""])

    return str(column_name)


def _find_column_by_keywords(
    *,
    df: pd.DataFrame,
    required_keywords: Sequence[str],
) -> Optional[Any]:
    """Find a dataframe column whose flattened name contains all keywords."""
    required_keywords_lower = [str(keyword).lower() for keyword in required_keywords]

    for column_name in df.columns:
        flat_name = _flatten_column_name(
            column_name=column_name,
        ).lower()

        if all(keyword in flat_name for keyword in required_keywords_lower):
            return column_name

    return None


def _extract_tf_from_any_name(
    *,
    value: Any,
    candidate_tf_names: Optional[set],
) -> Optional[str]:
    """
    Extract TF name from row index, regulon name, or text.

    Examples:
        Alx1(+) -> Alx1
        ('Alx1', 'motif_x') -> Alx1
    """
    if value is None:
        return None

    if isinstance(value, tuple):
        for item in value:
            tf = _extract_tf_from_any_name(
                value=item,
                candidate_tf_names=candidate_tf_names,
            )
            if tf is not None:
                return tf
        return None

    text = str(value).strip()

    if text == "":
        return None

    if text.startswith("Regulon for "):
        text = text.replace("Regulon for ", "", 1).strip()

    if "(" in text:
        text = text.split("(", 1)[0].strip()

    if "__" in text:
        text = text.split("__", 1)[0].strip()

    if candidate_tf_names is not None and text not in candidate_tf_names:
        return None

    return text


def _regex_recover_valid_genes_from_text(
    *,
    text: str,
    valid_gene_names: Optional[set],
) -> List[str]:
    """
    Last-resort recovery for unparseable TargetGenes strings.

    It never splits into characters. It only keeps tokens that exactly match
    expression gene names.
    """
    if valid_gene_names is None:
        return []

    import re

    candidate_tokens = re.split(
        pattern=r"[^A-Za-z0-9_.-]+",
        string=str(text),
    )

    out: List[str] = []

    for token in candidate_tokens:
        token = token.strip()
        if token in valid_gene_names:
            out.append(token)

    return out


def _parse_target_gene_pairs_from_value(
    *,
    value: Any,
    valid_gene_names: Optional[set],
) -> List[Tuple[str, float]]:
    """
    Parse pySCENIC TargetGenes into [(gene, weight), ...].

    This function is the key protection:
    it never calls list(...) on an unparsed string.
    """
    value = _literal_eval_if_possible(
        value=value,
    )

    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()

        if text == "":
            return []

        parsed_again = _literal_eval_if_possible(
            value=text,
        )

        if not isinstance(parsed_again, str):
            return _parse_target_gene_pairs_from_value(
                value=parsed_again,
                valid_gene_names=valid_gene_names,
            )

        gene = _clean_gene_name(
            value=text,
        )

        if valid_gene_names is not None and gene in valid_gene_names:
            return [
                (
                    gene,
                    1.0,
                )
            ]

        recovered_gene_list = _regex_recover_valid_genes_from_text(
            text=text,
            valid_gene_names=valid_gene_names,
        )

        return [
            (
                gene_name,
                1.0,
            )
            for gene_name in recovered_gene_list
        ]

    if isinstance(value, Mapping):
        out: List[Tuple[str, float]] = []

        for gene_raw, weight_raw in value.items():
            gene = _clean_gene_name(
                value=gene_raw,
            )

            if valid_gene_names is not None and gene not in valid_gene_names:
                continue

            out.append(
                (
                    gene,
                    _safe_float(
                        value=weight_raw,
                        default=1.0,
                    ),
                )
            )

        return out

    try:
        value_list = list(value)
    except TypeError:
        gene = _clean_gene_name(
            value=value,
        )

        if valid_gene_names is not None and gene not in valid_gene_names:
            return []

        return [
            (
                gene,
                1.0,
            )
        ]

    out: List[Tuple[str, float]] = []

    for element in value_list:
        element = _literal_eval_if_possible(
            value=element,
        )

        if isinstance(element, (tuple, list)) and len(element) >= 2:
            gene_raw = element[0]
            weight_raw = element[1]
        else:
            gene_raw = element
            weight_raw = 1.0

        gene = _clean_gene_name(
            value=gene_raw,
        )

        if valid_gene_names is not None and gene not in valid_gene_names:
            continue

        if _is_probably_character_split_artifact(
            target=gene,
        ):
            continue

        out.append(
            (
                gene,
                _safe_float(
                    value=weight_raw,
                    default=1.0,
                ),
            )
        )

    return out


def motifs_df_to_regulon_edges_safe(
    *,
    motifs_df: pd.DataFrame,
    candidate_tf_names: Optional[set],
    valid_gene_names: Optional[set],
) -> pd.DataFrame:
    """
    Build clean motif-pruned TF-target edges directly from motifs_df.

    This bypasses df2regulons(...) completely.
    """
    target_genes_column = _find_column_by_keywords(
        df=motifs_df,
        required_keywords=[
            "target",
            "gene",
        ],
    )

    if target_genes_column is None:
        raise KeyError(
            "Could not find TargetGenes column in motifs_df. "
            f"Available columns: {[str(c) for c in motifs_df.columns]}"
        )

    regulon_name_column = _find_column_by_keywords(
        df=motifs_df,
        required_keywords=[
            "regulon",
            "name",
        ],
    )

    tf_column = _find_column_by_keywords(
        df=motifs_df,
        required_keywords=[
            "tf",
        ],
    )

    rows: List[Dict[str, Any]] = []

    for row_index, row in motifs_df.iterrows():

        tf: Optional[str] = None

        if tf_column is not None:
            tf = _extract_tf_from_any_name(
                value=row[tf_column],
                candidate_tf_names=candidate_tf_names,
            )

        if tf is None and regulon_name_column is not None:
            tf = _extract_tf_from_any_name(
                value=row[regulon_name_column],
                candidate_tf_names=candidate_tf_names,
            )

        if tf is None:
            tf = _extract_tf_from_any_name(
                value=row_index,
                candidate_tf_names=candidate_tf_names,
            )

        if tf is None:
            continue

        regulon_name = f"{tf}(+)"

        if regulon_name_column is not None:
            raw_regulon_name = str(row[regulon_name_column]).strip()
            if raw_regulon_name not in {"", "nan", "None"}:
                regulon_name = raw_regulon_name
                if "(" not in regulon_name:
                    regulon_name = f"{tf}(+)"

        target_pair_list = _parse_target_gene_pairs_from_value(
            value=row[target_genes_column],
            valid_gene_names=valid_gene_names,
        )

        for target, weight in target_pair_list:
            if valid_gene_names is not None and target not in valid_gene_names:
                continue

            rows.append(
                {
                    "TF": str(tf),
                    "target": str(target),
                    "regulon": str(regulon_name),
                    "regulon_weight": float(weight),
                }
            )

    edge_df = pd.DataFrame(
        rows,
        columns=[
            "TF",
            "target",
            "regulon",
            "regulon_weight",
        ],
    )

    if edge_df.shape[0] > 0:
        edge_df = (
            edge_df.sort_values(
                by=[
                    "TF",
                    "regulon",
                    "target",
                ],
                ascending=[
                    True,
                    True,
                    True,
                ],
            )
            .drop_duplicates(
                subset=[
                    "TF",
                    "target",
                    "regulon",
                ],
                keep="first",
            )
            .reset_index(
                drop=True,
            )
        )

    log(
        "[ctx_fix] parsed clean regulon edges from motifs_df: "
        f"target_column={_flatten_column_name(target_genes_column)!r}, "
        f"n_edges={edge_df.shape[0]}, "
        f"n_regulons={edge_df['regulon'].nunique() if edge_df.shape[0] > 0 else 0}"
    )

    if edge_df.shape[0] > 0:
        log(
            "[ctx_fix] first 10 clean motif-pruned edges:\n"
            + edge_df.head(10).to_string(index=False)
        )

    return edge_df


def _make_official_or_simple_gene_signature(
    *,
    name: str,
    gene2weight: Dict[str, float],
    transcription_factor: Optional[str],
) -> Any:
    """
    Try official ctxcore/pyscenic GeneSignature/Regulon first.
    If all constructors fail, use SimpleGeneSignature.
    """
    class_candidates: List[Any] = []

    for module_name, class_name in [
        ("ctxcore.genesig", "Regulon"),
        ("ctxcore.genesig", "GeneSignature"),
        ("pyscenic.genesig", "Regulon"),
        ("pyscenic.genesig", "GeneSignature"),
    ]:
        try:
            module = __import__(
                module_name,
                fromlist=[
                    class_name,
                ],
            )
            class_candidates.append(
                getattr(
                    module,
                    class_name,
                )
            )
        except Exception:
            pass

    gene2weight_dict = {str(gene): float(weight) for gene, weight in gene2weight.items()}

    gene2weight_pairs = tuple(gene2weight_dict.items())

    for cls in class_candidates:

        try:
            signature = inspect.signature(
                cls,
            )
            parameter_names = set(signature.parameters.keys())

            kwargs: Dict[str, Any] = {}

            if "name" in parameter_names:
                kwargs["name"] = str(name)

            if "gene2weight" in parameter_names:
                kwargs["gene2weight"] = gene2weight_pairs

            if "gene2occurrence" in parameter_names:
                kwargs["gene2occurrence"] = []

            if "transcription_factor" in parameter_names:
                kwargs["transcription_factor"] = transcription_factor

            if "context" in parameter_names:
                kwargs["context"] = frozenset()

            if "score" in parameter_names:
                kwargs["score"] = 0.0

            if "name" in kwargs and "gene2weight" in kwargs:
                return cls(
                    **kwargs,
                )

        except Exception:
            pass

        try:
            return cls(
                str(name),
                gene2weight_pairs,
            )
        except Exception:
            pass

        try:
            return cls(
                str(name),
                gene2weight_dict,
            )
        except Exception:
            pass

    return SimpleGeneSignature(
        name=str(name),
        gene2weight=gene2weight_dict,
        transcription_factor=transcription_factor,
    )


def regulon_edges_to_clean_regulons(
    *,
    edge_df: pd.DataFrame,
    valid_gene_names: set,
) -> List[Any]:
    """Convert clean regulon edge table into AUCell-compatible regulons."""
    regulons: List[Any] = []

    if edge_df is None or edge_df.shape[0] == 0:
        return regulons

    for regulon_name, sub_df in edge_df.groupby(
        "regulon",
        sort=True,
    ):

        tf = str(sub_df["TF"].iloc[0])

        gene2weight: Dict[str, float] = {}

        for _, row in sub_df.iterrows():

            gene = str(row["target"])

            if gene not in valid_gene_names:
                continue

            weight = _safe_float(
                value=row["regulon_weight"],
                default=1.0,
            )

            if gene not in gene2weight:
                gene2weight[gene] = weight
            else:
                gene2weight[gene] = max(
                    gene2weight[gene],
                    weight,
                )

        if len(gene2weight) == 0:
            continue

        regulon = _make_official_or_simple_gene_signature(
            name=str(regulon_name),
            gene2weight=gene2weight,
            transcription_factor=tf,
        )

        regulons.append(
            regulon,
        )

    log(f"[ctx_fix] built AUCell-compatible clean regulons: n={len(regulons)}")

    return regulons


def _get_character_split_examples(
    *,
    edge_df: pd.DataFrame,
    valid_gene_names: Optional[set],
    n: int,
) -> pd.DataFrame:
    """Return suspicious edge rows where targets look character-split."""
    if edge_df is None or edge_df.shape[0] == 0:
        return pd.DataFrame()

    target_series = edge_df["target"].astype(str)

    bad_mask = target_series.isin(
        [
            "[",
            "]",
            "(",
            ")",
            "{",
            "}",
            "'",
            '"',
            ",",
            ":",
            ".",
        ]
    )

    one_char_mask = target_series.str.len() <= 1

    if valid_gene_names is not None:
        not_valid_gene_mask = ~target_series.isin(
            valid_gene_names,
        )
        bad_mask = bad_mask | (one_char_mask & not_valid_gene_mask)
    else:
        bad_mask = bad_mask | one_char_mask

    return edge_df.loc[
        bad_mask,
        [
            "TF",
            "target",
            "regulon",
            "regulon_weight",
        ],
    ].head(int(n))


def _edge_table_has_character_split_artifact(
    *,
    edge_df: pd.DataFrame,
    valid_gene_names: Optional[set],
) -> bool:
    """Check whether an edge table still has S/i/x/2/[/( target artifacts."""
    examples = _get_character_split_examples(
        edge_df=edge_df,
        valid_gene_names=valid_gene_names,
        n=1,
    )

    return examples.shape[0] > 0


def _literal_eval_if_possible(value: Any) -> Any:
    """
    Safely parse a string representation of a Python object.

    Why this is needed
    ------------------
    In some pySCENIC / ctxcore versions, regulon genes may appear as:
        "['Sox4', 'Sox11']"
        "('Sox4', 'Sox11')"
        "{'Sox4', 'Sox11'}"
        "frozenset({'Sox4', 'Sox11'})"

    If we directly call list(...) on such a string, Python splits the string
    into characters, which creates the bad target values:
        [
        (
        '
        S
        ...

    This helper tries to convert those string representations back to real
    Python containers before we iterate over them.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()

    if text == "":
        return value

    # Handle strings like: frozenset({'Sox4', 'Sox11'})
    if text.startswith("frozenset(") and text.endswith(")"):
        inner_text = text[len("frozenset(") : -1].strip()
        return _literal_eval_if_possible(inner_text)

    # Handle normal list/tuple/set/dict string representations.
    if (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("(") and text.endswith(")"))
        or (text.startswith("{") and text.endswith("}"))
    ):
        try:
            return ast.literal_eval(text)
        except Exception:
            return value

    return value


def _clean_gene_name(value: Any) -> str:
    """
    Convert one gene-like object into a clean gene-name string.

    This function also handles accidental tuple/list values such as:
        ('Sox4', 1.0)

    In that case, the first element is treated as the gene name.
    """
    value = _literal_eval_if_possible(value)

    # Sometimes an element may be a tuple like ('Sox4', 0.52).
    # In that case, the first element is the gene name.
    if isinstance(value, (tuple, list)) and len(value) > 0:
        value = value[0]

    gene_name = str(value).strip()

    # Remove simple wrapping quotes if present.
    if len(gene_name) >= 2:
        if (gene_name[0] == "'" and gene_name[-1] == "'") or (
            gene_name[0] == '"' and gene_name[-1] == '"'
        ):
            gene_name = gene_name[1:-1].strip()

    return gene_name


def _safe_float(value: Any, default: float = 1.0) -> float:
    """
    Convert a regulon weight into float safely.

    If conversion fails or the value is NaN/Inf, return default.
    """
    try:
        out = float(value)
    except Exception:
        return float(default)

    if not np.isfinite(out):
        return float(default)

    return float(out)


def _mapping_items_if_possible(value: Any) -> Optional[List[Tuple[Any, Any]]]:
    """
    Try to interpret value as a gene -> weight mapping.

    Why Mapping instead of dict?
    ----------------------------
    pySCENIC / ctxcore may store gene2weight as a special mapping object,
    not necessarily a pure Python dict. Therefore, checking only:

        isinstance(gene2weight, dict)

    is too strict.

    We accept:
        - dict
        - Mapping-like objects
        - objects with .items()
        - string representations of dicts
        - list/tuple of (gene, weight) pairs
    """
    value = _literal_eval_if_possible(value)

    if value is None:
        return None

    if isinstance(value, Mapping):
        return list(value.items())

    if hasattr(value, "items"):
        try:
            return list(value.items())
        except Exception:
            pass

    if isinstance(value, (list, tuple, set)):
        candidate_items: List[Tuple[Any, Any]] = []

        for element in value:
            element = _literal_eval_if_possible(element)

            if isinstance(element, (list, tuple)) and len(element) >= 2:
                candidate_items.append((element[0], element[1]))

        if len(candidate_items) > 0:
            return candidate_items

    return None


def _gene_list_if_possible(value: Any) -> List[str]:
    """
    Convert a regulon gene container into a list of gene names.

    Preserve complete gene symbols when containers are serialized as strings.

    If regulon.genes is a string like "['Sox4', 'Sox11']", then list(...) gives:
        ['[', "'", 'S', 'o', 'x', '4', "'", ',', ...]

    Correct behavior:
        first parse the string into a real list, then iterate over the list.
    """
    value = _literal_eval_if_possible(value)

    if value is None:
        return []

    if isinstance(value, Mapping):
        return [_clean_gene_name(gene) for gene in value.keys()]

    # Important:
    # A plain string is one gene name, not an iterable of characters.
    if isinstance(value, str):
        gene_name = _clean_gene_name(value)
        if gene_name in {"", "[]", "()", "{}", "set()", "frozenset()"}:
            return []
        return [gene_name]

    try:
        return [_clean_gene_name(gene) for gene in list(value)]
    except TypeError:
        return [_clean_gene_name(value)]


#     We do not want to reject all one-character strings blindly because in rare
#     cases a gene symbol could be short. But punctuation-only targets are always bad.
#     """
#     target = str(target).strip()

#     if target in {"[", "]", "(", ")", "{", "}", "'", '"', ",", ":"}:
#         return True

#     return False


def _is_probably_character_split_artifact(target: str) -> bool:
    """
    Detect character-split artifacts.

    Your bad output includes:
        S, i, x, 2, 3, ., T, f, ...

    Therefore this function now flags single-character targets too.
    """
    target = str(target).strip()

    if target in {"[", "]", "(", ")", "{", "}", "'", '"', ",", ":", "."}:
        return True

    if len(target) <= 1:
        return True

    return False


def regulons_to_target_edges(regulons: List[Any]) -> pd.DataFrame:
    """
    Convert pySCENIC regulons to a simple TF-target edge table.

    Output columns
    --------------
    TF:
        Transcription factor name, such as Sox4.
    target:
        Target gene name, such as Sox11.
    regulon:
        Regulon name, such as Sox4(+).
    regulon_weight:
        Target weight inside the regulon. If no weight is available, use 1.0.

    Main fix
    --------
    This version does NOT call list(...) directly on regulon.genes.
    It first checks gene2weight as a general Mapping, and if it has to fall back
    to regulon.genes, it safely parses string/list/set containers.
    """
    rows: List[Dict[str, Any]] = []

    n_regulons_total = 0
    n_regulons_from_gene2weight = 0
    n_regulons_from_genes = 0
    n_bad_character_targets = 0

    for regulon in regulons:
        n_regulons_total += 1

        tf = regulon_name_to_tf(regulon)
        reg_name = str(getattr(regulon, "name", tf))

        # ------------------------------------------------------------
        # 1. Preferred route:
        #    use regulon.gene2weight when available.
        #
        # Important:
        # We accept any Mapping-like object, not only dict.
        # ------------------------------------------------------------
        gene2weight = getattr(regulon, "gene2weight", None)
        mapping_items = _mapping_items_if_possible(gene2weight)

        if mapping_items is not None and len(mapping_items) > 0:
            n_regulons_from_gene2weight += 1

            for target_raw, weight_raw in mapping_items:
                target = _clean_gene_name(target_raw)

                if target == "":
                    continue

                if _is_probably_character_split_artifact(target):
                    n_bad_character_targets += 1
                    continue

                rows.append(
                    {
                        "TF": str(tf),
                        "target": str(target),
                        "regulon": reg_name,
                        "regulon_weight": _safe_float(weight_raw, default=1.0),
                    }
                )

            continue

        # ------------------------------------------------------------
        # 2. Fallback route:
        #    use regulon.genes, but parse safely.
        # ------------------------------------------------------------
        genes_raw = getattr(regulon, "genes", [])
        genes = _gene_list_if_possible(genes_raw)

        if len(genes) > 0:
            n_regulons_from_genes += 1

        for target in genes:
            target = _clean_gene_name(target)

            if target == "":
                continue

            if _is_probably_character_split_artifact(target):
                n_bad_character_targets += 1
                continue

            rows.append(
                {
                    "TF": str(tf),
                    "target": str(target),
                    "regulon": reg_name,
                    "regulon_weight": 1.0,
                }
            )

    edge_df = pd.DataFrame(
        rows,
        columns=[
            "TF",
            "target",
            "regulon",
            "regulon_weight",
        ],
    )

    # Remove duplicated TF-target-regulon rows.
    if edge_df.shape[0] > 0:
        edge_df = edge_df.drop_duplicates(
            subset=[
                "TF",
                "target",
                "regulon",
            ],
            keep="first",
        ).reset_index(drop=True)

    log(
        "[regulon_export] converted regulons to edges: "
        f"n_regulons_total={n_regulons_total}, "
        f"n_regulons_from_gene2weight={n_regulons_from_gene2weight}, "
        f"n_regulons_from_genes={n_regulons_from_genes}, "
        f"n_edges={edge_df.shape[0]}, "
        f"n_bad_character_targets_skipped={n_bad_character_targets}"
    )

    if edge_df.shape[0] > 0:
        log("[regulon_export] first 10 regulon edges:\n" + edge_df.head(10).to_string(index=False))

    return edge_df


def export_sparse_prior_matrix(
    *,
    edge_df: pd.DataFrame,
    gene_names: Sequence[str],
    out_raw_path: Path,
    out_row_norm_path: Path,
    out_col_norm_path: Path,
    out_metadata_path: Path,
    tf_col: str = "TF",
    target_col: str = "target",
    weight_col: str = "importance",
) -> None:
    """
    Save sparse GRN prior matrices.

    Matrix convention for FM integration:
        rows    = target genes affected by regulators
        columns = TF/regulator genes
        value   = edge weight

    If the expression vector x has shape [B, G], then a simple graph feature can be:
        graph_signal = x @ A.T
    where A[target, regulator] stores regulator -> target influence.
    """
    gene_names = list(map(str, gene_names))
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    if edge_df is None or edge_df.shape[0] == 0:
        log("[export] WARNING: empty edge table; writing empty sparse matrices")
    else:
        for rec in edge_df[[tf_col, target_col, weight_col]].itertuples(index=False, name=None):
            tf, target, weight = rec
            tf = str(tf)
            target = str(target)
            if tf not in gene_to_idx or target not in gene_to_idx:
                continue
            try:
                w = float(weight)
            except Exception:
                w = 1.0
            if not np.isfinite(w):
                continue
            rows.append(gene_to_idx[target])
            cols.append(gene_to_idx[tf])
            data.append(max(0.0, w))

    G = len(gene_names)
    A = sp.coo_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(G, G)).tocsr()
    A.sum_duplicates()

    # Row-normalized: each target receives a normalized mixture of regulators.
    row_sum = np.asarray(A.sum(axis=1)).reshape(-1)
    inv_row = np.zeros_like(row_sum, dtype=np.float32)
    inv_row[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    A_row = sp.diags(inv_row).dot(A).tocsr()

    # Column-normalized: each TF distributes its outgoing importance across targets.
    col_sum = np.asarray(A.sum(axis=0)).reshape(-1)
    inv_col = np.zeros_like(col_sum, dtype=np.float32)
    inv_col[col_sum > 0] = 1.0 / col_sum[col_sum > 0]
    A_col = A.dot(sp.diags(inv_col)).tocsr()

    sp.save_npz(out_raw_path, A)
    sp.save_npz(out_row_norm_path, A_row)
    sp.save_npz(out_col_norm_path, A_col)

    metadata = {
        "matrix_shape": [int(G), int(G)],
        "matrix_convention": "A[target_gene_index, regulator_TF_gene_index] = edge_weight",
        "fm_usage_example": "graph_signal = x @ A.T when x is [batch, genes]",
        "n_edges_nonzero": int(A.nnz),
        "gene_names_ordered": gene_names,
        "raw_matrix": str(out_raw_path),
        "row_normalized_matrix": str(out_row_norm_path),
        "column_normalized_matrix": str(out_col_norm_path),
    }
    save_json(metadata, out_metadata_path)
    log(f"[export] saved sparse prior matrices: nnz={A.nnz}, G={G}")


# def aggregate_auc_by_metadata(
#     *,
#     auc_mtx: pd.DataFrame,
#     selected_cells: np.ndarray,
#     obs: pd.DataFrame,
#     out_dir: Path,
#     time_col: str,
#     celltype_col: str,
# ) -> None:
#     """Save average regulon activity by time and by time/celltype."""
#     meta = obs.iloc[selected_cells].copy()
#     meta.index = auc_mtx.index

#     if time_col in meta.columns:
#         by_time = auc_mtx.groupby(meta[time_col], observed=True).mean()
#         by_time.to_csv(out_dir / "auc_mean_by_time.csv")
#         log(f"[export] saved AUC mean by time: {out_dir / 'auc_mean_by_time.csv'}")

#     if time_col in meta.columns and celltype_col in meta.columns:
#         group_key = pd.MultiIndex.from_frame(meta[[time_col, celltype_col]].astype(str))
#         by_time_type = auc_mtx.groupby(group_key, observed=True).mean()
#         by_time_type.index.names = [time_col, celltype_col]
#         by_time_type.to_csv(out_dir / "auc_mean_by_time_celltype.csv")
#         log(f"[export] saved AUC mean by time/celltype: {out_dir / 'auc_mean_by_time_celltype.csv'}")


def aggregate_auc_by_metadata(
    *,
    auc_mtx: pd.DataFrame,
    selected_cells: np.ndarray,
    obs: pd.DataFrame,
    out_dir: Path,
    time_col: str,
    celltype_col: str,
) -> None:
    """
    Save average regulon activity by:
        1. time
        2. time + celltype

    Why this safer version is needed
    --------------------------------
    The old version used:

        group_key = pd.MultiIndex.from_frame(...)
        by_time_type = auc_mtx.groupby(group_key, observed=True).mean()
        by_time_type.index.names = [time_col, celltype_col]

    In some pandas versions, grouping by an external MultiIndex can return
    an index that is treated as one-level rather than a true two-level
    MultiIndex. Then assigning two names causes:

        ValueError: Length of new names must be 1, got 2

    This version avoids that problem by:
        1. explicitly concatenating metadata columns with AUC columns
        2. grouping by real dataframe columns
        3. checking whether the result index is actually a MultiIndex
    """

    # ------------------------------------------------------------
    # 1. Safety check for selected cell indices
    # ------------------------------------------------------------
    selected_cells = np.asarray(
        a=selected_cells,
        dtype=np.int64,
    )

    if selected_cells.ndim != 1:
        raise ValueError(
            "`selected_cells` should be a 1-dimensional array of AnnData/obs row indices, "
            f"but got shape={selected_cells.shape}."
        )

    if selected_cells.shape[0] != auc_mtx.shape[0]:
        raise ValueError(
            "The number of selected cells must match the number of rows in auc_mtx.\n"
            f"selected_cells.shape[0] = {selected_cells.shape[0]}\n"
            f"auc_mtx.shape[0]       = {auc_mtx.shape[0]}"
        )

    # ------------------------------------------------------------
    # 2. Extract metadata for exactly the cells used in AUCell
    # ------------------------------------------------------------
    meta = obs.iloc[selected_cells].copy()

    # Make metadata index exactly match AUC matrix index.
    # This makes concat/groupby alignment safe.
    meta.index = auc_mtx.index

    # ------------------------------------------------------------
    # 3. Identify regulon columns
    # ------------------------------------------------------------
    regulon_column_list = list(auc_mtx.columns)

    if len(regulon_column_list) == 0:
        raise ValueError("`auc_mtx` has zero regulon columns. Cannot aggregate AUC values.")

    # ------------------------------------------------------------
    # 4. Build one combined dataframe:
    #       metadata columns + regulon AUC columns
    # ------------------------------------------------------------
    combined_df = pd.concat(
        objs=[
            meta,
            auc_mtx,
        ],
        axis=1,
    )

    # ------------------------------------------------------------
    # 5. Aggregate by time
    # ------------------------------------------------------------
    if time_col in combined_df.columns:

        by_time = combined_df.groupby(
            by=time_col,
            observed=True,
            sort=True,
        )[regulon_column_list].mean()

        by_time.index.name = time_col

        by_time_file_path = out_dir / "auc_mean_by_time.csv"

        by_time.to_csv(
            path_or_buf=by_time_file_path,
            index=True,
        )

        log(f"[export] saved AUC mean by time: {by_time_file_path}")

    else:

        log(
            f"[export][skip] Cannot save AUC mean by time because "
            f"time_col={time_col!r} is not in obs/meta columns."
        )

    # ------------------------------------------------------------
    # 6. Aggregate by time + celltype
    # ------------------------------------------------------------
    if time_col in combined_df.columns and celltype_col in combined_df.columns:

        # Important:
        # We group by real dataframe columns instead of passing an external
        # pd.MultiIndex. This is the main fix.
        by_time_type = combined_df.groupby(
            by=[
                time_col,
                celltype_col,
            ],
            observed=True,
            sort=True,
        )[regulon_column_list].mean()

        # In normal pandas behavior, this should be a MultiIndex.
        # But we check it anyway to avoid future pandas-version surprises.
        if isinstance(by_time_type.index, pd.MultiIndex):

            by_time_type.index.names = [
                time_col,
                celltype_col,
            ]

        else:

            # This should rarely happen, but if it does, do not crash.
            # Just give the one-level index a safe name.
            by_time_type.index.name = f"{time_col}_{celltype_col}"

            log(
                "[export][warning] Expected by_time_type.index to be a MultiIndex, "
                f"but got {type(by_time_type.index)}. "
                "Saved the table anyway with a one-level index name."
            )

        by_time_type_file_path = out_dir / "auc_mean_by_time_celltype.csv"

        by_time_type.to_csv(
            path_or_buf=by_time_type_file_path,
            index=True,
        )

        log(f"[export] saved AUC mean by time/celltype: {by_time_type_file_path}")

    else:

        missing_column_list = [
            column_name
            for column_name in [
                time_col,
                celltype_col,
            ]
            if column_name not in combined_df.columns
        ]

        log(
            f"[export][skip] Cannot save AUC mean by time/celltype because "
            f"these columns are missing from obs/meta: {missing_column_list}"
        )


def export_model_target_by_all_tf_prior(
    *,
    edge_df: pd.DataFrame,
    model_gene_names: Sequence[str],
    out_prefix: Path,
    tf_col: str = "TF",
    target_col: str = "target",
    weight_col: str = "importance",
) -> Dict[str, Any]:
    """
    Export a biologically useful bipartite prior B[target_model_gene, TF].

    Unlike the direct G x G prior, this representation retains regulators that
    are NOT part of the downstream FM HVG feature set. This is important when GRN
    inference is run on HVG2000 + all TFs but FM predicts only HVG200.

    It also exports a model-gene co-regulation prior C = B B^T, which transfers
    information from outside-model TFs into a G_model x G_model matrix that can
    be used directly by the current FM without expanding its expression input.
    """
    model_gene_names = [str(x) for x in model_gene_names]
    target_to_idx = {g: i for i, g in enumerate(model_gene_names)}

    if edge_df is None or edge_df.shape[0] == 0:
        filtered = pd.DataFrame(columns=[tf_col, target_col, weight_col])
    else:
        filtered = edge_df.loc[
            edge_df[target_col].astype(str).isin(target_to_idx.keys()),
            [tf_col, target_col, weight_col],
        ].copy()

    tf_names = (
        sorted(filtered[tf_col].astype(str).unique().tolist()) if filtered.shape[0] > 0 else []
    )
    tf_to_idx = {tf: i for i, tf in enumerate(tf_names)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for tf, target, weight in filtered.itertuples(index=False, name=None):
        tf = str(tf)
        target = str(target)
        if target not in target_to_idx or tf not in tf_to_idx:
            continue
        try:
            w = float(weight)
        except Exception:
            w = 1.0
        if not np.isfinite(w):
            continue
        rows.append(target_to_idx[target])
        cols.append(tf_to_idx[tf])
        data.append(max(0.0, w))

    G = int(len(model_gene_names))
    T = int(len(tf_names))
    B = sp.coo_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(G, T),
    ).tocsr()
    B.sum_duplicates()

    # Row-normalized target-by-TF matrix.
    row_sum = np.asarray(B.sum(axis=1)).reshape(-1)
    inv_row = np.zeros_like(row_sum, dtype=np.float32)
    inv_row[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    B_row = sp.diags(inv_row).dot(B).tocsr()

    # Column-normalized target-by-TF matrix.
    col_sum = np.asarray(B.sum(axis=0)).reshape(-1)
    inv_col = np.zeros_like(col_sum, dtype=np.float32)
    inv_col[col_sum > 0] = 1.0 / col_sum[col_sum > 0]
    B_col = B.dot(sp.diags(inv_col)).tocsr()

    sp.save_npz(str(out_prefix) + "_target_by_all_tf_raw.npz", B)
    sp.save_npz(str(out_prefix) + "_target_by_all_tf_row_norm.npz", B_row)
    sp.save_npz(str(out_prefix) + "_target_by_all_tf_col_norm.npz", B_col)
    pd.DataFrame({"tf_index": np.arange(T, dtype=np.int64), "TF": tf_names}).to_csv(
        str(out_prefix) + "_all_tf_order.tsv", sep="\t", index=False
    )

    # Co-regulation prior between model genes. We use the row-normalized B so a
    # target with many inferred TFs does not automatically dominate the graph.
    C = (B_row @ B_row.T).tocsr()
    C.setdiag(0.0)
    C.eliminate_zeros()
    c_row_sum = np.asarray(C.sum(axis=1)).reshape(-1)
    c_inv = np.zeros_like(c_row_sum, dtype=np.float32)
    c_inv[c_row_sum > 0] = 1.0 / c_row_sum[c_row_sum > 0]
    C_row = sp.diags(c_inv).dot(C).tocsr()

    sp.save_npz(str(out_prefix) + "_model_gene_coregulation_raw.npz", C)
    sp.save_npz(str(out_prefix) + "_model_gene_coregulation_row_norm.npz", C_row)

    targets_with_prior = int(np.sum(np.asarray(B.getnnz(axis=1)).reshape(-1) > 0)) if G > 0 else 0
    metadata = {
        "target_by_tf_shape": [G, T],
        "target_by_tf_nnz": int(B.nnz),
        "n_model_genes_with_at_least_one_tf": targets_with_prior,
        "fraction_model_genes_with_at_least_one_tf": float(targets_with_prior / max(1, G)),
        "n_unique_tfs": T,
        "coregulation_shape": [G, G],
        "coregulation_nnz": int(C.nnz),
        "target_by_tf_convention": "B[model_target_gene, TF] = nonnegative edge weight",
        "coregulation_convention": "C = row_normalize(B) @ row_normalize(B).T with zero diagonal",
        "recommended_current_fm_use": "Use the GxG co-regulation row-normalized prior when FM input contains only model HVGs; retain target-by-all-TF matrix for future latent TF/regulon modules.",
        "model_gene_names_ordered": model_gene_names,
        "tf_names_ordered": tf_names,
    }
    save_json(metadata, str(out_prefix) + "_bipartite_metadata.json")
    log(
        f"[export] bipartite/coregulation prior prefix={out_prefix}; "
        f"B_shape={B.shape} B_nnz={B.nnz} targets_covered={targets_with_prior}/{G} C_nnz={C.nnz}"
    )
    return metadata


def export_regulon_activity_dynamics(
    *,
    auc_mtx: pd.DataFrame,
    selected_cells: np.ndarray,
    obs: pd.DataFrame,
    out_dir: Path,
    time_col: str,
    celltype_col: str,
) -> None:
    """
    Export simple discovery-oriented regulon dynamics summaries.

    These tables are NOT causal claims. They are convenient ranked candidates
    for later biological interpretation/validation: regulons with large temporal
    range, strong first-to-last change, or cell-type-specific temporal dynamics.
    """
    selected_cells = np.asarray(selected_cells, dtype=np.int64)
    if selected_cells.size != auc_mtx.shape[0]:
        raise ValueError("selected_cells and auc_mtx row counts differ in dynamics export")

    meta = obs.iloc[selected_cells].copy()
    meta.index = auc_mtx.index
    regulon_columns = list(auc_mtx.columns)
    combined = pd.concat([meta, auc_mtx], axis=1)

    if time_col in combined.columns:
        by_time = combined.groupby(time_col, observed=True, sort=True)[regulon_columns].mean()
        rows: List[Dict[str, Any]] = []
        if by_time.shape[0] > 0:
            first_label = by_time.index[0]
            last_label = by_time.index[-1]
            for regulon in regulon_columns:
                values = by_time[regulon].to_numpy(dtype=np.float64)
                if values.size == 0:
                    continue
                peak_i = int(np.nanargmax(values))
                trough_i = int(np.nanargmin(values))
                rows.append(
                    {
                        "regulon": str(regulon),
                        "mean_activity_across_time": float(np.nanmean(values)),
                        "std_activity_across_time": float(np.nanstd(values)),
                        "range_activity_across_time": float(np.nanmax(values) - np.nanmin(values)),
                        "first_time": str(first_label),
                        "last_time": str(last_label),
                        "first_activity": float(values[0]),
                        "last_activity": float(values[-1]),
                        "last_minus_first": float(values[-1] - values[0]),
                        "peak_time": str(by_time.index[peak_i]),
                        "peak_activity": float(values[peak_i]),
                        "trough_time": str(by_time.index[trough_i]),
                        "trough_activity": float(values[trough_i]),
                    }
                )
        dynamics = pd.DataFrame(rows)
        if dynamics.shape[0] > 0:
            dynamics = dynamics.sort_values(
                ["range_activity_across_time", "std_activity_across_time"],
                ascending=[False, False],
            ).reset_index(drop=True)
        dynamics.to_csv(out_dir / "regulon_dynamics_by_time_ranked.tsv", sep="\t", index=False)
        log(
            f"[biology] saved ranked regulon temporal dynamics: {out_dir / 'regulon_dynamics_by_time_ranked.tsv'}"
        )

    if time_col in combined.columns and celltype_col in combined.columns:
        by_time_type = combined.groupby([celltype_col, time_col], observed=True, sort=True)[
            regulon_columns
        ].mean()
        rows2: List[Dict[str, Any]] = []
        if isinstance(by_time_type.index, pd.MultiIndex):
            for celltype_value in by_time_type.index.get_level_values(0).unique():
                one = by_time_type.xs(celltype_value, level=0)
                if one.shape[0] == 0:
                    continue
                for regulon in regulon_columns:
                    values = one[regulon].to_numpy(dtype=np.float64)
                    if values.size == 0:
                        continue
                    peak_i = int(np.nanargmax(values))
                    rows2.append(
                        {
                            "celltype": str(celltype_value),
                            "regulon": str(regulon),
                            "n_timepoints": int(values.size),
                            "mean_activity": float(np.nanmean(values)),
                            "std_activity": float(np.nanstd(values)),
                            "range_activity": float(np.nanmax(values) - np.nanmin(values)),
                            "last_minus_first": float(values[-1] - values[0]),
                            "peak_time": str(one.index[peak_i]),
                            "peak_activity": float(values[peak_i]),
                        }
                    )
        dynamics2 = pd.DataFrame(rows2)
        if dynamics2.shape[0] > 0:
            dynamics2 = dynamics2.sort_values(
                ["range_activity", "std_activity"], ascending=[False, False]
            ).reset_index(drop=True)
        dynamics2.to_csv(
            out_dir / "regulon_dynamics_by_celltype_time_ranked.tsv", sep="\t", index=False
        )
        log(
            f"[biology] saved celltype-specific regulon dynamics: {out_dir / 'regulon_dynamics_by_celltype_time_ranked.tsv'}"
        )


def export_for_fm(
    *,
    adjacencies: pd.DataFrame,
    regulons: Optional[List[Any]],
    grn_gene_names: List[str],
    grn_gene_original_indices: np.ndarray,
    model_gene_original_indices: np.ndarray,
    all_gene_names: List[str],
    out_dir: Path,
) -> None:
    """Export edge tables and sparse prior matrices for FM model integration."""
    export_dir = ensure_dir(out_dir / "fm_prior_exports")

    # 1) GRNBoost/GENIE3 adjacency edges.
    adj = adjacencies.copy()
    if "importance" not in adj.columns:
        raise KeyError("adjacencies must contain an 'importance' column")
    adj.to_csv(export_dir / "grnboost_adjacencies_for_fm.tsv", sep="\t", index=False)

    export_sparse_prior_matrix(
        edge_df=adj,
        gene_names=grn_gene_names,
        out_raw_path=export_dir / "grnboost_prior_A_target_by_tf_raw.npz",
        out_row_norm_path=export_dir / "grnboost_prior_A_target_by_tf_row_norm.npz",
        out_col_norm_path=export_dir / "grnboost_prior_A_target_by_tf_col_norm.npz",
        out_metadata_path=export_dir / "grnboost_prior_A_metadata.json",
        tf_col="TF",
        target_col="target",
        weight_col="importance",
    )

    # 2) Model-gene-only matrix, aligned to the downstream model feature order.
    all_gene_names_arr = np.asarray(all_gene_names, dtype=object)
    model_gene_names = [str(all_gene_names_arr[i]) for i in model_gene_original_indices]
    pd.DataFrame(
        {"gene_name": model_gene_names, "original_gene_index": model_gene_original_indices}
    ).to_csv(export_dir / "model_gene_order_for_fm.tsv", sep="\t", index=False)
    export_sparse_prior_matrix(
        edge_df=adj,
        gene_names=model_gene_names,
        out_raw_path=export_dir / "MODEL_GENES_grnboost_prior_A_target_by_tf_raw.npz",
        out_row_norm_path=export_dir / "MODEL_GENES_grnboost_prior_A_target_by_tf_row_norm.npz",
        out_col_norm_path=export_dir / "MODEL_GENES_grnboost_prior_A_target_by_tf_col_norm.npz",
        out_metadata_path=export_dir / "MODEL_GENES_grnboost_prior_A_metadata.json",
        tf_col="TF",
        target_col="target",
        weight_col="importance",
    )
    # Preserve regulators outside the FM HVG feature set and additionally
    # convert them into a model-gene co-regulation matrix usable by the current FM.
    export_model_target_by_all_tf_prior(
        edge_df=adj,
        model_gene_names=model_gene_names,
        out_prefix=export_dir / "MODEL_GENES_grnboost",
        tf_col="TF",
        target_col="target",
        weight_col="importance",
    )

    # 3) Regulon-pruned target edges, if ctx was run.
    if regulons is not None:

        reg_edges = regulons_to_target_edges(
            regulons=regulons,
        )

        # ------------------------------------------------------------
        # Safety check:
        # If this catches many punctuation/character targets, then the
        # regulon export is still broken and should not be used for FM.
        # ------------------------------------------------------------
        if reg_edges.shape[0] > 0:

            # bad_target_mask = reg_edges["target"].astype(str).isin(
            #     [
            #         "[",
            #         "]",
            #         "(",
            #         ")",
            #         "{",
            #         "}",
            #         "'",
            #         '"',
            #         ",",
            #         ":",
            #     ]
            # )

            # n_bad_targets = int(bad_target_mask.sum())

            valid_model_gene_names = set(model_gene_names)

            bad_examples = _get_character_split_examples(
                edge_df=reg_edges,
                valid_gene_names=valid_model_gene_names,
                n=20,
            )

            n_bad_targets = int(bad_examples.shape[0])

            if n_bad_targets > 0:
                raise RuntimeError(
                    "Regulon export still contains character-split targets.\n"
                    f"Examples:\n{bad_examples.to_string(index=False)}"
                )

            # if n_bad_targets > 0:
            #     bad_examples = reg_edges.loc[
            #         bad_target_mask,
            #         [
            #             "TF",
            #             "target",
            #             "regulon",
            #         ],
            #     ].head(20)

            #     raise RuntimeError(
            #         "Regulon export still contains character-split targets.\n"
            #         f"Number of bad rows: {n_bad_targets}\n"
            #         f"Examples:\n{bad_examples.to_string(index=False)}"
            #     )

        regulon_edges_file_path = export_dir / "regulon_pruned_tf_target_edges.tsv"

        reg_edges.to_csv(
            regulon_edges_file_path,
            sep="\t",
            index=False,
        )

        log(
            f"[export] saved regulon-pruned TF-target edges: "
            f"{regulon_edges_file_path}; shape={reg_edges.shape}"
        )

        if reg_edges.shape[0] > 0:

            reg_edges_for_matrix = reg_edges.rename(
                columns={
                    "regulon_weight": "importance",
                }
            )

            export_sparse_prior_matrix(
                edge_df=reg_edges_for_matrix,
                gene_names=model_gene_names,
                out_raw_path=export_dir / "MODEL_GENES_regulon_prior_A_target_by_tf_raw.npz",
                out_row_norm_path=export_dir
                / "MODEL_GENES_regulon_prior_A_target_by_tf_row_norm.npz",
                out_col_norm_path=export_dir
                / "MODEL_GENES_regulon_prior_A_target_by_tf_col_norm.npz",
                out_metadata_path=export_dir / "MODEL_GENES_regulon_prior_A_metadata.json",
                tf_col="TF",
                target_col="target",
                weight_col="importance",
            )
            export_model_target_by_all_tf_prior(
                edge_df=reg_edges_for_matrix,
                model_gene_names=model_gene_names,
                out_prefix=export_dir / "MODEL_GENES_regulon",
                tf_col="TF",
                target_col="target",
                weight_col="importance",
            )

        else:

            log(
                "[export][warning] regulon edge table is empty. "
                "The regulon prior matrix will not contain motif-pruned edges."
            )

    log(f"[export] FM prior exports written to {export_dir}")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pySCENIC GRN inference on project X/obs/var files and export GRN priors for FM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Project data paths.
    parser.add_argument("--adata_x_path", type=str, required=True)
    parser.add_argument("--adata_obs_path", type=str, required=True)
    parser.add_argument(
        "--adata_var_path",
        type=str,
        required=True,
        help="Var file defining the broad GRN inference gene set, e.g. HVG2000.",
    )
    parser.add_argument(
        "--model_var_path",
        type=str,
        default=None,
        help=(
            "Optional separate var file defining the exact downstream FM model genes, e.g. HVG200. "
            "If omitted, --adata_var_path is used for both GRN inference and FM export (legacy behavior)."
        ),
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--run_name", type=str, default=None, help="Optional stable subfolder name under out_dir."
    )
    parser.add_argument(
        "--append_timestamp",
        type=str2bool,
        default=False,
        help="Append a timestamp subfolder. Keep False when --resume True so reruns reuse the same artifacts.",
    )

    # Metadata columns.
    parser.add_argument("--time_col", type=str, default="time")
    parser.add_argument("--celltype_col", type=str, default="class_label")
    parser.add_argument("--gene_name_col", type=str, default="gene_short_name")
    parser.add_argument("--hvg_col", type=str, default="highly_variable")
    parser.add_argument("--use_hvg_mask", type=str2bool, default=True)

    # pySCENIC resources.
    parser.add_argument("--tf_names_path", type=str, default=None)
    parser.add_argument("--tf_fallback", type=str, default="error", choices=["error", "all_genes"])
    parser.add_argument("--ranking_db_paths", type=str, nargs="*", default=[])
    parser.add_argument("--motif_annotations_path", type=str, default=None)

    # Sampling / memory controls.
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="stratified_time_celltype",
        choices=["all", "random", "stratified_time", "stratified_time_celltype"],
    )
    parser.add_argument("--max_cells_total", type=int, default=50000)
    parser.add_argument("--max_cells_per_time", type=int, default=2000)
    parser.add_argument("--max_cells_per_time_celltype", type=int, default=500)
    parser.add_argument("--min_cells_per_gene", type=int, default=20)
    parser.add_argument(
        "--sampled_gene_variance_epsilon",
        type=float,
        default=0.0,
        help="Remove genes with variance <= epsilon after cell subsampling and before Arboreto.",
    )
    parser.add_argument("--include_tfs_outside_hvg", type=str2bool, default=True)
    parser.add_argument("--expr_dtype", type=str, default="float32", choices=["float32", "float64"])

    # pySCENIC controls.
    parser.add_argument("--method", type=str, default="grnboost2", choices=["grnboost2", "genie3"])
    parser.add_argument(
        "--steps",
        type=str,
        nargs="+",
        default=["grn", "ctx", "aucell", "export"],
        choices=["grn", "ctx", "aucell", "export"],
    )
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Parallel controls.
    parser.add_argument("--use_dask", type=str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--threads_per_worker", type=int, default=1)
    parser.add_argument("--worker_memory_limit", type=str, default="8GB")
    parser.add_argument(
        "--arboreto_dask_compatibility_patch",
        type=str2bool,
        default=True,
        help=(
            "Apply and preflight-test a narrow runtime fix for Arboreto 0.1.6 with modern Dask. "
            "Keep True unless the environment contains an Arboreto release whose create_graph bug is fixed."
        ),
    )
    parser.add_argument(
        "--pyscenic_numpy_pandas_compatibility_patch",
        type=str2bool,
        default=True,
        help=(
            "Apply and preflight-test in-memory compatibility aliases required by "
            "pySCENIC 0.12.1 with NumPy 2.x and pandas 2.x. Keep True for the "
            "current environment."
        ),
    )

    # Debug controls.
    parser.add_argument("--save_expression_tsv", type=str2bool, default=False)
    parser.add_argument(
        "--export_regulon_dynamics",
        type=str2bool,
        default=True,
        help="Export simple time/time-celltype regulon activity ranking tables for biological interpretation.",
    )
    return parser


def collect_environment_versions() -> Dict[str, Any]:
    """Collect package versions without making them hard dependencies."""
    versions: Dict[str, Any] = {"python": sys.version.replace("\n", " ")}
    package_names = [
        "pyscenic",
        "arboreto",
        "ctxcore",
        "dask",
        "distributed",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "pyarrow",
    ]
    try:
        from importlib.metadata import version as package_version
    except Exception:
        package_version = None
    for package_name in package_names:
        try:
            versions[package_name] = (
                package_version(package_name) if package_version is not None else None
            )
        except Exception:
            versions[package_name] = None
    return versions


def main() -> None:
    args = build_parser().parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    # Stable output folders are important for --resume True. The older script
    # always appended a fresh timestamp, which made resume ineffective across
    # separate invocations. Timestamping remains available explicitly.
    out_dir_base = Path(args.out_dir)
    if args.run_name is not None and str(args.run_name).strip() != "":
        out_dir_base = out_dir_base / str(args.run_name).strip()
    if bool(args.append_timestamp):
        out_dir_base = out_dir_base / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(out_dir_base)
    save_json(vars(args), out_dir / "run_config.json")
    save_json(collect_environment_versions(), out_dir / "environment_versions.json")
    log(f"[run] output directory: {out_dir}")

    # Output paths.
    selected_cells_path = out_dir / "selected_cells.tsv"
    selected_grn_genes_path = out_dir / "selected_grn_genes.tsv"
    adj_path = out_dir / "adjacencies.tsv"
    modules_path = out_dir / "modules.pkl"
    motifs_path = out_dir / "motifs.csv"
    regulons_path = out_dir / "regulons.pkl"
    auc_path = out_dir / "auc_mtx.csv.gz"
    expr_tsv_path = out_dir / "expression_sampled_cells_x_genes.tsv.gz"

    # Fail fast on the known Arboreto 0.1.6 / modern-Dask incompatibility.
    # This runs before loading the large sparse matrix, while the GRN function
    # calls it again defensively (the second call is an inexpensive no-op).
    if (
        "grn" in args.steps
        and not (bool(args.resume) and adj_path.exists())
        and bool(args.arboreto_dask_compatibility_patch)
    ):
        apply_arboreto_modern_dask_compatibility_patch()

    # Fail fast on the released pySCENIC 0.12.1 references to removed NumPy
    # and pandas aliases.  The same patch is applied to Dask workers after the
    # local cluster starts.
    if "ctx" in args.steps and bool(args.pyscenic_numpy_pandas_compatibility_patch):
        apply_pyscenic_numpy_pandas_compatibility_patch()

    # Load data and select genes.
    X_grn, obs, var, grn_gene_names, grn_gene_original_indices, model_gene_original_indices = (
        load_project_data(
            adata_x_path=args.adata_x_path,
            adata_obs_path=args.adata_obs_path,
            adata_var_path=args.adata_var_path,
            model_var_path=args.model_var_path,
            gene_name_col=args.gene_name_col,
            hvg_col=args.hvg_col,
            use_hvg_mask=bool(args.use_hvg_mask),
            include_tfs_outside_hvg=bool(args.include_tfs_outside_hvg),
            tf_names_path=args.tf_names_path,
            min_cells_per_gene=int(args.min_cells_per_gene),
        )
    )
    all_gene_names = choose_gene_names(var, requested_col=args.gene_name_col)

    pd.DataFrame(
        {
            "gene_name": grn_gene_names,
            "original_gene_index": grn_gene_original_indices,
        }
    ).to_csv(selected_grn_genes_path, sep="\t", index=False)
    log(f"[gene] saved selected GRN genes: {selected_grn_genes_path}")

    # Select cells.
    selected_cells = select_cells(
        obs=obs,
        time_col=args.time_col,
        celltype_col=args.celltype_col,
        sample_strategy=args.sample_strategy,
        max_cells_total=args.max_cells_total,
        max_cells_per_time=args.max_cells_per_time,
        max_cells_per_time_celltype=args.max_cells_per_time_celltype,
        seed=int(args.seed),
    )
    obs.iloc[selected_cells].assign(original_cell_index=selected_cells).to_csv(
        selected_cells_path, sep="\t", index=False
    )
    log(f"[cell] saved selected cells: {selected_cells_path}")

    # Remove genes that became constant in the actual selected-cell subset.
    X_grn, grn_gene_names, grn_gene_original_indices = filter_grn_genes_after_cell_sampling(
        X_grn=X_grn,
        selected_cells=selected_cells,
        grn_gene_names=grn_gene_names,
        grn_gene_original_indices=grn_gene_original_indices,
        variance_epsilon=float(args.sampled_gene_variance_epsilon),
    )
    pd.DataFrame(
        {
            "gene_name": grn_gene_names,
            "original_gene_index": grn_gene_original_indices,
        }
    ).to_csv(selected_grn_genes_path, sep="\t", index=False)
    log(
        f"[gene] updated selected GRN genes after sampled-cell variance filtering: {selected_grn_genes_path}"
    )

    # Build expression DataFrame once. This is the memory-heavy step.
    expr_df = sparse_to_expression_dataframe(
        X=X_grn,
        selected_cells=selected_cells,
        gene_names=grn_gene_names,
        dtype=args.expr_dtype,
    )
    if bool(args.save_expression_tsv):
        expr_df.to_csv(expr_tsv_path, sep="\t", compression="gzip")
        log(f"[expr] saved expression TSV: {expr_tsv_path}")

    # Start dask after the expression matrix is ready.
    client_or_address, cluster = make_dask_client(
        num_workers=int(args.num_workers),
        threads_per_worker=int(args.threads_per_worker),
        memory_limit=str(args.worker_memory_limit),
        use_dask=bool(args.use_dask),
    )

    adjacencies: Optional[pd.DataFrame] = None
    regulons: Optional[List[Any]] = None
    auc_mtx: Optional[pd.DataFrame] = None

    try:
        if "grn" in args.steps:
            adjacencies = run_grnboost_or_genie3(
                expr_df=expr_df,
                tf_names_path=args.tf_names_path,
                tf_fallback=args.tf_fallback,
                method=args.method,
                seed=int(args.seed),
                client_or_address=client_or_address,
                arboreto_dask_compatibility_patch=bool(args.arboreto_dask_compatibility_patch),
                out_path=adj_path,
                resume=bool(args.resume),
            )
        elif adj_path.exists():
            adjacencies = pd.read_csv(adj_path, sep="\t")
        else:
            raise RuntimeError("Step 'grn' was skipped but adjacencies.tsv does not exist.")

        if "ctx" in args.steps:
            if len(args.ranking_db_paths) == 0:
                raise ValueError("Step 'ctx' requires --ranking_db_paths")
            if args.motif_annotations_path is None:
                raise ValueError("Step 'ctx' requires --motif_annotations_path")
            motifs_df, regulons = run_ctx_pruning(
                expr_df=expr_df,
                adjacencies=adjacencies,
                ranking_db_paths=args.ranking_db_paths,
                motif_annotations_path=args.motif_annotations_path,
                client_or_address=client_or_address,
                num_workers=int(args.num_workers),
                out_motifs_path=motifs_path,
                out_regulons_path=regulons_path,
                out_modules_path=modules_path,
                resume=bool(args.resume),
                pyscenic_numpy_pandas_compatibility_patch=bool(
                    args.pyscenic_numpy_pandas_compatibility_patch
                ),
            )
        elif regulons_path.exists():
            with regulons_path.open("rb") as f:
                regulons = pickle.load(f)

        if "aucell" in args.steps:
            if regulons is None:
                raise RuntimeError(
                    "Step 'aucell' requires regulons. Run ctx first or provide regulons.pkl."
                )
            auc_mtx = run_aucell(
                expr_df=expr_df,
                regulons=regulons,
                num_workers=int(args.num_workers),
                out_auc_path=auc_path,
                resume=bool(args.resume),
            )
            aggregate_auc_by_metadata(
                auc_mtx=auc_mtx,
                selected_cells=selected_cells,
                obs=obs,
                out_dir=out_dir,
                time_col=args.time_col,
                celltype_col=args.celltype_col,
            )
            if bool(args.export_regulon_dynamics):
                export_regulon_activity_dynamics(
                    auc_mtx=auc_mtx,
                    selected_cells=selected_cells,
                    obs=obs,
                    out_dir=out_dir,
                    time_col=args.time_col,
                    celltype_col=args.celltype_col,
                )

        if "export" in args.steps:
            export_for_fm(
                adjacencies=adjacencies,
                regulons=regulons,
                grn_gene_names=grn_gene_names,
                grn_gene_original_indices=grn_gene_original_indices,
                model_gene_original_indices=model_gene_original_indices,
                all_gene_names=all_gene_names,
                out_dir=out_dir,
            )

        summary = {
            "status": "success",
            "n_cells_selected": int(selected_cells.size),
            "n_grn_genes": int(len(grn_gene_names)),
            "n_model_genes": int(len(model_gene_original_indices)),
            "n_adjacencies": int(adjacencies.shape[0]) if adjacencies is not None else None,
            "n_regulons": int(len(regulons)) if regulons is not None else None,
            "auc_shape": list(auc_mtx.shape) if auc_mtx is not None else None,
            "out_dir": str(out_dir),
        }
        save_json(summary, out_dir / "run_summary.json")
        log(f"[done] success. Summary: {json.dumps(summary, indent=2)}")

    except Exception as exc:
        error_info = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json(error_info, out_dir / "run_error.json")
        log(f"[failed] {type(exc).__name__}: {exc}")
        raise

    finally:
        # Close Dask cleanly and free memory.
        try:
            if hasattr(client_or_address, "close"):
                client_or_address.close()
        except Exception:
            pass
        try:
            if cluster is not None:
                cluster.close()
        except Exception:
            pass
        del expr_df
        gc.collect()


if __name__ == "__main__":
    main()
