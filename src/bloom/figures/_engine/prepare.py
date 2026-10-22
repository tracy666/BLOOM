"""Stream saved expression into reusable, truthful biological analysis artifacts.

Only observations fit coordinates. Generated rows never enter a reference fit.
Cell plots use a declared stratified sample; stage/state gene summaries retain all
saved rows unless an explicit analysis cap is requested.
"""

from __future__ import annotations

from concurrent.futures import as_completed
from .util import process_pool
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import traceback
import uuid

import numpy as np
import pandas as pd

VERSION = "biology-prepare-2.0-native-assays"
_UNKNOWN = {"", "unknown", "unlabelled", "unlabeled", "none", "nan", "-1"}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(_jsonable(data), indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def _token(data):
    return hashlib.sha256(
        json.dumps(_jsonable(data), sort_keys=True, default=str).encode()
    ).hexdigest()


def _file_stamp(path):
    path = Path(path)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _config_stamp(config):
    stamps = {}
    for dataset, settings in config.get("datasets", {}).items():
        stamps[dataset] = dict(settings)
        for key, path in settings.items():
            if key.endswith("_path") and path and Path(path).is_file():
                stamps[dataset][key] = _file_stamp(path)
    return stamps


def _pipeline_code_digest():
    folder = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("prepare.py", "raw.py", "discovery.py"):
        digest.update(name.encode())
        digest.update((folder / name).read_bytes())
    return digest.hexdigest()


def _fingerprint(source, config, options):
    return _token(
        {
            "version": VERSION,
            "pipeline_code": _pipeline_code_digest(),
            "source": source,
            "file": _file_stamp(source["path"]),
            "datasets": _config_stamp(config),
            "sample_cells": int(options.get("sample_cells", 20000)),
            "analysis_cap": int(options.get("analysis_cap", 0)),
            "seed": int(options.get("seed", 1729)),
        }
    )


def _source_meta(source):
    keys = (
        "source_id",
        "dataset",
        "model",
        "role",
        "panel",
        "scenario",
        "path",
        "time_kind",
        "parent_source_id",
        "population",
    )
    return {key: source.get(key) for key in keys}


def _group_codes(cells, state=False):
    cols = ["time", "time_label"] + (["state_supplied"] if state else [])
    key = pd.MultiIndex.from_frame(cells[cols])
    codes, unique = pd.factorize(key, sort=False)
    groups = pd.DataFrame(list(unique), columns=cols)
    return codes, groups


def _sample_rows(cells, maximum, seed):
    """Proportional time/state sampling with inverse-probability plot weights."""
    n = len(cells)
    if maximum <= 0 or maximum >= n:
        return np.arange(n, dtype=np.int64), np.ones(n, dtype=np.float64)
    codes, groups = _group_codes(cells, state=True)
    counts = np.bincount(codes, minlength=len(groups))
    if len(counts) > maximum:
        codes, groups = _group_codes(cells, state=False)
        counts = np.bincount(codes, minlength=len(groups))
    # If even stages exceed the requested sample, the finest cells are selected
    # uniformly; this uncommon condition is explicit in sample metadata.
    if len(counts) > maximum:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(n, maximum, replace=False))
        return rows, np.full(maximum, n / maximum, dtype=np.float64)
    allocations = np.minimum(counts, np.maximum(1, np.floor(maximum * counts / n).astype(int)))
    while allocations.sum() > maximum:
        candidates = np.flatnonzero(allocations > 1)
        j = candidates[np.argmax(allocations[candidates] / counts[candidates])]
        allocations[j] -= 1
    remainder = maximum - int(allocations.sum())
    if remainder:
        priority = np.argsort(-(maximum * counts / n - allocations))
        while remainder:
            changed = False
            for j in priority:
                if allocations[j] < counts[j]:
                    allocations[j] += 1
                    remainder -= 1
                    changed = True
                    if not remainder:
                        break
            if not changed:
                break
    rng = np.random.default_rng(seed)
    chunks, weights = [], []
    order = np.argsort(codes, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    for j, count in enumerate(counts):
        candidates = order[offsets[j] : offsets[j + 1]]
        chosen = rng.choice(candidates, int(allocations[j]), replace=False)
        chunks.append(chosen)
        weights.append(np.full(len(chosen), count / len(chosen)))
    rows, weight = np.concatenate(chunks), np.concatenate(weights)
    order = np.argsort(rows)
    return rows[order], weight[order]


def _summary_frame(groups, genes, sums, squares, detected, counts, state=False):
    valid = counts > 0
    groups = groups.loc[valid].reset_index(drop=True)
    counts = counts[valid]
    means = sums[valid] / counts[:, None]
    variance = np.maximum(0, squares[valid] / counts[:, None] - means**2)
    variance *= np.divide(counts, np.maximum(1, counts - 1))[:, None]
    n_gene = len(genes)
    frame = groups.loc[groups.index.repeat(n_gene)].reset_index(drop=True)
    if state:
        frame = frame.rename(columns={"state_supplied": "state"})
    frame["gene"] = np.tile(genes, len(groups))
    frame["n"] = np.repeat(counts, n_gene)
    frame["mean"] = means.ravel()
    frame["std"] = np.sqrt(variance).ravel()
    frame["pct"] = (detected[valid] / counts[:, None]).ravel()
    return frame


def _accumulate(values, codes, sums, squares, detected, counts):
    # A chunk generally contains only a few temporal/state groups. Group-level
    # reductions avoid constructing an N-by-group dummy matrix.
    for group in np.unique(codes):
        selected = values[codes == group]
        sums[group] += selected.sum(axis=0, dtype=np.float64)
        squares[group] += np.einsum("ij,ij->j", selected, selected, dtype=np.float64)
        detected[group] += np.count_nonzero(selected > 0, axis=0)
        counts[group] += len(selected)


def _read_native(source, config, options, cache_root):
    from . import raw

    fingerprint = _fingerprint(source, config, options)
    artifact_dir = Path(cache_root) / "native" / (str(source["source_id"]) + "_" + fingerprint[:14])
    meta_path = artifact_dir / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("native_complete") and meta.get("fingerprint") == fingerprint:
            required = [
                artifact_dir / "expression.npy",
                artifact_dir / "cells.parquet",
                artifact_dir / "stage_gene.parquet",
            ]
            if all(path.is_file() for path in required):
                meta.update(_source_meta(source))
                return meta, "reused"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    initial_stamp = _file_stamp(source["path"])
    result = raw.read_source(source, config)
    genes = list(map(str, result["genes"]))
    if len(set(genes)) != len(genes) or not genes:
        raise ValueError("Native gene identities must be nonempty and unique")
    cells = result["cells"].copy().reset_index(drop=True)
    for col in ("cell_id", "time_label", "state_supplied"):
        if col not in cells:
            cells[col] = "unknown" if col == "state_supplied" else cells.index.astype(str)
        cells[col] = cells[col].fillna("unknown").astype(str)
    cells["time"] = pd.to_numeric(cells["time"], errors="raise")
    if not np.isfinite(cells["time"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite developmental-time coordinate")
    n_total = len(cells)
    cap = int(options.get("analysis_cap", 0))
    if cap > 0 and cap < n_total:
        used_rows, _ = _sample_rows(cells, cap, int(options.get("seed", 1729)))
        # Explicit caps apply to summaries as well as plots and are declared.
        cells = cells.iloc[used_rows].reset_index(drop=True)
    else:
        used_rows = None
    rows, weights = _sample_rows(
        cells, int(options.get("sample_cells", 20000)), int(options.get("seed", 1729))
    )
    n, g = len(cells), len(genes)
    sampled = np.lib.format.open_memmap(
        artifact_dir / "expression.npy", mode="w+", dtype="float32", shape=(len(rows), g)
    )
    stages, stage_groups = _group_codes(cells)
    states, state_groups = _group_codes(cells, state=True)
    stage_acc = [np.zeros((len(stage_groups), g), np.float64) for _ in range(3)]
    state_acc = [np.zeros((len(state_groups), g), np.float64) for _ in range(3)]
    stage_n = np.zeros(len(stage_groups), np.int64)
    state_n = np.zeros(len(state_groups), np.int64)
    digest = hashlib.sha256()
    digest.update("\0".join(genes).encode())
    digest.update(
        pd.util.hash_pandas_object(cells[["cell_id", "time"]], index=False).values.tobytes()
    )
    offset = 0
    for start, stop, expression in raw.iter_expression(
        source, int(options.get("chunk_size", 65536))
    ):
        if used_rows is not None:
            begin = np.searchsorted(used_rows, start)
            end = np.searchsorted(used_rows, stop)
            if begin == end:
                continue
            expression = expression[used_rows[begin:end] - start]
            start, stop = begin, end
        expression = np.asarray(expression, dtype=np.float32)
        if expression.shape != (stop - start, g):
            raise ValueError(
                f"Expression chunk shape {expression.shape} disagrees with row/gene metadata"
            )
        if not np.isfinite(expression).all():
            raise ValueError(
                f"Nonfinite saved expression at rows {start}:{stop}; source was not silently repaired"
            )
        digest.update(expression.tobytes(order="C"))
        _accumulate(expression, stages[start:stop], *stage_acc, stage_n)
        _accumulate(expression, states[start:stop], *state_acc, state_n)
        a, b = np.searchsorted(rows, [start, stop])
        if b > a:
            sampled[a:b] = expression[rows[a:b] - start]
        offset += len(expression)
    if offset != n:
        raise ValueError(f"Expression reader delivered {offset} rows for {n} cell annotations")
    sampled.flush()
    sampled_cells = cells.iloc[rows].copy().reset_index(drop=True)
    sampled_cells["expression_row"] = np.arange(len(rows), dtype=np.int64)
    sampled_cells["sample_weight"] = weights
    sampled_cells["state"] = sampled_cells["state_supplied"]
    sampled_cells["x"] = np.nan
    sampled_cells["y"] = np.nan
    sampled_cells.to_parquet(artifact_dir / "cells.parquet", index=False)
    _summary_frame(stage_groups, genes, *stage_acc, stage_n).to_parquet(
        artifact_dir / "stage_gene.parquet", index=False
    )
    _summary_frame(state_groups, genes, *state_acc, state_n, state=True).to_parquet(
        artifact_dir / "state_gene.parquet", index=False
    )
    if _file_stamp(source["path"]) != initial_stamp:
        raise RuntimeError(
            "Source file changed during preparation; discard inconsistent cache and retry"
        )
    metadata = result.get("metadata", {})
    settings = result.get(
        "dataset_config", config.get("datasets", {}).get(source.get("dataset"), {})
    )
    warning = list(result.get("notes", []))
    if source.get("role") not in ("GT", "REFERENCE"):
        warning.append(
            "Supplied labels are target/template annotations, not independently predicted cell identities."
        )
    if used_rows is not None:
        warning.append(
            f"Explicit analysis cap: full-row summaries use {n} of {n_total} saved cells."
        )
    summary_files = {
        "stage": str(artifact_dir / "stage_gene.parquet"),
        "state": str(artifact_dir / "state_gene.parquet"),
    }
    meta = {
        **_source_meta(source),
        "artifact_dir": str(artifact_dir.resolve()),
        "genes": genes,
        "n_total": n_total,
        "n_analyzed": n,
        "n_sample": len(rows),
        "has_truth": bool(result.get("has_truth", source.get("role") == "GT")),
        "fingerprint": fingerprint,
        "expression_digest": digest.hexdigest(),
        "reference_policy": "not_yet_projected",
        "embedding_method": "none",
        "state_method": (
            "supplied_annotation"
            if (~cells["state_supplied"].str.lower().isin(_UNKNOWN)).any()
            else "unavailable"
        ),
        "capabilities": ["expression", "cells", "time", "stage_gene", "state_gene"]
        + (["state"] if (~cells["state_supplied"].str.lower().isin(_UNKNOWN)).any() else []),
        "summary_files": summary_files,
        "warning": warning,
        "source_metadata": metadata,
        "native_complete": True,
        "std_ddof": 1,
        "state_summary_policy": "full saved rows grouped by supplied annotation/conditioning state; this is not independently predicted abundance",
        "time_kind": result.get("time_kind", source.get("time_kind", "ordinal")),
        "sample_policy": "proportional time/state stratified; sample_weight is inverse sampling probability",
        "detection_policy": "fraction of saved expression values > 0; interpret using recorded expression scale",
        "detection_units": "fraction",
        "reference_input": {
            key: settings[key]
            for key in (
                "x_path",
                "obs_path",
                "var_path",
                "gene_name_column",
                "time_column",
                "actual_time_column",
                "state_column",
                "celltype_column",
                "obs_index_column",
                "time_map",
            )
            if settings.get(key)
        },
        "sample_size_requested": int(options.get("sample_cells", 20000)),
    }
    for key in ("author_genes_path", "programs_path", "edges_path", "species", "expression_scale"):
        if key in settings:
            meta[key] = settings[key]
    _write_json(meta_path, meta)
    return meta, "prepared"


def _read_perturbation(source, config, options, cache_root):
    """Compact edits are saved means/variances, never fabricated edited cells."""
    from . import raw
    import pyarrow as pa
    import pyarrow.parquet as pq

    fingerprint = _fingerprint(source, config, options)
    artifact_dir = Path(cache_root) / "native" / (str(source["source_id"]) + "_" + fingerprint[:14])
    meta_path = artifact_dir / "metadata.json"
    table_path = artifact_dir / "perturbation.parquet"
    if meta_path.is_file() and table_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("native_complete") and meta.get("fingerprint") == fingerprint:
            return meta, "reused"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    initial_stamp = _file_stamp(source["path"])
    writer, n_rows, genes_seen, interventions = None, 0, set(), 0
    try:
        for edit in raw.iter_perturbation(source):
            genes = list(map(str, edit["genes"]))
            genes_seen.update(genes)
            times = np.asarray(edit.get("times", edit.get("time", [0])), dtype=float).reshape(-1)
            baseline = np.asarray(edit["baseline_mean"], dtype=float)
            edited = np.asarray(edit["edited_mean"], dtype=float)
            if baseline.ndim == 1:
                baseline = baseline[None, :]
            if edited.ndim == 1:
                edited = edited[None, :]
            if baseline.shape != edited.shape or baseline.shape != (len(times), len(genes)):
                raise ValueError("Compact perturbation means do not share a time-by-gene axis")
            if not np.isfinite(baseline).all() or not np.isfinite(edited).all():
                raise ValueError("Nonfinite compact perturbation means")
            intervention = str(
                edit.get(
                    "intervention", edit.get("edit_name", edit.get("intervention_gene", "unknown"))
                )
            )
            intervention_gene = str(edit.get("intervention_gene", edit.get("gene", intervention)))
            frame = pd.DataFrame(
                {
                    "intervention_gene": intervention_gene,
                    "response_gene": np.tile(genes, len(times)),
                    "time": np.repeat(times, len(genes)),
                    "delta": (edited - baseline).ravel(),
                    "baseline_mean": baseline.ravel(),
                    "edited_mean": edited.ravel(),
                    "baseline_variance": np.nan,
                    "edited_variance": np.nan,
                    "model": str(source.get("model", "unknown")),
                    "intervention": intervention,
                    "scale": float(edit.get("scale", np.nan)),
                }
            )
            for key in ("baseline_variance", "edited_variance"):
                if key in edit and edit[key] is not None:
                    array = np.asarray(edit[key], dtype=float)
                    if array.shape != baseline.shape:
                        raise ValueError(f"Compact perturbation {key} axis mismatch")
                    frame[key] = array.ravel()
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(table_path, table.schema, compression="zstd")
            writer.write_table(table)
            n_rows += len(frame)
            interventions += 1
    finally:
        if writer is not None:
            writer.close()
    if not n_rows:
        raise ValueError("No saved matched baseline/edit summaries found in compact perturbation")
    if _file_stamp(source["path"]) != initial_stamp:
        raise RuntimeError("Compact source changed during preparation")
    meta = {
        **_source_meta(source),
        "artifact_dir": str(artifact_dir.resolve()),
        "genes": sorted(genes_seen),
        "n_total": 0,
        "n_analyzed": 0,
        "n_sample": 0,
        "n_interventions": interventions,
        "n_response_rows": n_rows,
        "has_truth": False,
        "fingerprint": fingerprint,
        "reference_policy": "matched_baseline_edit_saved_summaries",
        "embedding_method": "none",
        "state_method": "none",
        "capabilities": ["perturbation"],
        "summary_files": {"perturbation": str(table_path)},
        "warning": [
            "Compact means/variances support response summaries; no edited-cell embedding or per-cell response is claimed."
        ],
        "native_complete": True,
        "source_metadata": source.get("metadata", {}),
    }
    for key in ("author_genes_path", "programs_path", "edges_path", "species", "expression_scale"):
        if key in config.get("datasets", {}).get(source.get("dataset"), {}):
            meta[key] = config["datasets"][source["dataset"]][key]
    _write_json(meta_path, meta)
    return meta, "prepared"


def _native_task(source, config, options, cache_root):
    try:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            if source.get("scenario") == "perturbation" or source.get("matrix_key") is None:
                return _read_perturbation(source, config, options, cache_root)
            return _read_native(source, config, options, cache_root)
    except Exception as error:
        return {
            **_source_meta(source),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }, "error"


def _resources(options, tasks):
    """Honor affinity/cgroup CPU and memory limits as well as user reservation."""
    import multiprocessing

    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = multiprocessing.cpu_count()
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cpus = min(cpus, max(1, math.ceil(int(quota) / int(period))))
    except (FileNotFoundError, ValueError):
        pass
    cpus = max(1, cpus - int(options.get("cpu_reserve", 0)))
    requested = options.get("prepare_workers", options.get("workers", "auto"))
    if requested not in ("auto", None, 0, "0"):
        cpus = min(cpus, max(1, int(requested)))
    try:
        import psutil

        available = psutil.virtual_memory().available / 1024**3
    except ImportError:
        available = 8.0
    try:
        maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        current = int(Path("/sys/fs/cgroup/memory.current").read_text())
        if maximum != "max":
            memory_stats = dict(
                line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
            )
            reclaimable = int(memory_stats.get("inactive_file", 0))
            available = min(
                available,
                max(0.1, min(int(maximum), int(maximum) - current + reclaimable) / 1024**3),
            )
    except (FileNotFoundError, ValueError):
        pass
    if options.get("memory_gb"):
        available = min(available, float(options["memory_gb"]))
    worker_gb = max(
        float(options.get("prepare_gb_per_worker", options.get("worker_memory_gb", 0.8))),
        float(options.get("_estimated_native_gb_per_worker", 0)),
    )
    return max(1, min(cpus, tasks, int(max(0.25, available * 0.75) / worker_gb)))


def _deduplicate_observed(artifacts):
    """Retain incompatible truths rather than silently choose the nicest model."""
    kept, identities = [], {}
    # Keep the broadest compatible GT as the gallery's representative.
    artifacts = sorted(artifacts, key=lambda a: (-a.get("n_total", 0), a.get("source_id", "")))
    for meta in artifacts:
        if meta.get("role") not in ("GT", "REFERENCE"):
            kept.append(meta)
            continue
        group = (meta["dataset"], meta["scenario"], meta["role"], meta["panel"])
        identity = meta.get("expression_digest", "")
        existing = identities.setdefault(group, {})
        if identity in existing:
            original = existing[identity]
            original.setdefault("matching_observation_sources", []).append(meta["source_id"])
            if meta.get("parent_source_id"):
                original.setdefault("matching_parent_sources", []).append(meta["parent_source_id"])
            continue
        if existing:
            meta.setdefault("warning", []).append(
                "Observed target coverage or numerical truth differs across saved models; this is retained as an alternate observed artifact, not silently pooled."
            )
            meta["alternate_observation"] = True
            meta["model"] = str(meta["model"]) + "_alternate_" + str(len(existing) + 1)
        else:
            meta["alternate_observation"] = False
        existing[identity] = meta
        kept.append(meta)
    return kept


def _observed_scale_comparison(first, second):
    """Attest common native units from exact observed cell/stage/gene matches."""
    common = sorted(set(first["genes"]) & set(second["genes"]))
    if len(common) < 3:
        return "unverified", "fewer than three shared genes"
    a = pd.read_parquet(Path(first["artifact_dir"]) / "cells.parquet")
    b = pd.read_parquet(Path(second["artifact_dir"]) / "cells.parquet")
    keys = ["cell_id", "time"]
    if a.duplicated(keys).any() or b.duplicated(keys).any():
        return "unverified", "ambiguous repeated observed cell/stage keys"
    pairs = a[keys + ["expression_row"]].merge(
        b[keys + ["expression_row"]], on=keys, suffixes=("_a", "_b")
    )
    if len(pairs) < 3:
        return "unverified", "fewer than three exact observed cell/stage matches"
    # A bounded deterministic audit establishes common saved units; the full
    # arrays retain their native values and remain individually traceable.
    if len(pairs) > 1024:
        pairs = pairs.iloc[np.linspace(0, len(pairs) - 1, 1024).astype(int)]
    xa = np.load(Path(first["artifact_dir"]) / "expression.npy", mmap_mode="r")
    xb = np.load(Path(second["artifact_dir"]) / "expression.npy", mmap_mode="r")
    ca, cb = [first["genes"].index(g) for g in common], [second["genes"].index(g) for g in common]
    va = np.asarray(xa[np.ix_(pairs["expression_row_a"].to_numpy(int), ca)], dtype=np.float32)
    vb = np.asarray(xb[np.ix_(pairs["expression_row_b"].to_numpy(int), cb)], dtype=np.float32)
    if np.allclose(va, vb, atol=2e-5, rtol=2e-5):
        return (
            "verified",
            f"{len(pairs)} exact observed cell/stage matches × {len(common)} exact genes agree in saved native units",
        )
    return (
        "conflict",
        f"matched observed values differ (max absolute difference {np.max(np.abs(va - vb)):.6g}); native units are not pooled",
    )


def _assign_scale_groups(artifacts):
    ledger = []
    for dataset in sorted({a["dataset"] for a in artifacts}):
        sources = [
            a
            for a in artifacts
            if a["dataset"] == dataset and "expression" in a.get("capabilities", [])
        ]
        observed = [a for a in sources if a.get("role") in ("GT", "REFERENCE")]
        observed.sort(
            key=lambda a: (
                a["scenario"] != "measured",
                a.get("alternate_observation", False),
                -a["n_total"],
            )
        )
        comparisons, clusters = {}, []
        for i, first in enumerate(observed):
            for second in observed[i + 1 :]:
                status, reason = _observed_scale_comparison(first, second)
                comparisons[frozenset([first["source_id"], second["source_id"]])] = status
                ledger.append(
                    {
                        "dataset": dataset,
                        "first": first["source_id"],
                        "second": second["source_id"],
                        "status": status,
                        "reason": reason,
                    }
                )
        for observation in observed:
            chosen = None
            for cluster in clusters:
                results = [
                    comparisons.get(
                        frozenset([observation["source_id"], member["source_id"]]), "unverified"
                    )
                    for member in cluster
                ]
                if "verified" in results and "conflict" not in results:
                    chosen = cluster
                    break
            if chosen is None:
                chosen = []
                clusters.append(chosen)
            chosen.append(observation)
        parent_groups, model_groups = {}, {}
        for i, cluster in enumerate(clusters):
            group = dataset + "__scale_" + str(i + 1)
            for observation in cluster:
                observation["scale_group"] = group
                parent = observation.get("parent_source_id")
                if parent:
                    parent_groups[parent] = group
                for parent in observation.get("matching_parent_sources", []):
                    parent_groups[parent] = group
        for source in sources:
            if source["source_id"] in parent_groups:
                source["scale_group"] = parent_groups[source["source_id"]]
                if source["scenario"] == "measured":
                    model_groups[(source["model"], source["panel"])] = (
                        source["scale_group"],
                        source.get("source_metadata", {}).get("normalization"),
                    )
        for source in sources:
            if "scale_group" not in source:
                prior = model_groups.get((source["model"], source["panel"]))
                normalization = source.get("source_metadata", {}).get("normalization")
                if prior and (not normalization or not prior[1] or normalization == prior[1]):
                    source["scale_group"] = prior[0]
                    source.setdefault("warning", []).append(
                        "Scenario scale follows the same native model/panel measured scale; no unobserved truth is claimed."
                    )
                else:
                    source["scale_group"] = dataset + "__unverified_" + source["source_id"]
                    source.setdefault("warning", []).append(
                        "Common expression units could not be verified from exact observed matches; source remains on its native axis."
                    )
        for source in sources:
            source["scale_policy"] = (
                "verified_observed_common_values"
                if len(clusters) == 1 and len(clusters[0]) > 1
                else "explicit_observed_scale_group"
            )
            if len(clusters) > 1:
                source.setdefault("warning", []).append(
                    "Distinct or unverified saved expression units use separate observed reference fits; compare coordinate panels only within the same reference_id."
                )
    return ledger


def _reference_groups(artifacts):
    """Separate native assays before grouping verified observed units.

    An overlap of HVG and author genes never licenses a shared reference fit.
    Each assay keeps its own GT axis and its own biological coordinate system.
    """
    expression = [a for a in artifacts if "expression" in a.get("capabilities", [])]
    groups = []
    datasets = {a["dataset"] for a in expression}
    for dataset in sorted(datasets):
        dataset_sources = [a for a in expression if a["dataset"] == dataset]
        for panel in sorted({a.get("panel", "unknown") for a in dataset_sources}):
            panel_sources = [a for a in dataset_sources if a.get("panel", "unknown") == panel]
            scale_groups = sorted({a.get("scale_group", dataset) for a in panel_sources})
            for scale in scale_groups:
                sources = [a for a in panel_sources if a.get("scale_group", dataset) == scale]
                axis = set(sources[0]["genes"])
                for source in sources[1:]:
                    axis.intersection_update(source["genes"])
                groups.append(
                    (
                        scale + "__" + panel + "__native",
                        sources,
                        sorted(axis),
                        len(scale_groups) > 1,
                    )
                )
    return groups


def _reference_memory_available():
    try:
        import psutil

        available = psutil.virtual_memory().available
    except ImportError:
        available = 4 * 1024**3
    try:
        maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if maximum != "max":
            current = int(Path("/sys/fs/cgroup/memory.current").read_text())
            stats = dict(
                line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines()
            )
            available = min(available, int(maximum) - current + int(stats.get("inactive_file", 0)))
    except (OSError, ValueError):
        pass
    return max(0, available)


def _read_small_table(path, columns=None):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        if columns:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(path).schema_arrow.names)
            columns = [column for column in columns if column in available]
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(
        path, sep="\t" if path.suffix.lower() in (".tsv", ".txt") else ",", usecols=columns
    )


def _original_reference(observed, axis, options):
    """Optional full observed atlas, validated against native saved truth units.

    Compressed SciPy NPZ cannot stream its sparse index/data arrays. Only load it
    when its uncompressed array size fits a conservative available-memory budget;
    the dense sample is bounded separately. A skipped atlas does not block plots.
    """
    from scipy import sparse
    import zipfile

    notes = []
    if options.get("original_reference", "auto") in (False, "off", "none"):
        return None, notes
    attempted = set()
    for source in observed:
        settings = source.get("reference_input", {})
        paths = tuple(settings.get(key, "") for key in ("x_path", "obs_path", "var_path"))
        if not all(paths) or paths in attempted:
            continue
        attempted.add(paths)
        if not all(Path(path).is_file() for path in paths):
            notes.append(
                "Original observed NPZ/obs/var unavailable; saved observed-reference fallback retained."
            )
            continue
        try:
            with zipfile.ZipFile(paths[0]) as archive:
                estimate = sum(item.file_size for item in archive.infolist())
            memory = _reference_memory_available()
            if options.get("memory_gb"):
                memory = min(memory, float(options["memory_gb"]) * 1024**3)
            if estimate * 2 > memory * 0.65:
                notes.append(
                    "Original observed sparse NPZ exceeds bounded preparation memory; saved observed-reference fallback retained."
                )
                continue
            var = _read_small_table(paths[2])
            gene_column = settings.get("gene_name_column")
            if not gene_column:
                gene_column = next(
                    (c for c in ("gene_name", "gene", "symbol", "gene_symbol") if c in var), None
                )
            names = (
                var[gene_column].astype(str).tolist()
                if gene_column and gene_column in var
                else var.index.astype(str).tolist()
            )
            x = sparse.load_npz(paths[0]).tocsr()
            if x.shape[1] != len(names):
                if (
                    "highly_variable" in var
                    and int(var["highly_variable"].astype(bool).sum()) == x.shape[1]
                ):
                    names = (
                        var.loc[var["highly_variable"].astype(bool), gene_column]
                        .astype(str)
                        .tolist()
                        if gene_column
                        else var.index[var["highly_variable"].astype(bool)].astype(str).tolist()
                    )
                else:
                    raise ValueError("Original NPZ and var gene axes do not align exactly")
            if len(set(names)) != len(names) or not set(axis).issubset(names):
                raise ValueError("Original reference lacks unique exact common gene identities")
            columns = [names.index(g) for g in axis]
            obs_columns = [
                settings.get(key)
                for key in (
                    "time_column",
                    "actual_time_column",
                    "state_column",
                    "celltype_column",
                    "obs_index_column",
                )
                if settings.get(key)
            ]
            obs = _read_small_table(paths[1], list(dict.fromkeys(obs_columns)) or None)
            if len(obs) != x.shape[0]:
                raise ValueError("Original NPZ and obs row axes differ")
            local = pd.read_parquet(Path(source["artifact_dir"]) / "cells.parquet")
            key = settings.get("obs_index_column")
            if key and key in obs:
                if obs[key].astype(str).duplicated().any():
                    raise ValueError("Original observed index labels are duplicated")
                mapping = pd.Series(np.arange(len(obs)), index=obs[key].astype(str))
                original_rows = mapping.reindex(local.cell_id.astype(str)).to_numpy(float)
            else:
                original_rows = pd.to_numeric(local.cell_id, errors="coerce").to_numpy(float)
            valid = (
                np.isfinite(original_rows)
                & (original_rows == np.floor(original_rows))
                & (original_rows >= 0)
                & (original_rows < len(obs))
            )
            checked = np.flatnonzero(valid)
            if len(checked) < 3:
                raise ValueError(
                    "Original atlas cannot verify at least three exact saved observed cell IDs"
                )
            checked = checked[np.linspace(0, len(checked) - 1, min(1024, len(checked))).astype(int)]
            original_values = x[original_rows[checked].astype(int)][:, columns].toarray()
            saved = np.load(Path(source["artifact_dir"]) / "expression.npy", mmap_mode="r")
            saved_columns = [source["genes"].index(g) for g in axis]
            saved_values = np.asarray(saved[np.ix_(checked, saved_columns)])
            if not np.allclose(original_values, saved_values, atol=2e-5, rtol=2e-5):
                raise ValueError(
                    "Original observed values disagree with matched saved native truth units"
                )
            time_column = settings.get("time_column")
            if not time_column or time_column not in obs:
                raise ValueError("Original observed atlas lacks its declared numeric time column")
            times = pd.to_numeric(obs[time_column], errors="raise").to_numpy(float)
            if not np.isfinite(times).all():
                raise ValueError("Original observed numeric stage coordinate is nonfinite")
            time_map = settings.get("time_map", {})
            if time_map:
                mapped = [
                    time_map.get(str(int(v)) if v.is_integer() else str(v), time_map.get(str(v)))
                    for v in times
                ]
                if all(v is not None for v in mapped):
                    times = np.asarray(mapped, dtype=float)
            state_column = settings.get("state_column") or settings.get("celltype_column")
            labels = (
                obs[state_column].fillna("unknown").astype(str).to_numpy()
                if state_column in obs
                else np.full(len(obs), "unknown")
            )
            label_column = settings.get("actual_time_column")
            time_labels = (
                obs[label_column].astype(str).to_numpy()
                if label_column in obs
                else np.asarray([f"stage {v:g}" for v in times])
            )
            cells = pd.DataFrame(
                {
                    "cell_id": (
                        obs[key].astype(str).to_numpy()
                        if key and key in obs
                        else np.arange(len(obs)).astype(str)
                    ),
                    "time": times,
                    "time_label": time_labels,
                    "state_supplied": labels,
                }
            )
            rows, _ = _sample_rows(
                cells, int(options.get("reference_cells", 20000)), int(options.get("seed", 1729))
            )
            matrix = x[rows][:, columns].toarray().astype(np.float32)
            if not np.isfinite(matrix).all():
                raise ValueError("Original observed reference sample contains nonfinite values")
            notes.append(
                f"Original observed atlas fit uses {len(rows)} stratified cells from {len(cells)} available rows; saved native units verified on {len(checked)} exact observed IDs × {len(axis)} genes."
            )
            return (
                cells.iloc[rows].reset_index(drop=True),
                matrix,
                "fixed_original_observed_atlas_reference; descriptive_target_inclusion_not_training_claim",
                {key: _file_stamp(path) for key, path in zip(("x", "obs", "var"), paths)},
            ), notes
        except Exception as error:
            notes.append(f"Original observed-reference fallback: {type(error).__name__}: {error}")
    return None, notes


def _fit_reference(key, sources, axis, separate_axis, options, cache_root):
    """Build PCA/optional UMAP from observations only and persist one fit."""
    import joblib
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    observed = [
        a
        for a in sources
        if a.get("role") == "GT"
        and a.get("scenario") == "measured"
        and not a.get("alternate_observation")
    ]
    policy = "fixed_measured_GT_reference; descriptive_target_included"
    if not observed:
        observed = [
            a
            for a in sources
            if a.get("role") in ("GT", "REFERENCE") and not a.get("alternate_observation")
        ]
        policy = "fixed_saved_observed_anchor_or_boundary_reference; descriptive"
    if len(axis) < 3 or not observed:
        reason = (
            "Fewer than three shared genes"
            if len(axis) < 3
            else "No saved observed reference available"
        )
        return None, reason
    fingerprint = _token(
        {
            "version": VERSION,
            "key": key,
            "axis": axis,
            "observed": [(a["source_id"], a["fingerprint"]) for a in observed],
            "original_reference": options.get("original_reference", "auto"),
            "original_stamps": [
                _file_stamp(p)
                for a in observed
                for p in a.get("reference_input", {}).values()
                if isinstance(p, str) and Path(p).is_file()
            ],
            "embedding": options.get("embedding", "pca"),
            "reference_cells": int(options.get("reference_cells", 20000)),
            "reference_knn_cells": int(options.get("reference_knn_cells", 5000)),
            "seed": int(options.get("seed", 1729)),
        }
    )
    reference_dir = Path(cache_root) / "references" / (key + "_" + fingerprint[:14])
    reference_path = reference_dir / "reference.joblib"
    if reference_path.is_file():
        return joblib.load(reference_path), None
    reference_dir.mkdir(parents=True, exist_ok=True)
    frames, matrices = [], []
    for source in observed:
        folder = Path(source["artifact_dir"])
        cells = pd.read_parquet(folder / "cells.parquet")
        columns = [source["genes"].index(g) for g in axis]
        array = np.load(folder / "expression.npy", mmap_mode="r")
        frames.append(cells[["cell_id", "time", "time_label", "state_supplied"]])
        matrices.append(np.asarray(array[:, columns], dtype=np.float32))
    cells = pd.concat(frames, ignore_index=True)
    matrix = np.concatenate(matrices)
    # Native HVG/author GT can describe the same observed targets. Count each
    # exact cell/time once in the reference fit, keeping the deterministic first.
    # This affects the descriptive reference only; native truths remain saved.
    keep = ~cells.duplicated(["cell_id", "time"], keep="first")
    cells, matrix = cells.loc[keep].reset_index(drop=True), matrix[keep.to_numpy()]
    rows, _ = _sample_rows(
        cells, int(options.get("reference_cells", 20000)), int(options.get("seed", 1729))
    )
    cells, matrix = cells.iloc[rows].reset_index(drop=True), matrix[rows]
    original, original_notes = _original_reference(observed, axis, options)
    original_stamps = None
    if original is not None:
        cells, matrix, policy, original_stamps = original
    if len(matrix) < 3:
        return None, "Fewer than three observed cells in fixed reference"
    n_components = min(30, len(axis) - 1, len(matrix) - 1)
    with threadpool_limits(limits=1):
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=int(options.get("seed", 1729)),
        )
        pc = pca.fit_transform(matrix)
    embedding_method = "fixed_reference_pca"
    coordinates = pc[:, :2]
    umap_fit = None
    umap_warning = None
    if options.get("embedding", "pca") == "umap":
        try:
            import umap

            with threadpool_limits(limits=1):
                umap_fit = umap.UMAP(
                    n_neighbors=min(30, len(pc) - 1),
                    min_dist=0.3,
                    n_components=2,
                    random_state=int(options.get("seed", 1729)),
                    transform_seed=int(options.get("seed", 1729)),
                    n_jobs=1,
                )
                coordinates = umap_fit.fit_transform(pc)
            embedding_method = "fixed_reference_umap"
        except ImportError as error:
            umap_warning = f"Requested UMAP dependency unavailable; honest PCA fallback: {error}"
    known = ~cells["state_supplied"].str.lower().isin(_UNKNOWN)
    labelled_cells = cells.loc[known].reset_index(drop=True)
    labelled_pc = pc[known.to_numpy()]
    if len(labelled_cells):
        state_rows, _ = _sample_rows(
            labelled_cells,
            int(options.get("reference_knn_cells", 5000)),
            int(options.get("seed", 1729)),
        )
        labelled_pc = labelled_pc[state_rows]
        labels = labelled_cells.iloc[state_rows]["state_supplied"].astype(str).to_numpy()
    else:
        labelled_pc = np.empty((0, n_components), dtype=np.float32)
        labels = np.asarray([], dtype=str)
    cells["x"], cells["y"] = coordinates[:, 0], coordinates[:, 1]
    _atomic_parquet(cells, reference_dir / "reference_cells.parquet")
    reference_pc_temporary = reference_dir / ("reference_pc." + uuid.uuid4().hex + ".tmp.npy")
    np.save(reference_pc_temporary, pc.astype(np.float32))
    reference_pc_temporary.replace(reference_dir / "reference_pc.npy")
    reference = {
        "key": key,
        "fingerprint": fingerprint,
        "reference_dir": str(reference_dir.resolve()),
        "genes": axis,
        "pca": pca,
        "umap": umap_fit,
        "embedding_method": embedding_method,
        "reference_policy": policy,
        "separate_axis": separate_axis,
        "n_reference": len(matrix),
        "labelled_pc": labelled_pc.astype(np.float32),
        "labels": labels,
        "pc": pc.astype(np.float32),
        "warning": umap_warning,
        "observed_sources": [a["source_id"] for a in observed],
        "original_reference_notes": original_notes,
        "original_reference_stamps": original_stamps,
    }
    reference_temporary = reference_dir / ("reference." + uuid.uuid4().hex + ".tmp.joblib")
    joblib.dump(reference, reference_temporary)
    reference_temporary.replace(reference_path)
    _write_json(
        reference_dir / "metadata.json",
        {
            k: v
            for k, v in reference.items()
            if k not in ("pca", "umap", "labelled_pc", "labels", "pc")
        },
    )
    return reference, None


def _atomic_parquet(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _link_immutable(source, destination):
    """Publish an immutable native file without exposing an incomplete copy."""
    source, destination = Path(source), Path(destination)
    if destination.is_file():
        return
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _clear_projection(meta, warning):
    """Return native availability in the index without mutating cached files."""
    # Native cell tables already have unprojected coordinates. Completed
    # projection artifacts are immutable and are never cleared in place.
    for key in (
        "reference_id",
        "reference_genes",
        "reference_artifact_dir",
        "n_reference",
        "projection_fingerprint",
    ):
        meta.pop(key, None)
    meta["embedding_method"], meta["reference_policy"] = "none", "unavailable"
    meta["capabilities"] = [
        c
        for c in meta.get("capabilities", [])
        if c not in ("embedding", "reference_distance", "state_diagnostic")
    ]
    meta.setdefault("warning", []).append(warning)
    return meta


def _project_task(meta, reference_path, options):
    import joblib
    from sklearn.neighbors import NearestNeighbors
    from threadpoolctl import threadpool_limits

    native_meta = json.loads(json.dumps(_jsonable(meta)))
    try:
        reference = joblib.load(reference_path)
        projection_fingerprint = _token(
            {
                "reference": reference["fingerprint"],
                "native": meta["fingerprint"],
                "version": VERSION,
            }
        )
        native_folder = Path(meta["artifact_dir"])
        folder = (
            native_folder.parent.parent
            / "projected"
            / (
                str(meta["source_id"])
                + "_"
                + meta["fingerprint"][:10]
                + "_"
                + reference["fingerprint"][:10]
                + "_"
                + projection_fingerprint[:10]
            )
        )
        metadata_path = folder / "metadata.json"
        if metadata_path.is_file():
            cached = json.loads(metadata_path.read_text())
            required = [
                folder / name
                for name in (
                    "cells.parquet",
                    "expression.npy",
                    "stage_gene.parquet",
                    "state_gene.parquet",
                )
            ]
            if (
                cached.get("projection_complete")
                and cached.get("projection_fingerprint") == projection_fingerprint
                and all(path.is_file() for path in required)
            ):
                return cached, "reused"
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("expression.npy", "stage_gene.parquet", "state_gene.parquet"):
            _link_immutable(native_folder / name, folder / name)
        cells = pd.read_parquet(native_folder / "cells.parquet")
        expression = np.load(native_folder / "expression.npy", mmap_mode="r")
        columns = [meta["genes"].index(g) for g in reference["genes"]]
        matrix = np.asarray(expression[:, columns], dtype=np.float32)
        with threadpool_limits(limits=1):
            pc = reference["pca"].transform(matrix)
            coordinates = (
                reference["umap"].transform(pc) if reference["umap"] is not None else pc[:, :2]
            )
            # Distance to an observed cell is a descriptive novelty diagnostic,
            # not proof that a generated state is biologically valid.
            nearest = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1).fit(
                reference["pc"]
            )
            distance = np.empty(len(pc), dtype=np.float32)
            for start in range(0, len(pc), 2000):
                distance[start : start + 2000] = nearest.kneighbors(pc[start : start + 2000])[0][
                    :, 0
                ]
            cells["reference_distance"] = distance
            labelled_pc, labels = reference["labelled_pc"], reference["labels"]
            if len(labels):
                n_neighbors = min(5, len(labels))
                neighbors = NearestNeighbors(
                    n_neighbors=n_neighbors, algorithm="brute", n_jobs=1
                ).fit(labelled_pc)
                inferred, confidence = [], []
                for start in range(0, len(pc), 2000):
                    indices = neighbors.kneighbors(pc[start : start + 2000], return_distance=False)
                    for near in labels[indices]:
                        values, counts = np.unique(near, return_counts=True)
                        winner = int(np.argmax(counts))
                        inferred.append(values[winner])
                        confidence.append(float(counts[winner] / len(near)))
                cells["state_inferred"] = inferred
                cells["state_transfer_confidence"] = confidence
                # GT keeps its independently supplied observed annotation. Model
                # panels use expression-transferred states; supplied labels stay.
                cells["state"] = (
                    cells["state_supplied"]
                    if meta["role"] in ("GT", "REFERENCE")
                    else cells["state_inferred"]
                )
                meta["state_method"] = (
                    "observed_annotation"
                    if meta["role"] in ("GT", "REFERENCE")
                    else "fixed_observed_reference_knn_diagnostic"
                )
                meta["capabilities"] = sorted(
                    set(meta["capabilities"] + ["state_diagnostic", "state"])
                )
            else:
                meta.setdefault("warning", []).append(
                    "Observed reference lacks state labels; supplied labels remain descriptive, not predicted fate."
                )
        cells["x"], cells["y"] = coordinates[:, 0], coordinates[:, 1]
        # Store PC coordinates for distance/connectivity/program trajectory
        # diagnostics; the 2D display alone is never a lineage measurement.
        for j in range(min(10, pc.shape[1])):
            cells[f"pc{j + 1}"] = pc[:, j]
        _atomic_parquet(cells, folder / "cells.parquet")
        meta.update(
            {
                "artifact_dir": str(folder.resolve()),
                "native_artifact_dir": str(native_folder.resolve()),
                "summary_files": {
                    "stage": str(folder / "stage_gene.parquet"),
                    "state": str(folder / "state_gene.parquet"),
                },
                "projection_complete": True,
                "projection_fingerprint": projection_fingerprint,
                "reference_id": reference["key"] + "_" + reference["fingerprint"][:14],
                "reference_genes": reference["genes"],
                "n_reference": reference["n_reference"],
                "reference_artifact_dir": reference["reference_dir"],
                "reference_policy": reference["reference_policy"],
                "embedding_method": reference["embedding_method"],
                "capabilities": sorted(
                    set(meta["capabilities"] + ["embedding", "reference_distance"])
                ),
            }
        )
        if reference["separate_axis"]:
            warning = "Native-panel-specific observed fit: coordinates are comparable only within the same reference_id; do not overlay different reference axes."
            if warning not in meta["warning"]:
                meta["warning"].append(warning)
        meta["warning"].extend(reference.get("original_reference_notes", []))
        if reference["warning"]:
            meta["warning"].append(reference["warning"])
        meta["warning"] = list(dict.fromkeys(meta.get("warning", [])))
        _write_json(folder / "metadata.json", meta)
        return meta, "projected"
    except Exception as error:
        _clear_projection(
            native_meta,
            f"Projection unavailable: {type(error).__name__}: {error}; native gene views remain available.",
        )
        return native_meta, "projection_error"


def _notify(progress, event):
    if progress is not None:
        progress(event)


def prepare(config, options=None, progress=None):
    """Prepare available sources; return and persist the complete index.

    A failed source is recorded and does not remove other models or biological
    families. Raw files and previous trial outputs are never modified.
    """
    from .discovery import discover, FM_MODELS, TRANS_MODELS
    from .raw import truth_sources

    options = dict(options or {})
    cache_root = (
        Path(options.get("cache_root") or config.get("cache_root") or "biology_cache")
        .expanduser()
        .resolve()
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    discovered = discover(config)
    sources = discovered["sources"]
    config = json.loads(json.dumps(_jsonable(config)))
    for source in sources:
        settings = config.setdefault("datasets", {}).setdefault(
            source.get("dataset", "unknown"), {}
        )
        for key, path in source.get("annotations", {}).items():
            settings.setdefault(key, path)
    dataset_filter = options.get("dataset", "all")
    if dataset_filter not in (None, "all"):
        allowed = set(
            dataset_filter
            if isinstance(dataset_filter, (list, tuple))
            else str(dataset_filter).split(",")
        )
        sources = [s for s in sources if s.get("dataset") in allowed]
    scenario_filter = options.get("scenarios")
    if scenario_filter:
        scenarios = set(
            scenario_filter
            if isinstance(scenario_filter, (list, tuple))
            else str(scenario_filter).split(",")
        )
        # Measured observations also support future/history reference fits.
        scenarios.add("measured")
        sources = [s for s in sources if s.get("scenario") in scenarios]
    errors = list(discovered["errors"])
    expanded = list(sources)
    for source in sources:
        if source.get("scenario") == "perturbation":
            continue
        try:
            expanded.extend(truth_sources(source))
        except Exception as error:
            errors.append(
                {
                    "source_id": source["source_id"],
                    "phase": "observed_source",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    # Native row metadata and requested original obs columns matter more than
    # the fixed-sized expression chunks on very large datasets.
    largest = max((s.get("n_cells", 0) for s in expanded), default=0)
    largest_genes = max((s.get("n_genes", 0) for s in expanded), default=200)
    observation_rows, checked_paths = 0, set()
    for source in expanded:
        settings = {
            **source.get("dataset_config", {}),
            **config.get("datasets", {}).get(source.get("dataset"), {}),
        }
        obs_path = settings.get("obs_path")
        if (
            obs_path
            and obs_path not in checked_paths
            and Path(obs_path).is_file()
            and str(obs_path).endswith(".parquet")
        ):
            checked_paths.add(obs_path)
            try:
                import pyarrow.parquet as pq

                observation_rows += pq.ParquetFile(obs_path).metadata.num_rows
            except Exception:
                pass
    options["_estimated_native_gb_per_worker"] = (
        0.35
        + largest * 220 / 1024**3
        + observation_rows * 100 / 1024**3
        + int(options.get("chunk_size", 65536)) * largest_genes * 8 / 1024**3
    )
    workers = _resources(options, max(1, len(expanded)))
    artifacts = []
    _notify(
        progress,
        {
            "phase": "native",
            "done": 0,
            "total": len(expanded),
            "workers": workers,
            "detail": "Stream summaries once per source; figures reuse these caches.",
        },
    )
    if workers == 1:
        iterator = (
            (source, _native_task(source, config, options, str(cache_root))) for source in expanded
        )
        for done, (source, (result, status)) in enumerate(iterator, 1):
            (errors if status == "error" else artifacts).append(result)
            _notify(
                progress,
                {
                    "phase": "native",
                    "done": done,
                    "total": len(expanded),
                    "source_id": source["source_id"],
                    "status": status,
                },
            )
    else:
        with process_pool(workers) as pool:
            pending = {
                pool.submit(_native_task, source, config, options, str(cache_root)): source
                for source in expanded
            }
            for done, future in enumerate(as_completed(pending), 1):
                source = pending[future]
                try:
                    result, status = future.result()
                except Exception as error:
                    result, status = {
                        **_source_meta(source),
                        "error": f"WorkerFailure: {error}",
                    }, "error"
                (errors if status == "error" else artifacts).append(result)
                _notify(
                    progress,
                    {
                        "phase": "native",
                        "done": done,
                        "total": len(expanded),
                        "source_id": source["source_id"],
                        "status": status,
                    },
                )
    artifacts = _deduplicate_observed(artifacts)
    scale_ledger = _assign_scale_groups(artifacts)
    references, projection_jobs = [], []
    for key, group, axis, separate in _reference_groups(artifacts):
        try:
            reference, reason = _fit_reference(key, group, axis, separate, options, str(cache_root))
        except Exception as error:
            reference, reason = None, f"{type(error).__name__}: {error}"
        if reference is None:
            for meta in group:
                _clear_projection(
                    meta,
                    f"Embedding unavailable: {reason}; stage, state and gene biology remains available.",
                )
            errors.append({"phase": "reference", "reference_id": key, "error": reason})
        else:
            references.append(
                {
                    k: v
                    for k, v in reference.items()
                    if k not in ("pca", "umap", "labelled_pc", "labels", "pc")
                }
            )
            projection_jobs.extend(
                (meta, str(Path(reference["reference_dir"]) / "reference.joblib")) for meta in group
            )
    projected = {a["source_id"]: a for a in artifacts}
    _notify(
        progress,
        {
            "phase": "projection",
            "done": 0,
            "total": len(projection_jobs),
            "detail": "Project saved cells on one observed reference; generated cells never fit coordinates.",
        },
    )
    if workers == 1:
        iterator = (
            (meta, _project_task(meta, reference_path, options))
            for meta, reference_path in projection_jobs
        )
        for done, (meta, (result, status)) in enumerate(iterator, 1):
            projected[meta["source_id"]] = result
            _notify(
                progress,
                {
                    "phase": "projection",
                    "done": done,
                    "total": len(projection_jobs),
                    "source_id": meta["source_id"],
                    "status": status,
                },
            )
    elif projection_jobs:
        with process_pool(workers) as pool:
            pending = {
                pool.submit(_project_task, meta, reference_path, options): meta
                for meta, reference_path in projection_jobs
            }
            for done, future in enumerate(as_completed(pending), 1):
                meta = pending[future]
                try:
                    result, status = future.result()
                except Exception as error:
                    result, status = meta, "projection_error"
                    _clear_projection(
                        result,
                        f"Projection worker failed: {error}; native biology remains available.",
                    )
                projected[meta["source_id"]] = result
                _notify(
                    progress,
                    {
                        "phase": "projection",
                        "done": done,
                        "total": len(projection_jobs),
                        "source_id": meta["source_id"],
                        "status": status,
                    },
                )
    for meta in projected.values():
        meta["warning"] = list(dict.fromkeys(meta.get("warning", [])))
        for name in ("matching_parent_sources", "matching_observation_sources"):
            if name in meta:
                meta[name] = list(dict.fromkeys(meta[name]))
    artifacts = sorted(
        projected.values(),
        key=lambda a: (a["dataset"], a["scenario"], a.get("role", ""), a["model"], a["source_id"]),
    )
    cohorts = []
    for dataset, scenario in sorted({(a["dataset"], a["scenario"]) for a in artifacts}):
        cohort_sources = [
            a for a in artifacts if a["dataset"] == dataset and a["scenario"] == scenario
        ]
        settings = config.get("datasets", {}).get(dataset, {})
        cohort = {
            "key": dataset + "__" + scenario,
            "dataset": dataset,
            "scenario": scenario,
            "sources": cohort_sources,
            "configuration": settings,
            **{
                k: settings[k]
                for k in ("author_genes_path", "programs_path", "edges_path", "species")
                if settings.get(k)
            },
        }
        for panel in ("hvg", "author"):
            actual_fm = {
                a["model"] for a in cohort_sources if a["role"] == "FM" and a["panel"] == panel
            }
            expected_fm = [m for m in FM_MODELS if ("author" in m) == (panel == "author")]
            actual_trans = {
                a["model"] for a in cohort_sources if a["role"] == "TRANS" and a["panel"] == panel
            }
            cohort.setdefault("model_coverage", {})[panel] = {
                "missing_fm": [m for m in expected_fm if m not in actual_fm],
                "missing_trans": (
                    [m for m in TRANS_MODELS if m not in actual_trans]
                    if scenario == "measured"
                    else []
                ),
                "note": "Unavailable models are explicit N/A; completeness does not gate available biology.",
            }
        cohorts.append(cohort)
    index = {
        "schema_version": 1,
        "preparer_version": VERSION,
        "cohorts": cohorts,
        "sources": artifacts,
        "references": references,
        "configuration": config,
        "errors": errors,
        "scale_verification": scale_ledger,
        "discovery_excluded": discovered.get("excluded", 0),
        "discovery_exclusions": discovered.get("exclusions", []),
        "preparation": {
            "workers": workers,
            "sample_cells": int(options.get("sample_cells", 20000)),
            "analysis_cap": int(options.get("analysis_cap", 0)),
            "embedding": options.get("embedding", "pca"),
            "estimated_native_gb_per_worker": options["_estimated_native_gb_per_worker"],
        },
        "index_path": str(cache_root / "index.json"),
    }
    _write_json(cache_root / "index.json", index)
    return index
