"""Validated row metadata and streamed saved expression for biology plotting.

Developmental ordinal time is retained unless an explicitly attested reporting
axis is available. Expression is streamed from saved prediction files.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import pandas as pd


def _decode(values: Any) -> list[str]:
    return [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in np.asarray(values).reshape(-1)
    ]


def _attrs(handle: h5py.File) -> dict[str, Any]:
    def plain(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if hasattr(value, "tolist"):
            return plain(value.tolist())
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    return {str(k): plain(v) for k, v in handle.attrs.items()}


def _vector(handle: h5py.File, names: list[str], n: int, *, required=False) -> np.ndarray | None:
    for key in names:
        if key in handle:
            values = np.asarray(handle[key])
            if values.shape != (n,):
                raise ValueError(f"{key} shape {values.shape} is not row-aligned ({n},)")
            return values
    if required:
        raise ValueError(f"missing row vector: {names}")
    return None


def _config_for(source: dict[str, Any], config: dict[str, Any] | None) -> dict[str, Any]:
    saved = dict(source.get("dataset_config") or {})
    supplied = (config or {}).get("datasets", {}).get(source.get("dataset"), {})
    return {**saved, **supplied}


@lru_cache(maxsize=2)
def _read_obs_cached(path: str, columns: tuple[str, ...], modified: int, size: int) -> pd.DataFrame:
    """Read only required metadata columns; reused for prediction and its GT."""
    local = Path(path)
    if local.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        names = pq.read_schema(local).names
        selected = [c for c in columns if c in names]
        return pd.read_parquet(local, columns=selected or names[:1])
    requested = set(columns)
    return pd.read_csv(
        local,
        sep="\t" if local.suffix.lower() in {".tsv", ".txt"} else ",",
        usecols=(lambda c: c in requested) if requested else None,
    )


def _load_obs(source: dict[str, Any], settings: dict[str, Any]) -> pd.DataFrame | None:
    obs_path = settings.get("obs_path")
    if not obs_path:
        return None
    path = Path(obs_path).expanduser()
    if not path.is_absolute() and source.get("dataset_config_path"):
        path = Path(source["dataset_config_path"]).parent / path
    if not path.is_file():
        return None
    columns = tuple(
        sorted(
            set(
                str(v)
                for v in [
                    settings.get("obs_index_column"),
                    settings.get("state_column") or settings.get("celltype_column"),
                    settings.get("time_column"),
                    settings.get("stage_label_column") or settings.get("actual_time_column"),
                    *settings.get(
                        "annotation_columns", settings.get("additional_classification_columns", [])
                    ),
                ]
                if v
            )
        )
    )
    stat = path.stat()
    return _read_obs_cached(str(path), columns, stat.st_mtime_ns, stat.st_size)


def _attach_obs(
    cells: pd.DataFrame, source: dict[str, Any], settings: dict[str, Any], notes: list[str]
) -> None:
    obs = _load_obs(source, settings)
    if obs is None:
        if settings.get("obs_path"):
            notes.append(
                "Original obs file unavailable; saved H5 labels and biological stages retained."
            )
        return
    key = settings.get("obs_index_column")
    if key:
        if key not in obs or obs[key].astype(str).duplicated().any():
            raise ValueError(f"obs index column {key!r} is missing or duplicated")
        selected = obs.set_index(obs[key].astype(str)).reindex(cells["cell_id"].astype(str))
        found = cells["cell_id"].astype(str).isin(obs[key].astype(str)).to_numpy()
    else:
        indices = pd.to_numeric(cells["cell_id"], errors="coerce").to_numpy(float)
        found = (
            np.isfinite(indices)
            & (indices == np.floor(indices))
            & (indices >= 0)
            & (indices < len(obs))
        )
        selected = pd.DataFrame(index=np.arange(len(cells)), columns=obs.columns)
        if found.any():
            selected.loc[found] = obs.iloc[indices[found].astype(int)].to_numpy()
    if not found.any():
        notes.append(
            "Original obs could not join the saved target/template indices; no positional label substitution made."
        )
        return
    state_column = settings.get("state_column") or settings.get("celltype_column")
    if state_column in selected:
        labels = selected[state_column].astype("string").fillna("unknown").to_numpy(str)
        state_values = cells["state_supplied"].astype(str)
        coded = state_values.str.startswith("type_id:").to_numpy() & found
        if coded.any():
            mapping = (
                pd.DataFrame({"code": state_values.to_numpy()[coded], "label": labels[coded]})
                .groupby("code")["label"]
                .nunique()
            )
            if (mapping > 1).any():
                raise ValueError(
                    "Original obs exact-ID join maps one saved numeric state code to multiple state names"
                )
            notes.append(
                "Decoded saved numeric state codes using original obs labels joined by exact source-cell IDs."
            )
        known = (
            (state_values != "unknown") & ~state_values.str.startswith("type_id:")
        ).to_numpy() & found
        if np.any(known & (cells["state_supplied"].to_numpy(str) != labels)):
            raise ValueError(
                "Saved H5 cell-state labels conflict with original obs joined by exact target/template IDs"
            )
        cells.loc[found, "state_supplied"] = labels[found]
    stage_label_column = settings.get("stage_label_column") or settings.get("actual_time_column")
    if stage_label_column in selected and source.get("scenario") in {"measured", "reverse"}:
        labels = selected[stage_label_column].astype("string").fillna("unknown").to_numpy(str)
        cells.loc[found, "time_label"] = labels[found]
    for column in settings.get(
        "annotation_columns", settings.get("additional_classification_columns", [])
    ):
        if column in selected and column not in cells:
            cells[column] = selected[column].astype("string").fillna("unknown").to_numpy(str)
    notes.append(
        f"Joined original obs to {int(found.sum())}/{len(cells)} exact saved target/template IDs."
    )


def _validate_rollout(
    handle: h5py.File, cells: pd.DataFrame, scenario: str, attrs: dict[str, Any], notes: list[str]
) -> None:
    if "step" not in cells:
        raise ValueError(f"{scenario} lacks a step vector")
    steps = cells["step"].to_numpy()
    unique = np.unique(steps)
    if not np.array_equal(unique, np.arange(1, int(unique[-1]) + 1)):
        raise ValueError("Rollout does not contain a consecutive complete finite prefix")
    if "reference_expression" not in handle:
        raise ValueError("Rollout is missing its measured boundary reference")
    r = handle["reference_expression"].shape[0]
    if r <= 0 or len(cells) != r * len(unique) or not np.array_equal(steps, np.repeat(unique, r)):
        raise ValueError("Rollout step/reference blocks are incomplete or misordered")
    ids = cells["cell_id"].to_numpy().reshape(len(unique), r)
    if not np.all(ids == ids[:1]):
        raise ValueError("Rollout template IDs change across steps")
    times = cells["time"].to_numpy().reshape(len(unique), r)
    if not np.allclose(times, times[:, :1], rtol=0, atol=1e-6):
        raise ValueError("Rollout times differ within a step")
    differences = np.diff(times[:, 0])
    if (scenario == "future" and np.any(differences <= 0)) or (
        scenario == "history" and np.any(differences >= 0)
    ):
        raise ValueError("Rollout reporting times have wrong developmental direction")
    completed = int(attrs.get("rollout_completed_steps", attrs.get("rollout_steps", len(unique))))
    requested = int(attrs.get("rollout_steps_requested_effective", completed))
    if completed != len(unique) or requested < completed:
        raise ValueError("Rollout horizon declarations disagree with retained complete blocks")
    if requested > completed or bool(attrs.get("rollout_terminated_early", False)):
        notes.append(
            f"Finite prefix: {completed}/{requested} requested steps retained; no omitted/nonfinite steps plotted."
        )
    if attrs.get("evaluation_view_schema"):
        notes.append(
            "Explicit numerical-guard evaluation view: retained steps satisfy a declared magnitude heuristic; this does not validate biological plausibility or the excluded horizon."
        )
    notes.append(
        "Fixed boundary template identities and supplied template types; paths are model template paths, not tracked biological cell lineages. No unobserved GT."
    )


def read_source(
    source: dict[str, Any], config: dict[str, Any] | None = None, chunk_size: int | None = None
) -> dict[str, Any]:
    """Load only genes and row metadata. Use iter_expression for matrix chunks."""
    if source.get("format") == "h5ad" or str(source["path"]).endswith(".h5ad"):
        return _read_h5ad(source, config)
    settings = _config_for(source, config)
    notes: list[str] = []
    with h5py.File(source["path"], "r") as handle:
        attrs = _attrs(handle)
        matrix_key = source.get("matrix_key", "prediction_expression")
        matrix = handle[matrix_key]
        n, g = matrix.shape
        genes = _decode(handle[source.get("genes_key", "gene_names")])
        if len(genes) != g or len(set(genes)) != g or any(not x.strip() for x in genes):
            raise ValueError("Gene names must be unique and match the expression axis")
        scenario = source.get("scenario", "measured")
        population = source.get("population", "prediction")
        time_key = source.get("time_key")
        if not time_key:
            time_key = (
                "prediction_reporting_time"
                if "prediction_reporting_time" in handle
                else "prediction_time"
            )
        times = _vector(handle, [time_key], n, required=True).astype(float)
        if not np.isfinite(times).all():
            raise ValueError("Saved time contains nonfinite values")
        if population == "truth":
            id_names, type_names = ["truth_cell_index", "target_cell_index"], ["truth_type_id"]
        elif population == "reference":
            id_names, type_names = ["reference_cell_index"], ["reference_type_id"]
        else:
            id_names = ["prediction_cell_index", "template_cell_index", "target_cell_index"]
            type_names = [
                "predicted_type_id",
                "prediction_type_id",
                "conditioned_target_type_id",
                "template_type_id",
                "truth_type_id",
            ]
        ids = _vector(handle, id_names, n)
        if ids is None and population == "reference" and "target_cell_index" in handle:
            ids = np.asarray(handle["target_cell_index"][:n])
        if ids is None:
            ids = np.asarray([f"saved_row_{i}" for i in range(n)], dtype=object)
            notes.append(
                "No original cell IDs saved; source-local row IDs used, never paired across models."
            )
        elif ids.dtype.kind in "iuf" and (
            not np.isfinite(ids).all() or np.any(ids < 0) or np.any(ids != np.floor(ids))
        ):
            raise ValueError(
                "Numeric original cell/template IDs must be finite nonnegative integers"
            )
        labels = _vector(handle, type_names, n)
        vocabulary = _decode(handle["celltype_names"]) if "celltype_names" in handle else []
        if labels is not None and labels.dtype.kind in "iu" and vocabulary:
            if np.any(labels < 0) or np.any(labels >= len(vocabulary)):
                raise ValueError("Saved cell-state IDs are outside their vocabulary")
            supplied = np.asarray(vocabulary, dtype=object)[labels.astype(int)]
        elif labels is not None and labels.dtype.kind in "OSU":
            supplied = np.asarray(_decode(labels), dtype=object)
        elif labels is not None and labels.dtype.kind in "iu":
            if np.any(labels < 0):
                raise ValueError("Saved numeric state codes must be nonnegative")
            supplied = np.asarray([f"type_id:{int(value)}" for value in labels], dtype=object)
            notes.append(
                "Saved numeric state codes have no vocabulary; labels are displayed as type_id:<code>, with no anatomical names guessed."
            )
        else:
            supplied = np.full(n, "unknown", dtype=object)
        time_kind = source.get("time_kind", "ordinal")
        map_values = settings.get("time_map", {})
        original_times = times.copy()
        if time_kind == "ordinal" and map_values:

            def lookup(v):
                return map_values.get(
                    str(int(v)) if float(v).is_integer() else str(v), map_values.get(str(v))
                )

            mapped = [lookup(v) for v in times]
            if all(v is not None for v in mapped):
                times = np.asarray(mapped, dtype=float)
                if not np.isfinite(times).all():
                    raise ValueError("Configured time_map contains nonfinite values")
                ordered_stages = np.unique(original_times)
                ordered_axis = np.asarray(
                    [times[np.flatnonzero(original_times == stage)[0]] for stage in ordered_stages]
                )
                if np.any(np.diff(ordered_axis) < 0):
                    raise ValueError(
                        "Configured reporting time_map decreases in biological ordinal order; E/P phase reset needs an explicit valid convention"
                    )
                time_kind = "configured_reporting"
                notes.append(
                    "Reporting time uses explicit dataset time_map; model stage IDs retained separately."
                )
        time_labels = [f"stage {v:g}" if time_kind == "ordinal" else f"{v:g}" for v in times]
        cells = pd.DataFrame(
            {
                "cell_id": _decode(ids),
                "time": times,
                "time_label": time_labels,
                "state_supplied": supplied,
                "row_index": np.arange(n, dtype=np.int64),
            }
        )
        if labels is not None and labels.dtype.kind in "iu":
            cells["state_code"] = labels
        ordinal = _vector(
            handle,
            [
                (
                    "reference_time"
                    if population == "reference"
                    else "truth_time" if population == "truth" else "prediction_time"
                )
            ],
            n,
        )
        if ordinal is not None:
            cells["stage_ordinal"] = ordinal
        step = (
            None
            if population == "reference"
            else _vector(handle, ["history_step", "future_step"], n)
        )
        if step is not None:
            if step.dtype.kind not in "iu":
                raise ValueError("Rollout steps must be integers")
            cells["step"] = step
        generated = (
            None if population == "reference" else _vector(handle, ["prediction_is_generated"], n)
        )
        if generated is not None:
            cells["is_generated"] = generated.astype(bool)
        for key in ("source_anchor_ordinal", "right_endpoint_ordinal", "target_time_fraction"):
            vector = _vector(handle, [key], n)
            if vector is not None:
                cells[key] = vector
        if (
            scenario in {"measured", "reverse"}
            and population == "prediction"
            and "truth_expression" in handle
        ):
            if handle["truth_expression"].shape != matrix.shape:
                raise ValueError("Measured truth/prediction expression shapes differ")
            pids = _vector(handle, ["prediction_cell_index"], n)
            tids = _vector(handle, ["truth_cell_index"], n)
            if attrs.get("schema_version") == "v4.canonical_prediction.1" and (
                pids is None or tids is None
            ):
                raise ValueError(
                    "Canonical measured prediction requires exact prediction/truth target-cell indices"
                )
            if pids is not None and tids is not None and not np.array_equal(pids, tids):
                raise ValueError("Measured prediction/truth cell IDs are not matched")
            pt = _vector(handle, ["prediction_time"], n)
            tt = _vector(handle, ["truth_time"], n)
            if pt is not None and tt is not None and not np.array_equal(pt, tt):
                raise ValueError("Measured prediction/truth biological stages are not matched")
            if (
                pids is not None
                and pt is not None
                and pd.DataFrame({"cell": pids, "stage": pt}).duplicated().any()
            ):
                raise ValueError(
                    "Measured prediction has duplicated target-cell/stage identities; duplicate aggregation is unresolved"
                )
        if scenario in {"future", "history"} and population == "prediction":
            _validate_rollout(handle, cells, scenario, attrs, notes)
        if scenario == "interpolation":
            if generated is None:
                raise ValueError(
                    "Interpolation requires explicit stored anchor/generated row flags"
                )
            if population == "prediction" and not np.any(generated):
                raise ValueError("Interpolation contains no generated queries")
            if generated.dtype.kind != "b":
                raise ValueError("Interpolation anchor/generated flags must be Boolean")
            if all(
                key in cells
                for key in (
                    "source_anchor_ordinal",
                    "right_endpoint_ordinal",
                    "target_time_fraction",
                )
            ):
                left = cells["source_anchor_ordinal"].to_numpy()
                right = cells["right_endpoint_ordinal"].to_numpy()
                fractions = cells["target_time_fraction"].to_numpy(float)
                if (
                    np.any(left < 0)
                    or np.any(right < 0)
                    or not np.all(right[generated] == left[generated] + 1)
                ):
                    raise ValueError(
                        "Interpolation generated rows lack adjacent biological endpoints"
                    )
                if (
                    not np.all(left[~generated] == right[~generated])
                    or not np.isnan(fractions[~generated]).all()
                ):
                    raise ValueError(
                        "Interpolation anchor endpoint/fraction metadata is inconsistent"
                    )
                if (
                    not np.isfinite(fractions[generated]).all()
                    or np.any(fractions[generated] <= 0)
                    or np.any(fractions[generated] >= 1)
                ):
                    raise ValueError(
                        "Interpolation generated fractions must lie strictly inside (0,1)"
                    )
                anchors = {}
                for stage in np.unique(left[~generated]):
                    values = times[(~generated) & (left == stage)]
                    if not np.allclose(values, values[0], rtol=0, atol=1e-6):
                        raise ValueError(
                            "Interpolation stored-anchor time varies within a biological endpoint"
                        )
                    anchors[int(stage)] = values[0]
                if np.any(np.diff([anchors[k] for k in sorted(anchors)]) < -1e-6):
                    raise ValueError(
                        "Interpolation stored-anchor reporting ages decrease in biological endpoint order"
                    )
                for l, r in np.unique(np.column_stack([left[generated], right[generated]]), axis=0):
                    if int(l) not in anchors or int(r) not in anchors:
                        raise ValueError("Interpolation interval lacks a measured endpoint anchor")
                    rows = generated & (left == l) & (right == r)
                    expected = (1 - fractions[rows]) * anchors[int(l)] + fractions[rows] * anchors[
                        int(r)
                    ]
                    if not np.allclose(times[rows], expected, rtol=0, atol=1e-5):
                        raise ValueError(
                            "Interpolation reporting time disagrees with biological endpoints/fractions"
                        )
            notes.append(
                "Only stored anchor rows are observed; interior query rows have no measured GT. Supplied endpoint types are conditions, not predicted fate."
            )
        if scenario in {"measured", "reverse"} and time_kind == "ordinal":
            notes.append(
                "Biological stage ordinal retained; numeric E/P annotations are not silently treated as a monotonic physical age axis."
            )
        if attrs.get("uses_oracle_target_celltype") or scenario in {
            "interpolation",
            "future",
            "history",
        }:
            notes.append(
                "Supplied target/template state labels describe conditioning strata, not independently predicted states."
            )
        has_truth = "truth_expression" in handle and scenario in {"measured", "reverse"}
    if source.get("row_indices") is not None:
        indices = np.asarray(source["row_indices"], dtype=np.int64)
        if len(indices) and (
            np.any(indices < 0) or np.any(indices >= n) or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError(
                "Filtered source row indices must be strictly increasing and in bounds"
            )
        cells = cells.iloc[indices].reset_index(drop=True)
    _attach_obs(cells, source, settings, notes)
    result = {
        "genes": genes,
        "cells": cells,
        "metadata": attrs,
        "has_truth": has_truth,
        "n_cells": len(cells),
        "n_genes": g,
        "time_kind": time_kind,
        "notes": notes,
        "dataset_config": settings,
    }
    for key in ("cell_id", "time", "time_label", "state_supplied", "step", "is_generated"):
        if key in cells:
            result[key] = cells[key].to_numpy()
    return result


def iter_expression(
    source: dict[str, Any], chunk_size: int = 65536
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield logical start, stop and a finite float32 matrix; no full-array load."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if source.get("format") == "h5ad" or str(source["path"]).endswith(".h5ad"):
        import anndata as ad

        data = ad.read_h5ad(source["path"], backed="r")
        try:
            indices = np.asarray(source.get("row_indices", np.arange(data.n_obs)), dtype=int)
            for start in range(0, len(indices), chunk_size):
                stop = min(start + chunk_size, len(indices))
                block = data.X[indices[start:stop]]
                block = block.toarray() if hasattr(block, "toarray") else block
                block = np.asarray(block, dtype=np.float32)
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite expression rows {start}:{stop}")
                yield start, stop, block
        finally:
            data.file.close()
        return
    with h5py.File(source["path"], "r") as handle:
        matrix = handle[source.get("matrix_key", "prediction_expression")]
        indices = source.get("row_indices")
        n = len(indices) if indices is not None else matrix.shape[0]
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            block = np.asarray(
                (
                    matrix[np.asarray(indices[start:stop], dtype=int)]
                    if indices is not None
                    else matrix[start:stop]
                ),
                dtype=np.float32,
            )
            if not np.isfinite(block).all():
                raise ValueError(
                    f"Nonfinite expression in {source['path']} rows {start}:{stop}; no clipping or imputation performed"
                )
            yield start, stop, block


def read_rows(source: dict[str, Any], indices: Any) -> np.ndarray:
    """Read logical sample rows, preserving requested order including duplicates."""
    indices = np.asarray(indices, dtype=np.int64)
    if source.get("row_indices") is not None:
        indices = np.asarray(source["row_indices"], dtype=np.int64)[indices]
    unique, inverse = np.unique(indices, return_inverse=True)
    if source.get("format") == "h5ad" or str(source["path"]).endswith(".h5ad"):
        import anndata as ad

        data = ad.read_h5ad(source["path"], backed="r")
        try:
            values = data.X[unique]
            values = values.toarray() if hasattr(values, "toarray") else values
        finally:
            data.file.close()
    else:
        with h5py.File(source["path"], "r") as handle:
            values = handle[source.get("matrix_key", "prediction_expression")][unique]
    array = np.asarray(values, dtype=np.float32)[inverse]
    if not np.isfinite(array).all():
        raise ValueError("Selected saved expression contains nonfinite values")
    return array


def truth_sources(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Expose genuine observed cells, never invent truth at unobserved times."""
    if source.get("format") == "h5ad" or str(source["path"]).endswith(".h5ad"):
        return []
    scenario = source.get("scenario")
    panel = source.get("panel", "unknown")
    with h5py.File(source["path"], "r") as handle:
        if scenario in {"measured", "reverse"} and "truth_expression" in handle:
            key = (
                "truth_reporting_time"
                if "truth_reporting_time" in handle
                else "truth_time" if "truth_time" in handle else source["time_key"]
            )
            return [
                {
                    **source,
                    "source_id": source["source_id"] + "_truth",
                    "model": "GT_" + panel,
                    "role": "GT",
                    "matrix_key": "truth_expression",
                    "time_key": key,
                    "population": "truth",
                    "parent_source_id": source["source_id"],
                }
            ]
        if scenario == "interpolation" and "prediction_is_generated" in handle:
            rows = np.flatnonzero(
                ~np.asarray(handle["prediction_is_generated"], dtype=bool)
            ).tolist()
            if rows:
                return [
                    {
                        **source,
                        "source_id": source["source_id"] + "_anchors",
                        "model": "Observed_anchors_" + panel,
                        "role": "GT",
                        "population": "anchor",
                        "row_indices": rows,
                        "n_cells": len(rows),
                        "parent_source_id": source["source_id"],
                        "observed_anchors_only": True,
                    }
                ]
        if scenario in {"future", "history"} and "reference_expression" in handle:
            n = handle["reference_expression"].shape[0]
            return [
                {
                    **source,
                    "source_id": source["source_id"] + "_boundary",
                    "model": "Observed_boundary_" + panel,
                    "role": "REFERENCE",
                    "matrix_key": "reference_expression",
                    "time_key": "reference_time",
                    "population": "reference",
                    "n_cells": n,
                    "parent_source_id": source["source_id"],
                    "reference_only": True,
                }
            ]
    return []


def iter_perturbation(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one compact intervention summary, preserving stage/gene axes."""
    with h5py.File(source["path"], "r") as handle:
        genes = _decode(handle["gene_names"])
        if len(set(genes)) != len(genes):
            raise ValueError("Perturbation gene names are duplicated")
        times = np.asarray(handle["target_times"], dtype=float)
        indices = np.asarray(handle["intervention_gene_index"], dtype=int)
        if (
            len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(genes))
        ):
            raise ValueError("Perturbation interventions are duplicated or out of bounds")
        names = (
            _decode(handle["intervention_gene_name"])
            if "intervention_gene_name" in handle
            else [genes[i] for i in indices]
        )
        if names != [genes[i] for i in indices]:
            raise ValueError("Perturbation gene names and intervention indices disagree")
        baseline = np.asarray(handle["baseline_mean"], dtype=np.float32)
        variance = (
            np.asarray(handle["baseline_variance"], dtype=np.float32)
            if "baseline_variance" in handle
            else None
        )
        counts = (
            np.asarray(handle["target_cell_counts"], dtype=int)
            if "target_cell_counts" in handle
            else None
        )
        if (
            not np.isfinite(times).all()
            or baseline.shape != (len(times), len(genes))
            or not np.isfinite(baseline).all()
        ):
            raise ValueError("Invalid compact baseline/time/gene axes")
        if variance is not None and (
            variance.shape != baseline.shape
            or not np.isfinite(variance).all()
            or np.any(variance < 0)
        ):
            raise ValueError("Invalid compact baseline variance")
        for position, name in enumerate(names):
            edited = np.asarray(handle["perturbed_mean"][position], dtype=np.float32)
            edited_variance = (
                np.asarray(handle["perturbed_variance"][position], dtype=np.float32)
                if "perturbed_variance" in handle
                else None
            )
            if edited.shape != baseline.shape or not np.isfinite(edited).all():
                raise ValueError(f"Invalid compact edited mean for {name}")
            if edited_variance is not None and (
                edited_variance.shape != baseline.shape
                or not np.isfinite(edited_variance).all()
                or np.any(edited_variance < 0)
            ):
                raise ValueError(f"Invalid compact edited variance for {name}")
            yield {
                "intervention_gene": name,
                "intervention_index": int(indices[position]),
                "genes": genes,
                "times": times,
                "time_labels": [f"stage {v:g}" for v in times],
                "time_kind": "ordinal",
                "baseline_mean": baseline,
                "baseline_variance": variance,
                "edited_mean": edited,
                "edited_variance": edited_variance,
                "target_cell_counts": counts,
                "source_nonzero_counts": (
                    np.asarray(handle["source_nonzero_counts"][position])
                    if "source_nonzero_counts" in handle
                    else None
                ),
                "metadata": _attrs(handle),
            }


def _read_h5ad(source: dict[str, Any], config: dict[str, Any] | None) -> dict[str, Any]:
    import anndata as ad

    settings = _config_for(source, config)
    data = ad.read_h5ad(source["path"], backed="r")
    try:
        genes_column = settings.get("gene_name_column")
        genes = (
            data.var[genes_column].astype(str).tolist()
            if genes_column and genes_column in data.var
            else data.var_names.astype(str).tolist()
        )
        if len(set(genes)) != len(genes):
            raise ValueError("H5AD gene identities are duplicated")
        time_column = (
            source.get("time_column")
            or settings.get("time_column")
            or next((x for x in ("stage_ordinal", "time", "day_index") if x in data.obs), None)
        )
        if time_column is None:
            raise ValueError("H5AD requires an explicit numeric biological time column")
        times = pd.to_numeric(data.obs[time_column], errors="raise").to_numpy(float)
        state_column = (
            source.get("state_column")
            or settings.get("state_column")
            or settings.get("celltype_column")
        )
        labels = (
            data.obs[state_column].astype(str).to_numpy()
            if state_column in data.obs
            else np.full(data.n_obs, "unknown")
        )
        cells = pd.DataFrame(
            {
                "cell_id": data.obs_names.astype(str),
                "time": times,
                "time_label": data.obs[time_column].astype(str).to_numpy(),
                "state_supplied": labels,
                "row_index": np.arange(data.n_obs),
            }
        )
        if source.get("row_indices") is not None:
            cells = cells.iloc[source["row_indices"]].reset_index(drop=True)
        return {
            "genes": genes,
            "cells": cells,
            "metadata": dict(source.get("metadata", {})),
            "has_truth": False,
            "n_cells": len(cells),
            "n_genes": len(genes),
            "time_kind": source.get("time_kind", "ordinal"),
            "notes": [
                "Explicit prepared H5AD source; supplied labels are not automatically independent predicted fate."
            ],
            "dataset_config": settings,
            **{c: cells[c].to_numpy() for c in ("cell_id", "time", "time_label", "state_supplied")},
        }
    finally:
        data.file.close()
