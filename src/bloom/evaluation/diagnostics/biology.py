"""Biological-question readouts from existing V4/V5 exports; never runs inference.

Requires numpy and h5py for H5 input. Matplotlib is optional. TSV/CSV summary
input needs only numpy. Source expression is never modified or renormalized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SEMANTICS = [
    "Unmeasured-time curves describe model behavior; they are not accuracy estimates.",
    "A supplied target/template type is a condition, not an independently predicted identity.",
    "Program agreement can be circular when these same programs were conditioning inputs.",
    "Input-gene zeroing measures frozen-model sensitivity, not a validated causal knockout.",
    "Cell counts are descriptive; cells are not independent biological replicates.",
    "Half-range crossing is a grid-dependent expression descriptor, not validated biological onset.",
    "Expression means remain in the exported expression scale; no log-fold-change claim is made.",
]


def strings(values):
    return np.asarray(
        [
            v.decode("utf-8") if isinstance(v, bytes) else str(v)
            for v in np.asarray(values).reshape(-1)
        ],
        dtype=object,
    )


def first_key(handle, keys):
    return next((k for k in keys if k in handle), None)


def read_table(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(
            csv.DictReader(
                stream, delimiter="\t" if Path(path).suffix.lower() in {".tsv", ".txt"} else ","
            )
        )


def write_table(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)


def load_program_membership(
    path,
    program_column="program",
    gene_column="gene",
    weight_column="weight",
    duplicate_policy="auto",
):
    """Load explicit long-form membership and retain an auditable aggregation.

    The project's V5/V6/V7 model loaders add every driver score for a repeated
    (node, gene) pair. Therefore the declared node/gene/score schema uses a
    signed sum, including identical repeated rows. Generic membership tables
    retain their stricter require-equal behavior unless sum is explicitly
    requested. No sign is discarded and no conflicting value is overwritten.
    """
    if duplicate_policy not in {"auto", "sum", "require_equal"}:
        raise ValueError(f"Unknown duplicate program weight policy: {duplicate_policy}")
    project_driver_schema = (program_column, gene_column, weight_column) == (
        "node",
        "gene",
        "score",
    )
    resolved = (
        ("sum" if project_driver_schema else "require_equal")
        if duplicate_policy == "auto"
        else duplicate_policy
    )
    programs, grouped = defaultdict(dict), defaultdict(list)
    audit = {
        "input_path": str(Path(path).expanduser().resolve()) if path else None,
        "columns": {"program": program_column, "gene": gene_column, "weight": weight_column},
        "duplicate_policy_requested": duplicate_policy,
        "duplicate_policy_resolved": resolved,
        "policy_basis": (
            "project V5/V6/V7 driver loaders accumulate signed scores per node/gene"
            if duplicate_policy == "auto" and project_driver_schema
            else (
                "strict generic membership default"
                if duplicate_policy == "auto"
                else "explicit CLI selection"
            )
        ),
        "input_rows": 0,
        "unique_program_gene_pairs": 0,
        "duplicate_pairs": 0,
        "conflicting_duplicate_pairs": 0,
        "mixed_sign_pairs": 0,
        "zero_weight_pairs": 0,
    }
    if not path:
        return programs, audit, []
    for source_row, row in enumerate(read_table(path), start=1):
        if not row.get(program_column, "").strip() or not row.get(gene_column, "").strip():
            raise ValueError(
                f"Program table requires nonempty {program_column!r} and {gene_column!r} columns; use explicit --program-column/--gene-column when the source uses other names"
            )
        name, gene = row[program_column].strip(), row[gene_column].strip()
        weight = float(row.get(weight_column) or 1.0)
        if not math.isfinite(weight):
            raise ValueError(f"Non-finite program weight: {name}/{gene}")
        grouped[(name, gene)].append((source_row, weight))
        audit["input_rows"] += 1
    membership_rows = []
    for (name, gene), sources in sorted(grouped.items()):
        values = [weight for _, weight in sources]
        conflicting = len(set(values)) > 1
        if resolved == "require_equal" and conflicting:
            raise ValueError(f"Conflicting duplicate program weights: {name}/{gene}")
        try:
            weight = math.fsum(values) if resolved == "sum" else values[0]
        except OverflowError as exc:
            raise ValueError(f"Non-finite aggregated program weight: {name}/{gene}") from exc
        if not math.isfinite(weight):
            raise ValueError(f"Non-finite aggregated program weight: {name}/{gene}")
        programs[name][gene] = weight
        mixed_sign = min(values) < 0 < max(values)
        membership_rows.append(
            {
                "program": name,
                "gene": gene,
                "weight": weight,
                "aggregation": resolved,
                "n_source_rows": len(sources),
                "conflicting_source_weights": conflicting,
                "mixed_sign_source_weights": mixed_sign,
                "source_data_rows": json.dumps([index for index, _ in sources]),
                "source_weights": json.dumps(values),
                "zero_after_aggregation": weight == 0,
            }
        )
        audit["duplicate_pairs"] += len(sources) > 1
        audit["conflicting_duplicate_pairs"] += conflicting
        audit["mixed_sign_pairs"] += mixed_sign
        audit["zero_weight_pairs"] += weight == 0
    audit["unique_program_gene_pairs"] = len(grouped)
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    audit.update(
        {
            "input_sha256": digest.hexdigest(),
            "membership_table": "program_membership.tsv",
            "source_data_rows": "1-based parsed data-record numbers, excluding header",
            "score_normalization": "signed aggregate weights divided by sum of absolute available weights; zero aggregate genes excluded",
        }
    )
    return programs, audit, membership_rows


def load_programs(
    path,
    program_column="program",
    gene_column="gene",
    weight_column="weight",
    duplicate_policy="auto",
):
    """Compatibility wrapper returning the aggregated membership dictionaries."""
    return load_program_membership(
        path, program_column, gene_column, weight_column, duplicate_policy
    )[0]


def feature_definitions(genes, programs, min_program_genes=3, min_program_coverage=0.3):
    if len(set(genes)) != len(genes):
        raise ValueError("gene_names contains duplicates; align gene identities before analysis")
    index = {str(g): i for i, g in enumerate(genes)}
    features = [("gene", str(g), np.asarray([i]), np.asarray([1.0])) for i, g in enumerate(genes)]
    coverage = []
    for name, membership in sorted(programs.items()):
        present = [g for g, w in membership.items() if g in index and w != 0]
        requested = [g for g, w in membership.items() if w != 0]
        ratio = len(present) / len(requested) if requested else 0.0
        usable = len(present) >= min_program_genes and ratio >= min_program_coverage
        coverage.append(
            {
                "program": name,
                "n_requested": len(requested),
                "n_present": len(present),
                "coverage": ratio,
                "usable": usable,
                "missing_genes": ";".join(g for g in requested if g not in index),
            }
        )
        if usable:
            weights = np.asarray([membership[g] for g in present], dtype=float)
            features.append(
                (
                    "program",
                    name,
                    np.asarray([index[g] for g in present]),
                    weights / np.abs(weights).sum(),
                )
            )
    return features, coverage


def feature_scores(matrix, features):
    return np.column_stack([matrix[:, ix] @ weights for _, _, ix, weights in features])


def numeric_vector(handle, key, n):
    values = np.asarray(handle[key])
    if values.shape != (n,) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(
            f"{key} must be a finite numeric vector of length {n}; no ordinal/physical-time guessing is performed"
        )
    return values.astype(float)


def time_vector(handle, source, n):
    protocol = handle.attrs.get("prediction_protocol", "")
    protocol = protocol.decode("utf-8") if isinstance(protocol, bytes) else str(protocol)
    if protocol == "direct_cache_univariate_gene_forecast_v1":
        from . import read_direct_reporting

        axis, vectors = read_direct_reporting(handle)
        key = f"{source}_reporting_time"
        values = vectors[source]
        if values.shape != (n,):
            raise ValueError(f"{key} must be row-aligned to its expression population")
        return values.astype(float), axis["reporting_unit"], key
    actual = first_key(handle, [f"{source}_actual_time"])
    if actual:
        return numeric_vector(handle, actual, n), "actual_time", actual
    ordinary = first_key(handle, [f"{source}_time", f"{source}_day_index", "day_index"])
    if ordinary:
        return numeric_vector(handle, ordinary, n), "stored_time_coordinate", ordinary
    raise ValueError(f"Missing explicit {source} time coordinate")


def stratum_vector(handle, key, n, source):
    if not key:
        return np.full(n, "all_cells", dtype=object), "none"
    selected = key
    if source == "truth":
        selected = first_key(handle, ["truth_type_id", "truth_celltype"])
        if selected is None:
            pred_ix = first_key(handle, ["prediction_cell_index", "target_cell_index"])
            truth_ix = first_key(handle, ["truth_cell_index"])
            if not (
                pred_ix and truth_ix and np.array_equal(handle[pred_ix][:], handle[truth_ix][:])
            ):
                raise ValueError(
                    "Truth stratification needs truth_type_id or explicit row-alignment identities; unstratified reporting remains available"
                )
            selected = key
    if selected not in handle or handle[selected].shape != (n,):
        raise ValueError(f"Requested stratum {selected!r} is missing or not row-aligned")
    values = strings(handle[selected][:])
    vocab_key = first_key(handle, ["celltype_names", "cell_type_names"])
    if vocab_key and handle[selected].dtype.kind in "iu":
        vocab = strings(handle[vocab_key][:])
        ids = np.asarray(handle[selected])
        values = np.asarray(
            [vocab[i] if 0 <= i < len(vocab) else f"unmapped_id:{i}" for i in ids], dtype=object
        )
    return values, selected


def summarize_population(
    handle,
    source,
    matrix_key,
    features,
    metadata,
    chunk_rows=2048,
    stratify_key=None,
    stratum_role="unknown",
):
    matrix = handle[matrix_key]
    if matrix.ndim != 2 or matrix.shape[1] != len(handle["gene_names"]):
        raise ValueError(f"{matrix_key} must have shape [cells, gene_names]")
    n = matrix.shape[0]
    if n == 0:
        raise ValueError(f"{matrix_key} contains no cells")
    times, axis, time_key = time_vector(handle, source, n)
    reporting_labels = {}
    if time_key == f"{source}_reporting_time":
        # time_vector has validated both the axis and all reporting vectors.
        recorded_axis = json.loads(handle.attrs["reporting_time_axis_json"])
        reporting_labels = dict(
            zip(recorded_axis["reporting_values"], recorded_axis["display_labels"])
        )
    groups, group_key = stratum_vector(handle, stratify_key, n, source)
    generated = np.ones(n, dtype=bool)
    if source == "prediction" and "prediction_is_generated" in handle:
        if handle["prediction_is_generated"].shape != (n,):
            raise ValueError("prediction_is_generated is not row-aligned")
        generated = np.asarray(handle["prediction_is_generated"], dtype=bool)
    accum = {}
    for start in range(0, n, chunk_rows):
        stop = min(n, start + chunk_rows)
        block = np.asarray(matrix[start:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError(f"{matrix_key} contains non-finite expression in rows {start}:{stop}")
        scores = feature_scores(block, features)
        local = defaultdict(list)
        for j in range(stop - start):
            origin = (
                "observed_anchor" if source == "prediction" and not generated[start + j] else source
            )
            local[(origin, float(times[start + j]), str(groups[start + j]))].append(j)
        for key, indices in local.items():
            values = scores[indices]
            count, mean, m2 = len(values), values.mean(0), ((values - values.mean(0)) ** 2).sum(0)
            if key not in accum:
                accum[key] = [count, mean, m2]
            else:
                old_n, old_mean, old_m2 = accum[key]
                delta, total = mean - old_mean, old_n + count
                accum[key] = [
                    total,
                    old_mean + delta * count / total,
                    old_m2 + m2 + delta**2 * old_n * count / total,
                ]
    rows = []
    for (origin, time, group), (count, mean, m2) in sorted(accum.items()):
        for j, (kind, name, _, _) in enumerate(features):
            rows.append(
                {
                    **metadata,
                    "source": origin,
                    "time_axis": axis,
                    "time_key": time_key,
                    **({"time_label": reporting_labels[time]} if reporting_labels else {}),
                    "time": time,
                    "stratum": group,
                    "stratum_key": group_key,
                    "stratum_role": (
                        ("truth_annotation" if source == "truth" else stratum_role)
                        if group_key != "none"
                        else "none"
                    ),
                    "feature_kind": kind,
                    "feature": name,
                    "n_cells": count,
                    "mean": float(mean[j]),
                    "cell_sd": float(math.sqrt(max(0, m2[j]) / (count - 1))) if count > 1 else "",
                }
            )
    return rows


def half_range_descriptor(times, values):
    times, values = np.asarray(times, float), np.asarray(values, float)
    order = np.argsort(times)
    times, values = times[order], values[order]
    result = {
        "n_timepoints": len(times),
        "first_time": float(times[0]),
        "last_time": float(times[-1]),
        "peak_time": float(times[np.argmax(values)]),
        "net_change": float(values[-1] - values[0]),
        "range": float(np.ptp(values)),
        "crossing_status": "insufficient_timepoints",
        "crossing_time": "",
        "crossing_left": "",
        "crossing_right": "",
    }
    if len(times) < 3:
        return result
    if np.ptp(values) <= 1e-12:
        result["crossing_status"] = "constant"
        return result
    net = values[-1] - values[0]
    if abs(net) <= 1e-12:
        result["crossing_status"] = "no_net_direction"
        return result
    sign = 1 if net > 0 else -1
    half = (float(values.min()) + float(values.max())) / 2
    result.update(
        {
            "direction": "rising" if sign == 1 else "falling",
            "half_range_level": half,
            "fraction_intervals_in_net_direction": float(np.mean(sign * np.diff(values) >= 0)),
        }
    )
    if sign * (values[0] - half) >= 0:
        result["crossing_status"] = "already_beyond_half_at_first_time"
        return result
    for j in range(1, len(times)):
        if sign * (values[j - 1] - half) < 0 <= sign * (values[j] - half):
            fraction = (half - values[j - 1]) / (values[j] - values[j - 1])
            result.update(
                {
                    "crossing_status": "first_crossing",
                    "crossing_left": float(times[j - 1]),
                    "crossing_right": float(times[j]),
                    "crossing_time": float(times[j - 1] + fraction * (times[j] - times[j - 1])),
                }
            )
            break
    return result


def average_rank(x):
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(x, y):
    x, y = average_rank(np.asarray(x)), average_rank(np.asarray(y))
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return ""
    return float(np.corrcoef(x, y)[0, 1])


SERIES_KEYS = [
    "input_id",
    "dataset",
    "scenario",
    "source",
    "time_axis",
    "stratum",
    "stratum_role",
    "feature_kind",
]


def derive_temporal_tables(rows, min_cells=10):
    series, stages = defaultdict(list), defaultdict(dict)
    for row in rows:
        if int(row.get("n_cells", 0)) < min_cells:
            continue
        key = tuple(str(row.get(k, "unknown")) for k in SERIES_KEYS)
        series[key + (str(row["feature"]),)].append(row)
        stages[key].setdefault(float(row["time"]), {})[str(row["feature"])] = float(row["mean"])
    dynamics, ranks = [], []
    for key, local in sorted(series.items()):
        times = [float(row["time"]) for row in local]
        if len(set(times)) != len(times):
            raise ValueError(
                "Duplicate feature/time summaries: separate independent inputs, conditions, or replicate IDs before aggregation"
            )
        dynamics.append(
            {
                **dict(zip(SERIES_KEYS + ["feature"], key)),
                **half_range_descriptor(times, [float(row["mean"]) for row in local]),
            }
        )
    for key, local in sorted(stages.items()):
        times = sorted(local)
        for left, right in zip(times[:-1], times[1:]):
            common = sorted(set(local[left]) & set(local[right]))
            ranks.append(
                {
                    **dict(zip(SERIES_KEYS, key)),
                    "left_time": left,
                    "right_time": right,
                    "n_common_features": len(common),
                    "adjacent_rank_spearman": spearman(
                        [local[left][f] for f in common], [local[right][f] for f in common]
                    ),
                }
            )
    return dynamics, ranks


def measured_fidelity(rows):
    aligned = defaultdict(dict)
    keys = [
        "input_id",
        "dataset",
        "scenario",
        "time_axis",
        "time",
        "stratum",
        "feature_kind",
        "feature",
    ]
    for row in rows:
        if row["source"] not in {"prediction", "truth"}:
            continue
        key = tuple(str(row.get(k, "unknown")) for k in keys)
        if row["source"] in aligned[key]:
            raise ValueError("Duplicate source/time summaries prevent unambiguous truth comparison")
        aligned[key][row["source"]] = row
    result = []
    for key, pair in sorted(aligned.items()):
        if set(pair) != {"prediction", "truth"}:
            continue
        pred, true = pair["prediction"], pair["truth"]
        delta = float(pred["mean"]) - float(true["mean"])
        result.append(
            {
                **dict(zip(keys, key)),
                "prediction_mean": pred["mean"],
                "truth_mean": true["mean"],
                "signed_mean_error": delta,
                "absolute_mean_error": abs(delta),
                "prediction_n": pred["n_cells"],
                "truth_n": true["n_cells"],
                "comparison": "stage_stratum_pseudobulk_not_cell_identity_accuracy",
            }
        )
    return result


def compact_sensitivity(handle, features, metadata):
    required = ["baseline_mean", "perturbed_mean", "target_times", "intervention_gene_name"]
    for key in required:
        if key not in handle:
            raise ValueError(f"Compact sensitivity H5 is missing {key}")
    genes = strings(handle["gene_names"][:])
    interventions = strings(handle["intervention_gene_name"][:])
    base, changed = np.asarray(handle["baseline_mean"], float), handle["perturbed_mean"]
    times = np.asarray(handle["target_times"], float)
    if base.shape != (len(times), len(genes)) or changed.shape != (
        len(interventions),
        len(times),
        len(genes),
    ):
        raise ValueError("Compact sensitivity shapes are inconsistent")
    if not np.isfinite(base).all() or not np.isfinite(times).all():
        raise ValueError("Compact sensitivity contains non-finite baseline/time values")
    result = []
    for i, intervention in enumerate(interventions):
        delta = np.asarray(changed[i], float) - base
        if not np.isfinite(delta).all():
            raise ValueError(f"Compact sensitivity non-finite intervention {intervention}")
        downstream = np.flatnonzero(genes != intervention)
        for j, time in enumerate(times):
            result.append(
                {
                    **metadata,
                    "intervention": intervention,
                    "time": float(time),
                    "feature_kind": "global",
                    "feature": "all_other_genes",
                    "n_genes": len(downstream),
                    "signed_mean_response": (
                        float(delta[j, downstream].mean()) if len(downstream) else ""
                    ),
                    "rms_response": (
                        float(np.sqrt(np.mean(delta[j, downstream] ** 2)))
                        if len(downstream)
                        else ""
                    ),
                    "directly_intervened_gene_excluded": True,
                    "claim": "frozen_model_input_sensitivity",
                }
            )
            for kind, name, ix, weights in features:
                if kind != "program":
                    continue
                keep = genes[ix] != intervention
                selected, w = ix[keep], weights[keep]
                if not len(selected) or np.abs(w).sum() == 0:
                    continue
                w = w / np.abs(w).sum()
                result.append(
                    {
                        **metadata,
                        "intervention": intervention,
                        "time": float(time),
                        "feature_kind": kind,
                        "feature": name,
                        "n_genes": len(selected),
                        "signed_mean_response": float(delta[j, selected] @ w),
                        "rms_response": float(np.sqrt(np.mean(delta[j, selected] ** 2))),
                        "directly_intervened_gene_excluded": True,
                        "claim": "frozen_model_input_sensitivity",
                    }
                )
    return result


def read_h5(path, args, programs):
    import h5py

    path = Path(path).resolve()
    with h5py.File(path, "r") as handle:
        if "gene_names" not in handle:
            raise ValueError("H5 lacks gene_names; input schema is unsupported")
        genes = strings(handle["gene_names"][:])
        features, coverage = feature_definitions(
            genes, programs, args.min_program_genes, args.min_program_coverage
        )
        metadata = {
            "input_id": hashlib.sha256(str(path).encode()).hexdigest()[:12],
            "input_path": str(path),
            "dataset": args.dataset or str(handle.attrs.get("dataset_id", "unknown")),
            "scenario": str(handle.attrs.get("scenario", path.stem)),
            "program_provenance": args.program_provenance,
        }
        for row in coverage:
            row.update(metadata)
        if "baseline_mean" in handle and "perturbed_mean" in handle:
            return (
                [],
                coverage,
                compact_sensitivity(handle, features, metadata),
                {**metadata, "status": "complete", "kind": "sensitivity"},
            )
        pred_key = first_key(handle, ["prediction_expression", "prediction"])
        if pred_key is None:
            raise ValueError("H5 lacks prediction_expression/prediction; schema unsupported")
        rows = summarize_population(
            handle,
            "prediction",
            pred_key,
            features,
            metadata,
            args.chunk_rows,
            args.stratify_key,
            args.stratum_role,
        )
        truth_key = first_key(handle, ["truth_expression", "ground_truth", "truth"])
        if truth_key:
            rows.extend(
                summarize_population(
                    handle,
                    "truth",
                    truth_key,
                    features,
                    metadata,
                    args.chunk_rows,
                    args.stratify_key,
                    args.stratum_role,
                )
            )
        attrs = {
            k: str(handle.attrs[k])
            for k in [
                "uses_oracle_target_celltype",
                "uses_oracle_target_count",
                "oracle_target_inputs",
                "actual_time_unit",
                "temporal_direction",
            ]
            if k in handle.attrs
        }
        return (
            rows,
            coverage,
            [],
            {
                **metadata,
                "status": "complete",
                "kind": "temporal",
                "has_truth": bool(truth_key),
                "source_attributes": attrs,
            },
        )


def normalize_summary(path, dataset=None):
    rows = read_table(path)
    result = []
    for row in rows:
        feature = row.get("feature") or row.get("program")
        time = row.get("time") or row.get("actual_time")
        mean = row.get("mean") or row.get("program_score_mean")
        count = row.get("n_cells") or row.get("n")
        if feature is None or time is None or mean is None or count is None:
            raise ValueError(
                "Summary needs feature/program, time/actual_time, mean/program_score_mean and n_cells/n columns"
            )
        if int(count) < 1 or not math.isfinite(float(time)) or not math.isfinite(float(mean)):
            raise ValueError("Summary has invalid time, mean or n_cells")
        result.append(
            {
                **row,
                "input_id": row.get("input_id")
                or hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:12],
                "input_path": str(Path(path).resolve()),
                "dataset": dataset or row.get("dataset", "unknown"),
                "scenario": row.get("scenario", Path(path).stem),
                "source": row.get("source", "prediction"),
                "time_axis": row.get(
                    "time_axis", "actual_time" if "actual_time" in row else "stored_time_coordinate"
                ),
                "stratum": row.get("stratum", "all_cells"),
                "stratum_role": row.get("stratum_role", "unknown"),
                "feature_kind": row.get("feature_kind", "program" if "program" in row else "gene"),
                "feature": feature,
                "time": float(time),
                "mean": float(mean),
                "n_cells": int(count),
            }
        )
    seen = set()
    for row in result:
        key = tuple(str(row.get(k, "unknown")) for k in SERIES_KEYS + ["feature", "time"])
        if key in seen:
            raise ValueError(
                "Duplicate summary feature/time rows: provide distinct input IDs or strata for separate conditions/replicates"
            )
        seen.add(key)
    return result


def make_plots(rows, out, max_features=12):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "matplotlib is not installed; all numerical tables are still generated",
        }
    subsets = defaultdict(list)
    for row in rows:
        if row["stratum"] == "all_cells":
            subsets[(row["input_id"], row["feature_kind"], row["time_axis"])].append(row)
    outputs = []
    for (input_id, kind, axis), local in sorted(subsets.items()):
        candidates = defaultdict(list)
        for row in local:
            candidates[row["feature"]].append(float(row["mean"]))
        names = sorted(candidates, key=lambda f: (-float(np.var(candidates[f])), f))[:max_features]
        if not names:
            continue
        fig, axes = plt.subplots(
            math.ceil(len(names) / 3),
            min(3, len(names)),
            figsize=(10, 2.6 * math.ceil(len(names) / 3)),
            squeeze=False,
        )
        colors = {"prediction": "#578FC8", "truth": "#3B745A", "observed_anchor": "#505050"}
        for ax, name in zip(axes.flat, names):
            for source in sorted({r["source"] for r in local}):
                data = sorted(
                    [r for r in local if r["feature"] == name and r["source"] == source],
                    key=lambda r: float(r["time"]),
                )
                if data:
                    ax.plot(
                        [float(r["time"]) for r in data],
                        [float(r["mean"]) for r in data],
                        "o-",
                        ms=3,
                        lw=1.3,
                        color=colors.get(source, "#A37DAF"),
                        label=source,
                    )
            ax.set_title(name, fontsize=10)
            axis_label = {
                "actual_time": "Stored actual time",
                "ordinal_stage": "Biological stage (ordinal)",
                "source_annotation_unit": "Observed age (source units)",
            }.get(axis, "Stored time coordinate")
            ax.set_xlabel(axis_label, fontsize=8)
            if axis == "ordinal_stage":
                tick_labels = {
                    float(row["time"]): row.get("time_label", str(row["time"])) for row in local
                }
                ticks = sorted(tick_labels)
                if len(ticks) > 12:
                    ticks = [
                        ticks[index]
                        for index in np.unique(np.linspace(0, len(ticks) - 1, 10).astype(int))
                    ]
                ax.set_xticks(ticks, [tick_labels[tick] for tick in ticks], rotation=45, ha="right")
            ax.set_ylabel(
                "Mean exported expression" if kind == "gene" else "Weighted expression score",
                fontsize=8,
            )
            ax.spines[["right", "top"]].set_visible(False)
            ax.tick_params(labelsize=8)
        for ax in axes.flat[len(names) :]:
            ax.axis("off")
        axes.flat[0].legend(frameon=False, fontsize=7)
        fig.suptitle(
            "Descriptive temporal dynamics · displayed features selected by variance", fontsize=11
        )
        fig.tight_layout()
        stem = out / f"{input_id}_{kind}_temporal_dynamics"
        fig.savefig(stem.with_suffix(".png"), dpi=250, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        outputs.extend([stem.with_suffix(".png").name, stem.with_suffix(".pdf").name])
    return {
        "status": "complete" if outputs else "not_applicable",
        "files": outputs,
        "selection": "up to max_features by variance for display only; tables retain all features; stratified-only runs are table-first",
    }


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--h5",
        action="append",
        default=[],
        help="Existing canonical/scenario H5; repeat for multiple files",
    )
    p.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Existing long-format temporal TSV/CSV; repeat",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--dataset")
    p.add_argument("--programs", help="TSV/CSV with program,gene and optional weight columns")
    p.add_argument(
        "--program-column",
        default="program",
        help="Explicit membership-set column; e.g. node for project driver tables",
    )
    p.add_argument("--gene-column", default="gene")
    p.add_argument(
        "--weight-column", default="weight", help="Optional weight column; absent values use 1.0"
    )
    p.add_argument(
        "--duplicate-program-weights",
        choices=["auto", "sum", "require_equal"],
        default="auto",
        help="auto sums signed project node/gene/score rows like training; generic membership duplicates must agree. All resolved memberships are audited.",
    )
    p.add_argument(
        "--program-provenance",
        choices=["used_for_conditioning", "external_heldout", "unknown"],
        default="unknown",
    )
    p.add_argument(
        "--stratify-key",
        help="Explicit row-vector H5 key, e.g. target_type_id or template_type_id; default is all cells",
    )
    p.add_argument(
        "--stratum-role",
        choices=["supplied_condition", "truth_annotation", "inferred_label", "unknown"],
        default="unknown",
    )
    p.add_argument("--min-program-genes", type=int, default=3)
    p.add_argument("--min-program-coverage", type=float, default=0.3)
    p.add_argument(
        "--min-cells",
        type=int,
        default=10,
        help="Minimum cells/time/stratum for timing and rank descriptors",
    )
    p.add_argument("--chunk-rows", type=int, default=2048)
    p.add_argument("--max-plot-features", type=int, default=12)
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.h5 and not args.summary:
        parser().error("Provide at least one --h5 or --summary")
    if (
        args.chunk_rows < 1
        or args.min_program_genes < 1
        or args.min_cells < 1
        or args.max_plot_features < 1
        or not 0 <= args.min_program_coverage <= 1
    ):
        parser().error("Counts must be positive and minimum coverage must lie in [0,1]")
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        programs, program_audit, membership_rows = load_program_membership(
            args.programs,
            args.program_column,
            args.gene_column,
            args.weight_column,
            args.duplicate_program_weights,
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        manifest = {
            "schema": "v5.biological_questions.1",
            "status": "failed",
            "parameters": vars(args),
            "inputs": [
                {
                    "input_path": args.programs,
                    "status": "failed",
                    "kind": "program_membership",
                    "error": str(exc),
                }
            ],
            "claim_boundaries": SEMANTICS,
        }
        (out / "biological_report_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"status": "failed", "output": str(out), "error": str(exc)}))
        return 2
    rows, coverage, sensitivity, inputs = [], [], [], []
    for path in dict.fromkeys(args.h5):
        try:
            new_rows, new_coverage, new_sensitivity, info = read_h5(path, args, programs)
            rows.extend(new_rows)
            coverage.extend(new_coverage)
            sensitivity.extend(new_sensitivity)
            inputs.append(info)
        except (ValueError, OSError, KeyError, ImportError, TypeError) as exc:
            inputs.append({"input_path": str(path), "status": "failed", "error": str(exc)})
    for path in dict.fromkeys(args.summary):
        try:
            rows.extend(normalize_summary(path, args.dataset))
            inputs.append({"input_path": str(path), "status": "complete", "kind": "summary"})
        except (ValueError, OSError, KeyError, TypeError) as exc:
            inputs.append({"input_path": str(path), "status": "failed", "error": str(exc)})
    dynamics, ranks = derive_temporal_tables(rows, args.min_cells)
    comparisons = measured_fidelity(rows)
    tables = {
        "temporal_summary.tsv": rows,
        "temporal_descriptors.tsv": dynamics,
        "adjacent_rank_stability.tsv": ranks,
        "measured_pseudobulk_fidelity.tsv": comparisons,
        "program_coverage.tsv": coverage,
        "input_sensitivity.tsv": sensitivity,
        "program_membership.tsv": membership_rows,
    }
    for name, values in tables.items():
        write_table(out / name, values)
    plots = (
        {"status": "disabled"} if args.no_plots else make_plots(rows, out, args.max_plot_features)
    )
    status = "partial_failure" if any(i["status"] == "failed" for i in inputs) else "complete"
    manifest = {
        "schema": "v5.biological_questions.1",
        "status": status,
        "inputs": inputs,
        "table_rows": {k: len(v) for k, v in tables.items()},
        "plots": plots,
        "parameters": vars(args),
        "claim_boundaries": SEMANTICS,
        "program_membership": program_audit,
        "program_score": "weighted mean of available exported expression; weights normalized by L1 norm; not Scanpy score_genes or AUCell",
        "timing_order": "descriptors sort biological/stored time numerically ascending; source H5 order and prediction direction are unchanged",
        "replicates": "no biological-replicate confidence intervals or hypothesis-test P values are computed",
    }
    (out / "biological_report_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    notes = "# Biological question report\n\n" + "\n".join(f"- {s}" for s in SEMANTICS)
    notes += "\n\nAll tables were derived from the listed input files. Empty tables mean the relevant evidence was not supplied; no values were simulated. Inspect biological_report_manifest.json for per-input failures and exact parameters.\n"
    (out / "README.md").write_text(notes, encoding="utf-8")
    print(json.dumps({"status": status, "output": str(out), "tables": manifest["table_rows"]}))
    return 2 if status != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
