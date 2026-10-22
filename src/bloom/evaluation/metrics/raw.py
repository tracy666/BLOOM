"""Full-row, paired metrics on native V5 forward-prediction HDF5 files.

PCC means follow the uploaded V6 training code's fast backend: calculate a
correlation within each day, apply its zero/skip constant-vector policy, then
average days equally. Pooled-cell and pooled-gene statistics are separate.
No expression is clipped, rescaled, sampled or imputed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import rankdata


def _strings(values):
    a = np.asarray(values)
    if a.ndim != 1:
        raise ValueError("Identity and gene vectors must be one-dimensional")
    if a.dtype.kind in "iuf":
        if not np.isfinite(a).all() or np.any(a < 0) or np.any(a != np.floor(a)):
            raise ValueError("Numeric cell IDs must be finite nonnegative integers")
        return np.asarray([str(int(x)) for x in a], dtype=str)
    result = np.asarray([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in a], dtype=str)
    if any(not x.strip() or "\0" in x for x in result):
        raise ValueError("Identity and gene names cannot be empty or contain NUL")
    return result


def _attr(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _known(value):
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "unknown", "unspecified", "none", "null"} and not text.startswith("unknown") and not text.endswith("unspecified")


def pcc(x, y, eps=1e-12):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.clip(np.dot(x, y) / den, -1., 1.)) if den > eps else float("nan")


def cosine(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.clip(np.dot(x, y) / den, -1., 1.)) if den > 0 and np.isfinite(den) else float("nan")


def _policy(values, policy, minimum):
    """Training mean plus transparent, additive policy contributions.

    For skip, G/n_valid multiplication means the mean of G contributions equals
    the skip mean. Invalid components are zero and remain invalid in raw PCC.
    """
    v = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(v)
    fraction = float(valid.mean()) if len(v) else 0.
    if policy == "zero":
        selected = np.where(valid, v, 0.)
        return float(selected.mean()) if len(v) else 0., float(np.median(selected)) if len(v) else 0., selected, fraction
    if not valid.any() or fraction < minimum:
        return 0., 0., np.zeros_like(v), fraction
    components = np.where(valid, v / fraction, 0.)
    return float(v[valid].mean()), float(np.median(v[valid])), components, fraction


class Accumulator:
    """Stable centered merge, with O(genes + cells) extra working memory.

    Row correlation vectors are retained for exact median statistics; matrices
    are streamed. Per-gene correlations use centered parallel moments rather
    than cancellation-prone sum(x*x)-sum(x)**2/n subtraction.
    """
    def __init__(self, n_genes):
        self.g = n_genes
        self.n = 0
        self.mx = np.zeros(n_genes, np.float64)
        self.my = np.zeros(n_genes, np.float64)
        self.vx = np.zeros(n_genes, np.float64)
        self.vy = np.zeros(n_genes, np.float64)
        self.cross = np.zeros(n_genes, np.float64)
        self.sse = np.zeros(n_genes, np.float64)
        self.absolute = np.zeros(n_genes, np.float64)
        self.dx = np.zeros(n_genes, np.int64)
        self.dy = np.zeros(n_genes, np.int64)
        self.cell_corr = []
        self.cell_cosine = []

    def update(self, x, y, eps):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        n = len(x)
        if not n:
            return
        bx, by = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - bx, y - by
        weight = self.n * n / (self.n + n)
        delta_x, delta_y = bx - self.mx, by - self.my
        self.vx += np.einsum("ij,ij->j", xc, xc) + delta_x * delta_x * weight
        self.vy += np.einsum("ij,ij->j", yc, yc) + delta_y * delta_y * weight
        self.cross += np.einsum("ij,ij->j", xc, yc) + delta_x * delta_y * weight
        self.mx += delta_x * n / (self.n + n)
        self.my += delta_y * n / (self.n + n)
        self.n += n
        self.dx += np.sum(x > 0, axis=0)
        self.dy += np.sum(y > 0, axis=0)
        difference = x - y
        self.sse += np.einsum("ij,ij->j", difference, difference)
        self.absolute += np.abs(difference).sum(axis=0)
        xr, yr = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
        numerator = np.einsum("ij,ij->i", xr, yr)
        denominator = np.sqrt(np.einsum("ij,ij->i", xr, xr) * np.einsum("ij,ij->i", yr, yr))
        corr = np.full(n, np.nan)
        valid = denominator > eps
        corr[valid] = np.clip(numerator[valid] / denominator[valid], -1., 1.)
        self.cell_corr.append(corr)
        numerator = np.einsum("ij,ij->i", x, y)
        denominator = np.sqrt(np.einsum("ij,ij->i", x, x) * np.einsum("ij,ij->i", y, y))
        cos = np.full(n, np.nan)
        valid = (denominator > 0) & np.isfinite(denominator)
        cos[valid] = np.clip(numerator[valid] / denominator[valid], -1., 1.)
        self.cell_cosine.append(cos)

    def gene_correlations(self, eps):
        den = np.sqrt(np.maximum(self.vx, 0.) * np.maximum(self.vy, 0.))
        valid = (den > eps) & (self.n > 1)
        result = np.full(self.g, np.nan)
        result[valid] = np.clip(self.cross[valid] / den[valid], -1., 1.)
        return result

    def merge(self, other):
        """Merge a day accumulator without reprocessing its expression rows."""
        if not other.n:
            return
        total = self.n + other.n
        factor = self.n * other.n / total
        delta_x, delta_y = other.mx - self.mx, other.my - self.my
        self.vx += other.vx + delta_x * delta_x * factor
        self.vy += other.vy + delta_y * delta_y * factor
        self.cross += other.cross + delta_x * delta_y * factor
        self.mx += delta_x * other.n / total
        self.my += delta_y * other.n / total
        self.n = total
        self.sse += other.sse
        self.absolute += other.absolute
        self.dx += other.dx
        self.dy += other.dy
        # Share these arrays: medians do not depend on original row order.
        self.cell_corr.extend(other.cell_corr)
        self.cell_cosine.extend(other.cell_cosine)

    def statistics(self, policy, eps, minimum):
        cell = np.concatenate(self.cell_corr)
        gene = self.gene_correlations(eps)
        cp, cm, _, cf = _policy(cell, policy, minimum)
        gp, gm, contribution, gf = _policy(gene, policy, minimum)
        cell_cosine = np.concatenate(self.cell_cosine)
        finite_cosine = np.isfinite(cell_cosine)
        cosmean = float(cell_cosine[finite_cosine].mean()) if finite_cosine.any() else float("nan")
        cosfraction = float(finite_cosine.mean())
        total_sst = float(self.vy.sum() + self.n * np.sum((self.my - self.my.mean()) ** 2))
        detection = (self.dx - self.dy) / self.n
        variance_x = self.vx / (self.n - 1) if self.n > 1 else np.full(self.g, np.nan)
        variance_y = self.vy / (self.n - 1) if self.n > 1 else np.full(self.g, np.nan)
        variance_delta = variance_x - variance_y
        values = {
            "mean_cell_pcc": cp, "median_cell_pcc": cm,
            "mean_gene_pcc": gp, "median_gene_pcc": gm,
            "mean_row_cosine": cosmean,
            "mse": float(self.sse.sum() / (self.n * self.g)),
            "rmse": float(np.sqrt(self.sse.sum() / (self.n * self.g))),
            "mae": float(self.absolute.sum() / (self.n * self.g)),
            "r2": float(1. - self.sse.sum() / total_sst) if total_sst > 0 else float("nan"),
            "signed_bias": float(np.mean(self.mx - self.my)),
            "pseudobulk_pcc": pcc(self.mx, self.my, eps),
            "pseudobulk_spearman": pcc(rankdata(self.mx), rankdata(self.my), eps),
            "pseudobulk_rmse": float(np.sqrt(np.mean((self.mx - self.my) ** 2))),
            "pseudobulk_mae": float(np.mean(np.abs(self.mx - self.my))),
            "pseudobulk_cosine": cosine(self.mx, self.my),
            "variance_pcc": pcc(variance_x, variance_y, eps),
            "variance_rmse": float(np.sqrt(np.mean(variance_delta ** 2))),
            "detection_fraction_rmse": float(np.sqrt(np.mean(detection ** 2))),
            "detection_fraction_mae": float(np.mean(np.abs(detection))),
        }
        return values, gene, contribution, {"cell": cf, "gene": gf, "cosine": cosfraction}


def _identity_and_axes(handle, source):
    pred_key = source.get("matrix_key", "prediction_expression")
    required = [pred_key, "truth_expression", "prediction_cell_index", "truth_cell_index", "prediction_time", "truth_time", "gene_names"]
    missing = [key for key in required if key not in handle]
    if missing:
        raise ValueError("Canonical measured input is missing: " + ", ".join(missing))
    pred, truth = handle[pred_key], handle["truth_expression"]
    if len(pred.shape) != 2 or pred.shape != truth.shape or not all(pred.shape):
        raise ValueError("Prediction/truth must have identical nonempty cell-by-gene shape")
    n, g = pred.shape
    genes = _strings(handle["gene_names"][:])
    if len(genes) != g or len(set(genes)) != g:
        raise ValueError("Native gene names must be unique and match the matrix axis")
    for key in ("prediction_gene_names", "truth_gene_names"):
        if key in handle and not np.array_equal(_strings(handle[key][:]), genes):
            raise ValueError(f"{key} differs from the shared native gene axis")
    ids = _strings(handle["prediction_cell_index"][:])
    truth_ids = _strings(handle["truth_cell_index"][:])
    if len(ids) != n or not np.array_equal(ids, truth_ids):
        raise ValueError("Prediction/truth target cell identities are not exactly row paired")
    times, true_times = np.asarray(handle["prediction_time"][:], np.float64), np.asarray(handle["truth_time"][:], np.float64)
    if times.shape != (n,) or true_times.shape != (n,) or not np.isfinite(times).all() or not np.isfinite(true_times).all():
        raise ValueError("Prediction/truth developmental times must be finite row vectors")
    if not np.array_equal(times, true_times):
        raise ValueError("Prediction/truth target developmental times differ")
    attrs = {str(k): _attr(v) for k, v in handle.attrs.items()}
    if str(attrs.get('temporal_direction','forward')).lower() != 'forward':
        raise ValueError('Only forward measured prediction is supported')
    if str(attrs.get('model_version','')).lower() == 'v7' or str(source.get('model','')).startswith('v7_'):
        raise ValueError('Bidirectional BLOOM is excluded from this benchmark')
    if "row_aligned" in attrs:
        declaration = attrs["row_aligned"]
        if isinstance(declaration, str):
            declaration = declaration.lower() in {"true", "1", "yes"}
        if not bool(declaration):
            raise ValueError("Input declares unpaired rows")
    scenario = str(source.get("scenario", "measured")).lower()
    if scenario not in {"measured", "forward", "measured_forward", "normal"}:
        raise ValueError("Benchmark metrics only support measured forward prediction with saved GT")
    for key in ("scenario", "prediction_scenario", "task"):
        declared = str(attrs.get(key, "")).lower()
        if any(t in declared for t in ("history", "future", "reverse", "interpol", "perturb")):
            raise ValueError(f"Input {key} declares a non-forward benchmark task")
    panel = str(source.get("panel", "")).lower()
    if panel not in {"hvg", "author"}:
        raise ValueError("Native panel must be explicitly hvg or author")
    declared_panel = str(attrs.get("gene_panel", "")).lower()
    if declared_panel in {"200", "hvg", "hvg200", "hvg_200"}:
        declared_panel = "hvg"
    if declared_panel in {"hvg", "author"} and declared_panel != panel:
        raise ValueError("Input native panel disagrees with source panel")
    weighting = source.get("weighting", "unique_target")
    if weighting not in {"unique_target", "occurrence"}:
        raise ValueError("weighting must be unique_target or occurrence")
    duplicate_count = int(n - len(set(ids)))
    if weighting == "unique_target" and duplicate_count:
        raise ValueError("Unique-target weighting requires unique target cell IDs; duplicate predictions must first be averaged upstream")
    # Repeated occurrence GT for the same target is validated below during
    # digest construction, rather than silently permitting conflicting truth.
    return pred_key, ids, times, genes, attrs, duplicate_count


def _metadata_value(source, attrs, key, aliases, path_identity):
    supplied = source.get(key)
    declared = [attrs[k] for k in (key, *aliases) if k in attrs and _known(attrs[k])]
    if len(set(map(str, declared))) > 1:
        raise ValueError(f"Conflicting saved {key} metadata")
    if _known(supplied) and declared and str(supplied) != str(declared[0]):
        raise ValueError(f"Source and saved {key} metadata disagree")
    value = supplied if _known(supplied) else declared[0] if declared else None
    return str(value) if _known(value) else f"unknown_{key}:isolated_source:{path_identity}"


def _truth_digest(handle, source, ids, times, genes, chunk, normalization, split):
    """Hash ALL saved truth values in canonical identity/time/gene order.

    HDF5 fancy selections must be increasing, so each canonical-order chunk is
    read in source order and restored before hashing. This verifies complete
    numerical GT equality rather than relying on a sampled identity audit.
    """
    gene_order = np.argsort(genes, kind="stable")
    row_order = np.lexsort((np.arange(len(ids)), ids, times))
    digest = hashlib.sha256()
    header = {"dataset": source["dataset"], "panel": source["panel"], "normalization": normalization,
              "split": split, "weighting": source.get("weighting", "unique_target"),
              "contract_id": source.get("contract_id"), "gene_names": genes[gene_order].tolist(), "n_rows": len(ids)}
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    # Bound metadata encoding memory by the same chunk setting. Delimited IDs
    # and contiguous little-endian times produce identical byte streams for
    # any chunk size; expression bytes are hashed once per matrix chunk.
    digest.update(b"\0canonical_row_ids\0")
    for start in range(0, len(row_order), chunk):
        selected = row_order[start:start + chunk]
        digest.update(("\0".join(ids[selected]) + "\0").encode("utf-8"))
    digest.update(b"canonical_row_times\0")
    for start in range(0, len(row_order), chunk):
        selected_times = times[row_order[start:start + chunk]]
        digest.update(np.where(selected_times == 0., 0., selected_times).astype("<f8", copy=False).tobytes())
    digest.update(b"canonical_truth_matrix\0")
    previous_identity = None
    previous_truth = None
    for start in range(0, len(row_order), chunk):
        selected = row_order[start:start + chunk]
        ordered = np.argsort(selected)
        read_indices = selected[ordered]
        if len(read_indices) == 1 or np.all(np.diff(read_indices) == 1):
            block = handle["truth_expression"][int(read_indices[0]):int(read_indices[-1]) + 1]
        else:
            block = handle["truth_expression"][read_indices]
        values = np.asarray(block, np.float64)[np.argsort(ordered)][:, gene_order]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite saved truth; no imputation or clipping is performed")
        chunk_ids, chunk_times = ids[selected], times[selected]
        first_identity = (str(chunk_ids[0]), float(chunk_times[0]))
        if previous_identity == first_identity and not np.array_equal(previous_truth, values[0]):
            raise ValueError("Duplicate target occurrences contain conflicting numerical GT")
        duplicates = (chunk_ids[1:] == chunk_ids[:-1]) & (chunk_times[1:] == chunk_times[:-1])
        if duplicates.any() and not np.array_equal(values[1:][duplicates], values[:-1][duplicates]):
            raise ValueError("Duplicate target occurrences contain conflicting numerical GT")
        # Positive/negative zero are numerically equal. Canonicalize zero signs.
        canonical_values = np.where(values == 0., 0., values).astype("<f8", copy=False)
        digest.update(canonical_values.tobytes())
        previous_identity = (str(chunk_ids[-1]), float(chunk_times[-1]))
        previous_truth = values[-1].copy()
    return digest.hexdigest()[:24], header


def measure_source(source, options=None):
    """Return canonical long metric rows and a complete pairing/provenance audit."""
    options = options or {}
    policy = options.get("pcc_nan_policy", "zero")
    if policy not in {"zero", "skip"}:
        raise ValueError("pcc_nan_policy must be zero or skip")
    eps = float(options.get("pcc_eps", 1e-12))
    minimum = float(options.get("min_valid_fraction", options.get("pcc_min_valid_fraction", .05)))
    chunk = int(options.get("chunk_size", 32768))
    if not np.isfinite(eps) or eps <= 0 or not 0 <= minimum <= 1 or chunk < 1:
        raise ValueError("Require positive finite PCC EPS/chunk_size and min_valid_fraction in [0,1]")
    if options.get("pcc_backend", "fast") != "fast":
        raise ValueError("Raw recomputation supports the training fast PCC backend; SciPy allclose constant detection has a different definition")
    path = Path(source["path"]).expanduser().resolve()
    path_identity = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    with h5py.File(path, "r") as handle:
        pred_key, ids, times, genes, attrs, duplicate_count = _identity_and_axes(handle, source)
        normalization = _metadata_value(source, attrs, "normalization", (), path_identity)
        split = _metadata_value(source, attrs, "split", ("evaluation_split", "eval_split"), path_identity)
        cohort, contract = _truth_digest(handle, source, ids, times, genes, chunk, normalization, split)
        days = np.unique(times)
        full = Accumulator(len(genes))
        daily = {day: Accumulator(len(genes)) for day in days}
        for start in range(0, len(ids), chunk):
            stop = min(start + chunk, len(ids))
            x, y = np.asarray(handle[pred_key][start:stop], np.float64), np.asarray(handle["truth_expression"][start:stop], np.float64)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Nonfinite saved prediction/truth; no imputation or clipping is performed")
            chunk_days = np.unique(times[start:stop])
            if len(chunk_days) == 1:
                daily[chunk_days[0]].update(x, y, eps)
            else:
                for day in chunk_days:
                    mask = times[start:stop] == day
                    daily[day].update(x[mask], y[mask], eps)
        for day in days:
            full.merge(daily[day])

    base = {k: source.get(k, "") for k in ("dataset", "panel", "model", "family", "readout", "seed")}
    base.update(cohort_id=cohort, route="recomputed", weighting=source.get("weighting", "unique_target"),
                split=split, normalization=normalization, source_path=str(path),
                target_days=json.dumps([float(d) for d in days], separators=(",", ":")),
                comparison_status="verified" if not normalization.startswith("unknown_") and not split.startswith("unknown_") else "isolated_unknown_contract",
                pcc_nan_policy=policy, pcc_eps=eps, min_valid_fraction=minimum, pcc_backend="fast_centered", variance_ddof=1)
    rows = []
    definition_suffix = f"policy={policy};eps={eps:.17g};minimum={minimum:.17g};weighting={base['weighting']}"

    def emit(metric, value, scope, acc, day="", gene="", aggregation="", valid_fraction="", status=None):
        finite = bool(np.isfinite(value))
        rows.append({**base, "metric": metric, "value": float(value), "scope": scope, "day": day, "gene": gene,
                     "definition_id": f"raw_v1:{metric}:{aggregation}:{definition_suffix}", "aggregation": aggregation,
                     "status": status or ("available" if finite else "undefined_constant_or_insufficient_vector"),
                     "n_cells": acc.n, "n_genes": acc.g, "n_days": len(days) if scope in {"global", "gene"} else 1,
                     "valid_fraction": valid_fraction})

    daily_values = {}
    daily_components = []
    daily_gene_raw = []
    for scope, day, acc in [("global", "", full), *[("day", float(day), daily[day]) for day in days]]:
        values, gcorr, contribution, fractions = acc.statistics(policy, eps, minimum)
        if scope == "day":
            daily_values[day] = values
            daily_components.append(contribution)
            daily_gene_raw.append(gcorr)
        for metric, value in values.items():
            aggregation = "pooled_cells" if metric in {"mean_cell_pcc", "median_cell_pcc"} else "pooled_finite_row_cosines" if metric == "mean_row_cosine" else "pooled_genes" if metric in {"mean_gene_pcc", "median_gene_pcc"} else "all_cell_gene_entries" if metric in {"mse", "rmse", "mae", "r2", "signed_bias"} else "population_gene_statistics_ddof1" if metric.startswith("variance_") else "population_gene_means" if metric.startswith("pseudobulk_") else "fractions_of_saved_expression_gt0"
            vf = fractions["cell"] if "cell_pcc" in metric else fractions["gene"] if "gene_pcc" in metric else fractions["cosine"] if metric == "mean_row_cosine" else ""
            emit(metric, value, scope, acc, day, aggregation=aggregation, valid_fraction=vf)
        for j, gene in enumerate(genes):
            gene_scope = "gene" if scope == "global" else "day_gene"
            emit("gene_pcc", gcorr[j], gene_scope, acc, day, gene, "correlation_across_matched_cells", fractions["gene"])
            emit("gene_pcc_contribution", contribution[j], gene_scope, acc, day, gene, "policy_component" if policy == "zero" else "policy_component_g_over_n_valid", fractions["gene"])
            error_values = {"mse": acc.sse[j] / acc.n, "rmse": np.sqrt(acc.sse[j] / acc.n), "mae": acc.absolute[j] / acc.n,
                            "r2": 1. - acc.sse[j] / acc.vy[j] if acc.vy[j] > 0 else np.nan,
                            "signed_bias": acc.mx[j] - acc.my[j], "detection_fraction_error": (acc.dx[j] - acc.dy[j]) / acc.n,
                            "gene_variance_error": (acc.vx[j] - acc.vy[j]) / (acc.n - 1) if acc.n > 1 else np.nan}
            for metric, value in error_values.items():
                aggregation = "pred_minus_truth_population_variance_ddof1" if metric == "gene_variance_error" else "pred_minus_truth_fraction_gt0" if metric == "detection_fraction_error" else "matched_cells_for_one_gene"
                emit(metric, value, gene_scope, acc, day, gene, aggregation)

    for suffix, original in [("cell", "mean_cell_pcc"), ("gene", "mean_gene_pcc")]:
        emit(f"mean_day_pcc_{suffix}", np.mean([v[original] for v in daily_values.values()]), "global", full, aggregation="equal_day_mean", valid_fraction=np.mean([np.mean(np.isfinite(np.concatenate(daily[d].cell_corr))) if suffix == "cell" else np.mean(np.isfinite(daily[d].gene_correlations(eps))) for d in days]))
    daily_pb = [v["pseudobulk_pcc"] for v in daily_values.values()]
    valid_pb = np.asarray(daily_pb)[np.isfinite(daily_pb)]
    emit("mean_day_pseudobulk_pcc", valid_pb.mean() if len(valid_pb) else np.nan, "global", full, aggregation="equal_day_mean_valid_cross_gene_pcc", valid_fraction=len(valid_pb) / len(days))
    pm = np.vstack([daily[d].mx for d in days])
    tm = np.vstack([daily[d].my for d in days])
    emit("time_gene_pseudobulk_pcc", pcc(pm.ravel(), tm.ravel(), eps), "global", full, aggregation="flatten_equal_day_gene_means")
    components = np.vstack(daily_components)
    gene_daily_raw = np.vstack(daily_gene_raw)
    for j, gene in enumerate(genes):
        gene_mean = float(components[:, j].mean()) if policy == "zero" else _policy(gene_daily_raw[:, j], policy, minimum)[0]
        emit("mean_day_pcc_gene", gene_mean, "gene", full, gene=gene, aggregation="equal_day_policy_gene_component" if policy == "zero" else "mean_valid_day_gene_pcc_not_global_decomposition", valid_fraction=np.isfinite(gene_daily_raw[:, j]).mean())
        emit("gene_trajectory_pcc", pcc(pm[:, j], tm[:, j], eps), "gene", full, gene=gene, aggregation="correlation_across_equal_day_population_means")
        emit("gene_temporal_rmse", np.sqrt(np.mean((pm[:, j] - tm[:, j]) ** 2)), "gene", full, gene=gene, aggregation="equal_day_population_mean_error")
    audit = {"source_path": str(path), "cohort_id": cohort, "truth_verification": "all_saved_truth_values_exact_canonical_identity_time_gene_hash",
             "contract": contract, "n_cells": full.n, "n_genes": len(genes), "n_days": len(days), "days": days.tolist(),
             "duplicate_target_occurrences": duplicate_count, "pairing": "exact_row_aligned_cell_ids_times_native_gene_axis",
             "pcc_backend": "fast_centered", "numerical_note": "Centered moment merges avoid training streaming sum-of-squares cancellation; no input rounding. Printed logs have limited precision.",
             "normalization": normalization, "split": split, "comparability": base["comparison_status"],
             "variance_ddof": 1,
             "constant_policy": {"policy": policy, "eps": eps, "minimum": minimum},
             "per_gene_day_pcc_note": "Zero policy per-gene equal-day PCCs average to the global score. With skip policy, each gene's valid-day mean is descriptive and their mean need not equal the global score, whose valid-gene denominator varies by day. gene_pcc_contribution skip values are weighted additive components, may exceed [-1,1], and are not Pearson correlations.",
             "detection_note": "Fraction of saved expression values >0; only interpretable as biological detection if expression normalization preserves meaningful zero.",
             "cosine_note": "Row cosine averages only finite, nonzero-norm rows, independent of PCC zero/skip policy; all-zero norms give undefined cosine.",
             "r2_note": "Global/day R2 uses grand-mean flattened truth; per-gene R2 uses that gene's truth mean. Constant truth gives undefined R2 without force-finite replacement.",
             "inference_note": "Correlations across cells/genes are descriptive; these rows do not establish independent biological replication."}
    return {"rows": rows, "audit": audit, "status": "complete"}
