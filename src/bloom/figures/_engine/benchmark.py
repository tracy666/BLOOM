"""Optional paired measured-expression metrics, separate from biological questions.

Day-wise cell/gene PCC reproduces the supplied benchmark's equal-day averaging
and default constant-vector policy. Rankings stay within identical saved target,
gene and scale cohorts; mismatched model evaluation populations are not pooled.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .discovery import discover, FM_MODELS, TRANS_MODELS, _modern_identity
from .raw import read_source

METRICS = {
    "pseudobulk_pcc": (True, "PCC across genes between full saved-population mean expression"),
    "mean_day_pseudobulk_pcc": (True, "Equal-day mean of PCC across each day's gene means"),
    "time_gene_pseudobulk_pcc": (
        True,
        "PCC of flattened day-by-gene means; days equally represented",
    ),
    "mean_day_pcc_gene": (
        True,
        "Within day PCC across matched cells per gene, then equal-day mean",
    ),
    "mean_day_pcc_cell": (
        True,
        "Within day PCC across genes per matched cell, then equal-day mean",
    ),
    "rmse": (False, "Root mean squared residual over all saved matched cell-gene values"),
    "mae": (False, "Mean absolute residual over all saved matched cell-gene values"),
    "mean_gene_expression_pcc": (
        True,
        "PCC across genes of full population mean expression; equals pseudobulk_pcc",
    ),
    "detection_fraction_rmse": (False, "RMSE across genes of fractions of saved values > 0"),
    "mean_gene_variance_pcc": (
        True,
        "PCC across genes between saved population expression variances",
    ),
}


def _pcc(x, y, eps=1e-12):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.clip(np.sum(x * y) / den, -1, 1)) if den > eps else float("nan")


def _mean_valid(values, policy="zero", minimum=0.05):
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    if policy == "zero":
        return float(np.where(finite, values, 0).mean())
    return float(values[finite].mean()) if finite.mean() >= minimum else 0.0


class DayAccumulator:
    """Centered parallel moment merge makes correlations independent of chunks."""

    def __init__(self, n_genes):
        self.g = n_genes
        self.n = 0
        self.mx = np.zeros(n_genes)
        self.my = np.zeros(n_genes)
        self.vx = np.zeros(n_genes)
        self.vy = np.zeros(n_genes)
        self.cross = np.zeros(n_genes)
        self.dx = np.zeros(n_genes, np.int64)
        self.dy = np.zeros(n_genes, np.int64)
        self.cell_sum = 0.0
        self.cell_valid = 0
        self.sse = 0.0
        self.absolute = 0.0

    def update(self, x, y, eps=1e-12):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if not len(x):
            return
        n = len(x)
        bx, by = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - bx, y - by
        factor = self.n * n / (self.n + n)
        deltax, deltay = bx - self.mx, by - self.my
        self.vx += np.einsum("ij,ij->j", xc, xc) + deltax * deltax * factor
        self.vy += np.einsum("ij,ij->j", yc, yc) + deltay * deltay * factor
        self.cross += np.einsum("ij,ij->j", xc, yc) + deltax * deltay * factor
        self.mx += deltax * n / (self.n + n)
        self.my += deltay * n / (self.n + n)
        self.n += n
        self.dx += (x > 0).sum(axis=0)
        self.dy += (y > 0).sum(axis=0)
        cx, cy = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
        numerator = np.einsum("ij,ij->i", cx, cy)
        denominator = np.sqrt(np.einsum("ij,ij->i", cx, cx) * np.einsum("ij,ij->i", cy, cy))
        valid = denominator > eps
        self.cell_sum += np.clip(numerator[valid] / denominator[valid], -1, 1).sum()
        self.cell_valid += valid.sum()
        difference = x - y
        self.sse += np.einsum("ij,ij->", difference, difference)
        self.absolute += np.abs(difference).sum()

    def finalize(self, day, *, policy="zero", eps=1e-12, minimum=0.05):
        den = np.sqrt(np.maximum(self.vx, 0) * np.maximum(self.vy, 0))
        valid = (den > eps) & (self.n > 1)
        genes = np.full(self.g, np.nan)
        genes[valid] = np.clip(self.cross[valid] / den[valid], -1, 1)
        cell = (
            self.cell_sum / self.n
            if policy == "zero"
            else (
                self.cell_sum / self.cell_valid
                if self.cell_valid and self.cell_valid / self.n >= minimum
                else 0.0
            )
        )
        return {
            "day": float(day),
            "n_cells": self.n,
            "n_genes": self.g,
            "pcc_gene": _mean_valid(genes, policy, minimum),
            "pcc_cell": float(cell),
            "valid_gene_fraction": float(valid.mean()),
            "valid_cell_fraction": float(self.cell_valid / self.n),
            "pseudobulk_pcc": _pcc(self.mx, self.my, eps),
            "rmse": float(np.sqrt(self.sse / (self.n * self.g))),
            "mae": float(self.absolute / (self.n * self.g)),
        }


def _cohort_signature(source, meta):
    cells = meta["cells"]
    keys = ["cell_id", "stage_ordinal"] if "stage_ordinal" in cells else ["cell_id", "time"]
    rows = cells[keys].copy()
    rows[keys[-1]] = rows[keys[-1]].astype(float)
    rows = rows.sort_values(keys)
    attrs = meta["metadata"]
    dataset = source.get("dataset_config", {})
    scale = (
        attrs.get("normalization")
        or dataset.get("normalization")
        or "unknown_scale:" + source["source_id"]
    )
    split = (
        attrs.get("split")
        or attrs.get("evaluation_split")
        or attrs.get("eval_split")
        or source.get("run_spec", {}).get("split", "unspecified")
    )
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(rows, index=False).values.tobytes())
    digest.update(
        json.dumps(
            [source["dataset"], source["panel"], sorted(meta["genes"]), scale, split],
            sort_keys=True,
        ).encode()
    )
    # Confirm saved numerical GT at a deterministic exact-ID subset. This is
    # a declared bounded numerical audit, not a claim to have compared every
    # truth value across every model file.
    selected = rows.iloc[
        np.linspace(0, len(rows) - 1, min(64, len(rows))).astype(int)
    ].index.to_numpy(int)
    unique, inverse = np.unique(selected, return_inverse=True)
    with h5py.File(source["path"], "r") as h:
        values = np.asarray(h["truth_expression"][unique], dtype=np.float32)[inverse]
    gene_order = np.argsort(np.asarray(meta["genes"], dtype=str))
    digest.update(values[:, gene_order].tobytes())
    return digest.hexdigest()[:20], scale, split


def measure_source(source, options=None):
    """Compute all paired native-axis metrics in one full-row streaming pass."""
    options = options or {}
    if source.get("scenario") != "measured" or source.get("format") == "h5ad":
        raise ValueError("Metrics require a measured forward canonical H5 with matched GT")
    # Obs naming is irrelevant to metric pairing; avoid loading a huge original
    # metadata table a second time for this optional route.
    meta = read_source(source, {"datasets": {source["dataset"]: {"obs_path": None}}})
    if not meta["has_truth"]:
        raise ValueError("Measured source has no saved GT")
    cells = meta["cells"]
    times = (
        cells["stage_ordinal"].to_numpy() if "stage_ordinal" in cells else cells["time"].to_numpy()
    )
    policy = options.get("pcc_nan_policy", "zero")
    if policy not in {"zero", "skip"}:
        raise ValueError("pcc_nan_policy must be zero or skip")
    eps, minimum = float(options.get("pcc_eps", 1e-12)), float(
        options.get("pcc_min_valid_fraction", 0.05)
    )
    if eps <= 0 or not 0 <= minimum <= 1:
        raise ValueError("pcc_eps must be positive and pcc_min_valid_fraction must lie in [0,1]")
    g = len(meta["genes"])
    accumulators = {day: DayAccumulator(g) for day in np.unique(times)}
    full = DayAccumulator(g)
    chunk = int(options.get("chunk_size", 32768))
    if chunk < 1:
        raise ValueError("chunk_size must be positive")
    with h5py.File(source["path"], "r") as h:
        if "prediction_cell_index" not in h or "truth_cell_index" not in h:
            raise ValueError(
                "Benchmark requires exact saved prediction/truth target IDs; local row indices are insufficient"
            )
        if (
            h["truth_expression"].shape
            != h[source.get("matrix_key", "prediction_expression")].shape
        ):
            raise ValueError("Saved prediction/GT gene and cell axes differ")
        for start in range(0, len(cells), chunk):
            stop = min(start + chunk, len(cells))
            x = np.asarray(h[source.get("matrix_key", "prediction_expression")][start:stop], float)
            y = np.asarray(h["truth_expression"][start:stop], float)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Nonfinite saved prediction or truth; no clipping/imputation")
            full.update(x, y, eps)
            for day in np.unique(times[start:stop]):
                mask = times[start:stop] == day
                accumulators[day].update(x[mask], y[mask], eps)
    daily = [
        accumulators[d].finalize(d, policy=policy, eps=eps, minimum=minimum)
        for d in sorted(accumulators)
    ]
    px, py = np.vstack([accumulators[d].mx for d in sorted(accumulators)]), np.vstack(
        [accumulators[d].my for d in sorted(accumulators)]
    )
    cohort, scale, split = _cohort_signature(source, meta)
    pbulk = _pcc(full.mx, full.my, eps)
    row = {k: source.get(k) for k in ("source_id", "dataset", "model", "role", "panel", "path")}
    row.update(
        status="COMPLETE",
        cohort_id=cohort,
        n_cells=full.n,
        n_genes=g,
        n_days=len(daily),
        normalization=scale,
        split=split,
        scope="full_saved_unique_target_rows",
        pcc_nan_policy=policy,
        pseudobulk_pcc=pbulk,
        mean_day_pseudobulk_pcc=(
            float(np.nanmean([r["pseudobulk_pcc"] for r in daily]))
            if any(np.isfinite(r["pseudobulk_pcc"]) for r in daily)
            else float("nan")
        ),
        time_gene_pseudobulk_pcc=_pcc(px.ravel(), py.ravel(), eps),
        mean_day_pcc_gene=float(np.mean([r["pcc_gene"] for r in daily])),
        mean_day_pcc_cell=float(np.mean([r["pcc_cell"] for r in daily])),
        rmse=float(np.sqrt(full.sse / (full.n * g))),
        mae=float(full.absolute / (full.n * g)),
        mean_gene_expression_pcc=pbulk,
        detection_fraction_rmse=float(np.sqrt(np.mean(((full.dx - full.dy) / full.n) ** 2))),
        mean_gene_variance_pcc=_pcc(
            full.vx / max(1, full.n - 1), full.vy / max(1, full.n - 1), eps
        ),
        valid_gene_fraction_mean=float(np.mean([r["valid_gene_fraction"] for r in daily])),
        valid_cell_fraction_mean=float(np.mean([r["valid_cell_fraction"] for r in daily])),
        cohort_verification="exact saved target IDs/stages/gene identities/normalization/split plus <=64 exact-ID numeric GT rows; full cross-file numerical truth equality untested",
    )
    for r in daily:
        r.update(
            {k: row[k] for k in ("source_id", "dataset", "model", "role", "panel", "cohort_id")}
        )
    return row, daily


def _measure_task(source, options):
    try:
        row, daily = measure_source(source, options)
        return {"row": row, "daily": daily}
    except Exception as exc:
        return {
            "row": {
                **{
                    k: source.get(k)
                    for k in ("source_id", "dataset", "model", "role", "panel", "path")
                },
                "status": "UNSUPPORTED_INPUT",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "daily": [],
        }


def _plots(table, daily, out, options):
    from . import plot
    import matplotlib.pyplot as plt

    plot.style()
    files = []
    cols = [
        "pseudobulk_pcc",
        "mean_day_pcc_gene",
        "mean_day_pcc_cell",
        "rmse",
        "detection_fraction_rmse",
    ]
    for cohort, group in table[table["status"] == "COMPLETE"].groupby("cohort_id"):
        if len(group) < 2:
            continue
        labels = [plot.model_label(row, options) for row in group.to_dict("records")]
        directory = out / "figures" / str(cohort)
        for metric in cols:
            higher = METRICS[metric][0]
            local = group.sort_values(metric, ascending=not higher)
            fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(local))))
            colors = [
                plot.PALETTE[0] if role == "FM" else plot.PALETTE[1] for role in local["role"]
            ]
            ax.barh(range(len(local)), local[metric], color=colors)
            ax.set_yticks(
                range(len(local)),
                [plot.model_label(row, options) for row in local.to_dict("records")],
                fontsize=8,
            )
            ax.invert_yaxis()
            ax.set_xlabel(metric)
            ax.set_title(f"Matched cohort · {metric}")
            fig.tight_layout()
            files.extend(plot.save(fig, directory / f"rank_{metric}", options))
            plt.close(fig)
        values = group[cols].to_numpy(float)
        normalized = np.empty_like(values)
        for i, metric in enumerate(cols):
            v = values[:, i]
            span = np.nanmax(v) - np.nanmin(v) if np.isfinite(v).any() else 0
            normalized[:, i] = (v - np.nanmin(v)) / span if span > 0 else 0.5
            if not METRICS[metric][0]:
                normalized[:, i] = 1 - normalized[:, i]
        pd.DataFrame(normalized, index=group["model"], columns=cols).to_csv(
            directory / "radar_normalization.csv", index_label="model"
        )
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.28 * len(group))))
        im = ax.imshow(normalized, aspect="auto", vmin=0, vmax=1, cmap=plot.EXPRESSION_CMAP)
        ax.set_yticks(range(len(group)), labels, fontsize=8)
        ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=8)
        fig.colorbar(im, ax=ax, label="Within-cohort min–max score; higher = better")
        ax.set_title("Metric profile · identical evaluation cohort")
        fig.tight_layout()
        files.extend(plot.save(fig, directory / "metric_profile_heatmap", options))
        plt.close(fig)
        # Radars use four FM/leading TRANS only; 24 overlapping traces obscure.
        chosen = list(np.flatnonzero(group["role"].eq("FM").to_numpy()))
        trans = group["role"].eq("TRANS").to_numpy()
        chosen += sorted(np.flatnonzero(trans), key=lambda j: -values[j, 0])[:2]
        chosen = chosen[:6]
        if chosen:
            angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False)
            fig, ax = plt.subplots(figsize=(7, 5.8), subplot_kw={"projection": "polar"})
            for j in chosen:
                ax.plot(
                    np.r_[angles, angles[0]],
                    np.r_[normalized[j], normalized[j, 0]],
                    lw=1.7,
                    label=labels[j],
                )
            ax.set_xticks(angles, cols, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(
                "Within-cohort normalized metrics\nFM plus two highest pseudobulk TRANS", pad=25
            )
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=8)
            fig.tight_layout()
            files.extend(plot.save(fig, directory / "metric_profile_radar", options))
            plt.close(fig)
        local_daily = daily[daily["cohort_id"] == cohort]
        for column in ("pcc_gene", "pcc_cell", "rmse"):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for model, group_day in local_daily.groupby("model"):
                group_day = group_day.sort_values("day")
                ax.plot(
                    group_day["day"],
                    group_day[column],
                    "o-",
                    lw=1.2,
                    ms=3,
                    alpha=0.8,
                    label=options.get("aliases", {}).get(model, model),
                )
            ax.set_xlabel("Biological stage ordinal")
            ax.set_ylabel(column)
            ax.set_title("Matched per-stage metric")
            ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)
            fig.tight_layout()
            files.extend(plot.save(fig, directory / f"stage_{column}_lines", options))
            plt.close(fig)
    return [str(p) for p in files]


def _saved_metrics(config, sources, out):
    """Retain previously computed metrics with their original scope/definitions.

    V2 summary means, covariance/MMD approximations and V1 outputs have their
    own semantics. These are exported separately, never substituted for a
    verified paired full-row benchmark or silently combined with training logs.
    """
    runs = set()
    for value in config.get("v5_roots", []):
        root = Path(value).expanduser().resolve()
        for dataset in ("big", "small", "sted"):
            for model in FM_MODELS:
                run = root / dataset / "FM" / model
                if run.is_dir():
                    runs.add(run)
            for panel in ("200", "author"):
                for model in TRANS_MODELS:
                    run = root / dataset / "TRANS" / f"{model}_{panel}"
                    if run.is_dir():
                        runs.add(run)
    for source in sources:
        if source.get("run_root"):
            runs.add(Path(source["run_root"]))
        else:
            for run in Path(source["path"]).parents:
                if (run / "02_v2_quantitative").is_dir() or (run / "01_v1_evaluation").is_dir():
                    runs.add(run)
                    break
    frames, errors, found = [], [], []
    paths = [
        "02_v2_quantitative/02_result2_expression/tables/model_summary_metrics.csv",
        "02_v2_quantitative/02_result2_expression/tables/per_stage_metrics.csv",
        "01_v1_evaluation/02_measured_expression/all_model_summary.csv",
        "01_v1_evaluation/02_measured_expression/per_model_per_stage_metrics.csv",
    ]
    for run in sorted(runs):
        identity = _modern_identity(run) or {}
        for prefix in ("", "04_reverse_direction/"):
            for relative in paths:
                path = run / (prefix + relative)
                if not path.is_file():
                    continue
                try:
                    frame = pd.read_csv(path)
                    frame.insert(0, "saved_metric_table_path", str(path))
                    frame.insert(1, "saved_direction", "reverse" if prefix else "forward")
                    frame.insert(
                        2,
                        "saved_metric_scope",
                        "previous evaluation table; preserve original definitions and population/sampling policy",
                    )
                    for key in ("dataset", "model", "role", "panel"):
                        frame["slot_" + key] = identity.get(key, "legacy_unknown")
                    frames.append(frame)
                    found.append(str(path))
                except Exception as exc:
                    errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    combined = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else pd.DataFrame(
            columns=[
                "saved_metric_table_path",
                "saved_direction",
                "saved_metric_scope",
                "slot_dataset",
                "slot_model",
                "slot_role",
                "slot_panel",
            ]
        )
    )
    output = out / "PREVIOUS_EVALUATION_METRICS_SEPARATE.csv"
    combined.to_csv(output, index=False)
    return {"table": str(output), "source_tables": found, "rows": len(combined), "errors": errors}


def export_metrics(config, out, options=None):
    """Export full-row metrics and honest per-native-panel/cohort ranked CSVs."""
    options = dict(options or {})
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = discover(config)
    sources = [
        s
        for s in report["sources"]
        if s["scenario"] == "measured" and s["role"] in {"FM", "TRANS", "legacy"}
    ]
    if options.get("dataset", "all") != "all":
        sources = [s for s in sources if s["dataset"] == options["dataset"]]
    if options.get("panel", "all") != "all":
        panel = "hvg" if str(options["panel"]) == "200" else options["panel"]
        sources = [s for s in sources if s["panel"] == panel]
    from .prepare import _resources

    workers = _resources(options, max(1, len(sources)))
    rows, days = [], []
    from tqdm.auto import tqdm

    if workers == 1:
        results = (_measure_task(s, options) for s in sources)
        for result in tqdm(
            results, total=len(sources), desc="Paired measured metrics", unit="model"
        ):
            rows.append(result["row"])
            days.extend(result["daily"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_measure_task, s, options) for s in sources]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Paired measured metrics",
                unit="model",
            ):
                result = future.result()
                rows.append(result["row"])
                days.extend(result["daily"])
    table = pd.DataFrame(rows)
    daily = pd.DataFrame(days)
    if table.empty:
        table = pd.DataFrame(
            columns=[
                "source_id",
                "dataset",
                "model",
                "role",
                "panel",
                "status",
                "cohort_id",
                *METRICS,
            ]
        )
    if "cohort_id" not in table:
        table["cohort_id"] = ""
    table = table.sort_values(["dataset", "panel", "role", "model"]).reset_index(drop=True)
    if daily.empty:
        daily = pd.DataFrame(
            columns=[
                "source_id",
                "dataset",
                "model",
                "role",
                "panel",
                "cohort_id",
                "day",
                "n_cells",
                "n_genes",
                "pcc_gene",
                "pcc_cell",
                "pseudobulk_pcc",
                "rmse",
                "mae",
                "valid_gene_fraction",
                "valid_cell_fraction",
            ]
        )
    else:
        daily = daily.sort_values(["dataset", "panel", "role", "model", "day"]).reset_index(
            drop=True
        )
    table.to_csv(out / "ALL_MODEL_METRICS.csv", index=False)
    daily.to_csv(out / "PER_DAY_METRICS.csv", index=False)
    ranked_paths = []
    for (dataset, panel), group in table.groupby(["dataset", "panel"]):
        directory = out / str(dataset) / str(panel)
        directory.mkdir(parents=True, exist_ok=True)
        for metric, (higher, _) in METRICS.items():
            local = group.copy()
            if metric not in local:
                local[metric] = np.nan
            local["rank_within_identical_cohort"] = local.groupby("cohort_id")[metric].rank(
                method="min", ascending=not higher
            )
            local["n_models_in_identical_cohort"] = local.groupby("cohort_id")["status"].transform(
                lambda x: int(x.eq("COMPLETE").sum())
            )
            local["fm_is_best_in_cohort"] = False
            local["best_fm_score_in_cohort"] = local.groupby("cohort_id")[metric].transform(
                lambda x: np.nan
            )
            for cohort, sub in local[local["status"] == "COMPLETE"].groupby("cohort_id"):
                fm = sub[sub["role"] == "FM"][metric].dropna()
                if len(fm):
                    best = fm.max() if higher else fm.min()
                    local.loc[sub.index, "best_fm_score_in_cohort"] = best
                    overall = sub[metric].max() if higher else sub[metric].min()
                    local.loc[sub.index, "fm_is_best_in_cohort"] = bool(
                        len(sub) > 1 and np.isfinite(overall) and np.isclose(best, overall)
                    )
            local = local.sort_values(
                ["cohort_id", "rank_within_identical_cohort"], na_position="last"
            )
            path = directory / f"RANKED_{metric}.csv"
            local.to_csv(path, index=False)
            ranked_paths.append(str(path))
        group.to_csv(directory / "NATIVE_PANEL_METRICS.csv", index=False)
    figures = (
        _plots(
            table,
            daily,
            out,
            {
                "formats": ["png"],
                "variants": ["legend", "no_legend"],
                **options,
                "aliases": {**config.get("aliases", {}), **options.get("aliases", {})},
            },
        )
        if options.get("figures", True) and len(days)
        else []
    )
    coverage = []
    for dataset in sorted({s["dataset"] for s in sources}):
        for panel in ("hvg", "author"):
            expected = [
                ("FM", model) for model in FM_MODELS if ("author" in model) == (panel == "author")
            ] + [("TRANS", model) for model in TRANS_MODELS]
            for role, model in expected:
                present = table[
                    (table["dataset"] == dataset)
                    & (table["panel"] == panel)
                    & (table["model"] == model)
                    & (table["role"] == role)
                ]
                coverage.append(
                    {
                        "dataset": dataset,
                        "panel": panel,
                        "role": role,
                        "model": model,
                        "status": (
                            present.iloc[0]["status"]
                            if len(present)
                            else "MISSING_SAVED_PREDICTION"
                        ),
                    }
                )
    pd.DataFrame(coverage, columns=["dataset", "panel", "role", "model", "status"]).to_csv(
        out / "EXPECTED_MODEL_COVERAGE.csv", index=False
    )
    saved = _saved_metrics(config, sources, out)
    audit = {
        "definitions": {
            k: {"higher_is_better": high, "definition": definition}
            for k, (high, definition) in METRICS.items()
        },
        "pcc_nan_policy": options.get("pcc_nan_policy", "zero"),
        "pcc_eps": options.get("pcc_eps", 1e-12),
        "pcc_min_valid_fraction": options.get("pcc_min_valid_fraction", 0.05),
        "training_code_semantics": "code_v5/version2::evaluate_predictions_unified and StreamingDayPCCAccumulator",
        "scope": "Saved unique-target-cell measured predictions. Training occurrence-weighted or differently split log metrics are not silently merged.",
        "rank_rule": "Rank only within exact saved target-ID/stage/gene-identity/normalization/split cohort, plus <=64 deterministic exact-ID numeric GT rows; full numerical GT equality across files is untested. Unknown scales stay source-specific. Different cohorts in the same native-panel CSV remain separate rankings.",
        "radar_rule": "Within each comparable cohort, each metric min-max scaled to 0..1; error directions inverted; constant metrics shown at .5. The normalization CSV is exported beside the radar; displayed models are FM plus two highest pseudobulk TRANS.",
        "gene_selection": "Metrics do not filter biological genes or biological questions.",
        "discovery_errors": report["errors"],
        "sources": len(sources),
        "complete": int(table["status"].eq("COMPLETE").sum()),
    }
    audit["previous_evaluation_tables"] = saved
    (out / "METRIC_DEFINITIONS.json").write_text(json.dumps(audit, indent=2) + "\n")
    return {
        "status": (
            "complete"
            if audit["complete"]
            else "saved_tables_only" if saved["rows"] else "no_eligible_inputs"
        ),
        "sources": len(sources),
        "complete": audit["complete"],
        "tables": [
            str(out / "ALL_MODEL_METRICS.csv"),
            str(out / "PER_DAY_METRICS.csv"),
            str(out / "EXPECTED_MODEL_COVERAGE.csv"),
            saved["table"],
            *ranked_paths,
        ],
        "figures": figures,
        "notes": [
            audit["scope"],
            audit["rank_rule"],
            "Previous evaluation tables are retained separately with original source paths and scope; covariance/MMD columns are not redefined or silently mixed with newly streamed metrics.",
        ],
    }
