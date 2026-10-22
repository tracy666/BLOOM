"""Biology-first measured/reverse expression views from prepared saved arrays.

No model inference occurs here. Common reference geometry is supplied by prepare;
full-row summaries and declared per-cell drawing samples have separate roles.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from . import plot

VIEWS = (
    "atlas_time",
    "atlas_state",
    "stage_storyboard",
    "trajectory",
    "state_trajectory",
    "density",
    "density_residual",
    "density_contour",
    "density_state",
    "gene_feature",
    "gene_timecourse",
    "gene_distribution",
    "gene_detection",
    "gene_phase",
    "marker_dotplot",
    "marker_matrix",
    "marker_violin",
    "cascade",
    "program_heatmap",
    "program_timecourse",
    "program_feature",
    "state_graph",
    "state_occupancy",
    "coexpression",
    "grn_network",
    "grn_activity",
    "grn_support",
    "candidate_evidence",
    "regulator_evidence",
    "author_graph",
    "edge_rescue",
)
GENE_VIEWS = {
    "gene_feature",
    "gene_timecourse",
    "gene_distribution",
    "gene_detection",
    "gene_phase",
}
PROGRAM_VIEWS = {"program_timecourse", "program_feature"}


def _finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def _limits(a, quantiles=(0.005, 0.995)):
    a = _finite(a)
    if not len(a):
        return (-1.0, 1.0)
    lo, hi = np.quantile(a, quantiles)
    pad = max((hi - lo) * 0.04, abs(lo) * 0.01, 1e-5)
    return (float(lo - pad), float(hi + pad))


def _missing(ax, text="Not covered by this native gene panel"):
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=8,
        color="#59636C",
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _cells(d):
    c = d["cells"]
    if not len(c) or not {"x", "y"}.issubset(c.columns):
        return pd.DataFrame()
    return c.loc[np.isfinite(c["x"]) & np.isfinite(c["y"])]


def _values(d, gene, cells=None):
    genes = list(map(str, d["genes"]))
    if gene not in genes or d.get("expression") is None:
        return None
    cells = d["cells"] if cells is None else cells
    rows = (
        cells["expression_row"].to_numpy(int)
        if "expression_row" in cells
        else cells.index.to_numpy(int)
    )
    expression = d["expression"]
    if rows.size and (rows.min() < 0 or rows.max() >= expression.shape[0]):
        raise ValueError("Prepared expression_row exceeds expression sample dimensions")
    return np.asarray(expression[rows, genes.index(gene)], dtype=float)


def _fraction(values, d):
    values = np.asarray(values, dtype=float)
    unit = str(d.get("meta", {}).get("detection_units", "")).lower()
    observed = []
    for key in ("stage", "state"):
        table = d.get(key, pd.DataFrame())
        if "pct" in table:
            observed.extend(_finite(table["pct"]).tolist())
    percent = unit in ("percent", "percentage", "0..100") or (
        not unit and observed and max(observed) > 1.0 + 1e-8
    )
    return values / 100.0 if percent else values


def _groups(data, entity):
    # One GT on the same available native gene axis; preserve all model panels.
    gt = [d for d in data if d["source"].get("role") == "GT" and entity in set(d["genes"])]
    chosen = gt[0] if gt else None
    return ([chosen] if chosen is not None else []) + [
        d for d in data if d["source"].get("role") != "GT"
    ]


def _label(d, options):
    try:
        return plot.model_label(d["source"], options)
    except AttributeError:
        m = d["source"].get("model", "unknown")
        return options.get("aliases", {}).get(m, m)


def _cfg(cohort, options):
    if options.get("dataset_config"):
        return options["dataset_config"]
    return options.get("datasets", {}).get(cohort.get("dataset"), cohort.get("dataset_config", {}))


def _read_annotation(path):
    if not path or not Path(path).is_file():
        return pd.DataFrame()
    p = Path(path)
    return (
        pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, sep=None, engine="python")
    )


def _absolute_program(program):
    """Aggregate supplied signed rows using the legacy absolute-weight score."""
    if program.empty:
        return program.copy()
    table = program.dropna(subset=["gene"]).copy()
    if "program" not in table:
        table["program"] = "program"
    table["gene"] = table["gene"].astype(str)
    table = table[table["gene"].ne("")]
    signed = (
        pd.to_numeric(table["weight"], errors="coerce").to_numpy(float)
        if "weight" in table
        else np.ones(len(table))
    )
    invalid = ~np.isfinite(signed)
    table["weight"] = np.where(invalid, 0.0, np.abs(signed))
    if "raw_signed_weight" in table:
        raw = pd.to_numeric(table["raw_signed_weight"], errors="coerce").to_numpy(float)
        table["raw_signed_weight"] = np.where(np.isfinite(raw), raw, 0.0)
    else:
        table["raw_signed_weight"] = np.where(invalid, 0.0, signed)
    if "input_rows" not in table:
        table["input_rows"] = 1
    if "invalid_weight_rows" not in table:
        table["invalid_weight_rows"] = invalid.astype(int)
    if "scoring_basis" not in table:
        table["scoring_basis"] = "normalized absolute supplied weights"
    if "weight_source" not in table:
        table["weight_source"] = "supplied program rows"
    return table.groupby(["program", "gene"], sort=False, as_index=False).agg(
        weight=("weight", "sum"),
        raw_signed_weight=("raw_signed_weight", "sum"),
        input_rows=("input_rows", "sum"),
        invalid_weight_rows=("invalid_weight_rows", "sum"),
        scoring_basis=("scoring_basis", "first"),
        weight_source=("weight_source", "first"),
    )


def _programs(cohort, options, grn=False, data=None):
    cfg = _cfg(cohort, options)
    table = _read_annotation(cfg.get("edges_path" if grn else "programs_path"))
    if table.empty:
        if grn or not data:
            return pd.DataFrame(columns=["program", "gene", "weight"])
        # Reference-state marker programs are useful data-backed biology when
        # no author program file exists. They are explicitly named and never
        # passed off as author-curated pathways.
        rows = []
        allowed = set(options.get("selected_gene_names", []))
        seen = set()
        for d in data:
            if d["source"].get("role") != "GT":
                continue
            t = d["state"]
            if t.empty or not {"state", "gene", "mean", "n"}.issubset(t):
                continue
            t = t[t["gene"].isin(allowed)] if allowed else t
            for state, g in t.groupby("state", sort=True):
                effects = []
                for gene, q in g.groupby("gene", sort=False):
                    n = q["n"].to_numpy(float)
                    inside = float(np.average(q["mean"], weights=np.maximum(n, 1)))
                    outside = t[(t["gene"] == gene) & (t["state"] != state)]
                    if outside.empty:
                        continue
                    out = float(np.average(outside["mean"], weights=np.maximum(outside["n"], 1)))
                    effects.append((str(gene), inside - out))
                for gene, effect in sorted(effects, key=lambda x: x[1], reverse=True)[:6]:
                    key = (str(state), gene)
                    if effect > 0 and key not in seen:
                        rows.append(
                            {
                                "program": "Reference-state markers: " + str(state),
                                "gene": gene,
                                "weight": float(effect),
                            }
                        )
                        seen.add(key)
        result = pd.DataFrame(rows, columns=["program", "gene", "weight"])
        result["scoring_basis"] = "normalized positive observed state-contrast weights"
        result["weight_source"] = (
            "observed full-row state summaries; exploratory state marker program"
        )
        return _absolute_program(result)
    low = {str(c).lower(): c for c in table}
    first = next(
        (
            low[c]
            for c in (
                ("source", "tf", "regulator", "from")
                if grn
                else (
                    "program",
                    "node",
                    "module",
                    "program_name",
                    "driver_program",
                    "celltype",
                    "cell_type",
                    "cluster",
                )
            )
            if c in low
        ),
        None,
    )
    gene = next(
        (
            low[c]
            for c in (
                ("target", "gene", "to") if grn else ("gene", "gene_name", "target", "symbol")
            )
            if c in low
        ),
        None,
    )
    weight = next(
        (
            low[c]
            for c in ("weight", "prior_weight", "importance", "score", "signed_weight")
            if c in low
        ),
        None,
    )
    if first is None or gene is None:
        return pd.DataFrame(columns=["program", "gene", "weight"])
    result = pd.DataFrame(
        {
            "program": table[first].astype(str),
            "gene": table[gene].astype(str),
            "weight": 1.0 if weight is None else pd.to_numeric(table[weight], errors="coerce"),
        }
    )
    if weight is None:
        result = result.drop_duplicates(["program", "gene"])
        result["scoring_basis"] = "uniform membership weights; no supplied weight column"
    else:
        result["scoring_basis"] = "normalized absolute supplied weights"
    result["weight_source"] = str(cfg.get("edges_path" if grn else "programs_path"))
    return _absolute_program(result)


def _reference_match(d, data):
    candidates = [x for x in data if x["source"].get("role") in ("GT", "REFERENCE")]
    if not candidates:
        return None, "no observed native reference"
    scale = d.get("meta", {}).get("scale_group")
    if scale:
        candidates = [x for x in candidates if x.get("meta", {}).get("scale_group") == scale]
        if not candidates:
            return None, "no observed reference in the model scale group"
    genes = set(d["genes"])
    native = plot.source_panel({**d.get("meta", {}), **d["source"]})
    same_panel = [
        x
        for x in candidates
        if native is not None and plot.source_panel({**x.get("meta", {}), **x["source"]}) == native
    ]
    # Gene overlap does not make two native assays interchangeable.
    if not same_panel:
        return None, "no observed reference in the native assay"
    candidates = same_panel

    def metadata(item):
        return {**item.get("meta", {}), **item.get("source", {})}

    own = metadata(d)
    identities = {str(own.get(key)) for key in ("source_id", "parent_source_id") if own.get(key)}
    if d["source"].get("role") in ("GT", "REFERENCE"):
        found = next(
            (
                ref
                for ref in candidates
                if ref is d
                or (own.get("source_id") and metadata(ref).get("source_id") == own.get("source_id"))
            ),
            None,
        )
        if found is not None:
            return found, "observed population itself"
    exact = []

    def identity_values(value):
        if value is None:
            return set()
        if isinstance(value, str):
            return {value}
        return set(map(str, value))

    for ref in candidates:
        meta = metadata(ref)
        parents = {str(meta.get("parent_source_id"))} if meta.get("parent_source_id") else set()
        parents.update(identity_values(meta.get("matching_parent_sources")))
        observations = {str(meta.get("source_id"))} if meta.get("source_id") else set()
        observations.update(identity_values(meta.get("matching_observation_sources")))
        # Exact identity/provenance, never fuzzy path or gene-name matching.
        if identities & parents or identities & observations:
            exact.append(ref)
    if exact:
        chosen = max(
            exact,
            key=lambda ref: (ref["source"].get("role") == "GT", len(genes & set(ref["genes"]))),
        )
        return chosen, "exact parent/source observation binding"
    chosen = max(
        candidates, key=lambda x: (len(genes & set(x["genes"])), x["source"].get("role") == "GT")
    )
    return chosen, "descriptive same-native reference; exact model truth binding unavailable"


def _reference(d, data):
    return _reference_match(d, data)[0]


def _reference_provenance(d, data):
    ref, method = _reference_match(d, data)
    target = set(
        pd.to_numeric(
            d.get("stage", pd.DataFrame()).get("time", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
    )
    observed = (
        set(
            pd.to_numeric(
                ref.get("stage", pd.DataFrame()).get("time", pd.Series(dtype=float)),
                errors="coerce",
            ).dropna()
        )
        if ref is not None
        else set()
    )
    return {
        "observed_model": ref["source"].get("model", "") if ref is not None else "",
        "observed_source_id": ref["source"].get("source_id", "") if ref is not None else "",
        "observed_reference_id": (
            ref.get("meta", {}).get("reference_id", "") if ref is not None else ""
        ),
        "reference_match": method,
        "paired_source": method
        in ("exact parent/source observation binding", "observed population itself"),
        "model_target_stage_count": len(target),
        "observed_target_stage_count": len(observed),
        "matched_target_stage_count": len(target & observed),
        "model_target_stages": ";".join(map(str, sorted(target))),
        "observed_target_stages": ";".join(map(str, sorted(observed))),
    }


def _moments(d):
    table = d["stage"]
    if table.empty or not {"gene", "mean"}.issubset(table):
        return {}
    result = {}
    for gene, g in table.groupby("gene", sort=False):
        n = g["n"].to_numpy(float) if "n" in g else np.ones(len(g))
        means = g["mean"].to_numpy(float)
        std = g["std"].to_numpy(float) if "std" in g else np.full(len(g), np.nan)
        valid = np.isfinite(n) & np.isfinite(means) & (n > 0)
        n, means, std = n[valid], means[valid], std[valid]
        if not len(n):
            continue
        total = float(n.sum())
        mu = float(np.average(means, weights=n))
        ddof = int(d.get("meta", {}).get("std_ddof", 1))
        within = (
            float(np.sum(np.maximum(n - ddof, 0) * std**2)) if np.isfinite(std).all() else np.nan
        )
        variance = (
            (within + float(np.sum(n * (means - mu) ** 2))) / (total - 1) if total > 1 else np.nan
        )
        result[str(gene)] = (mu, math.sqrt(max(variance, 0)) if np.isfinite(variance) else np.nan)
    return result


def _program_policy(d, genes, data, standardized=True):
    ref = _reference(d, data)
    moments = _moments(ref) if ref is not None else {}
    if not standardized:
        mode, reason = "raw_expression", "raw scoring requested"
    elif ref is None:
        mode, reason = "raw_expression", "no compatible observed native/scale reference"
    elif not all(
        g in moments and np.isfinite(moments[g][1]) and moments[g][1] > 1e-12 for g in genes
    ):
        mode, reason = (
            "raw_expression",
            "missing, undefined or constant observed member SD; no artificial SD floor",
        )
    else:
        mode, reason = (
            "observed_reference_standardized",
            "full-row observed mean and pooled sample SD",
        )
    return mode, reason, ref, moments


def _program_audit(d, program, data, standardized=True):
    p = _absolute_program(program)
    members = p["gene"].astype(str).tolist()
    weights = p.set_index("gene")["weight"].to_dict()
    covered = [g for g in members if g in d["genes"]]
    scoring = [g for g in covered if weights[g] > 0]
    mode, reason, ref, _ = _program_policy(d, scoring, data, standardized)
    full = float(sum(weights.values()))
    mass = float(sum(weights[g] for g in covered))
    if not scoring:
        mode, reason = "undefined", "no positive native covered weight mass"
    return {
        "model": d["source"].get("model"),
        "panel": d["source"].get("panel"),
        "program": str(p["program"].iloc[0]) if len(p) else "",
        "total_genes": len(members),
        "n_covered": len(covered),
        "n_scoring_members": len(scoring),
        "coverage": len(covered) / len(members) if members else np.nan,
        "original_members": ";".join(members),
        "covered_members": ";".join(covered),
        "missing_members": ";".join(g for g in members if g not in covered),
        "absolute_weights": json.dumps(weights, sort_keys=True),
        "raw_signed_weights": json.dumps(
            p.set_index("gene")["raw_signed_weight"].to_dict(), sort_keys=True
        ),
        "full_weight_mass": full,
        "covered_weight_mass": mass,
        "weighted_coverage": mass / full if full > 0 else np.nan,
        "scoring_basis": str(p["scoring_basis"].iloc[0]) if len(p) else "",
        "weight_source": str(p["weight_source"].iloc[0]) if len(p) else "",
        "input_rows": int(p["input_rows"].sum()),
        "invalid_weight_rows": int(p["invalid_weight_rows"].sum()),
        "score_scale": mode,
        "scale_reason": reason,
        "observed_model": ref["source"].get("model") if ref is not None else "",
        "observed_source_id": ref["source"].get("source_id", "") if ref is not None else "",
        **_reference_provenance(d, data),
        "weight_formula": "sum(abs(supplied row weight) * member value) / covered absolute weight mass; duplicate rows aggregated by exact gene",
        "reference_sd_formula": "sqrt((sum((n-std_ddof)*std^2)+sum(n*(stage_mean-global_mean)^2))/(sum(n)-1))",
    }


def _program_scores(d, program, data, standardized=True):
    program = _absolute_program(program)
    lookup = {g: i for i, g in enumerate(d["genes"])}
    inside = program[program["gene"].isin(lookup)]
    if inside.empty or d.get("expression") is None:
        return None, []
    genes = inside["gene"].astype(str).tolist()
    ids = np.array([lookup[g] for g in genes], int)
    weights = inside["weight"].to_numpy(float)
    if weights.sum() <= 0:
        return None, genes
    weights = weights / weights.sum()
    x = np.asarray(d["expression"][:, ids], float)
    positive = np.flatnonzero(weights > 0)
    mode, _, _, moments = _program_policy(d, [genes[i] for i in positive], data, standardized)
    if mode == "observed_reference_standardized":
        x[:, positive] = (
            x[:, positive] - np.array([moments[genes[i]][0] for i in positive])
        ) / np.array([moments[genes[i]][1] for i in positive])
    return x @ weights, genes


def _aggregate_program(d, program, data):
    program = _absolute_program(program)
    stage = d["stage"]
    if stage.empty:
        return pd.DataFrame()
    genes = program["gene"].astype(str).tolist()
    selected = stage[stage["gene"].isin(genes)].copy()
    if selected.empty:
        return pd.DataFrame()
    weights = program.set_index("gene")["weight"].to_dict()
    genes = selected["gene"].unique().tolist()
    norm = sum(weights[g] for g in genes)
    if norm <= 0:
        return pd.DataFrame()
    selected["weight"] = selected["gene"].map(weights)
    selected["value"] = pd.to_numeric(selected["mean"], errors="coerce").astype(np.float64)
    mode, reason, _, moments = _program_policy(d, [g for g in genes if weights[g] > 0], data)
    if mode == "observed_reference_standardized":
        positive = selected["weight"].gt(0)
        selected.loc[positive, "value"] = (
            selected.loc[positive, "mean"]
            - selected.loc[positive, "gene"].map({g: moments[g][0] for g in genes if g in moments})
        ) / selected.loc[positive, "gene"].map({g: moments[g][1] for g in genes if g in moments})
    selected = selected[np.isfinite(selected["value"])]
    selected["value"] *= selected["weight"]
    result = selected.groupby("time", sort=True, as_index=False).agg(
        value=("value", "sum"), covered_weight_mass=("weight", "sum"), n_covered=("gene", "nunique")
    )
    result["value"] /= result["covered_weight_mass"].replace(0, np.nan)
    result["score_scale"] = mode
    result["scale_reason"] = reason
    return result


def _state_occupancy(d, states, known_states=None):
    """Inverse-sampling-weight composition, with actual drawn counts recorded."""
    c = d["cells"].copy()
    if c.empty or not {"time", "state"}.issubset(c):
        return pd.DataFrame()
    if "sample_weight" in c:
        weights = pd.to_numeric(c["sample_weight"], errors="coerce").to_numpy(float)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("State occupancy requires finite nonnegative sample_weight")
        method = "sum of declared inverse sampling probability sample_weight"
    else:
        weights = np.ones(len(c))
        method = "unit drawing weights; no sample_weight supplied"
    c["_weight"] = weights
    unknown = "Unknown/unassigned"
    c["state"] = c["state"].fillna(unknown).astype(str)
    unusable = (
        c["state"]
        .str.lower()
        .isin(("unknown", "unavailable", "nan", "none", "", "unknown/unassigned"))
    )
    omitted = ~c["state"].isin(states) & ~unusable
    c.loc[omitted, "state"] = "Other observed/model states"
    c.loc[unusable, "state"] = unknown
    columns = list(states)
    for label in ("Other observed/model states", unknown):
        if c["state"].eq(label).any() and label not in columns:
            columns.append(label)
    count = c.groupby(["time", "state"], observed=True).agg(
        sample_count=("_weight", "size"), weight_mass=("_weight", "sum")
    )
    grid = pd.MultiIndex.from_product(
        [sorted(c["time"].dropna().unique()), columns], names=["time", "state"]
    )
    result = count.reindex(grid, fill_value=0).reset_index()
    totals = c.groupby("time", observed=True).agg(
        total_sample_count=("_weight", "size"), total_weight_mass=("_weight", "sum")
    )
    result = result.merge(totals, on="time", how="left")
    result["fraction"] = result["weight_mass"] / result["total_weight_mass"].replace(0, np.nan)
    result["sample_count"] = result["sample_count"].astype(int)
    result["method"] = method
    result["state_assignment_method"] = d.get("meta", {}).get("state_method", "unspecified")
    return result


def _density(xy, extent, bins=72, sigma=1.3, weights=None):
    if not len(xy):
        return np.zeros((bins, bins))
    h, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=bins, range=[extent[:2], extent[2:]], weights=weights
    )
    h = gaussian_filter(h.T, sigma=sigma)
    return h / max(h.sum(), 1.0)


def _geometry(data):
    arrays = [_cells(d)[["x", "y"]].to_numpy(float) for d in data if len(_cells(d))]
    allxy = np.concatenate(arrays) if arrays else np.zeros((1, 2))
    # Entire sampled support is retained; rare model outliers extend axes.
    xl = _limits(allxy[:, 0], (0.0, 1.0))
    yl = _limits(allxy[:, 1], (0.0, 1.0))
    return [*xl, *yl]


def _state_labels(data):
    return sorted(
        {
            str(s)
            for d in data
            if "state" in d["cells"]
            for s in d["cells"]["state"].dropna()
            if str(s).lower() not in ("unknown", "unavailable", "nan", "none", "")
        }
    )


def _state_selection(data, primary, options, requested=()):
    """Freeze a readable state subset from actual observed counts, never model wins."""
    labels = set(_state_labels(data))
    full = {}
    sampled = {}
    counts = []
    if primary is not None:
        t = primary.get("state", pd.DataFrame())
        if not t.empty and {"state", "n"}.issubset(t):
            group = ["state"] + (["time"] if "time" in t else [])
            # Every gene repeats the same state count; never sum across genes.
            n = t.groupby(group, dropna=False)["n"].max().groupby("state").sum()
            full = {str(k): float(v) for k, v in n.items() if np.isfinite(v) and v > 0}
        c = primary.get("cells", pd.DataFrame())
        if "state" in c:
            weights = pd.to_numeric(
                c.get("sample_weight", pd.Series(1.0, index=c.index)), errors="coerce"
            ).fillna(0.0)
            sampled = weights.groupby(c["state"].astype(str)).sum().to_dict()
    labels.update(full)
    labels.update(sampled)
    usable = {
        s
        for s in labels
        if s.lower() not in ("unknown", "unavailable", "nan", "none", "", "unknown/unassigned")
    }
    # Coordinate states may be expression-transferred while the full-row
    # summary stores conditioning strata. Freeze names on the actual GT map
    # vocabulary when available, then use matching full-row counts to rank.
    observed = {s for s in usable if sampled.get(s, 0) > 0}
    if not observed:
        observed = {s for s in usable if full.get(s, 0) > 0}
    requested = set(map(str, requested or ()))
    pool = observed & requested if requested else observed
    # A GT-free reverse fixture remains drawable, with explicit provenance.
    basis = "observed full-row state counts, sample weighted mass as fallback"
    if primary is None:
        pool = usable & requested if requested else usable
        basis = "no observed state vocabulary; source names only, descriptive"
    ranked = sorted(pool, key=lambda s: (-full.get(s, 0), -sampled.get(s, 0), s))
    cap = int(options.get("max_display_states", 12))
    selected = ranked if cap == 0 else ranked[: max(cap, 0)]
    for state in sorted(usable):
        row = {
            "state": state,
            "selected": state in selected,
            "gt_supported": state in observed,
            "gt_full_row_count": full.get(state, np.nan),
            "gt_sample_weight_mass": sampled.get(state, 0.0),
            "display_rank": selected.index(state) + 1 if state in selected else np.nan,
            "selection_basis": basis,
            "source_label_preserved": True,
            "source_name_status": (
                "coded source label; anatomy not inferred"
                if state.isdigit() or state.startswith("type_id:")
                else "named source annotation"
            ),
            "reason": (
                "GT-supported named state selected by observed abundance"
                if state in selected and primary is not None
                else (
                    "descriptive source state; no GT state vocabulary"
                    if state in selected
                    else (
                        "not present in observed state vocabulary"
                        if state not in observed
                        else (
                            "not in explicitly requested state subset"
                            if requested and state not in requested
                            else "outside display cap; mass remains visible in grey/Other states"
                        )
                    )
                )
            ),
        }
        for d in data:
            c = d.get("cells", pd.DataFrame())
            mask = (
                c["state"].astype(str).eq(state)
                if "state" in c
                else pd.Series(False, index=c.index)
            )
            weights = pd.to_numeric(
                c.get("sample_weight", pd.Series(1.0, index=c.index)), errors="coerce"
            ).fillna(0.0)
            counts.append(
                {
                    **row,
                    "model": d["source"].get("model"),
                    "role": d["source"].get("role"),
                    "sample_count": int(mask.sum()),
                    "sample_weight_mass": float(weights[mask].sum()),
                }
            )
    return selected, pd.DataFrame(counts)


def _regulator_tables(primary, data, selected, programs, prior, baseline):
    """Keep GT biological evidence and model reconstruction evidence separate."""
    if primary is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    selected_info = selected.set_index("gene").to_dict("index") if "gene" in selected else {}
    evidence = []
    reconstruction = []
    coverage = []
    edge_errors_by_model = {}
    gtstage = primary.get("stage", pd.DataFrame())
    gtstate = primary.get("state", pd.DataFrame())
    gtmoments = _moments(primary)
    for gene in primary["genes"]:
        t = gtstage[gtstage["gene"].astype(str).eq(gene)] if "gene" in gtstage else pd.DataFrame()
        means = t["mean"].to_numpy(float) if "mean" in t else np.array([])
        means = means[np.isfinite(means)]
        st = (
            gtstate[gtstate["gene"].astype(str).eq(gene)].copy()
            if "gene" in gtstate
            else pd.DataFrame()
        )
        state_effect = np.nan
        state_span = np.nan
        if not st.empty and {"state", "mean"}.issubset(st):
            w = pd.to_numeric(st.get("n", pd.Series(1.0, index=st.index)), errors="coerce").fillna(
                0.0
            )
            st["_mass"] = w
            st["_value"] = pd.to_numeric(st["mean"], errors="coerce") * w
            a = st.groupby("state")[["_mass", "_value"]].sum()
            sm = a["_value"] / a["_mass"].replace(0, np.nan)
            sm = sm[np.isfinite(sm)]
            state_span = float(sm.max() - sm.min()) if len(sm) else np.nan
            moment = gtmoments.get(gene, (np.nan, np.nan))
            state_effect = (
                state_span / moment[1] if np.isfinite(moment[1]) and moment[1] > 0 else np.nan
            )
        out = prior[prior["program"].astype(str).eq(gene)] if not prior.empty else pd.DataFrame()
        inc = prior[prior["gene"].astype(str).eq(gene)] if not prior.empty else pd.DataFrame()
        member = (
            programs[programs["gene"].astype(str).eq(gene)]
            if not programs.empty
            else pd.DataFrame()
        )
        info = selected_info.get(gene, {})
        evidence.append(
            {
                "gene": gene,
                "gt_temporal_variance": float(np.var(means)) if len(means) > 1 else np.nan,
                "gt_temporal_range": float(np.ptp(means)) if len(means) else np.nan,
                "gt_state_mean_range": state_span,
                "gt_state_standardized_range": state_effect,
                "prior_outdegree": out["gene"].nunique() if len(out) else 0,
                "prior_indegree": inc["program"].nunique() if len(inc) else 0,
                "author_program_count": member["program"].nunique() if len(member) else 0,
                "author_programs": (
                    ";".join(sorted(member["program"].astype(str).unique())) if len(member) else ""
                ),
                "author_gene_member": bool(info.get("author_gene_member", False))
                or str(info.get("source_kind", "")).lower()
                in ("author_gene_list", "author_membership", "native_author_axis")
                or str(info.get("reason", ""))
                .lower()
                .startswith(("member of supplied ", "declared author-panel membership")),
                "prior_available": not prior.empty,
                "author_program_available": not programs.empty,
                "eligible": bool(info.get("eligible", False)),
                "selected": bool(info.get("selected", False)),
                "annotation_reason": info.get("reason", ""),
                "annotation_source": info.get("evidence", ""),
                "reference_model": primary["source"].get("model"),
                "reference_source_id": primary["source"].get("source_id", ""),
                "reference_panel": primary["source"].get("panel"),
                "screening_is_not_significance": True,
            }
        )
    for d in data:
        ref = _reference(d, data)
        stage = d.get("stage", pd.DataFrame())
        covered = set(d["genes"])
        refgenes = set(ref["genes"]) if ref is not None else set()
        edge_values = []
        if ref is not None and not prior.empty:
            for tf, g in prior.groupby("program", sort=False):
                targets = g["gene"].astype(str).tolist()
                assoc = _edge_support(d, str(tf), targets)
                truth = _edge_support(ref, str(tf), targets)
                for target in set(assoc) & set(truth):
                    if np.isfinite(assoc[target]) and np.isfinite(truth[target]):
                        edge_values.append((str(tf), target, abs(assoc[target] - truth[target])))
        edge_errors_by_model[d["source"].get("model")] = {
            (src, tgt): error for src, tgt, error in edge_values
        }
        for gene in primary["genes"]:
            a = stage[stage["gene"].astype(str).eq(gene)] if "gene" in stage else pd.DataFrame()
            b = (
                ref["stage"][ref["stage"]["gene"].astype(str).eq(gene)]
                if ref is not None and "gene" in ref["stage"]
                else pd.DataFrame()
            )
            pair = (
                a[["time", "mean"]].merge(
                    b[["time", "mean"]], on="time", suffixes=("_model", "_gt")
                )
                if not a.empty and not b.empty
                else pd.DataFrame()
            )
            if not pair.empty:
                pair = pair[np.isfinite(pair["mean_model"]) & np.isfinite(pair["mean_gt"])]
            errors = [err for src, tgt, err in edge_values if gene in (src, tgt)]
            trend = np.nan
            if len(pair) > 1 and pair["mean_model"].std() > 0 and pair["mean_gt"].std() > 0:
                trend = float(pair[["mean_model", "mean_gt"]].corr().iloc[0, 1])
            reconstruction.append(
                {
                    "model": d["source"].get("model"),
                    "role": d["source"].get("role"),
                    "panel": d["source"].get("panel"),
                    "gene": gene,
                    "native_available": gene in covered,
                    "observed_available": gene in refgenes,
                    "n_matched_stages": len(pair),
                    "stage_mean_mae": (
                        float((pair["mean_model"] - pair["mean_gt"]).abs().mean())
                        if len(pair)
                        else np.nan
                    ),
                    "stage_mean_pcc": trend,
                    "incident_prior_edge_count": len(errors),
                    "mean_incident_prior_association_error": (
                        float(np.mean(errors)) if errors else np.nan
                    ),
                    "baseline_model": baseline,
                    "baseline_error_minus_model_error": np.nan,
                    "baseline_observed_source_id": "",
                    "contrast_comparable": False,
                    "contrast_reason": "actual named baseline unavailable",
                    "n_shared_incident_edges_for_contrast": 0,
                    "baseline_error_on_shared_incident_edges": np.nan,
                    "model_error_on_shared_incident_edges": np.nan,
                    "observed_model": ref["source"].get("model") if ref is not None else "",
                    "observed_source_id": (
                        ref["source"].get("source_id", "") if ref is not None else ""
                    ),
                    **_reference_provenance(d, data),
                    "association_basis": "within-age residual correlation of declared drawing sample; supplied prior edges, not causality",
                }
            )
        coverage.append(
            {
                "model": d["source"].get("model"),
                "panel": d["source"].get("panel"),
                "native_genes": len(covered),
                "observed_common_genes": len(covered & refgenes),
                "evaluated_prior_edges": len(edge_values),
                "observed_model": ref["source"].get("model") if ref is not None else "",
                "baseline_model": baseline,
                "baseline_available": any(x["source"].get("model") == baseline for x in data),
                "no_composite_score": True,
                "selection_uses_model_error": False,
                **_reference_provenance(d, data),
            }
        )
    reconstruction = pd.DataFrame(reconstruction)
    if not reconstruction.empty and baseline in set(reconstruction["model"]):
        base = reconstruction[reconstruction["model"].eq(baseline)].set_index("gene")
        reconstruction["baseline_observed_source_id"] = reconstruction["gene"].map(
            base["observed_source_id"]
        )
        reconstruction["contrast_comparable"] = reconstruction["observed_source_id"].eq(
            reconstruction["gene"].map(base["observed_source_id"])
        ) & reconstruction["model_target_stages"].eq(
            reconstruction["gene"].map(base["model_target_stages"])
        )
        reconstruction["contrast_reason"] = np.where(
            reconstruction["contrast_comparable"],
            "same observed identity and model target stages; exact/descriptive binding disclosed",
            "different observed target identity or target coverage; contrast withheld",
        )
        baseline_edges = edge_errors_by_model.get(baseline, {})
        for index, row in reconstruction.iterrows():
            model_edges = edge_errors_by_model.get(row["model"], {})
            shared = [
                edge for edge in set(baseline_edges) & set(model_edges) if row["gene"] in edge
            ]
            reconstruction.at[index, "n_shared_incident_edges_for_contrast"] = len(shared)
            if row["contrast_comparable"] and shared:
                old = float(np.mean([baseline_edges[edge] for edge in shared]))
                new = float(np.mean([model_edges[edge] for edge in shared]))
                reconstruction.at[index, "baseline_error_on_shared_incident_edges"] = old
                reconstruction.at[index, "model_error_on_shared_incident_edges"] = new
                reconstruction.at[index, "baseline_error_minus_model_error"] = old - new
            elif row["contrast_comparable"]:
                reconstruction.at[index, "contrast_comparable"] = False
                reconstruction.at[index, "contrast_reason"] = (
                    "no shared finite incident prior edges; contrast withheld"
                )
    return pd.DataFrame(evidence), reconstruction, pd.DataFrame(coverage)


def _distribution_tables(group, gene, lower, upper):
    """Exact declared sample and histogram/violin drawing evidence, not fake cells."""
    summaries = []
    bins = []
    edges = np.linspace(lower, upper, 35)
    for d in group:
        c = d["cells"]
        values = _values(d, gene)
        if values is None or "time" not in c:
            continue
        ages = sorted(c["time"].dropna().unique())
        histogram_ages = (
            ages
            if len(ages) <= 6
            else [ages[i] for i in np.unique(np.linspace(0, len(ages) - 1, 6).astype(int))]
        )
        box_ages = (
            ages
            if len(ages) <= 8
            else [ages[i] for i in np.unique(np.linspace(0, len(ages) - 1, 8).astype(int))]
        )
        for age in ages:
            mask = np.isclose(c["time"], age)
            x = values[mask]
            weight = (
                c.loc[mask, "sample_weight"].to_numpy(float)
                if "sample_weight" in c
                else np.ones(len(x))
            )
            valid = np.isfinite(x) & np.isfinite(weight) & (weight >= 0)
            x = x[valid]
            weight = weight[valid]
            if not len(x):
                continue
            q = np.quantile(x, [0, 0.25, 0.5, 0.75, 1])
            mass = float(weight.sum())
            summaries.append(
                {
                    "model": d["source"].get("model"),
                    "gene": gene,
                    "time": age,
                    "sample_count": len(x),
                    "sample_weight_mass": mass,
                    "sample_mean": float(x.mean()),
                    "sample_sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                    "minimum": q[0],
                    "q25": q[1],
                    "median": q[2],
                    "q75": q[3],
                    "maximum": q[4],
                    "box_violin_age_displayed": age in box_ages,
                    "histogram_age_displayed": age in histogram_ages,
                    "below_histogram_range_weight": float(weight[x < lower].sum()),
                    "above_histogram_range_weight": float(weight[x > upper].sum()),
                    "quantile_basis": "unweighted actual declared drawing sample; boxplot basis",
                    "histogram_basis": "declared inverse-sampling weights; density normalized within displayed range",
                }
            )
            hist, _ = np.histogram(x, bins=edges, weights=weight)
            normal = hist.sum()
            density = (
                hist / (normal * np.diff(edges)) if normal > 0 else np.zeros_like(hist, dtype=float)
            )
            smooth = gaussian_filter(hist.astype(float), sigma=1.2)
            width = smooth / max(smooth.max(), 1.0) * 0.4
            for i in range(len(hist)):
                bins.append(
                    {
                        "model": d["source"].get("model"),
                        "gene": gene,
                        "time": age,
                        "bin_left": edges[i],
                        "bin_right": edges[i + 1],
                        "weighted_count": hist[i],
                        "histogram_density": density[i],
                        "violin_half_width": width[i],
                        "histogram_age_displayed": age in histogram_ages,
                        "violin_age_displayed": age in box_ages,
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(bins)


def _marker_summary(d, genes, states):
    table = d.get("state", pd.DataFrame())
    if not table.empty and {"state", "gene", "mean", "pct"}.issubset(table):
        t = table[table["gene"].isin(genes) & table["state"].astype(str).isin(states)].copy()
        if "n" in t:
            t["weighted_mean"] = t["mean"] * t["n"]
            t["weighted_pct"] = t["pct"] * t["n"]
            t = t.groupby(["gene", "state"], as_index=False)[
                ["n", "weighted_mean", "weighted_pct"]
            ].sum()
            t["mean"] = t["weighted_mean"] / t["n"].replace(0, np.nan)
            t["pct"] = t["weighted_pct"] / t["n"].replace(0, np.nan)
        t["basis"] = "full-row supplied/template state strata"
    else:
        c = d["cells"]
        rows = []
        for state in states:
            q = c[c["state"].astype(str) == state]
            for gene in genes:
                x = _values(d, gene, q)
                if x is not None and len(x):
                    rows.append(
                        {
                            "state": state,
                            "gene": gene,
                            "n": len(x),
                            "mean": x.mean(),
                            "pct": (x > 0).mean(),
                        }
                    )
        t = pd.DataFrame(rows)
        if t.empty:
            return t
        t["basis"] = "unweighted declared expression-inferred drawing sample fallback"
    t["fraction_stored_expression_gt_0"] = _fraction(t["pct"].to_numpy(), d)
    t["model"] = d["source"].get("model")
    t["native_panel"] = d["source"].get("panel")
    return t


def _make_panels(data, options, callback):
    return [
        (
            _label(d, options),
            d["source"].get("role", "unknown"),
            (lambda ax, legend=True, d=d: callback(ax, d, legend)),
        )
        for d in data
    ]


def _record(paths, view, entity=""):
    records = []
    for p in paths:
        stem = Path(p).stem
        model = (
            stem.split("__individual_", 1)[1].rsplit("__", 1)[0]
            if "__individual_" in stem
            else ("native " + entity if view.startswith("scanpy_") else "comparison")
        )
        records.append({"path": str(p), "view": view, "model": model, "entity": entity})
    return records


def _notify(progress, view, entity, count):
    if progress is not None:
        progress({"view": view, "entity": entity, "figures": count})


def _scatter_background(ax, d, options):
    c = _cells(d)
    return ax.scatter(
        c["x"],
        c["y"],
        s=3,
        c=options.get("background_color", "#969FA7"),
        alpha=options.get("background_alpha", 0.8),
        linewidths=0,
        rasterized=True,
    )


def _format_geometry(ax, extent):
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Reference axis 1")
    ax.set_ylabel("Reference axis 2")


def _simple_mosaic(data, output, view, options, callback, title, entity=""):
    return plot.compare_mosaics(
        _make_panels(data, options, callback),
        output / (view + (("__" + _slug(entity)) if entity else "")),
        options,
        title=title,
    )


def _slug(value):
    import re

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")[:120] or "entity"


def render_question(
    question: dict, cohort: dict, output: Path, options: dict, progress=None
) -> dict:
    """Draw biological views from actual prepared cells/summary tables."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    options = dict(options)
    options["selection_family"] = question.get("family", "")
    # One assay per comparison, including invisible reference sources.
    sources = plot.available_models(cohort, options)
    data = [plot.load_artifact(s) for s in sources]
    existing = {str(d["source"].get("artifact_dir")) for d in data}
    for source in plot.native_sources(cohort, options, sources=cohort.get("reference_sources", [])):
        if source.get("role") == "GT" and str(source.get("artifact_dir")) not in existing:
            data.append(plot.load_artifact(source))
            existing.add(str(source.get("artifact_dir")))
    if not data:
        return {
            "status": "unsupported",
            "figures": [],
            "tables": [],
            "notes": ["No prepared measured/reverse expression source is available."],
        }
    for d in data:
        d.setdefault(
            "source",
            next(
                (
                    s
                    for s in sources
                    if s.get("artifact_dir") == d.get("meta", {}).get("artifact_dir")
                ),
                d.get("meta", {}),
            ),
        )
        d.setdefault("genes", list(d.get("meta", {}).get("genes", [])))
        d.setdefault("stage", pd.DataFrame())
        d.setdefault("state", pd.DataFrame())
    gt = [d for d in data if d["source"].get("role") == "GT"]
    native = str(options.get("panel", cohort.get("panel", "hvg"))).lower()
    native = "hvg" if native in ("200", "hvg", "hvg200") else native
    cohort_native = native
    primary = next(
        (
            d
            for d in gt
            if str(d["source"].get("panel")).lower()
            in (("hvg", "200", "hvg200") if native == "hvg" else ("author",))
        ),
        gt[0] if gt else None,
    )
    main = ([primary] if primary is not None else []) + [
        d for d in data if d["source"].get("role") != "GT"
    ]
    from .selection import select_genes

    selected = select_genes(cohort, options)
    selection_path = output / "selected_entities.csv"
    selected.to_csv(selection_path, index=False)
    genes = (
        selected.loc[selected["selected"].astype(bool), "gene"].astype(str).tolist()
        if len(selected) and {"gene", "selected"}.issubset(selected)
        else []
    )
    limit = int(options.get("max_entities", 10))
    genes = genes[:limit] if limit > 0 else genes
    options["selected_gene_names"] = genes
    plot.style()
    extent = _geometry(main)
    all_states = _state_labels(main)
    requested_states = question.get("states", question.get("options", {}).get("states", []))
    states, state_selection = _state_selection(main, primary, options, requested_states)
    reference_ids = {
        str(d["meta"].get("reference_id")) for d in main if d["meta"].get("reference_id")
    }
    extent_by_reference = {
        rid: _geometry([d for d in data if str(d["meta"].get("reference_id")) == rid])
        for rid in reference_ids
    }

    def _extent(d):
        return extent_by_reference.get(str(d["meta"].get("reference_id")), extent)

    def _matching_ref(d):
        ref = _reference(d, data)
        rid = d["meta"].get("reference_id")
        if rid and (ref is None or ref["meta"].get("reference_id") != rid):
            return None
        return ref

    palette = list(plot.PALETTE) + [plt.get_cmap("tab20b")(i) for i in (0, 4, 8, 12, 16)]
    statecolors = {s: palette[i % len(palette)] for i, s in enumerate(states)}
    timevals = (
        np.concatenate(
            [
                _cells(d)["time"].to_numpy(float)
                for d in main
                if len(_cells(d)) and "time" in d["cells"]
            ]
        )
        if any(len(_cells(d)) and "time" in d["cells"] for d in main)
        else np.array([0.0, 1.0])
    )
    timelim = _limits(timevals, (0.0, 1.0))
    timenorm = Normalize(*timelim)
    state_selection_path = output / "STATE_SELECTION.csv"
    state_selection.to_csv(state_selection_path, index=False)
    tables = [str(selection_path), str(state_selection_path)]
    figures = []
    notes = []
    incomplete_semantics = False
    if set(all_states) - set(states):
        notes.append(
            "Named states were frozen by observed support and abundance; omitted/model-only states remain visible as grey background or Other states. STATE_SELECTION.csv records all source names and counts."
        )
    if any(
        "descriptive" in _reference_match(d, data)[1]
        for d in main
        if d["source"].get("role") not in ("GT", "REFERENCE")
    ):
        notes.append(
            "Some sources lack exact own-truth provenance. Their same-assay observed comparisons are descriptive; MODEL_ROWS.csv and numerical sidecars disclose reference_match, paired_source and target coverage. An alternate same-unit GT is never silently substituted as paired truth."
        )
    if len(reference_ids) > 1:
        notes.append(
            "Multiple verified native observed reference fits are present; coordinate panels compare only within the same reference_id. Density residuals require a matching observed reference."
        )
    views = question.get("views", [])
    maximum = int(options.get("max_views_per_entity", 0))
    per_entity = {}
    capped_views = set()

    def emit(view, callback, title, entity="", group=None):
        if entity and maximum > 0 and per_entity.get(entity, 0) >= maximum:
            capped_views.add(view)
            return
        if len(reference_ids) > 1 and view in (
            "atlas_time",
            "atlas_state",
            "trajectory",
            "state_trajectory",
            "stage_storyboard",
            "density",
            "density_residual",
            "density_contour",
            "density_state",
            "gene_feature",
            "program_feature",
            "state_graph",
        ):
            title = (
                title.replace("one measured reference", "observed native references")
                + " · native axes; reference IDs in MODEL_ROWS.csv"
            )
        paths = _simple_mosaic(group or main, output, view, options, callback, title, entity)
        figures.extend(_record(paths, view, entity))
        per_entity[entity] = per_entity.get(entity, 0) + 1
        _notify(progress, view, entity, len(figures))

    def write(frame, name):
        p = output / name
        frame.to_csv(p, index=False)
        tables.append(str(p))

    spatial_views = {
        "atlas_time",
        "atlas_state",
        "stage_storyboard",
        "trajectory",
        "state_trajectory",
        "density",
        "density_residual",
        "density_contour",
        "density_state",
        "gene_feature",
        "program_feature",
        "state_graph",
    }
    if set(views) & spatial_views:
        coordinate_rows = []
        for d in main:
            c = _cells(d).copy()
            if c.empty:
                continue
            cols = [
                col
                for col in (
                    "cell_id",
                    "expression_row",
                    "time",
                    "time_label",
                    "state",
                    "state_supplied",
                    "x",
                    "y",
                    "sample_weight",
                    "is_generated",
                    "reference_knn_distance",
                )
                if col in c
            ]
            c = c[cols]
            c["model"] = d["source"].get("model")
            c["role"] = d["source"].get("role")
            c["native_panel"] = d["source"].get("panel")
            c["reference_id"] = d["meta"].get("reference_id", "")
            c["selected_display_state"] = (
                c["state"].astype(str).isin(states) if "state" in c else False
            )
            coordinate_rows.append(c)
        if coordinate_rows:
            write(pd.concat(coordinate_rows, ignore_index=True), "DRAWING_COORDINATES.csv")
    if "gene_feature" in views and genes:
        for d in main:
            c = _cells(d).copy()
            names = [g for g in genes if g in d["genes"]]
            if c.empty or not names:
                continue
            cols = [
                col
                for col in ("cell_id", "expression_row", "time", "state", "x", "y", "sample_weight")
                if col in c
            ]
            frame = c[cols].copy()
            for gene in names:
                frame["expression:" + gene] = _values(d, gene, c)
            frame["model"] = d["source"].get("model")
            frame["native_panel"] = d["source"].get("panel")
            write(frame, "feature_values__" + _slug(d["source"].get("model")) + ".csv")
    for view in views:
        if view not in VIEWS:
            notes.append(f"Unknown measured view: {view}")
            continue
        if view == "atlas_time":

            def draw(ax, d, legend):
                c = _cells(d)
                if c.empty:
                    return _missing(ax, "No shared-reference coordinates")
                im = ax.scatter(
                    c["x"],
                    c["y"],
                    c=c["time"],
                    cmap="viridis",
                    norm=timenorm,
                    s=5,
                    alpha=0.8,
                    linewidths=0,
                    rasterized=True,
                )
                _format_geometry(ax, _extent(d))
                if legend:
                    ax.figure.colorbar(
                        im, ax=ax, label="Saved developmental age", fraction=0.035, pad=0.02
                    )

            emit(view, draw, "Developmental populations on one measured reference")
        elif view == "atlas_state":
            if not states:
                notes.append("No usable expression-derived or supplied state labels.")
                continue

            def draw(ax, d, legend):
                c = _cells(d)
                background = _scatter_background(ax, d, options)
                if (~c["state"].astype(str).isin(states)).any():
                    background.set_label("Other/unassigned states")
                for state in states:
                    q = c[c["state"].astype(str) == state]
                    ax.scatter(
                        q["x"],
                        q["y"],
                        s=5,
                        color=statecolors[state],
                        alpha=0.8,
                        linewidths=0,
                        rasterized=True,
                        label=state,
                    )
                _format_geometry(ax, _extent(d))
                if legend and len(states) <= 18:
                    ax.legend(fontsize=5, frameon=False, markerscale=2, loc="best")

            emit(
                view,
                draw,
                "Cell states read from saved expression; see state method in MODEL_ROWS.csv",
            )
        elif view == "stage_storyboard":
            stages = sorted(set(np.asarray(timevals)[np.isfinite(timevals)]))
            if limit > 0 and len(stages) > limit:
                stages = [
                    stages[i] for i in np.unique(np.linspace(0, len(stages) - 1, limit).astype(int))
                ]
            for age in stages:

                def draw(ax, d, legend, age=age):
                    c = _cells(d)
                    _scatter_background(ax, d, options)
                    q = c[np.isclose(c["time"], age)]
                    if q.empty:
                        _missing(ax, f"No saved cells at age {age:g}")
                    else:
                        ax.scatter(
                            q["x"],
                            q["y"],
                            s=7,
                            color="#287C8E",
                            alpha=0.9,
                            linewidths=0,
                            rasterized=True,
                        )
                    _format_geometry(ax, _extent(d))

                emit(
                    view,
                    draw,
                    f"Developmental state-space occupancy at saved age {age:g}",
                    f"age_{age:g}",
                )
        elif view in ("trajectory", "state_trajectory"):
            choices = states if view == "state_trajectory" else ["all_states"]
            if limit > 0:
                choices = choices[:limit]
            for state in choices:
                coords = []

                def draw(ax, d, legend, state=state):
                    c = _cells(d)
                    _scatter_background(ax, d, {**options, "background_alpha": 0.35})
                    if state != "all_states":
                        c = c[c["state"].astype(str) == state]
                    if c.empty:
                        return _missing(ax, "No saved cells for this state")
                    if "sample_weight" in c:
                        cent = pd.DataFrame(
                            [
                                {
                                    "time": age,
                                    "x": np.average(q["x"], weights=q["sample_weight"]),
                                    "y": np.average(q["y"], weights=q["sample_weight"]),
                                }
                                for age, q in c.groupby("time", sort=True)
                            ]
                        ).set_index("time")
                    else:
                        cent = c.groupby("time", sort=True)[["x", "y"]].mean()
                    ax.plot(cent["x"], cent["y"], color="#376D83", linewidth=1.6)
                    im = ax.scatter(
                        cent["x"],
                        cent["y"],
                        c=cent.index,
                        cmap="viridis",
                        norm=timenorm,
                        s=35,
                        zorder=3,
                    )
                    for i in range(len(cent) - 1):
                        ax.annotate(
                            "",
                            xy=cent.iloc[i + 1],
                            xytext=cent.iloc[i],
                            arrowprops={"arrowstyle": "->", "color": "#376D83", "lw": 1.1},
                        )
                    _format_geometry(ax, _extent(d))
                    if legend:
                        ax.figure.colorbar(
                            im, ax=ax, label="Saved developmental age", fraction=0.035, pad=0.02
                        )
                    local = cent.reset_index()
                    local["model"] = d["source"].get("model")
                    local["state"] = state
                    local["sample_count"] = local["time"].map(c.groupby("time").size())
                    local["sample_weight_mass"] = (
                        local["time"].map(c.groupby("time")["sample_weight"].sum())
                        if "sample_weight" in c
                        else local["sample_count"]
                    )
                    local["method"] = (
                        "declared sample-weight centroid"
                        if "sample_weight" in c
                        else "unweighted sample centroid"
                    )
                    coords.append(local)

                emit(
                    view,
                    draw,
                    "Population centroid path; descriptive, not lineage"
                    + (f": {state}" if state != "all_states" else ""),
                    state,
                )
                if coords:
                    write(
                        pd.concat(coords, ignore_index=True).drop_duplicates(),
                        view + "__" + _slug(state) + ".csv",
                    )
        elif view in ("density", "density_residual", "density_contour", "density_state"):
            choices = states if view == "density_state" else ["all_states"]
            if limit > 0:
                choices = choices[:limit]
            for state in choices:

                def coords(d):
                    c = _cells(d)
                    if state != "all_states":
                        c = c[c["state"].astype(str) == state]
                    return c[["x", "y"]].to_numpy(float)

                def weights(d):
                    c = _cells(d)
                    if state != "all_states":
                        c = c[c["state"].astype(str) == state]
                    return c["sample_weight"].to_numpy(float) if "sample_weight" in c else None

                truth_density = (
                    _density(coords(primary), _extent(primary), weights=weights(primary))
                    if primary is not None
                    else None
                )
                dens = [_density(coords(d), _extent(d), weights=weights(d)) for d in main + gt]
                maxd = max([float(a.max()) for a in dens] + [1e-8])
                bound = maxd
                grids = {}
                grid_rows = []
                observed_grids = {}
                for d in main:
                    xy = coords(d)
                    local_extent = _extent(d)
                    if not len(xy):
                        continue
                    h = _density(xy, local_extent, weights=weights(d))
                    if view == "density_contour":
                        xx = np.linspace(local_extent[0], local_extent[1], 72)
                        yy = np.linspace(local_extent[2], local_extent[3], 72)
                        X, Y = np.meshgrid(xx, yy)
                        try:
                            ix = np.linspace(0, len(xy) - 1, min(len(xy), 2000)).astype(int)
                            w = weights(d)
                            w = None if w is None else w[ix]
                            z = gaussian_kde(xy[ix].T, weights=w)(
                                np.vstack([X.ravel(), Y.ravel()])
                            ).reshape(X.shape)
                        except (ValueError, np.linalg.LinAlgError):
                            z = h
                        h = z / max(z.sum(), 1e-12)
                    else:
                        xedge = np.linspace(local_extent[0], local_extent[1], 73)
                        yedge = np.linspace(local_extent[2], local_extent[3], 73)
                        X, Y = np.meshgrid(
                            (xedge[:-1] + xedge[1:]) / 2, (yedge[:-1] + yedge[1:]) / 2
                        )
                    ref = _matching_ref(d)
                    refkey = (id(ref), tuple(local_extent))
                    if ref is not None and refkey not in observed_grids:
                        refxy = coords(ref)
                        observed = _density(refxy, local_extent, weights=weights(ref))
                        if view == "density_contour" and len(refxy):
                            try:
                                ix = np.linspace(0, len(refxy) - 1, min(len(refxy), 2000)).astype(
                                    int
                                )
                                w = weights(ref)
                                w = None if w is None else w[ix]
                                observed = gaussian_kde(refxy[ix].T, weights=w)(
                                    np.vstack([X.ravel(), Y.ravel()])
                                ).reshape(X.shape)
                            except (ValueError, np.linalg.LinAlgError):
                                pass
                            observed = observed / max(observed.sum(), 1e-12)
                        observed_grids[refkey] = observed
                    observed = observed_grids.get(refkey, np.full_like(h, np.nan))
                    grids[id(d)] = (h, X, Y, observed)
                    frame = pd.DataFrame(
                        {
                            "x": X.ravel(),
                            "y": Y.ravel(),
                            "density": h.ravel(),
                            "gt_density": observed.ravel(),
                            "model_minus_gt": (h - observed).ravel(),
                        }
                    )
                    frame["model"] = d["source"].get("model")
                    frame["state"] = state
                    frame["reference_id"] = d["meta"].get("reference_id", "")
                    for key, value in _reference_provenance(d, data).items():
                        frame[key] = value
                    frame["method"] = (
                        "bounded sample weighted KDE with smoothed histogram fallback"
                        if view == "density_contour"
                        else "72-bin sample weighted histogram; Gaussian sigma=1.3; normalized mass"
                    )
                    grid_rows.append(frame)
                if grid_rows:
                    write(
                        pd.concat(grid_rows, ignore_index=True),
                        view + "__" + _slug(state) + "__grid.csv",
                    )

                def draw(ax, d, legend):
                    xy = coords(d)
                    if not len(xy):
                        return _missing(ax, "No cells for density in this state")
                    local_extent = _extent(d)
                    h, X, Y, observed = grids[id(d)]
                    if view == "density_residual":
                        ref = _matching_ref(d)
                        if ref is None:
                            return _missing(ax, "No observed population on the same reference axes")
                        im = ax.imshow(
                            h - observed,
                            origin="lower",
                            extent=local_extent,
                            aspect="auto",
                            cmap=plot.DIVERGING_CMAP,
                            vmin=-bound,
                            vmax=bound,
                        )
                        clabel = "Model − GT normalized density"
                    elif view == "density_contour":
                        # Optional KDE keeps the old contour interpretation but
                        # caps only the drawing calculation, not summary rows.
                        im = ax.contourf(
                            X, Y, h, levels=np.linspace(0, maxd, 13), extend="max", cmap="viridis"
                        )
                        ax.contour(X, Y, h, levels=6, colors="#214A61", linewidths=0.45)
                        clabel = "Reference-space sample KDE"
                    else:
                        im = ax.imshow(
                            h,
                            origin="lower",
                            extent=local_extent,
                            aspect="auto",
                            cmap="viridis",
                            vmin=0,
                            vmax=maxd,
                        )
                        clabel = "Normalized sample density"
                    _format_geometry(ax, _extent(d))
                    if legend:
                        ax.figure.colorbar(im, ax=ax, label=clabel, fraction=0.035, pad=0.02)

                emit(
                    view,
                    draw,
                    "Developmental region density"
                    + (f": {state}" if state != "all_states" else ""),
                    state,
                )
        elif view in GENE_VIEWS:
            if not genes:
                notes.append(f"{view}: reference-based selection found no eligible genes.")
                continue
            for gene in genes:
                group = _groups(
                    ([primary] if primary is not None else [])
                    + [d for d in data if d is not primary],
                    gene,
                )
                vals = [_values(d, gene) for d in group]
                joined = (
                    np.concatenate([a for a in vals if a is not None and len(a)])
                    if any(a is not None and len(a) for a in vals)
                    else np.array([0.0, 1.0])
                )
                vlo, vhi = _limits(joined, (0.01, 0.99))
                if view == "gene_distribution":
                    summary, bins = _distribution_tables(group, gene, vlo, vhi)
                    if not summary.empty:
                        write(summary, "gene_distribution__" + _slug(gene) + "__quantiles.csv")
                    if not bins.empty:
                        write(bins, "gene_distribution__" + _slug(gene) + "__bins.csv")
                if view == "gene_feature":
                    write(
                        pd.DataFrame(
                            [
                                {
                                    "gene": gene,
                                    "vmin": vlo,
                                    "vmax": vhi,
                                    "color_limits_basis": "1st/99th quantile of pooled same-assay declared drawing values; axis provenance in MODEL_ROWS.csv",
                                }
                            ]
                        ),
                        "gene_feature__" + _slug(gene) + "__color_limits.csv",
                    )

                def draw(ax, d, legend, gene=gene):
                    c = _cells(d)
                    v = _values(d, gene, c)
                    table = d["stage"]
                    table = (
                        table[table["gene"].astype(str) == gene]
                        if "gene" in table
                        else pd.DataFrame()
                    )
                    if v is None:
                        return _missing(ax)
                    if view == "gene_feature":
                        order = np.argsort(v)
                        im = ax.scatter(
                            c["x"].to_numpy()[order],
                            c["y"].to_numpy()[order],
                            c=v[order],
                            cmap=plot.EXPRESSION_CMAP,
                            vmin=vlo,
                            vmax=vhi,
                            s=6,
                            alpha=0.9,
                            linewidths=0,
                            rasterized=True,
                        )
                        _format_geometry(ax, _extent(d))
                        if legend:
                            ax.figure.colorbar(
                                im,
                                ax=ax,
                                label=f"{gene} stored expression",
                                fraction=0.035,
                                pad=0.02,
                            )
                    elif view in ("gene_timecourse", "gene_detection"):
                        if table.empty:
                            return _missing(ax, "No full-row stage summary for this gene")
                        table = table.sort_values("time")
                        if view == "gene_detection":
                            if "pct" not in table:
                                return _missing(ax, "No saved detection fraction")
                            ax.plot(
                                table["time"],
                                _fraction(table["pct"], d),
                                "o-",
                                color="#287C8E",
                                ms=3,
                            )
                            ax.set_ylabel("Fraction saved expression > 0")
                            ax.set_ylim(0, 1)
                        else:
                            ax.plot(
                                table["time"],
                                table["mean"],
                                "o-",
                                color="#287C8E",
                                ms=3,
                                label="Full-row mean",
                            )
                            ax.fill_between(
                                table["time"],
                                table["mean"] - table["std"],
                                table["mean"] + table["std"],
                                color="#287C8E",
                                alpha=0.18,
                                label="Cell SD",
                            )
                            if len(c) and "time" in c:
                                q = (
                                    pd.DataFrame({"time": c["time"].to_numpy(), "value": v})
                                    .groupby("time")["value"]
                                    .quantile([0.25, 0.75])
                                    .unstack()
                                )
                                ax.fill_between(
                                    q.index,
                                    q[0.25],
                                    q[0.75],
                                    color="#A97F87",
                                    alpha=0.2,
                                    label="Drawing sample IQR",
                                )
                            ax.set_ylabel("Stored expression")
                            if legend:
                                ax.legend(fontsize=5, frameon=False)
                        ax.set_xlabel("Saved developmental age")
                    elif view == "gene_distribution":
                        times = sorted(c["time"].dropna().unique())
                        if len(times) > 8:
                            times = [
                                times[i]
                                for i in np.unique(np.linspace(0, len(times) - 1, 8).astype(int))
                            ]
                        groups = [v[np.isclose(c["time"], t)] for t in times]
                        groups = [x[np.isfinite(x)] for x in groups]
                        if not groups:
                            return _missing(ax, "No sample distribution")
                        # Boxplot describes actual sampled cells, no fabricated
                        # distributions from only saved mean and SD.
                        box = ax.boxplot(
                            groups,
                            positions=range(len(times)),
                            widths=0.6,
                            patch_artist=True,
                            showfliers=False,
                        )
                        for patch in box["boxes"]:
                            patch.set_facecolor("#68A7AD")
                            patch.set_alpha(0.7)
                        ax.set_xticks(
                            range(len(times)),
                            [f"{t:g}" for t in times],
                            rotation=45,
                            ha="right",
                            fontsize=6,
                        )
                        ax.set_xlabel("Saved age")
                        ax.set_ylabel("Sampled-cell expression")
                    elif view == "gene_phase":
                        if primary is None:
                            return _missing(ax, "No observed expression reference")
                        ref = _reference(d, data)
                        if ref is not None and gene not in ref["genes"]:
                            ref = None
                        if ref is None:
                            return _missing(ax, "Gene absent from observed native panels")
                        a = ref["stage"]
                        a = a[a["gene"].astype(str) == gene][["time", "mean"]]
                        b = table[["time", "mean"]]
                        pair = a.merge(b, on="time", suffixes=("_gt", "_model"))
                        if pair.empty:
                            return _missing(ax, "No matched saved ages")
                        lim = _limits(np.r_[pair["mean_gt"], pair["mean_model"]], (0, 1))
                        ax.plot(lim, lim, color="#88929A", ls="--", lw=0.8)
                        im = ax.scatter(
                            pair["mean_gt"],
                            pair["mean_model"],
                            c=pair["time"],
                            cmap="viridis",
                            norm=timenorm,
                            s=25,
                        )
                        ax.set_xlabel("GT mean expression")
                        ax.set_ylabel("Model mean expression")
                        ax.set_xlim(lim)
                        ax.set_ylim(lim)
                        if legend:
                            ax.figure.colorbar(
                                im, ax=ax, label="Saved age", fraction=0.035, pad=0.02
                            )

                title = {
                    "gene_feature": "Developmental marker localization",
                    "gene_timecourse": "Developmental marker timecourse",
                    "gene_distribution": "Developmental marker distribution",
                    "gene_detection": "Developmental marker detection",
                    "gene_phase": "Stage-matched expression fidelity",
                }[view]
                emit(view, draw, title + ": " + gene, gene, group)
                if view == "gene_distribution":

                    def hist(ax, d, legend, gene=gene):
                        c = d["cells"]
                        v = _values(d, gene)
                        if v is None:
                            return _missing(ax)
                        times = sorted(c["time"].dropna().unique())
                        if len(times) > 6:
                            times = [
                                times[i]
                                for i in np.unique(np.linspace(0, len(times) - 1, 6).astype(int))
                            ]
                        for i, age in enumerate(times):
                            mask = np.isclose(c["time"], age)
                            weights = (
                                c.loc[mask, "sample_weight"].to_numpy(float)
                                if "sample_weight" in c
                                else None
                            )
                            h, edges = np.histogram(
                                v[mask], bins=np.linspace(vlo, vhi, 35), weights=weights
                            )
                            density = (
                                h / (h.sum() * np.diff(edges))
                                if h.sum() > 0
                                else np.zeros(len(h), float)
                            )
                            ax.stairs(
                                density,
                                edges,
                                lw=1.4,
                                color=plot.PALETTE[i % len(plot.PALETTE)],
                                label=f"age {age:g}",
                            )
                        ax.set_xlabel("Stored expression")
                        ax.set_ylabel("Weighted sample density")
                        if legend:
                            ax.legend(fontsize=5, frameon=False)

                    emit(
                        "gene_histogram",
                        hist,
                        "Stage-specific expression distributions: " + gene,
                        gene,
                        group,
                    )

                    def violin(ax, d, legend, gene=gene):
                        c = d["cells"]
                        v = _values(d, gene)
                        if v is None:
                            return _missing(ax)
                        times = sorted(c["time"].dropna().unique())
                        if len(times) > 8:
                            times = [
                                times[i]
                                for i in np.unique(np.linspace(0, len(times) - 1, 8).astype(int))
                            ]
                        for i, age in enumerate(times):
                            mask = np.isclose(c["time"], age)
                            x = v[mask]
                            w = (
                                c.loc[mask, "sample_weight"].to_numpy(float)
                                if "sample_weight" in c
                                else None
                            )
                            h, edges = np.histogram(
                                x, bins=np.linspace(vlo, vhi, 35), weights=w, density=False
                            )
                            h = gaussian_filter(h.astype(float), sigma=1.2)
                            h = h / max(h.max(), 1) * 0.4
                            yy = (edges[:-1] + edges[1:]) / 2
                            ax.fill_betweenx(
                                yy,
                                i - h,
                                i + h,
                                color=plot.PALETTE[i % len(plot.PALETTE)],
                                alpha=0.8,
                                lw=0.5,
                            )
                        ax.set_xticks(
                            range(len(times)),
                            [f"{t:g}" for t in times],
                            rotation=45,
                            ha="right",
                            fontsize=6,
                        )
                        ax.set_xlabel("Saved age")
                        ax.set_ylabel("Stored expression")

                    emit(
                        "gene_violin",
                        violin,
                        "Actual-cell developmental expression distributions: " + gene,
                        gene,
                        group,
                    )
                if view in ("gene_timecourse", "gene_detection", "gene_phase"):
                    frames = []
                    for d in group:
                        t = d["stage"]
                        if "gene" in t:
                            q = t[t["gene"].astype(str) == gene].copy()
                            q["model"] = d["source"].get("model")
                            for key, value in _reference_provenance(d, data).items():
                                q[key] = value
                            ref = _reference(d, data)
                            if view == "gene_phase" and ref is not None and not ref["stage"].empty:
                                observed = ref["stage"]
                                observed = observed[observed["gene"].astype(str).eq(gene)][
                                    ["time", "mean"]
                                ].rename(columns={"mean": "observed_mean"})
                                q = q.merge(observed, on="time", how="left")
                            frames.append(q)
                    if frames:
                        write(
                            pd.concat(frames, ignore_index=True), view + "__" + _slug(gene) + ".csv"
                        )
        elif view in ("marker_dotplot", "marker_matrix", "marker_violin"):
            if not states or not genes:
                notes.append(
                    f"{view}: needs reference-selected genes and usable cell-state labels."
                )
                continue
            marker_pages = [genes[i : i + 30] for i in range(0, len(genes), 30)]
            marker_rows = [_marker_summary(d, genes, states) for d in main]
            marker_rows = [t for t in marker_rows if not t.empty]
            if marker_rows:
                write(pd.concat(marker_rows, ignore_index=True), view + "__values.csv")
            if view == "marker_violin":
                for d in main:
                    c = d["cells"]
                    q = c[c["state"].astype(str).isin(states)].copy()
                    cols = [
                        col
                        for col in ("cell_id", "time", "state", "expression_row", "sample_weight")
                        if col in q
                    ]
                    frame = q[cols].copy()
                    for gene in genes:
                        val = _values(d, gene, q)
                        if val is not None:
                            frame["expression:" + gene] = val
                    frame["model"] = d["source"].get("model")
                    if len(frame):
                        write(frame, "marker_sample__" + _slug(d["source"].get("model")) + ".csv")
            write(
                pd.DataFrame(
                    [
                        dict(
                            view=view,
                            gene=gene,
                            model=d["source"].get("model"),
                            native_available=gene in d["genes"],
                            aggregate_page=i // 30 + 1,
                        )
                        for i, gene in enumerate(genes)
                        for d in main
                    ]
                ),
                view + "__gene_coverage.csv",
            )
            if len(marker_pages) > 1:
                notes.append(
                    view
                    + ": all selected genes paginated in groups of at most 30; native Scanpy groups at most 12."
                )
            for marker_page, usegenes in enumerate(marker_pages):
                marker_values = (
                    np.concatenate(
                        [
                            d["state"]
                            .loc[d["state"]["gene"].isin(usegenes), "mean"]
                            .to_numpy(float)
                            for d in main
                            if not d["state"].empty and {"gene", "mean"}.issubset(d["state"])
                        ]
                    )
                    if any(
                        not d["state"].empty and {"gene", "mean"}.issubset(d["state"]) for d in main
                    )
                    else np.array([0.0, 1.0])
                )
                marker_norm = Normalize(*_limits(marker_values, (0.01, 0.99)))

                def draw(ax, d, legend):
                    native = usegenes
                    if not any(g in d["genes"] for g in native):
                        return _missing(ax)
                    table = d["state"]
                    if not table.empty and {"state", "gene", "mean", "pct"}.issubset(table):
                        t = table[
                            table["gene"].isin(native) & table["state"].astype(str).isin(states)
                        ].copy()
                        if "n" in t:
                            t["weighted_mean"] = t["mean"] * t["n"]
                            t["weighted_pct"] = t["pct"] * t["n"]
                            t = t.groupby(["gene", "state"], as_index=False)[
                                ["n", "weighted_mean", "weighted_pct"]
                            ].sum()
                            t["mean"] = t["weighted_mean"] / t["n"].replace(0, np.nan)
                            t["pct"] = t["weighted_pct"] / t["n"].replace(0, np.nan)
                        m = t.pivot_table(
                            index="gene", columns="state", values="mean", aggfunc="mean"
                        ).reindex(index=native, columns=states)
                        p = t.pivot_table(
                            index="gene", columns="state", values="pct", aggfunc="mean"
                        ).reindex(index=native, columns=states)
                    else:
                        c = d["cells"]
                        rows = []
                        for state in states:
                            q = c[c["state"].astype(str) == state]
                            for gene in native:
                                x = _values(d, gene, q)
                                if x is not None and len(x):
                                    rows.append(
                                        {
                                            "state": state,
                                            "gene": gene,
                                            "mean": x.mean(),
                                            "pct": (x > 0).mean(),
                                        }
                                    )
                        t = pd.DataFrame(rows)
                        if t.empty:
                            return _missing(ax, "No state expression summary")
                        m = t.pivot(index="gene", columns="state", values="mean").reindex(
                            index=native, columns=states
                        )
                        p = t.pivot(index="gene", columns="state", values="pct").reindex(
                            index=native, columns=states
                        )
                    p = pd.DataFrame(_fraction(p.to_numpy(), d), index=p.index, columns=p.columns)
                    if view == "marker_matrix":
                        im = ax.imshow(
                            m,
                            aspect="auto",
                            cmap=plot.EXPRESSION_CMAP,
                            norm=marker_norm,
                            interpolation="nearest",
                        )
                        if legend:
                            ax.figure.colorbar(
                                im, ax=ax, label="Mean expression", fraction=0.035, pad=0.02
                            )
                    elif view == "marker_dotplot":
                        yy, xx = np.indices(m.shape)
                        im = ax.scatter(
                            xx.ravel(),
                            yy.ravel(),
                            c=m.to_numpy().ravel(),
                            s=8 + 60 * np.nan_to_num(p.to_numpy().ravel()),
                            cmap=plot.EXPRESSION_CMAP,
                            norm=marker_norm,
                            linewidths=0.3,
                            edgecolors="#52606A",
                        )
                        if legend:
                            ax.figure.colorbar(
                                im,
                                ax=ax,
                                label="Mean; size: fraction > 0",
                                fraction=0.035,
                                pad=0.02,
                            )
                    else:
                        # Stacked actual-cell violins on gene rows, categorical
                        # state positions, each gene independently scaled.
                        c = d["cells"]
                        for i, gene in enumerate(native):
                            samples = [
                                _values(d, gene, c[c["state"].astype(str) == state])
                                for state in states
                            ]
                            for j, x in enumerate(samples):
                                if x is None or len(x) < 2 or np.ptp(x) < 1e-10:
                                    continue
                                lo, hi = _limits(x, (0, 1))
                                hist, edges = np.histogram(x, bins=25, density=True)
                                hist = hist / max(hist.max(), 1e-12) * 0.36
                                yy = i + (edges[:-1] - lo) / max(hi - lo, 1e-12) * 0.7 - 0.35
                                ax.fill_betweenx(
                                    yy, j - hist, j + hist, color="#579AA5", alpha=0.8, lw=0.4
                                )
                        ax.set_ylim(len(native) - 0.5, -0.5)
                    ax.set_xticks(range(len(states)), states, rotation=70, ha="right", fontsize=5)
                    ax.set_yticks(range(len(native)), native, fontsize=6)
                    ax.set_xlabel(
                        "Expression-inferred sample state"
                        if view == "marker_violin"
                        else "Supplied/template state strata"
                    )
                    ax.set_ylabel("Reference-selected marker")

                emit(
                    view,
                    draw,
                    (
                        "Expression-inferred sampled state markers: "
                        if view == "marker_violin"
                        else "Template-stratified developmental marker programs: "
                    )
                    + view.replace("_", " "),
                    "" if len(marker_pages) == 1 else f"genes_{marker_page+1:03d}",
                )
                if options.get("scanpy"):
                    method = {
                        "marker_dotplot": "dotplot",
                        "marker_matrix": "matrixplot",
                        "marker_violin": "stacked_violin",
                    }[view]
                    try:
                        import anndata as ad
                        import scanpy as sc

                        native_sources = (
                            main
                            if options.get("scanpy_models") == "all"
                            else [
                                d for d in main if d["source"].get("role") in ("GT", "FM", "legacy")
                            ]
                        )
                        for d in native_sources:
                            native_genes = [g for g in usegenes if g in d["genes"]]
                            for native_start in range(0, len(native_genes), 12):
                                native = native_genes[native_start : native_start + 12]
                                if not native or d.get("expression") is None:
                                    continue
                                c = d["cells"]
                                c = c[c["state"].astype(str).isin(states)].copy()
                                if c.empty:
                                    continue
                                rows = (
                                    c["expression_row"].to_numpy(int)
                                    if "expression_row" in c
                                    else c.index.to_numpy(int)
                                )
                                matrix = np.asarray(
                                    d["expression"][rows][:, [d["genes"].index(g) for g in native]],
                                    dtype=np.float32,
                                )
                                adata = ad.AnnData(
                                    X=matrix,
                                    obs=pd.DataFrame(
                                        {
                                            "state": pd.Categorical(
                                                c["state"].astype(str).to_numpy()
                                            )
                                        },
                                        index=[str(i) for i in range(len(c))],
                                    ),
                                    var=pd.DataFrame(index=native),
                                )
                                function = getattr(sc.pl, method)
                                kw = {"stripplot": False} if method == "stacked_violin" else {}
                                result = function(
                                    adata,
                                    native,
                                    groupby="state",
                                    show=False,
                                    return_fig=True,
                                    **kw,
                                )
                                page_suffix = (
                                    ""
                                    if len(marker_pages) == 1 and len(native_genes) <= 12
                                    else f"__genes_{marker_page+1:03d}_{native_start//12+1:03d}"
                                )
                                result.make_figure()
                                for axis_name, legend_axis in result.get_axes().items():
                                    if "legend" in axis_name:
                                        legend_axis._bloom_colorbar = True
                                paths = plot.save(
                                    result.fig,
                                    output
                                    / (
                                        "scanpy_"
                                        + method
                                        + "__"
                                        + _slug(_label(d, options))
                                        + page_suffix
                                    ),
                                    options,
                                )
                                plt.close(result.fig)
                                figures.extend(
                                    _record(paths, "scanpy_" + method, _label(d, options))
                                )
                                _notify(
                                    progress, "scanpy_" + method, _label(d, options), len(figures)
                                )
                        notes.append(
                            "Native Scanpy "
                            + method
                            + " generated from declared sampled cells; native gene coverage retained."
                        )
                    except ImportError as exc:
                        notes.append(
                            "Scanpy unavailable ("
                            + str(exc)
                            + "); corresponding custom biological view was generated."
                        )
                    except Exception as exc:
                        notes.append(
                            "Native Scanpy "
                            + method
                            + " failed: "
                            + type(exc).__name__
                            + ": "
                            + str(exc)
                            + "; custom biological view remains available."
                        )
        elif view == "cascade":
            if not genes:
                notes.append("Cascade has no eligible reference-dynamic genes.")
                continue
            # One contextual GT freezes order/scale; model residuals below use
            # the model's own bound truth rather than pooling alternate GTs.
            ref = primary["stage"].copy() if primary is not None else pd.DataFrame()
            if ref.empty:
                notes.append("Cascade needs a measured reference temporal gene summary.")
                continue
            t = ref[ref["gene"].isin(genes)].pivot_table(
                index="gene", columns="time", values="mean", aggfunc="mean"
            )
            if t.empty:
                continue
            ordered = t.idxmax(axis=1).sort_values().index.tolist()
            center = t.mean(axis=1)
            scale = t.std(axis=1).replace(0, 1)
            cascade_rows = []
            for d in main:
                stage = d["stage"]
                if stage.empty:
                    continue
                q = stage[stage["gene"].isin(ordered)].copy()
                q["model"] = d["source"].get("model")
                q["gene_order"] = q["gene"].map({g: i + 1 for i, g in enumerate(ordered)})
                q["gt_temporal_center"] = q["gene"].map(center)
                q["gt_temporal_sd"] = q["gene"].map(scale)
                q["frozen_gt_temporal_z"] = (q["mean"] - q["gt_temporal_center"]) / q[
                    "gt_temporal_sd"
                ]
                ownref = _reference(d, data)
                baseline = (
                    ownref["stage"][["gene", "time", "mean"]].rename(columns={"mean": "gt_mean"})
                    if ownref is not None and not ownref["stage"].empty
                    else pd.DataFrame(columns=["gene", "time", "gt_mean"])
                )
                q = q.merge(baseline, on=["gene", "time"], how="left")
                q["model_minus_gt_temporal_z"] = (q["mean"] - q["gt_mean"]) / q["gt_temporal_sd"]
                for key, value in _reference_provenance(d, data).items():
                    q[key] = value
                cascade_rows.append(q)
            if cascade_rows:
                write(pd.concat(cascade_rows, ignore_index=True), "cascade_values.csv")

            def draw(ax, d, legend):
                stage = d["stage"]
                if stage.empty:
                    return _missing(ax, "No temporal expression summary")
                h = (
                    stage[stage["gene"].isin(ordered)]
                    .pivot_table(index="gene", columns="time", values="mean", aggfunc="mean")
                    .reindex(ordered)
                )
                h = h.sub(center, axis=0).div(scale, axis=0)
                if not np.isfinite(h.to_numpy()).any():
                    return _missing(ax)
                im = ax.imshow(
                    h,
                    aspect="auto",
                    cmap=plot.DIVERGING_CMAP,
                    vmin=-3,
                    vmax=3,
                    interpolation="nearest",
                )
                ax.set_xticks(
                    range(len(h.columns)),
                    [f"{x:g}" for x in h.columns],
                    rotation=60,
                    ha="right",
                    fontsize=5,
                )
                ax.set_yticks(range(len(ordered)), ordered, fontsize=6)
                ax.set_xlabel("Saved developmental age")
                if legend:
                    ax.figure.colorbar(
                        im, ax=ax, label="Frozen GT temporal z scale", fraction=0.035, pad=0.02
                    )

            emit(view, draw, "Truth-peak-ordered developmental gene cascade")

            def residual(ax, d, legend):
                stage = d["stage"]
                if stage.empty:
                    return _missing(ax, "No temporal expression summary")
                h = (
                    stage[stage["gene"].isin(ordered)]
                    .pivot_table(index="gene", columns="time", values="mean", aggfunc="mean")
                    .reindex(ordered)
                )
                ownref = _reference(d, data)
                if ownref is None or ownref["stage"].empty:
                    return _missing(ax, "No compatible bound observed gene summary")
                baseline = (
                    ownref["stage"]
                    .pivot_table(index="gene", columns="time", values="mean", aggfunc="mean")
                    .reindex(index=ordered, columns=h.columns)
                )
                delta = (h - baseline).div(scale, axis=0)
                if not np.isfinite(delta.to_numpy()).any():
                    return _missing(ax, "No matched observed ages/genes")
                im = ax.imshow(
                    delta,
                    aspect="auto",
                    cmap=plot.DIVERGING_CMAP,
                    vmin=-3,
                    vmax=3,
                    interpolation="nearest",
                )
                ax.set_xticks(
                    range(len(h.columns)),
                    [f"{x:g}" for x in h.columns],
                    rotation=60,
                    ha="right",
                    fontsize=5,
                )
                ax.set_yticks(range(len(ordered)), ordered, fontsize=6)
                ax.set_xlabel("Saved developmental age")
                if legend:
                    ax.figure.colorbar(
                        im, ax=ax, label="(Model − GT) / GT temporal SD", fraction=0.035, pad=0.02
                    )

            emit("cascade_residual", residual, "Observed developmental gene program residuals")
        elif view.startswith("program_") or view == "grn_activity":
            programs = _programs(cohort, options, grn=view == "grn_activity", data=data)
            if programs.empty:
                notes.append(f"{view}: no compatible program/TF-target annotation supplied.")
                continue
            coverage = (
                programs[programs["gene"].isin(set(g for d in main for g in d["genes"]))]
                .groupby("program")["gene"]
                .nunique()
                .sort_values(ascending=False)
            )
            names = coverage.index.tolist()
            names = names[:limit] if limit > 0 else names
            prefix = "grn_program" if view == "grn_activity" else "program"
            write(programs, prefix + "_membership.csv")
            audits = [
                _program_audit(d, programs[programs["program"] == name], data)
                for name in names
                for d in main
            ]
            write(pd.DataFrame(audits), prefix + "_coverage.csv")
            activity = []
            for d in main:
                for name in names:
                    a = _aggregate_program(d, programs[programs["program"] == name], data)
                    if not a.empty:
                        a = a.copy()
                        a["program"] = name
                        a["model"] = d["source"].get("model")
                        for key, value in _reference_provenance(d, data).items():
                            a[key] = value
                        activity.append(a)
            if activity:
                write(pd.concat(activity, ignore_index=True), prefix + "_activity.csv")
            if view == "program_feature":
                for d in main:
                    c = _cells(d)
                    cols = [
                        col
                        for col in (
                            "cell_id",
                            "time",
                            "state",
                            "expression_row",
                            "sample_weight",
                            "x",
                            "y",
                        )
                        if col in c
                    ]
                    frame = c[cols].copy()
                    ix = (
                        c["expression_row"].to_numpy(int)
                        if "expression_row" in c
                        else c.index.to_numpy(int)
                    )
                    for name in names:
                        values, _ = _program_scores(d, programs[programs["program"] == name], data)
                        if values is not None:
                            frame["program:" + str(name)] = values[ix]
                    frame["model"] = d["source"].get("model")
                    if len(frame):
                        write(
                            frame,
                            "program_sample_scores__" + _slug(d["source"].get("model")) + ".csv",
                        )
            if len({a["score_scale"] for a in audits if a["score_scale"] != "undefined"}) > 1:
                note = (
                    "Program panels contain raw-expression and observed-reference standardized modes, labelled per panel and audited in "
                    + prefix
                    + "_coverage.csv; magnitudes across these modes are not comparable."
                )
                if note not in notes:
                    notes.append(note)
            if view in PROGRAM_VIEWS:
                for name in names:
                    prog = programs[programs["program"] == name]
                    scores = [_program_scores(d, prog, data)[0] for d in main]
                    joined = (
                        np.concatenate([s for s in scores if s is not None and len(s)])
                        if any(s is not None and len(s) for s in scores)
                        else np.array([0.0, 1.0])
                    )
                    lim = _limits(joined, (0.01, 0.99))

                    def draw(ax, d, legend, prog=prog):
                        values, members = _program_scores(d, prog, data)
                        if values is None:
                            return _missing(ax, "Program genes absent from native panel")
                        audit = _program_audit(d, prog, data)
                        unit = (
                            "Observed-reference z weighted mean"
                            if audit["score_scale"] == "observed_reference_standardized"
                            else "Raw expression weighted mean"
                        )
                        if view == "program_feature":
                            c = _cells(d)
                            rows = (
                                c["expression_row"].to_numpy(int)
                                if "expression_row" in c
                                else c.index.to_numpy(int)
                            )
                            im = ax.scatter(
                                c["x"],
                                c["y"],
                                c=values[rows],
                                cmap=plot.DIVERGING_CMAP,
                                vmin=lim[0],
                                vmax=lim[1],
                                s=6,
                                alpha=0.85,
                                linewidths=0,
                                rasterized=True,
                            )
                            _format_geometry(ax, _extent(d))
                            if legend:
                                ax.figure.colorbar(im, ax=ax, label=unit, fraction=0.035, pad=0.02)
                        else:
                            a = _aggregate_program(d, prog, data)
                            if a.empty:
                                return _missing(ax, "No temporal program summary")
                            ax.plot(a["time"], a["value"], "o-", color="#287C8E", ms=3)
                            ax.set_xlabel("Saved developmental age")
                            ax.set_ylabel(unit)
                        ax.text(
                            0.02,
                            0.98,
                            f'{len(members)}/{len(prog)} genes · {audit["score_scale"]}',
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=6,
                            color="#52606A",
                        )

                    emit(view, draw, "Known developmental program: " + str(name), str(name))
            else:
                all_program_values = []
                for modeldata in main:
                    for name in names:
                        temp = _aggregate_program(
                            modeldata, programs[programs["program"] == name], data
                        )
                        if not temp.empty:
                            all_program_values.extend(temp["value"].tolist())
                program_limit = _limits(all_program_values, (0.01, 0.99))

                def draw(ax, d, legend):
                    rows = []
                    for name in names:
                        a = _aggregate_program(d, programs[programs["program"] == name], data)
                        if not a.empty:
                            a["program"] = name
                            rows.append(a)
                    if not rows:
                        return _missing(ax, "No covered programs")
                    h = (
                        pd.concat(rows)
                        .pivot(index="program", columns="time", values="value")
                        .reindex(names)
                    )
                    im = ax.imshow(
                        h,
                        aspect="auto",
                        cmap=plot.DIVERGING_CMAP,
                        vmin=program_limit[0],
                        vmax=program_limit[1],
                        interpolation="nearest",
                    )
                    ax.set_yticks(range(len(names)), names, fontsize=5)
                    ax.set_xticks(
                        range(len(h.columns)),
                        [f"{x:g}" for x in h.columns],
                        rotation=60,
                        ha="right",
                        fontsize=5,
                    )
                    ax.set_xlabel("Saved developmental age")
                    modes = {
                        a["score_scale"]
                        for a in audits
                        if a["model"] == d["source"].get("model")
                        and a["score_scale"] != "undefined"
                    }
                    unit = (
                        "Observed-reference z weighted mean"
                        if modes == {"observed_reference_standardized"}
                        else (
                            "Raw expression weighted mean"
                            if modes == {"raw_expression"}
                            else "Mixed raw/z; see coverage CSV"
                        )
                    )
                    if legend:
                        ax.figure.colorbar(im, ax=ax, label=unit, fraction=0.035, pad=0.02)

                emit(
                    view,
                    draw,
                    (
                        "Prior TF-target modules"
                        if view == "grn_activity"
                        else "Author developmental programs"
                    )
                    + " across stages",
                )
        elif view == "state_occupancy":
            if not states:
                notes.append("No usable state labels for occupancy.")
                continue
            rows = []

            def draw(ax, d, legend):
                q = _state_occupancy(d, states, all_states)
                if q.empty:
                    return _missing(ax, "No state occupancy")
                table = q.pivot(index="time", columns="state", values="fraction")
                if table.empty:
                    return _missing(ax, "No state occupancy")
                displayed = table.columns.tolist()
                ax.stackplot(
                    table.index,
                    *[table[s] for s in displayed],
                    colors=[statecolors.get(s, plot.BACKGROUND) for s in displayed],
                    labels=displayed,
                    alpha=0.85,
                )
                ax.set_xlabel("Saved developmental age")
                ax.set_ylabel("Weighted drawing-sample state fraction")
                ax.set_ylim(0, 1)
                if legend and len(displayed) <= 12:
                    ax.legend(fontsize=5, frameon=False, loc="upper left")
                q["model"] = d["source"].get("model")
                rows.append(q)

            emit(
                view, draw, "Developmental state occupancy; inference/conditioning method disclosed"
            )
            if rows:
                write(pd.concat(rows, ignore_index=True).drop_duplicates(), "state_occupancy.csv")
        elif view == "state_graph":
            if len(states) < 2:
                notes.append("State graph needs at least two usable states.")
                continue
            import networkx as nx

            connect = {}
            nodes = {}
            for d in main:
                c = _cells(d)
                if c.empty:
                    continue
                # Geometry is shared reference-expression projection, not
                # lineage or PAGA. Avoid dense cell adjacency on Big.
                pc_cols = sorted(
                    [col for col in c if str(col).startswith("pc") and str(col)[2:].isdigit()],
                    key=lambda col: int(str(col)[2:]),
                )
                xy = c[pc_cols].to_numpy(float) if pc_cols else c[["x", "y"]].to_numpy(float)
                labs = c["state"].astype(str).to_numpy()
                ages = c["time"].to_numpy(float)
                if len(xy) > 8000:
                    ix = np.linspace(0, len(xy) - 1, 8000).astype(int)
                    xy = xy[ix]
                    labs = labs[ix]
                    ages = ages[ix]
                k = min(12, len(xy) - 1)
                if k < 1:
                    continue
                nearest = (
                    NearestNeighbors(n_neighbors=k + 1)
                    .fit(xy)
                    .kneighbors(xy, return_distance=False)[:, 1:]
                )
                es = {}
                n = {}
                for state in states:
                    mask = labs == state
                    n[state] = {
                        "n": int(mask.sum()),
                        "age": float(np.nanmedian(ages[mask])) if mask.any() else np.nan,
                    }
                for i, nei in enumerate(nearest):
                    a = labs[i]
                    if a not in states:
                        continue
                    for j in nei:
                        b = labs[j]
                        if b not in states or a == b:
                            continue
                        pair = tuple(sorted([a, b]))
                        es[pair] = es.get(pair, 0) + 1
                es = {
                    pair: v / max(k * (n[pair[0]]["n"] + n[pair[1]]["n"]), 1)
                    for pair, v in es.items()
                }
                connect[id(d)] = es
                nodes[id(d)] = n
            graph = nx.Graph()
            graph.add_nodes_from(states)
            if primary is not None:
                graph.add_edges_from(connect.get(id(primary), {}))
            else:
                for es in connect.values():
                    graph.add_edges_from(es)
            pos = nx.spring_layout(graph, seed=42, weight=None)
            rows = []

            def draw(ax, d, legend):
                n = nodes.get(id(d), {})
                es = connect.get(id(d), {})
                if not n:
                    return _missing(ax, "No state-neighborhood graph")
                g = nx.Graph()
                g.add_nodes_from([s for s in states if n[s]["n"] > 0])
                g.add_edges_from(es)
                weights = [0.4 + 8 * es[e] for e in g.edges]
                nx.draw_networkx_edges(
                    g, pos, ax=ax, width=weights, edge_color="#6F7D85", alpha=0.75
                )
                nx.draw_networkx_nodes(
                    g,
                    pos,
                    ax=ax,
                    node_size=[
                        50 + 500 * n[s]["n"] / max(sum(x["n"] for x in n.values()), 1) for s in g
                    ],
                    node_color=[statecolors[s] for s in g],
                    edgecolors="#42535D",
                    linewidths=0.5,
                )
                if legend:
                    nx.draw_networkx_labels(g, pos, ax=ax, font_size=5)
                ax.axis("off")
                for (a, b), weight in es.items():
                    rows.append(
                        {
                            "model": d["source"].get("model"),
                            "source": a,
                            "target": b,
                            "connectivity": weight,
                            "method": (
                                "fixed_reference_PC_knn_connectivity"
                                if any(
                                    str(col).startswith("pc") and str(col)[2:].isdigit()
                                    for col in d["cells"]
                                )
                                else "shared_reference_2d_knn_connectivity"
                            ),
                        }
                    )

            emit(view, draw, "Expression-neighborhood state organization; edges are associations")
            if rows:
                write(pd.DataFrame(rows).drop_duplicates(), "state_graph_edges.csv")
            write(
                pd.DataFrame(
                    [
                        {
                            "model": d["source"].get("model"),
                            "state": state,
                            "sample_count": entry["n"],
                            "median_age": entry["age"],
                            "displayed_state": True,
                        }
                        for d in main
                        for state, entry in nodes.get(id(d), {}).items()
                    ]
                ),
                "state_graph_nodes.csv",
            )
        elif view in ("coexpression", "grn_network", "grn_support", "edge_rescue"):
            prior = _programs(cohort, options, grn=True)
            if view != "coexpression" and prior.empty:
                notes.append(f"{view}: no compatible supplied TF-target edges.")
                continue
            if view == "coexpression":
                names = genes[: min(len(genes), 25)]
                if len(names) < 2:
                    notes.append("Coexpression needs at least two eligible genes.")
                    continue
                matrices = {}
                correlation_rows = []
                for d in main:
                    native = [g for g in names if g in d["genes"]]
                    if len(native) < 2:
                        continue
                    x = np.asarray(d["expression"][:, [d["genes"].index(g) for g in native]], float)
                    ages = d["cells"]["time"].to_numpy(float)
                    for age in np.unique(ages):
                        x[np.isclose(ages, age)] -= x[np.isclose(ages, age)].mean(axis=0)
                    corr = _corr_matrix(x)
                    matrices[id(d)] = (native, corr)
                    for i, source in enumerate(native):
                        for j, target in enumerate(native):
                            correlation_rows.append(
                                {
                                    "model": d["source"].get("model"),
                                    "source": source,
                                    "target": target,
                                    "within_age_residual_correlation": corr[i, j],
                                    "n_declared_sample_cells": len(x),
                                    "method": "unweighted declared drawing-sample association after within-age centering; not causality",
                                }
                            )
                if correlation_rows:
                    write(pd.DataFrame(correlation_rows), "coexpression_values.csv")

                def draw(ax, d, legend):
                    if id(d) not in matrices:
                        return _missing(ax, "Fewer than two selected native genes")
                    native, corr = matrices[id(d)]
                    im = ax.imshow(corr, aspect="equal", cmap=plot.DIVERGING_CMAP, vmin=-1, vmax=1)
                    ax.set_xticks(range(len(native)), native, rotation=80, ha="right", fontsize=5)
                    ax.set_yticks(range(len(native)), native, fontsize=5)
                    if legend:
                        ax.figure.colorbar(
                            im,
                            ax=ax,
                            label="Within-age residual association",
                            fraction=0.035,
                            pad=0.02,
                        )

                emit(view, draw, "Developmental marker coexpression after within-age centering")
            else:
                names = (
                    prior.groupby("program")["gene"]
                    .nunique()
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                names = names[:limit] if limit > 0 else names
                rows = []
                for tf in names:
                    prog = prior[prior["program"] == tf]
                    targets = prog["gene"].astype(str).tolist()[: min(20, len(prog))]
                    support = {id(d): _edge_support(d, tf, targets) for d in main}
                    observed = {id(d): _reference(d, data) for d in main}
                    reference_support = {
                        id(d): (
                            _edge_support(observed[id(d)], tf, targets)
                            if observed[id(d)] is not None
                            else {}
                        )
                        for d in main
                    }
                    prior_absolute = prog.set_index("gene")["weight"].to_dict()
                    prior_signed = prog.set_index("gene")["raw_signed_weight"].to_dict()
                    import networkx as nx

                    g = nx.DiGraph()
                    g.add_edges_from([(tf, gene) for gene in targets if gene != tf])
                    pos = nx.spring_layout(g, seed=42)

                    def draw(ax, d, legend, tf=tf, targets=targets):
                        assoc = support[id(d)]
                        ref = observed[id(d)]
                        refs = reference_support[id(d)]
                        if not assoc:
                            return _missing(ax, "Regulator/targets absent from native panel")
                        shared = sorted(
                            t
                            for t in set(assoc) & set(refs)
                            if np.isfinite(assoc[t]) and np.isfinite(refs[t])
                        )
                        if view == "grn_network":
                            colors = [assoc.get(n, np.nan) if n != tf else 0 for n in g]
                            nx.draw_networkx_edges(
                                g,
                                pos,
                                ax=ax,
                                edge_color="#78848B",
                                arrows=True,
                                arrowsize=8,
                                width=0.8,
                            )
                            nx.draw_networkx_nodes(
                                g,
                                pos,
                                ax=ax,
                                node_size=[260 if n == tf else 140 for n in g],
                                node_color=colors,
                                cmap=plot.DIVERGING_CMAP,
                                vmin=-1,
                                vmax=1,
                                edgecolors="#3F535D",
                                linewidths=0.4,
                            )
                            if legend:
                                nx.draw_networkx_labels(g, pos, ax=ax, font_size=5)
                            ax.axis("off")
                        else:
                            if not shared:
                                return _missing(ax, "No covered observed TF-target support")
                            x = np.array([refs[t] for t in shared])
                            y = np.array([assoc[t] for t in shared])
                            if view == "edge_rescue":
                                ax.barh(range(len(shared)), np.abs(y - x), color="#508F9A")
                                ax.set_yticks(range(len(shared)), shared, fontsize=5)
                                ax.set_xlabel("|Model − GT correlation|")
                            else:
                                ax.plot([-1, 1], [-1, 1], ls="--", color="#89959C", lw=0.8)
                                ax.scatter(x, y, color="#397F8E", s=18)
                                ax.set_xlabel("GT within-age association")
                                ax.set_ylabel("Model within-age association")
                                ax.set_xlim(-1, 1)
                                ax.set_ylim(-1, 1)
                                if len(shared) < 2:
                                    ax.text(
                                        0.02,
                                        0.98,
                                        "n=1 supported target; descriptive",
                                        transform=ax.transAxes,
                                        ha="left",
                                        va="top",
                                        fontsize=6,
                                        color="#59636C",
                                    )
                            if legend and view == "grn_support" and len(shared) <= 10:
                                for a, b, t in zip(x, y, shared):
                                    ax.annotate(t, (a, b), fontsize=5)
                        for target, value in assoc.items():
                            rows.append(
                                {
                                    "model": d["source"].get("model"),
                                    "source": tf,
                                    "target": target,
                                    "within_age_residual_correlation": value,
                                    "gt_support": refs.get(target, np.nan),
                                    "prior_supplied": True,
                                    "observed_model": (
                                        ref["source"].get("model") if ref is not None else ""
                                    ),
                                    "observed_source_id": (
                                        ref["source"].get("source_id", "")
                                        if ref is not None
                                        else ""
                                    ),
                                    "observed_panel": (
                                        ref["source"].get("panel") if ref is not None else ""
                                    ),
                                    "model_scale_group": d.get("meta", {}).get(
                                        "scale_group", "unspecified"
                                    ),
                                    "observed_scale_group": (
                                        ref.get("meta", {}).get("scale_group", "unspecified")
                                        if ref is not None
                                        else ""
                                    ),
                                    "prior_absolute_weight": prior_absolute.get(target, np.nan),
                                    "prior_raw_signed_weight": prior_signed.get(target, np.nan),
                                    "n_finite_supported_targets": len(shared),
                                    "association_error_formula": "abs(model within-age residual correlation - observed correlation)",
                                    **_reference_provenance(d, data),
                                }
                            )

                    emit(view, draw, "Supplied TF-target association: " + tf, tf)
                if rows:
                    edge_table = pd.DataFrame(rows).drop_duplicates()
                    write(edge_table, "grn_edge_support.csv")
                if view == "edge_rescue" and rows:
                    pivot = edge_table.assign(
                        error=(
                            edge_table["within_age_residual_correlation"] - edge_table["gt_support"]
                        ).abs()
                    ).pivot_table(
                        index=["source", "target"], columns="model", values="error", aggfunc="mean"
                    )
                    baseline = options.get(
                        "edge_baseline_model",
                        "v6_200_new" if cohort_native == "hvg" else "v6_author",
                    )
                    if baseline not in pivot:
                        baseline = next(
                            (
                                d["source"].get("model")
                                for d in main
                                if d["source"].get("role") == "FM"
                                and d["source"].get("model") in pivot
                            ),
                            None,
                        )
                    if baseline is not None:
                        comparison_metadata = (
                            edge_table[
                                [
                                    "model",
                                    "observed_source_id",
                                    "model_target_stages",
                                    "paired_source",
                                    "reference_match",
                                ]
                            ]
                            .drop_duplicates("model")
                            .set_index("model")
                        )
                        baseline_metadata = comparison_metadata.loc[baseline]
                        comparable_models = {
                            model
                            for model, row in comparison_metadata.iterrows()
                            if row["observed_source_id"] == baseline_metadata["observed_source_id"]
                            and row["model_target_stages"]
                            == baseline_metadata["model_target_stages"]
                        }
                        contrast_pivot = pivot.copy()
                        for model in contrast_pivot:
                            if model not in comparable_models:
                                contrast_pivot[model] = np.nan
                        contrast_table = (
                            pivot.rename_axis(columns="model")
                            .stack()
                            .rename("absolute_gt_correlation_error")
                            .reset_index()
                        )
                        baseline_errors = pivot[baseline].rename("baseline_error").reset_index()
                        contrast_table = contrast_table.merge(
                            baseline_errors, on=["source", "target"], how="left"
                        )
                        contrast_table = contrast_table.merge(
                            comparison_metadata.reset_index(), on="model", how="left"
                        )
                        contrast_table["contrast_comparable"] = contrast_table["model"].isin(
                            comparable_models
                        )
                        contrast_table["baseline_observed_source_id"] = baseline_metadata[
                            "observed_source_id"
                        ]
                        contrast_table["contrast_reason"] = np.where(
                            contrast_table["contrast_comparable"],
                            "same observed identity and target stages; exact/descriptive binding disclosed",
                            "different observed identity or target coverage; contrast withheld",
                        )
                        contrast_table["baseline_error_minus_model_error"] = (
                            contrast_table["baseline_error"]
                            - contrast_table["absolute_gt_correlation_error"]
                        ).where(contrast_table["contrast_comparable"])
                        contrast_table["baseline_model"] = baseline
                        contrast_table["contrast_formula"] = (
                            "abs(baseline correlation - bound observed correlation) - abs(model correlation - bound observed correlation)"
                        )
                        contrast_table["positive_means"] = (
                            "smaller own-observed association error; no ablation or causality inference"
                        )
                        write(contrast_table, "PRIOR_EDGE_ERROR_CONTRAST.csv")
                        network = nx.DiGraph()
                        network.add_edges_from(pivot.index.tolist())
                        position = nx.spring_layout(network, seed=42)
                        all_changes = contrast_pivot.subtract(
                            contrast_pivot[baseline], axis=0
                        ).to_numpy(float)
                        contrast_limit = max(float(np.nanmax(np.abs(all_changes))), 1e-8)

                        def contrast(ax, d, legend):
                            model = d["source"].get("model")
                            if model not in pivot:
                                return _missing(ax, "No matched reference TF-target support")
                            change = contrast_pivot[baseline] - contrast_pivot[model]
                            eligible = (
                                change.dropna()
                                .sort_values(key=lambda v: v.abs(), ascending=False)
                                .head(40)
                            )
                            g = nx.DiGraph()
                            g.add_edges_from(eligible.index.tolist())
                            if not g.number_of_edges():
                                return _missing(ax, "No same-target contrast edges")
                            bound = contrast_limit
                            nx.draw_networkx_nodes(
                                g,
                                position,
                                ax=ax,
                                node_size=100,
                                node_color="#96B1B4",
                                edgecolors="#4C646A",
                                linewidths=0.4,
                            )
                            nx.draw_networkx_edges(
                                g,
                                position,
                                ax=ax,
                                edge_color=eligible.reindex(list(g.edges)).to_numpy(),
                                edge_cmap=plot.DIVERGING_CMAP,
                                edge_vmin=-bound,
                                edge_vmax=bound,
                                width=1.3,
                                arrowsize=6,
                            )
                            if legend:
                                nx.draw_networkx_labels(g, position, ax=ax, font_size=4)
                            ax.axis("off")
                            if legend:
                                from matplotlib.cm import ScalarMappable

                                ax.figure.colorbar(
                                    ScalarMappable(
                                        norm=Normalize(-bound, bound), cmap=plot.DIVERGING_CMAP
                                    ),
                                    ax=ax,
                                    label="Baseline error − model error",
                                    fraction=0.035,
                                    pad=0.02,
                                )

                        emit(
                            "prior_edge_error_contrast",
                            contrast,
                            "Supplied edges closer to GT than "
                            + str(options.get("aliases", {}).get(baseline, baseline)),
                        )
                    notes.append(
                        "Edge-network contrast uses a named actual current baseline; positive values mean smaller observed-reference error. No bio0/bio6 ablation is inferred."
                    )
        elif view == "author_graph":
            cfg = _cfg(cohort, options)
            path = cfg.get("graph_path") or cfg.get("author_graph_path")
            table = _read_annotation(path)
            if table.empty:
                notes.append("Author topology needs a supplied author_graph_path/graph_path.")
                continue
            low = {str(c).lower(): c for c in table}
            a = next(
                (
                    low[k]
                    for k in ("source", "source_id", "source_node", "parent", "parent_id", "from")
                    if k in low
                ),
                None,
            )
            b = next(
                (
                    low[k]
                    for k in ("target", "target_id", "target_node", "child", "child_id", "to")
                    if k in low
                ),
                None,
            )
            if a is None or b is None:
                notes.append("Author graph lacks source/target identifiers.")
                continue
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_edges_from(table[[a, b]].astype(str).itertuples(index=False, name=None))
            pos = nx.spring_layout(graph.to_undirected(), seed=42)
            write(table, "author_graph_supplied_edges.csv")
            candidates = list(
                dict.fromkeys(
                    [
                        cfg.get("author_graph_state_column"),
                        cfg.get("graph_node_column"),
                        "state",
                        "state_supplied",
                        *cfg.get(
                            "annotation_columns", cfg.get("additional_classification_columns", [])
                        ),
                    ]
                )
            )
            reference = primary if primary is not None else main[0]
            graphnames = set(map(str, graph.nodes))
            overlaps = {}
            for col in candidates:
                if col and col in reference["cells"]:
                    overlaps[col] = len(
                        set(reference["cells"][col].dropna().astype(str)) & graphnames
                    )
            mapping = (
                max(overlaps, key=overlaps.get) if overlaps and max(overlaps.values()) > 0 else None
            )
            matching = (
                (set(reference["cells"][mapping].dropna().astype(str)) & graphnames)
                if mapping
                else set()
            )
            if not matching:
                # Useful supplied topology remains visible, but no empty
                # occupancy is presented as a generated reconstruction.
                fig, ax = plt.subplots(figsize=(8, 6))
                nx.draw_networkx(
                    graph,
                    pos,
                    ax=ax,
                    node_size=80,
                    font_size=4,
                    node_color="#79AAB0",
                    edge_color="#8A989F",
                    arrowsize=5,
                    with_labels=False,
                )
                ax.axis("off")
                ax.set_title("Supplied author topology; exact observed node mapping unavailable")
                paths = plot.save(fig, output / "author_topology_only", options)
                plt.close(fig)
                figures.extend(_record(paths, view))
                write(
                    pd.DataFrame(
                        [
                            {
                                "author_node": str(n),
                                "mapping_status": "unavailable",
                                "mapping_column": "",
                                "sample_count": np.nan,
                                "weighted_mass": np.nan,
                            }
                            for n in graph
                        ]
                    ),
                    "author_graph_node_mapping.csv",
                )
                notes.append(
                    "Author nodes do not match the observed prepared state/explicit annotation vocabulary: topology shown alone; no invented occupancy. Configure author_graph_state_column and include it in annotation_columns for an exact-ID source obs join."
                )
                incomplete_semantics = True
                continue
            reference_count = reference["cells"][mapping].astype(str).value_counts()
            named_nodes = sorted(matching, key=lambda n: (-reference_count.get(n, 0), n))
            nodecap = int(options.get("max_display_states", 12))
            if nodecap > 0:
                named_nodes = named_nodes[:nodecap]
            node_rows = []
            for d in main:
                c = d["cells"]
                labels = (
                    c[mapping].astype(str)
                    if mapping in c
                    else pd.Series("unavailable", index=c.index)
                )
                weights = pd.to_numeric(
                    c.get("sample_weight", pd.Series(1.0, index=c.index)), errors="coerce"
                ).fillna(0.0)
                for n in graph:
                    mask = labels.eq(str(n))
                    ages = (
                        pd.to_numeric(c.loc[mask, "time"], errors="coerce")
                        if "time" in c
                        else pd.Series(dtype=float)
                    )
                    node_rows.append(
                        {
                            "model": d["source"].get("model"),
                            "author_node": str(n),
                            "mapping_column": mapping,
                            "mapping_status": (
                                "explicit matched prepared annotation"
                                if mapping in c
                                else "column unavailable in this model"
                            ),
                            "sample_count": int(mask.sum()) if mapping in c else np.nan,
                            "weighted_mass": float(weights[mask].sum()) if mapping in c else np.nan,
                            "median_age": float(ages.median()) if len(ages) else np.nan,
                            "selected_named_node": str(n) in named_nodes,
                            "source_labels_are_conditioning_annotations": mapping != "state",
                            "state_method": d["meta"].get("state_method", "unknown"),
                        }
                    )
            node_table = pd.DataFrame(node_rows)
            write(node_table, "author_graph_node_mapping.csv")

            def draw(ax, d, legend):
                if mapping not in d["cells"]:
                    return _missing(ax, "Exact author-node annotation unavailable")
                local = node_table[node_table["model"].eq(d["source"].get("model"))].set_index(
                    "author_node"
                )
                counts = local["weighted_mass"]
                sizes = [40 + 350 * counts.get(str(n), 0) / max(counts.max(), 1) for n in graph]
                nx.draw_networkx_edges(
                    graph, pos, ax=ax, edge_color="#89969C", arrowsize=7, alpha=0.7
                )
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    ax=ax,
                    node_size=sizes,
                    node_color=["#508F9A" if str(n) in named_nodes else "#ADB5BC" for n in graph],
                    edgecolors="#52636C",
                    linewidths=0.3,
                )
                if legend:
                    nx.draw_networkx_labels(
                        graph,
                        pos,
                        ax=ax,
                        labels={n: str(n) for n in graph if str(n) in named_nodes},
                        font_size=5,
                    )
                ax.axis("off")

            emit(
                view,
                draw,
                "Supplied author topology; exact "
                + str(mapping)
                + " annotation, GT-supported named nodes",
            )
            notes.append(
                "Author node occupancy uses the explicitly matched "
                + str(mapping)
                + " vocabulary. Supplied/template node counts describe conditioning coverage; they do not prove autonomous developmental fate generation. Complete node mass and named-node selection are in author_graph_node_mapping.csv."
            )
        elif view == "candidate_evidence":
            cols = [c for c in ("score", "effect", "effect_size", "q_value") if c in selected]
            if not cols or selected.empty:
                notes.append("No reference gene-selection evidence to visualize.")
                continue
            t = selected.loc[selected["gene"].isin(genes), ["gene", *cols]].set_index("gene")
            fig, ax = plt.subplots(figsize=(max(5, len(cols) * 1.3), max(3, len(t) * 0.24)))
            for col in cols:
                a = pd.to_numeric(t[col], errors="coerce")
                t[col] = (a - a.min()) / max(a.max() - a.min(), 1e-12)
            im = ax.imshow(t.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_yticks(range(len(t)), t.index, fontsize=7)
            ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
            ax.set_title("Reference-only developmental marker screening evidence")
            ax.figure.colorbar(im, ax=ax, label="Within-column scaled evidence")
            paths = plot.save(fig, output / view, options)
            plt.close(fig)
            figures.extend(_record(paths, view))
            _notify(progress, view, "", len(figures))
        elif view == "regulator_evidence":
            if primary is None:
                notes.append("Regulator evidence needs an observed native-panel expression source.")
                continue
            programs = _programs(cohort, options, data=None)
            prior = _programs(cohort, options, grn=True)
            baseline = options.get(
                "edge_baseline_model", "v6_200_new" if cohort_native == "hvg" else "v6_author"
            )
            evidence, reconstruction, coverage = _regulator_tables(
                primary, main, selected, programs, prior, baseline
            )
            write(evidence, "REGULATOR_BIOLOGICAL_EVIDENCE.csv")
            write(reconstruction, "REGULATOR_MODEL_RECONSTRUCTION.csv")
            write(coverage, "REGULATOR_COVERAGE.csv")
            if not prior.empty:
                write(prior, "REGULATOR_PRIOR_EDGES.csv")
            if not programs.empty:
                write(programs, "REGULATOR_AUTHOR_PROGRAM_MEMBERSHIP.csv")
            named = [g for g in genes if g in set(evidence["gene"])]
            if not named:
                notes.append(
                    "Observed regulator evidence was exported but no biologically eligible display genes were selected."
                )
                continue
            features = [
                "gt_temporal_variance",
                "gt_state_standardized_range",
                "prior_outdegree",
                "prior_indegree",
                "author_program_count",
            ]
            shown = evidence.set_index("gene").reindex(named)[features]
            # Each independent component is shown. A tiny TF star does not
            # boost a composite score, and prediction wins never select genes.
            ranks = shown.apply(
                lambda col: col.rank(pct=True, method="average") if col.notna().any() else col
            )
            fig, ax = plt.subplots(
                figsize=(max(6, len(features) * 1.15), max(3, len(named) * 0.28))
            )
            im = ax.imshow(
                ranks.to_numpy(float), aspect="auto", cmap=plot.EXPRESSION_CMAP, vmin=0, vmax=1
            )
            ax.set_yticks(range(len(named)), named, fontsize=7)
            ax.set_xticks(
                range(len(features)),
                [
                    "GT temporal\nvariance",
                    "GT state\nspecificity",
                    "Supplied prior\noutdegree",
                    "Supplied prior\nindegree",
                    "Author program\nmembership",
                ],
                fontsize=7,
            )
            ax.set_title("Developmental regulator evidence; components shown separately")
            ax.set_ylabel("GT-selected biological gene")
            ax.figure.colorbar(
                im,
                ax=ax,
                label="Within-component display percentile; no combined priority/significance",
                fraction=0.035,
                pad=0.02,
            )
            paths = plot.save(fig, output / "regulator_evidence", options)
            plt.close(fig)
            figures.extend(_record(paths, view))
            _notify(progress, view, "", len(figures))
            errors = reconstruction[reconstruction["gene"].isin(named)]
            max_error = (
                max(float(errors["stage_mean_mae"].max()), 1e-8)
                if errors["stage_mean_mae"].notna().any()
                else 1.0
            )

            def draw_regulator(ax, d, legend):
                local = (
                    errors[errors["model"].eq(d["source"].get("model"))]
                    .set_index("gene")
                    .reindex(named)
                )
                if not local["stage_mean_mae"].notna().any():
                    return _missing(ax, "No same-assay matched observed stages")
                ax.barh(range(len(named)), local["stage_mean_mae"], color="#397F8E")
                ax.invert_yaxis()
                ax.set_yticks(range(len(named)), named, fontsize=6)
                ax.set_xlim(0, max_error * 1.04)
                ax.set_xlabel("Matched stage-mean absolute error")
                if legend:
                    ax.text(
                        0.98,
                        0.98,
                        "Lower = closer observed means\nPrior association error in CSV",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=6,
                        color="#53636D",
                    )

            emit(
                "regulator_reconstruction",
                draw_regulator,
                "Selected regulator expression reconstruction within the native assay",
            )
            notes.append(
                "Regulator components are independent GT/prior evidence, not a significance score or causal regulator proof. Current named-baseline association-error contrast is separate in REGULATOR_MODEL_RECONSTRUCTION.csv; no bio0/bio6 ablation is inferred and genes are never selected by a model win."
            )
    source_rows = []
    for d in data:
        m = d["meta"]
        s = d["source"]
        source_rows.append(
            {
                "model": s.get("model"),
                "role": s.get("role"),
                "panel": s.get("panel"),
                "n_total": m.get("n_total"),
                "n_sample": len(d["cells"]),
                "state_method": m.get("state_method", s.get("state_method", "unknown")),
                "reference_policy": m.get("reference_policy"),
                "reference_id": m.get("reference_id"),
                "scale_group": m.get("scale_group"),
                "embedding_method": m.get("embedding_method"),
                "time_kind": m.get("time_kind"),
                "native_gene_count": len(d["genes"]),
                **_reference_provenance(d, data),
            }
        )
    write(pd.DataFrame(source_rows), "MODEL_ROWS.csv")
    notes.extend(
        [
            "Atlas/centroid/neighbor views are descriptive expression geometry, not lineage evidence.",
            "Drawing distributions use the declared cell sample; density/centroid/occupancy use sample weights when provided; temporal means, SD and detection use saved full-row summaries.",
            "Full-row state_gene marker summaries use supplied/template state strata, not autonomous predicted composition. Sample cell-state maps/violins use the declared state inference method. Fraction > 0 refers to saved expression units, not molecular RNA detection if the data were scaled. Native gene/program coverage is explicit.",
        ]
    )
    rendered = {f["view"] for f in figures}
    missing_views = sorted(set(views) - rendered - capped_views)
    if capped_views:
        notes.append(
            "Explicit per-entity view cap omitted additional variants: "
            + ", ".join(sorted(capped_views))
        )
    if missing_views:
        notes.append(
            "Requested views without a drawable supported result: " + ", ".join(missing_views)
        )
    status = (
        "complete"
        if figures and not missing_views and not incomplete_semantics
        else ("partial" if figures else "unsupported")
    )
    return {"status": status, "figures": figures, "tables": tables, "notes": notes}


def _corr_matrix(x):
    x = np.asarray(x, float)
    x = x - x.mean(axis=0, keepdims=True)
    norm = np.sqrt((x * x).sum(axis=0))
    den = np.outer(norm, norm)
    return np.divide(x.T @ x, den, out=np.full(den.shape, np.nan), where=den > 1e-12)


def _edge_support(d, tf, targets):
    if d is None or tf not in d["genes"]:
        return {}
    names = [g for g in targets if g in d["genes"] and g != tf]
    if not names:
        return {}
    ids = [d["genes"].index(g) for g in [tf, *names]]
    x = np.asarray(d["expression"][:, ids], float)
    ages = d["cells"]["time"].to_numpy(float)
    for age in np.unique(ages):
        mask = np.isclose(ages, age)
        x[mask] -= x[mask].mean(axis=0)
    corr = _corr_matrix(x)
    return dict(zip(names, corr[0, 1:]))
