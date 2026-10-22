"""Audited population and developmental-change metrics.

Primary source: evaluation V5 ``engine/vendor/v2/v2lib/metrics.py``.
Formula provenance is retained; undefined covariance/MMD/DE cases are repaired,
and developmental signed overlap now requires the predicted change's sign.
The old rank-only overlap is exported as a clearly named diagnostic.

This module never infers a previous stage from an arbitrary matrix.  DE requires
an explicitly verified, observed prior-stage reference.  Cell-level Mann--
Whitney p/q values describe observed DEG calls, not independent biological
replication and not a test that one model outperforms another.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


EPS = 1e-12
DEFINITION_VERSION = "population-audited-v1"
SOURCE = "evaluation_v5/engine/vendor/v2/v2lib/metrics.py"
POPULATION_NAMES = (
    "pseudobulk_pcc", "pseudobulk_spearman", "pseudobulk_mae", "pseudobulk_rmse",
    "covariance_relative_frobenius", "covariance_edge_pcc", "variance_pcc",
    "variance_rmse", "mmd_u", "sliced_wasserstein", "variogram",
)
DE_NAMES = (
    "de_score", "de_direction", "n_true_up", "n_true_down", "signed_overlap",
    "chance_overlap", "legacy_ranked_signed_overlap", "legacy_ranked_de_score",
)
SKILL_NAMES = ("de_score", "de_direction", "mmd_u", "variogram")
ANCHOR_NAMES = tuple(f"{name}_{suffix}" for name in SKILL_NAMES for suffix in ("baseline", "upper", "skill"))
DEFAULTS = {
    "seed": 42, "max_distribution_cells": 2000, "max_covariance_cells": 5000,
    "max_covariance_genes": 500, "variogram_gene_pairs": 20000,
    "pca_components": 30, "projections": 128, "deg_fdr": .05,
    "deg_min_effect": .25, "deg_max_cells": 10000, "paired_rows": False,
    "max_gene_distribution_cells": 256,
}


def _pcc(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float).reshape(-1), np.asarray(y, float).reshape(-1)
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return float("nan")
    x, y = x - x.mean(), y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip(np.dot(x, y) / den, -1., 1.)) if den > EPS else float("nan")


def _sample_indices(size: int, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or size <= limit:
        return np.arange(size)
    return np.sort(np.random.default_rng(seed).choice(size, limit, replace=False))


def _sample_pair(pred: np.ndarray, truth: np.ndarray, limit: int, seed: int,
                 paired: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Every model gets the same truth indices, fit and pair/projection seed.
    ti = _sample_indices(len(truth), limit, seed)
    pi = ti if paired and len(pred) == len(truth) else _sample_indices(len(pred), limit, seed + 1)
    return pred[pi], truth[ti], {
        "truth_cells": len(ti), "prediction_cells": len(pi), "truth_seed": seed,
        "prediction_seed": seed if paired and len(pred) == len(truth) else seed + 1,
        "paired_indices": bool(paired and len(pred) == len(truth)),
        "truth_sampled": len(ti) < len(truth), "prediction_sampled": len(pi) < len(pred),
    }


def _truth_pca(pred: np.ndarray, truth: np.ndarray, components: int,
               seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(truth) < 2 or not truth.shape[1]:
        raise ValueError("PCA requires at least two truth cells and one gene")
    if np.linalg.norm(truth - truth.mean(0)) <= EPS:
        raise ValueError("truth has no variation; truth-fitted PCA and MMD bandwidth are undefined")
    k = min(int(components), len(truth) - 1, truth.shape[1])
    if k < 1:
        raise ValueError("pca_components must be positive")
    # Reference axes are fitted on GT only; no model-specific joint PCA.
    pca = PCA(n_components=k, svd_solver="randomized", random_state=int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        truth_z = pca.fit_transform(np.asarray(truth, float))
    return pca.transform(np.asarray(pred, float)), truth_z


def _mmd_from_axes(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    bw = y[_sample_indices(len(y), min(1000, len(y)), seed + 17)]
    distances = pairwise_distances(bw, bw, metric="sqeuclidean")
    triangle = distances[np.triu_indices_from(distances, k=1)]
    triangle = triangle[np.isfinite(triangle) & (triangle > EPS)]
    if not len(triangle):
        return float("nan")
    gamma0 = 1. / float(np.median(triangle))
    dxx = pairwise_distances(x, x, metric="sqeuclidean")
    dyy = pairwise_distances(y, y, metric="sqeuclidean")
    dxy = pairwise_distances(x, y, metric="sqeuclidean")
    values = []
    for scale in (.25, .5, 1., 2., 4.):
        kxx, kyy, kxy = np.exp(-scale * gamma0 * dxx), np.exp(-scale * gamma0 * dyy), np.exp(-scale * gamma0 * dxy)
        xx = (kxx.sum() - np.trace(kxx)) / (len(x) * (len(x) - 1))
        yy = (kyy.sum() - np.trace(kyy)) / (len(y) * (len(y) - 1))
        values.append(float(xx + yy - 2 * kxy.mean()))
    # U-statistic may legitimately be negative.  Never clamp to zero.
    return float(np.mean(values))


def _distribution(pred: np.ndarray, truth: np.ndarray, opts: dict[str, Any],
                  seed: int, paired: bool) -> tuple[dict[str, float], dict[str, Any], dict[str, str]]:
    values: dict[str, float] = {}
    audit: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    x, y, sample = _sample_pair(pred, truth, int(opts["max_distribution_cells"]), seed + 11, paired)
    audit["distribution_sample"] = sample
    if len(x) < 2 or len(y) < 2:
        for key in ("mmd_u", "sliced_wasserstein"):
            values[key], reasons[key] = float("nan"), "requires at least two prediction and truth cells"
    else:
        try:
            xz, yz = _truth_pca(x, y, int(opts["pca_components"]), seed)
            audit["pca_components_actual"] = int(yz.shape[1])
            values["mmd_u"] = _mmd_from_axes(xz, yz, seed)
            if not np.isfinite(values["mmd_u"]):
                reasons["mmd_u"] = "no positive truth pairwise distances for Gaussian bandwidth"
            projections = int(opts["projections"])
            if projections < 1:
                raise ValueError("projections must be positive")
            directions = np.random.default_rng(seed + 29).normal(size=(xz.shape[1], projections))
            directions /= np.maximum(np.linalg.norm(directions, axis=0, keepdims=True), EPS)
            px, py = xz @ directions, yz @ directions
            # Exact one-dimensional W1 also supports unequal population sizes;
            # for equal sizes this is the source's sorted absolute difference.
            values["sliced_wasserstein"] = float(np.mean([
                stats.wasserstein_distance(px[:, i], py[:, i]) for i in range(projections)
            ]))
        except ValueError as exc:
            for key in ("mmd_u", "sliced_wasserstein"):
                values[key], reasons[key] = float("nan"), str(exc)

    x, y, sample = _sample_pair(pred, truth, int(opts["max_covariance_cells"]), seed + 31, paired)
    audit["variogram_sample"] = sample
    ng = pred.shape[1]
    npairs = int(opts["variogram_gene_pairs"])
    if ng < 2 or npairs < 1:
        values["variogram"] = float("nan")
        reasons["variogram"] = "requires two genes and at least one sampled gene pair"
    else:
        rng = np.random.default_rng(seed + 35)
        first = rng.integers(0, ng, npairs)
        second = rng.integers(0, ng - 1, npairs)
        second += second >= first
        squared = 0.
        incident_sum = np.zeros(ng, float)
        incident_count = np.zeros(ng, int)
        # Bound intermediates: source created two cells x 20,000 arrays.
        for start in range(0, npairs, 128):
            a, b = first[start:start + 128], second[start:start + 128]
            vx = np.mean(np.sqrt(np.abs(x[:, a] - x[:, b])), axis=0)
            vy = np.mean(np.sqrt(np.abs(y[:, a] - y[:, b])), axis=0)
            pair_error = (vx - vy) ** 2
            squared += float(np.sum(pair_error))
            np.add.at(incident_sum, a, pair_error)
            np.add.at(incident_sum, b, pair_error)
            np.add.at(incident_count, a, 1)
            np.add.at(incident_count, b, 1)
        values["variogram"] = squared / npairs
        audit["variogram_pairs"] = npairs
        audit["gene_variogram_components"] = [
            {"gene_index": i, "gene_variogram_incident_error": float(incident_sum[i] / incident_count[i])
             if incident_count[i] else float("nan"), "incident_pair_count": int(incident_count[i])}
            for i in range(ng)
        ]
    return values, audit, reasons


def _covariance(pred: np.ndarray, truth: np.ndarray, opts: dict[str, Any],
                seed: int, paired: bool) -> tuple[dict[str, float], dict[str, Any], dict[str, str]]:
    names = ("covariance_relative_frobenius", "covariance_edge_pcc", "variance_pcc", "variance_rmse")
    values = dict.fromkeys(names, float("nan"))
    reasons: dict[str, str] = {}
    x, y, sample = _sample_pair(pred, truth, int(opts["max_covariance_cells"]), seed + 1, paired)
    genes = _sample_indices(truth.shape[1], int(opts["max_covariance_genes"]), seed + 3)
    audit = {"covariance_sample": sample, "covariance_gene_indices": genes.tolist(), "covariance_ddof": 1}
    if min(len(x), len(y)) < 2:
        return values, audit, dict.fromkeys(names, "sample covariance requires at least two cells per population")
    cx, cy = np.atleast_2d(np.cov(x[:, genes], rowvar=False)), np.atleast_2d(np.cov(y[:, genes], rowvar=False))
    denom = float(np.linalg.norm(cy, ord="fro"))
    if denom > EPS:
        values["covariance_relative_frobenius"] = float(np.linalg.norm(cx - cy, ord="fro") / denom)
    else:
        reasons["covariance_relative_frobenius"] = "truth covariance norm is zero; relative error is undefined"
    upper = np.triu_indices_from(cx, k=1)
    values["covariance_edge_pcc"] = _pcc(cx[upper], cy[upper])
    values["variance_pcc"] = _pcc(np.diag(cx), np.diag(cy))
    values["variance_rmse"] = float(np.sqrt(np.mean((np.diag(cx) - np.diag(cy)) ** 2)))
    for name in ("covariance_edge_pcc", "variance_pcc"):
        if not np.isfinite(values[name]):
            reasons[name] = "fewer than two eligible entries or constant covariance/variance vector"
    row_squared_error = np.sum((cx - cy) ** 2, axis=1)
    audit["gene_covariance_components"] = []
    for j, gene_index in enumerate(genes):
        others = np.arange(len(genes)) != j
        audit["gene_covariance_components"].append({
            "gene_index": int(gene_index),
            "gene_covariance_row_squared_error": float(row_squared_error[j]),
            "gene_covariance_relative_frobenius_component": float(np.sqrt(row_squared_error[j]) / denom)
            if denom > EPS else float("nan"),
            "gene_covariance_profile_pcc": _pcc(cx[j, others], cy[j, others]),
        })
    return values, audit, reasons


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini--Hochberg correction, finite tests only."""
    p = np.asarray(pvalues, float)
    q = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid):
        order = valid[np.argsort(p[valid], kind="stable")]
        ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
        q[order] = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    return q


def _signed_sets(delta: np.ndarray, nup: int, ndown: int, corrected: bool = True) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(delta, kind="stable")
    up, down = order[::-1], order
    if corrected:
        up, down = up[delta[up] > 0], down[delta[down] < 0]
    return up[:nup], down[:ndown]


def _overlap(up: np.ndarray, down: np.ndarray, true_up: np.ndarray, true_down: np.ndarray) -> float:
    n = len(true_up) + len(true_down)
    if not n:
        return float("nan")
    return float((len(set(up) & set(true_up)) + len(set(down) & set(true_down))) / n)


def _de(pred: np.ndarray, truth: np.ndarray, reference: np.ndarray,
        opts: dict[str, Any], seed: int) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, str]]:
    t = truth[_sample_indices(len(truth), int(opts["deg_max_cells"]), seed + 41)]
    r = reference[_sample_indices(len(reference), int(opts["deg_max_cells"]), seed + 43)]
    effect = t.mean(0) - r.mean(0)
    # Vectorized scipy call is identical to per-column Mann--Whitney calls.
    p = np.asarray(stats.mannwhitneyu(t, r, axis=0, alternative="two-sided", method="asymptotic").pvalue)
    q = bh_adjust(p)
    sig = (q <= float(opts["deg_fdr"])) & (np.abs(effect) >= float(opts["deg_min_effect"]))
    delta_true, delta_pred = truth.mean(0) - reference.mean(0), pred.mean(0) - reference.mean(0)
    up_true, down_true = np.flatnonzero(sig & (delta_true > 0)), np.flatnonzero(sig & (delta_true < 0))
    nup, ndown = len(up_true), len(down_true)
    mean_ref = reference.mean(0)
    values: dict[str, float] = {"n_true_up": nup, "n_true_down": ndown}
    reasons: dict[str, str] = {}
    for corrected, suffix in ((True, ""), (False, "legacy_ranked_")):
        sets = _signed_sets(delta_pred, nup, ndown, corrected)
        overlap = _overlap(*sets, up_true, down_true)
        # Preserve the source's reference-level ranking null (not a theoretical
        # chance probability); corrected signs apply only to model changes.
        plus = _signed_sets(mean_ref, nup, ndown, False)
        minus = _signed_sets(-mean_ref, nup, ndown, False)
        chance = max(_overlap(*plus, up_true, down_true), _overlap(*minus, up_true, down_true))
        score = (overlap - chance) / (1 - chance) if np.isfinite(overlap) and np.isfinite(chance) and (1 - chance) > EPS else float("nan")
        values[f"{suffix}signed_overlap"], values[f"{suffix}de_score"] = overlap, score
        if not suffix:
            values["chance_overlap"] = chance
        if not np.isfinite(overlap):
            reasons[f"{suffix}signed_overlap"] = "no truth DEG passes FDR and minimum effect thresholds"
        if not np.isfinite(score):
            reasons[f"{suffix}de_score"] = "no truth DEG, undefined overlap, or reference-level null overlap equals one"
    rp, rt, rr = stats.rankdata(delta_pred), stats.rankdata(delta_true), stats.rankdata(mean_ref)
    cpt, cpr, ctr = _pcc(rp, rt), _pcc(rp, rr), _pcc(rt, rr)
    den = math.sqrt(max((1 - cpr ** 2) * (1 - ctr ** 2), 0)) if np.isfinite(cpr) and np.isfinite(ctr) else float("nan")
    values["de_direction"] = float(np.clip((cpt - cpr * ctr) / den, -1, 1)) if np.isfinite(cpt) and np.isfinite(den) and den > EPS else float("nan")
    if not np.isfinite(values["de_direction"]):
        reasons["de_direction"] = "constant ranks or singular reference-adjusted partial correlation"
    return values, {"delta_true": delta_true, "delta_pred": delta_pred, "pvalue": p,
                    "qvalue": q, "significant": sig, "deg_sample_effect": effect}, reasons


def _skill(value: float, baseline: float, upper: float) -> float:
    if not all(np.isfinite(v) for v in (value, baseline, upper)):
        return float("nan")
    d = abs(upper - baseline)
    return float(d / (d + abs(upper - value))) if d > EPS else float("nan")


def compute_population(pred: Any, truth: Any, reference: Any = None,
                       options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``values``, per-gene diagnostics, and a machine-readable audit.

    ``reference_verified=True`` and ``reference_kind='observed_prior_stage'``
    are required to enable DE.  Caller must verify native genes/units/dataset,
    prior-stage identity and observed provenance before setting these flags.
    ``paired_rows`` is used for synchronized sampling only; population metrics
    themselves do not require row correspondence.
    """
    opts = {**DEFAULTS, **(options or {})}
    p, t = np.asarray(pred, float), np.asarray(truth, float)
    values = {name: float("nan") for name in (*POPULATION_NAMES, *DE_NAMES, *ANCHOR_NAMES, "virtual_embryo_t1_score")}
    audit: dict[str, Any] = {
        "definition_version": DEFINITION_VERSION, "source": SOURCE,
        "seed": int(opts["seed"]), "invalid_reasons": {},
        "benchmark_excluded": [*DE_NAMES[2:4], "chance_overlap", "legacy_ranked_signed_overlap", "legacy_ranked_de_score", *ANCHOR_NAMES, "virtual_embryo_t1_score"],
        "warnings": [
            "Truth-fitted PCA distribution metrics measure retained GT axes; discarded directions can contain prediction error.",
            "MMD U-statistic can be negative; on matched/paired cells its classical independent-sample unbiasedness claim does not apply.",
            "Bounded sampling and common truth/PCA seeds are shared across models; this differs from legacy model-specific sampling seeds.",
            "DE cell-level p/q values are descriptive DEG diagnostics and do not establish model superiority or biological replicate significance.",
            "chance_overlap is the source's reference-expression ranking null, not a theoretical random chance probability.",
            "Anchored skills favor closeness to an empirical split-GT anchor and can penalize improvement beyond it; they and the composite are diagnostic only.",
            "de_direction_baseline is structurally undefined: reference predicted from itself has zero change and constant ranks; its skill and the no-imputation composite are therefore undefined.",
        ],
        "formula_changes": [
            "Undefined constant-truth MMD and zero-norm relative covariance now return NaN with reason.",
            "Signed DEG overlap requires actual positive/negative predicted changes; rank-only source values are separate diagnostics.",
            "Sliced Wasserstein uses exact 1D W1 for unequal sample sizes without changing the shared truth sample.",
            "Composite is NaN if any constituent skill is undefined; legacy missing-skill zero imputation was removed.",
        ],
    }
    if p.ndim != 2 or t.ndim != 2 or not len(p) or not len(t) or not t.shape[1] or p.shape[1] != t.shape[1]:
        audit["invalid_reasons"] = dict.fromkeys(values, "requires non-empty two-dimensional arrays with identical native gene columns")
        return {"values": values, "genes": [], "audit": audit}
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(t)):
        audit["invalid_reasons"] = dict.fromkeys(values, "non-finite expression; no cell/gene was silently removed")
        return {"values": values, "genes": [], "audit": audit}
    paired = bool(opts["paired_rows"]) and len(p) == len(t)
    seed = int(opts["seed"])
    pm, tm = p.mean(0), t.mean(0)
    mean_delta = pm - tm
    values.update({"pseudobulk_pcc": _pcc(pm, tm), "pseudobulk_spearman": _pcc(stats.rankdata(pm), stats.rankdata(tm)),
                   "pseudobulk_mae": float(np.mean(np.abs(mean_delta))), "pseudobulk_rmse": float(np.sqrt(np.mean(mean_delta ** 2)))})
    for key in ("pseudobulk_pcc", "pseudobulk_spearman"):
        if not np.isfinite(values[key]):
            audit["invalid_reasons"][key] = "fewer than two genes or constant pseudobulk vector"
    cov, ca, cr = _covariance(p, t, opts, seed, paired)
    dist, da, dr = _distribution(p, t, opts, seed, paired)
    values.update(cov); values.update(dist)
    audit.update(ca); audit.update(da)
    audit["invalid_reasons"].update(cr); audit["invalid_reasons"].update(dr)
    names = opts.get("gene_names")
    if names is None:
        names = [str(i) for i in range(t.shape[1])]
    if len(names) != t.shape[1]:
        raise ValueError("gene_names must match native matrix columns exactly")
    pv = np.var(p, axis=0, ddof=1) if len(p) >= 2 else np.full(t.shape[1], np.nan)
    tv = np.var(t, axis=0, ddof=1) if len(t) >= 2 else np.full(t.shape[1], np.nan)
    genes = [{"gene_index": i, "gene": str(name), "truth_mean": float(tm[i]), "prediction_mean": float(pm[i]),
              "pseudobulk_signed_error": float(mean_delta[i]), "pseudobulk_absolute_error": float(abs(mean_delta[i])),
              "pseudobulk_squared_error": float(mean_delta[i] ** 2), "truth_variance_full_rows": float(tv[i]),
              "prediction_variance_full_rows": float(pv[i]), "variance_signed_error_full_rows": float(pv[i] - tv[i])}
             for i, name in enumerate(names)]
    # These are components or separately defined univariate endpoints, never
    # single-gene substitutes for the original multigene global metrics.
    for component_key in ("gene_covariance_components", "gene_variogram_components"):
        for component in audit.get(component_key, []):
            i = component["gene_index"]
            genes[i].update({key: value for key, value in component.items() if key != "gene_index"})
    gene_limit = int(opts["max_gene_distribution_cells"])
    if gene_limit < 0:
        raise ValueError("max_gene_distribution_cells must be nonnegative (0 disables per-gene distributions)")
    if gene_limit:
        gx, gy, sample = _sample_pair(p, t, gene_limit, seed + 51, paired)
        audit["gene_distribution_sample"] = sample
        audit["gene_distribution_max_cells"] = gene_limit
        audit["gene_distribution_definition"] = "Native expression axes; empirical 1D W1 and five-bandwidth Gaussian U-MMD; no PCA. These differ from global multigene metrics."
        audit["gene_distribution_invalid_reasons"] = {}
        for i, row in enumerate(genes):
            row["gene_wasserstein_1d"] = float(stats.wasserstein_distance(gx[:, i], gy[:, i]))
            row["gene_mmd_u_1d"] = _mmd_from_axes(gx[:, [i]], gy[:, [i]], seed + 53 + i)
            if not np.isfinite(row["gene_mmd_u_1d"]):
                audit["gene_distribution_invalid_reasons"][str(names[i])] = "MMD requires at least two cells per population and positive truth pairwise distances; constant-truth MMD is undefined"
    else:
        audit["gene_distribution_status"] = "disabled by max_gene_distribution_cells=0"
    audit["gene_component_definitions"] = {
        "gene_covariance_row_squared_error": "Sum_j (Cpred[g,j]-Ctruth[g,j])^2 over the global covariance sampled gene set, with the identical global sampled cells.",
        "gene_covariance_relative_frobenius_component": "sqrt(row squared error)/||Ctruth||F; sum of squared gene components equals global relative Frobenius error squared.",
        "gene_covariance_profile_pcc": "Correlation of this gene's incident off-diagonal covariance entries over the same sampled genes; a separate diagnostic, not the global edge PCC.",
        "gene_variogram_incident_error": "Mean squared variogram error of sampled pairs incident to this gene. Incidence-count-weighted mean reconstructs global variogram.",
        "incident_pair_count": "Count of sampled pair incidences, including repeated pairs; each global pair contributes to two genes.",
    }
    refmeta = opts.get("reference_metadata", {}) or {}
    refkind = opts.get("reference_kind", refmeta.get("kind"))
    verified = opts.get("reference_verified", refmeta.get("verified", False))
    if reference is None:
        de_reason = "no verified observed prior-stage reference was supplied"
    elif not verified or refkind != "observed_prior_stage":
        de_reason = "reference provenance must be verified and kind must be observed_prior_stage"
    else:
        ref = np.asarray(reference, float)
        if ref.ndim != 2 or ref.shape[1] != t.shape[1] or len(ref) < 2 or len(t) < 4:
            de_reason = "DE requires matching native gene columns, at least two reference cells and four truth cells"
        elif not np.all(np.isfinite(ref)):
            de_reason = "reference contains non-finite expression"
        else:
            de_reason = ""
    if de_reason:
        audit["invalid_reasons"].update(dict.fromkeys((*DE_NAMES, *ANCHOR_NAMES, "virtual_embryo_t1_score"), de_reason))
        return {"values": values, "genes": genes, "audit": audit}

    de, arrays, reasons = _de(p, t, ref, opts, seed)
    values.update(de); audit["invalid_reasons"].update(reasons)
    audit["reference_metadata"] = refmeta
    audit["deg_fdr"], audit["deg_min_effect"], audit["deg_max_cells"] = opts["deg_fdr"], opts["deg_min_effect"], opts["deg_max_cells"]
    for i, row in enumerate(genes):
        row.update({key: bool(value[i]) if key == "significant" else float(value[i]) for key, value in arrays.items()})
    baseline_de, _, _ = _de(ref, t, ref, opts, seed + 103)
    baseline_dist, _, _ = _distribution(ref, t, opts, seed + 101, False)
    order = np.random.default_rng(seed + 107).permutation(len(t))
    middle = len(t) // 2
    upper_pred, upper_truth = t[order[:middle]], t[order[middle:]]
    upper_de, _, _ = _de(upper_pred, upper_truth, ref, opts, seed + 113)
    upper_dist, _, _ = _distribution(upper_pred, upper_truth, opts, seed + 109, False)
    for name in SKILL_NAMES:
        baseline = (baseline_de if name.startswith("de_") else baseline_dist)[name]
        upper = (upper_de if name.startswith("de_") else upper_dist)[name]
        values[f"{name}_baseline"], values[f"{name}_upper"] = baseline, upper
        values[f"{name}_skill"] = _skill(values[name], baseline, upper)
        if not np.isfinite(values[f"{name}_skill"]):
            audit["invalid_reasons"][f"{name}_skill"] = "undefined metric/anchor or identical baseline and split-GT anchors"
    weights = {"de_score": .25, "de_direction": .25, "mmd_u": .30, "variogram": .20}
    if all(np.isfinite(values[f"{name}_skill"]) for name in weights):
        values["virtual_embryo_t1_score"] = 100 * sum(weight * values[f"{name}_skill"] for name, weight in weights.items())
    else:
        audit["invalid_reasons"]["virtual_embryo_t1_score"] = "one or more constituent skills are undefined; no missing-skill zero imputation"
    return {"values": values, "genes": genes, "audit": audit}
