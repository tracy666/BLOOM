"""Transparent, model-independent benchmark scaling and aspect averaging.

Raw metrics remain the scientifically interpretable endpoints.  ``overall`` is
an exploratory equal-aspect summary, not a newly validated biological metric.
Display bounds control marker area only: changing them cannot change the main
average or ranking.  Every compared model uses the same bounds and weights.
"""

from __future__ import annotations

import math
import json
from collections import defaultdict
from typing import Any

OVERALL = "overall"
_CONTRACT_FIELDS = (
    "route",
    "cohort_id",
    "definition_id",
    "aggregation",
    "weighting",
    "normalization",
    "split",
    "pcc_nan_policy",
    "pcc_eps",
    "min_valid_fraction",
)
_DESCRIPTIVE = {"descriptive_different_targets", "legacy_unverified", "descriptive"}


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def _raw(row: dict) -> float:
    for name in ("value", "rawvalue", "raw_value"):
        if name in row and row[name] not in (None, ""):
            return _number(row[name])
    return math.nan


def _issue(
    issues: list, severity: str, setting: tuple, metric: str, reason: str, detail: str
) -> None:
    issues.append(
        {
            "severity": severity,
            "dataset": setting[0],
            "panel": setting[1],
            "metric": metric,
            "reason": reason,
            "detail": detail,
        }
    )


def _rank_details(
    values: dict[str, float], higher: bool = True, tolerance: float = 1e-12
) -> dict[str, dict]:
    """Average ranks for ties; rank 1 is best; all equal => rank score 0.5.

    Ties are anchored to the first value in a sorted block, avoiding chained
    approximate ties.  The same tie policy is used by all score/rank outputs.
    """
    usable = [(m, v) for m, v in values.items() if math.isfinite(v)]
    usable.sort(key=lambda item: ((-item[1] if higher else item[1]), item[0]))
    result = {}
    n = len(usable)
    start = 0
    while start < n:
        stop = start + 1
        anchor = usable[start][1]
        while stop < n and math.isclose(
            usable[stop][1], anchor, rel_tol=tolerance, abs_tol=tolerance
        ):
            stop += 1
        # Zero-based start and exclusive stop -> one-based mean rank.
        mean_rank = (start + 1 + stop) / 2.0
        rank_score = (n - mean_rank) / (n - 1) if n > 1 else 0.5
        for model, _ in usable[start:stop]:
            result[model] = {
                "rank": mean_rank,
                "rank_min": start + 1,
                "rank_max": stop,
                "rank_score": rank_score,
                "tied_models": stop - start,
            }
        start = stop
    return result


def _scale(value: float, low: float, high: float, direction: str) -> float:
    if not all(math.isfinite(v) for v in (value, low, high)):
        return math.nan
    if high == low:
        return 0.5
    result = (value - low) / (high - low)
    if direction == "lower":
        result = 1.0 - result
    return min(1.0, max(0.0, result))


def _nice_bounds(low: float, high: float) -> tuple[float, float]:
    """Expanded, round bounds derived from all models, never from one model.

    A range 0.88..0.93 becomes 0.85..0.95.  Constant vectors get a finite
    symmetric display range while their benchmark score remains exactly 0.5.
    """
    span = high - low
    if span <= 0:
        span = max(abs(low) * 0.1, 0.1)
        low, high = low - span / 2, high + span / 2
    expanded_low, expanded_high = low - 0.1 * span, high + 0.1 * span
    approximate_step = span / 2
    exponent = 10 ** math.floor(math.log10(approximate_step))
    fraction = approximate_step / exponent
    multiplier = next((x for x in (1, 2, 2.5, 5, 10) if x >= fraction), 10)
    step = multiplier * exponent
    return (math.floor(expanded_low / step) * step, math.ceil(expanded_high / step) * step)


def _range_value(ranges: dict, metric: str, dataset: str, panel: str) -> Any:
    """Accept default and dataset/panel overrides in several documented forms.

    Preferred: {metric: {"default": [lo,hi], "big/author": [lo,hi]}}.
    Also supported: {"big/author": {metric: [lo,hi]}} and nested
    {"big": {"author": {metric: [lo,hi]}}}.
    """
    setting_key = f"{dataset}/{panel}"
    setting = ranges.get(setting_key)
    if isinstance(setting, dict) and metric in setting:
        return setting[metric]
    nested = ranges.get(dataset)
    if isinstance(nested, dict):
        cohort = nested.get(panel)
        if isinstance(cohort, dict) and metric in cohort:
            return cohort[metric]
    spec = ranges.get(metric)
    if isinstance(spec, dict):
        if setting_key in spec:
            return spec[setting_key]
        nested = spec.get(dataset)
        if isinstance(nested, dict) and panel in nested:
            return nested[panel]
        settings = spec.get("settings", {})
        if isinstance(settings, dict) and setting_key in settings:
            return settings[setting_key]
        return spec.get("default", spec if "min" in spec else None)
    return spec


def _bounds(spec: Any) -> tuple[float, float] | None:
    if spec is None:
        return None
    if isinstance(spec, dict):
        raw = (spec.get("min", spec.get("low")), spec.get("max", spec.get("high")))
    elif isinstance(spec, (list, tuple)) and len(spec) == 2:
        raw = spec
    else:
        raise ValueError("Display range must be [low, high] or {min, max}.")
    low, high = map(_number, raw)
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("Display range requires finite low < high.")
    return low, high


def _validated_plan(plan: dict) -> tuple[list[dict], dict[str, float], list[dict]]:
    specs = []
    seen = set()
    for entry in plan.get("metrics", []):
        spec = dict(entry)
        name = spec.get("name", spec.get("metric"))
        if not name or name in seen or name == OVERALL:
            raise ValueError("Metric names must be unique, nonempty and cannot be 'overall'.")
        seen.add(name)
        direction = spec.get("direction", "higher")
        if direction not in {"higher", "lower"}:
            raise ValueError(f"Unknown direction for {name}: {direction}")
        weight = _number(spec.get("weight", 1))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Metric weight must be positive and finite: {name}")
        specs.append(
            {
                **spec,
                "name": name,
                "label": spec.get("label", name),
                "aspect": spec.get("aspect", name),
                "direction": direction,
                "weight": weight,
            }
        )
    if not specs:
        raise ValueError("At least one declared metric is required.")
    aspects = sorted({s["aspect"] for s in specs})
    declared = plan.get("aspect_weights", {})
    if not isinstance(declared, dict):
        raise ValueError("aspect_weights must be a mapping.")
    unknown = set(declared) - set(aspects)
    if unknown:
        raise ValueError(f"Aspect weights refer to undeclared aspects: {sorted(unknown)}")
    weights = {aspect: _number(declared.get(aspect, 1)) for aspect in aspects}
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Aspect weights must be positive and finite.")
    total = sum(weights.values())
    weights = {aspect: weight / total for aspect, weight in weights.items()}
    per_aspect = {
        aspect: sum(s["weight"] for s in specs if s["aspect"] == aspect) for aspect in aspects
    }
    metadata = [
        {
            "metric": s["name"],
            "label": s["label"],
            "aspect": s["aspect"],
            "direction": s["direction"],
            "declared_metric_weight": s["weight"],
            "metric_weight_within_aspect": s["weight"] / per_aspect[s["aspect"]],
            "aspect_weight_normalized": weights[s["aspect"]],
            "overall_effective_weight": weights[s["aspect"]]
            * s["weight"]
            / per_aspect[s["aspect"]],
        }
        for s in specs
    ]
    return specs, weights, metadata


def _average(
    model: str, specs: list[dict], values: dict, aspect_weights: dict, exclude: str | None = None
) -> float:
    aspect_values = defaultdict(list)
    for spec in specs:
        if spec["aspect"] == exclude:
            continue
        value = values.get((model, spec["name"]), math.nan)
        if not math.isfinite(value):
            return math.nan  # Never drop a missing endpoint from a model average.
        aspect_values[spec["aspect"]].append((value, spec["weight"]))
    if not aspect_values:
        return math.nan
    total_weight = sum(aspect_weights[a] for a in aspect_values)
    return (
        sum(
            (
                sum(value * weight for value, weight in entries)
                / sum(weight for _, weight in entries)
            )
            * aspect_weights[aspect]
            for aspect, entries in aspect_values.items()
        )
        / total_weight
    )


def score_rows(rows: list[dict], plan: dict) -> dict:
    """Score a fixed metric plan without optimizing weights or endpoints.

    ``expected_models`` is either an integer (default 23), or the exact model
    names.  Every metric requires that complete set; missing or conflicting
    endpoints block that setting's overall average for every model.  Setting
    boundaries are exactly (dataset, native panel), never pooled across assays.
    Descriptive comparisons must be explicitly authorized in the plan or by
    upstream ``descriptive_comparison_authorized=True`` on every such row.
    """
    specs, aspect_weights, weight_metadata = _validated_plan(plan)
    summary_label = (
        "Equal-aspect average"
        if len(set(aspect_weights.values())) == 1
        else "Aspect-weighted average"
    )
    if not isinstance(plan.get("display_ranges", {}), dict):
        raise ValueError("display_ranges must be a mapping.")
    names = {s["name"] for s in specs}
    expected = plan.get("expected_models", 23)
    if isinstance(expected, int):
        if expected < 1:
            raise ValueError("expected_models must be positive.")
        declared_models = None
    elif isinstance(expected, (list, tuple)) and expected and len(set(expected)) == len(expected):
        declared_models = set(map(str, expected))
    else:
        raise ValueError("expected_models must be a positive count or unique model-name list.")
    tolerance = _number(plan.get("tie_tolerance", 1e-12))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tie_tolerance must be finite and nonnegative.")
    groups = defaultdict(list)
    for row in rows:
        if row.get("metric") in names:
            groups[(str(row.get("dataset", "")), str(row.get("panel", "")))].append(dict(row))
    frame, rankings, normalization, issues, sensitivity = [], [], [], [], []
    meta_by_name = {entry["metric"]: entry for entry in weight_metadata}
    if not groups:
        _issue(
            issues, "error", ("", ""), "", "no_selected_metrics", "No rows match the declared plan."
        )
    for setting, setting_rows in sorted(groups.items()):
        dataset, panel = setting
        observed_models = {str(r.get("model", "")) for r in setting_rows}
        models = sorted(declared_models or observed_models)
        model_count_ok = (
            observed_models == declared_models
            if declared_models is not None
            else len(observed_models) == expected
        )
        if not model_count_ok:
            _issue(
                issues,
                "error",
                setting,
                "",
                "incomplete_model_set",
                f"Expected {sorted(declared_models) if declared_models else expected}; "
                f"found {sorted(observed_models)}. Overall scores are blocked.",
            )
        index = defaultdict(list)
        for row in setting_rows:
            index[(str(row.get("model", "")), row["metric"])].append(row)
        scores, rank_scores, display_scores, labels = {}, {}, {}, {}
        descriptive_setting = False
        for spec in specs:
            metric = spec["name"]
            chosen, metric_ok = {}, model_count_ok
            for model in models:
                candidates = index.get((model, metric), [])
                if len(candidates) > 1:
                    comparable = [
                        (
                            _raw(r),
                            tuple(str(r.get(f, "")) for f in _CONTRACT_FIELDS),
                            r.get("comparison_status", ""),
                        )
                        for r in candidates
                    ]
                    if all(item == comparable[0] for item in comparable[1:]):
                        _issue(
                            issues, "info", setting, metric, "identical_duplicate_collapsed", model
                        )
                        candidates = candidates[:1]
                    else:
                        _issue(issues, "error", setting, metric, "conflicting_duplicate", model)
                        metric_ok = False
                        candidates = []
                if not candidates or not math.isfinite(_raw(candidates[0])):
                    _issue(issues, "error", setting, metric, "missing_finite_endpoint", model)
                    metric_ok = False
                    continue
                row = candidates[0]
                if str(row.get("status", "available")) not in {
                    "available",
                    "ok",
                    "computed",
                    "complete",
                    "",
                }:
                    _issue(issues, "error", setting, metric, "unavailable_endpoint", model)
                    metric_ok = False
                chosen[model] = row
                labels[model] = row.get("model_label", model)
                if row.get("comparison_status") not in (_DESCRIPTIVE | {"verified_matched"}):
                    _issue(
                        issues,
                        "error",
                        setting,
                        metric,
                        "unknown_comparison_status",
                        f"{model}: {row.get('comparison_status', '')}",
                    )
                    metric_ok = False
            descriptive = any(r.get("comparison_status") in _DESCRIPTIVE for r in chosen.values())
            descriptive_setting |= descriptive
            unauthorized = [
                model
                for model, row in chosen.items()
                if row.get("comparison_status") in _DESCRIPTIVE
                and not (
                    _bool(
                        plan.get(
                            "allow_descriptive_different_targets",
                            plan.get("allow_descriptive", False),
                        )
                    )
                    or _bool(row.get("descriptive_comparison_authorized", False))
                )
            ]
            if unauthorized:
                _issue(
                    issues,
                    "error",
                    setting,
                    metric,
                    "descriptive_comparison_not_authorized",
                    ", ".join(unauthorized),
                )
                metric_ok = False
            elif descriptive:
                _issue(
                    issues,
                    "warning",
                    setting,
                    metric,
                    "descriptive_different_targets",
                    "Native evaluation protocols may compare FM all-target coverage with TRANS "
                    "trajectory-supported coverage. Numeric ordering describes those native protocols; "
                    "it does not establish matched-population or statistical superiority.",
                )
            # Native-protocol authorization permits differing target populations
            # only.  It never licenses comparing distinct definitions, scales,
            # splits, aggregation/weighting rules, routes, or PCC policies.
            contract_fields = tuple(
                f for f in _CONTRACT_FIELDS if not (descriptive and f == "cohort_id")
            )
            contracts = {
                tuple(str(row.get(f, "")) for f in contract_fields) for row in chosen.values()
            }
            if len(contracts) > 1:
                differing = [
                    f
                    for f in contract_fields
                    if len({str(row.get(f, "")) for row in chosen.values()}) > 1
                ]
                _issue(
                    issues,
                    "error",
                    setting,
                    metric,
                    "incompatible_comparison_contract",
                    "Models differ in protected comparison fields: "
                    + ", ".join(differing)
                    + ". Native-protocol authorization permits cohort differences only.",
                )
                metric_ok = False
            values = {model: _raw(row) for model, row in chosen.items()}
            low = min(values.values()) if values else math.nan
            high = max(values.values()) if values else math.nan
            bounds_spec = _range_value(plan.get("display_ranges", {}), metric, dataset, panel)
            declared_bounds = _bounds(bounds_spec)
            display_low, display_high = (
                declared_bounds or _nice_bounds(low, high) if values else (math.nan, math.nan)
            )
            clipping = sum(v < display_low or v > display_high for v in values.values())
            raw_ranks = _rank_details(values, spec["direction"] == "higher", tolerance)
            norm = {
                "dataset": dataset,
                "panel": panel,
                "metric": metric,
                **meta_by_name[metric],
                "score_low": low,
                "score_high": high,
                "score_normalization": "within_setting_all_models_empirical_minmax",
                "display_low": display_low,
                "display_high": display_high,
                "display_range_source": (
                    "declared" if declared_bounds else "expanded_nice_all_models"
                ),
                "display_clipped_count": clipping,
                "n_models": len(values),
                "expected_models": len(models) if declared_models else expected,
                "coverage_complete": bool(metric_ok),
                "constant_metric": low == high,
                "tie_policy": "average_rank",
                "tie_tolerance": tolerance,
                "source_population_counts_json": json.dumps(
                    {
                        model: {
                            "n_cells": row.get("n_cells", ""),
                            "n_genes": row.get("n_genes", ""),
                            "n_days": row.get("n_days", ""),
                            "target_days": row.get("target_days", ""),
                            "cohort_id": row.get("cohort_id", ""),
                            "weighting": row.get("weighting", ""),
                            "route": row.get("route", ""),
                        }
                        for model, row in chosen.items()
                    },
                    sort_keys=True,
                ),
                "comparison_status": (
                    "descriptive_different_targets" if descriptive else "verified_matched"
                ),
            }
            normalization.append(norm)
            if clipping:
                _issue(
                    issues,
                    "warning",
                    setting,
                    metric,
                    "display_values_clipped",
                    f"{clipping} marker values outside [{display_low}, {display_high}]. "
                    "Raw values are retained; main average is unaffected.",
                )
            for model in models:
                value = values.get(model, math.nan)
                score = _scale(value, low, high, spec["direction"]) if metric_ok else math.nan
                rank_score = (
                    raw_ranks.get(model, {}).get("rank_score", math.nan) if metric_ok else math.nan
                )
                display_score = _scale(value, display_low, display_high, spec["direction"])
                if metric_ok and low == high:
                    display_score = 0.5
                scores[(model, metric)] = score
                rank_scores[(model, metric)] = rank_score
                display_scores[(model, metric)] = display_score if metric_ok else math.nan
                frame.append(
                    {
                        **chosen.get(model, {}),
                        "dataset": dataset,
                        "panel": panel,
                        "model": model,
                        "model_label": labels.get(model, model),
                        "metric": metric,
                        "metric_label": spec["label"],
                        "label": spec["label"],
                        "aspect": spec["aspect"],
                        "direction": spec["direction"],
                        "value": value,
                        "raw_value": value,
                        "score": score,
                        "display_score": display_score,
                        **raw_ranks.get(
                            model,
                            {
                                "rank": math.nan,
                                "rank_min": math.nan,
                                "rank_max": math.nan,
                                "tied_models": 0,
                            },
                        ),
                        "rank_score": rank_score,
                        "display_low": display_low,
                        "display_high": display_high,
                        "display_clipped": bool(
                            math.isfinite(value) and (value < display_low or value > display_high)
                        ),
                        "coverage_complete": bool(metric_ok),
                        "comparison_status": norm["comparison_status"],
                        "metric_weight_within_aspect": meta_by_name[metric][
                            "metric_weight_within_aspect"
                        ],
                        "aspect_weight_normalized": aspect_weights[spec["aspect"]],
                        "overall_effective_weight": meta_by_name[metric][
                            "overall_effective_weight"
                        ],
                    }
                )
        methods = {
            "empirical_minmax": scores,
            "rank_normalized": rank_scores,
            "display_range_diagnostic": display_scores,
        }
        method_averages = {
            method: {model: _average(model, specs, values, aspect_weights) for model in models}
            for method, values in methods.items()
        }
        method_ranks = {
            method: _rank_details(values, True, tolerance)
            for method, values in method_averages.items()
        }
        for model in models:
            average = method_averages["empirical_minmax"][model]
            rank_average = method_averages["rank_normalized"][model]
            details = method_ranks["empirical_minmax"].get(model, {})
            status = "descriptive_different_targets" if descriptive_setting else "verified_matched"
            rank_row = {
                "dataset": dataset,
                "panel": panel,
                "model": model,
                "model_label": labels.get(model, model),
                "average_score": average,
                "score": average,
                "rank_average_score": rank_average,
                "rank_normalized_rank": method_ranks["rank_normalized"]
                .get(model, {})
                .get("rank", math.nan),
                **details,
                "complete": math.isfinite(average),
                "comparison_status": status,
                "n_metrics": len(specs),
                "n_aspects": len(aspect_weights),
                "summary_label": summary_label,
                "tie_policy": "average_rank",
            }
            rankings.append(rank_row)
            frame.append(
                {
                    "dataset": dataset,
                    "panel": panel,
                    "model": model,
                    "model_label": labels.get(model, model),
                    "metric": OVERALL,
                    "metric_label": summary_label,
                    "label": summary_label,
                    "aspect": "summary",
                    "direction": "higher",
                    "value": average,
                    "raw_value": average,
                    "score": average,
                    "display_score": average,
                    "rank_score": rank_average,
                    "display_low": 0.0,
                    "display_high": 1.0,
                    "display_clipped": False,
                    "coverage_complete": math.isfinite(average),
                    "comparison_status": status,
                    **details,
                    "definition_id": "exploratory_equal_aspect_minmax_v1",
                    "aggregation": "fixed_weight_within_aspect_then_across_aspects",
                }
            )
            for method in methods:
                sensitivity.append(
                    {
                        "dataset": dataset,
                        "panel": panel,
                        "model": model,
                        "model_label": labels.get(model, model),
                        "method": method,
                        "excluded_aspect": "",
                        "score": method_averages[method][model],
                        **method_ranks[method].get(model, {}),
                        "main_score": method == "empirical_minmax",
                        "comparison_status": status,
                    }
                )
        for aspect in sorted(aspect_weights):
            if len(aspect_weights) == 1:
                continue
            values = {
                model: _average(model, specs, scores, aspect_weights, exclude=aspect)
                for model in models
            }
            ranks = _rank_details(values, True, tolerance)
            for model in models:
                sensitivity.append(
                    {
                        "dataset": dataset,
                        "panel": panel,
                        "model": model,
                        "model_label": labels.get(model, model),
                        "method": "leave_one_aspect_out",
                        "excluded_aspect": aspect,
                        "score": values[model],
                        **ranks.get(model, {}),
                        "main_score": False,
                        "comparison_status": (
                            "descriptive_different_targets"
                            if descriptive_setting
                            else "verified_matched"
                        ),
                    }
                )
    return {
        "frame": frame,
        "rankings": rankings,
        "normalization": normalization,
        "issues": issues,
        "sensitivity": sensitivity,
        "weights": weight_metadata,
        "metadata": {
            "summary_label": summary_label,
            "overall_metric": OVERALL,
            "main_normalization": "within_setting_all_models_empirical_minmax",
            "display_ranges_affect_main_score": False,
            "tie_policy": "average_rank",
            "tie_tolerance": tolerance,
            "metric_selection_optimized": False,
            "weights_optimized": False,
            "missing_endpoint_policy": "block_overall_no_omission",
            "population_coverage_policy": "retain_declared_native_evaluation_protocols",
            "aspect_weights": aspect_weights,
            "weights": weight_metadata,
        },
    }
