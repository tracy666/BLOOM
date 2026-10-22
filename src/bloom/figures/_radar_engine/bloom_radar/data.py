"""Prepare truthful radar coordinates and a global, auditable metric screen."""

from collections import defaultdict
import math
from bloom.figures._radar_engine.bloom_benchmark.data import DATASETS, PANELS, load_globals
from bloom.figures._radar_engine.bloom_benchmark.scoring import (
    score_rows,
    _bounds,
    _range_value,
    _scale,
)


def prepare(root, config, dataset="all", panel="all"):
    datasets = DATASETS if dataset == "all" else (dataset,)
    panels = PANELS if panel == "all" else (panel,)
    core_plan = {
        **config["core_plan"],
        "display_ranges": config.get("display_ranges", {}),
        "allow_descriptive_different_targets": True,
    }
    broad_plan = {
        "metrics": config["candidate_metrics"],
        "expected_models": 23,
        "display_ranges": config.get("display_ranges", {}),
        "allow_descriptive_different_targets": True,
    }
    core_data = load_globals(root, core_plan, dataset=dataset, panel=panel, mode="descriptive")
    broad_data = load_globals(root, broad_plan, dataset=dataset, panel=panel, mode="descriptive")
    core = score_rows(core_data["rows"], core_plan)
    broad = score_rows(broad_data["rows"], broad_plan)
    for ds in datasets:
        for assay in panels:
            for spec in core_plan["metrics"]:
                rows = [
                    r
                    for r in core["frame"]
                    if (r["dataset"], r["panel"], r["metric"]) == (ds, assay, spec["name"])
                ]
                if len(rows) != 23 or not all(
                    r.get("coverage_complete") and math.isfinite(r["value"]) for r in rows
                ):
                    raise ValueError(
                        f'Core metric {spec["name"]} lacks one verified, compatible value for all 23 models in {ds}/{assay}. Inspect source/route definitions.'
                    )
    scope = [(ds, assay) for ds in datasets for assay in panels]
    diagnostics, selected = select_extended(
        broad["frame"], config.get("extended_selection", {}), settings=scope
    )
    # Extended averages are deliberately discarded: the displayed average always
    # retains the original four endpoints and three fixed evaluation aspects.
    full = [r for r in broad["frame"] if r["metric"] != "overall"]
    present = {(r["dataset"], r["panel"], r["model"], r["metric"]) for r in full}
    full.extend(
        r
        for r in core["frame"]
        if r["metric"] != "overall"
        and (r["dataset"], r["panel"], r["model"], r["metric"]) not in present
    )
    full.extend(r for r in core["frame"] if r["metric"] == "overall")
    spans = defaultdict(set)
    for row in full:
        spans[row["dataset"], row["panel"], row["metric"]].add(row["value"])
    core_weights = {r["metric"]: r["overall_effective_weight"] for r in core["weights"]}
    for row in full:
        for key in (
            "metric_weight_within_aspect",
            "aspect_weight_normalized",
            "overall_effective_weight",
        ):
            row.pop(key, None)
        row["core_average_effective_weight"] = core_weights.get(row["metric"], 0.0)
        if row["metric"] == "overall":
            row["metric_label"] = row["label"] = "Equal-aspect average"
            custom = _bounds(
                _range_value(
                    config.get("display_ranges", {}), "overall", row["dataset"], row["panel"]
                )
            )
            if custom is not None:
                row["display_low"], row["display_high"] = custom
                row["display_score"] = _scale(row["value"], *custom, "higher")
                row["display_clipped"] = not custom[0] <= row["value"] <= custom[1]
        elif (
            config.get("respect_metric_domain", True)
            and _range_value(
                config.get("display_ranges", {}), row["metric"], row["dataset"], row["panel"]
            )
            is None
        ):
            low, high = row["display_low"], row["display_high"]
            metric = row["metric"]
            if any(term in metric for term in ("pcc", "cosine", "spearman")):
                low, high = max(-1.0, low), min(1.0, high)
            elif metric == "r2":
                high = min(1.0, high)
            elif metric.startswith("detection_fraction_"):
                low, high = max(0.0, low), min(1.0, high)
            elif row["direction"] == "lower" and metric != "mmd_u":
                low = max(0.0, low)
            if low < high:
                row["display_low"], row["display_high"] = low, high
                constant = len(spans[row["dataset"], row["panel"], metric]) == 1
                row["display_score"] = (
                    0.5 if constant else _scale(row["value"], low, high, row["direction"])
                )
                row["display_clipped"] = not low <= row["value"] <= high
    # Export normalization tables with the final radar display bounds, while
    # leaving empirical score bounds and the fixed core averages unchanged.
    lookup = {(r["dataset"], r["panel"], r["metric"]): r for r in full}
    for result in (core, broad):
        for norm in result["normalization"]:
            row = lookup.get((norm["dataset"], norm["panel"], norm["metric"]))
            if row:
                changed = (
                    norm["display_low"] != row["display_low"]
                    or norm["display_high"] != row["display_high"]
                )
                norm["display_low"], norm["display_high"] = row["display_low"], row["display_high"]
                if changed:
                    norm["display_range_source"] = (
                        "expanded_nice_all_models_bounded_to_metric_domain"
                    )
    return {
        "frame": full,
        "core": core,
        "broad": broad,
        "core_data": core_data,
        "broad_data": broad_data,
        "selection": diagnostics,
        "extended_metrics": selected,
        "selection_scope": [{"dataset": ds, "panel": assay} for ds, assay in scope],
    }


def select_extended(frame, rule, settings=None):
    """Apply one exclusion set to all three datasets and both native panels.

    Severe means bottom-five rank AND poor empirical-range quality in the same
    setting. This post-hoc screen affects Version 5 only, never Versions 1–4.
    """
    rank_threshold = float(rule.get("severe_rank_at_least", 19))
    quality_threshold = float(rule.get("severe_quality_at_most", 0.25))
    if not 1 <= rank_threshold <= 23 or not 0 <= quality_threshold <= 1:
        raise ValueError("Extended selection requires rank in [1,23] and quality in [0,1].")
    apply_screen = bool(rule.get("exclude_severe", True))
    forced = set(rule.get("exclude_metrics", []))
    grouped = defaultdict(list)
    for row in frame:
        if row["metric"] != "overall":
            grouped[row["metric"]].append(row)
    selected, diagnostics = [], []
    settings = settings or [(ds, panel) for ds in DATASETS for panel in PANELS]
    for metric, rows in grouped.items():
        reasons, setting_rows = [], []
        constant_everywhere = True
        for ds, panel in settings:
            cohort = [r for r in rows if r["dataset"] == ds and r["panel"] == panel]
            bloom = [r for r in cohort if r["model_label"] == "BLOOM"]
            complete = (
                len(cohort) == 23 and len({r["model"] for r in cohort}) == 23 and len(bloom) == 1
            )
            complete = complete and all(
                r.get("coverage_complete") and math.isfinite(r["value"]) for r in cohort
            )
            if not complete:
                reasons.append(f"incomplete_or_incompatible:{ds}/{panel}")
                setting_rows.append({"dataset": ds, "panel": panel, "complete": False})
                continue
            b = bloom[0]
            values = [r["value"] for r in cohort]
            constant = max(values) == min(values)
            constant_everywhere &= constant
            severe = b["rank"] >= rank_threshold and b["score"] <= quality_threshold
            if apply_screen and severe:
                reasons.append(f'severe:{ds}/{panel}:rank={b["rank"]:g},quality={b["score"]:.6g}')
            setting_rows.append(
                {
                    "dataset": ds,
                    "panel": panel,
                    "complete": True,
                    "bloom_value": b["value"],
                    "bloom_rank": b["rank"],
                    "bloom_empirical_quality": b["score"],
                    "severe": severe,
                    "constant_all_models": constant,
                }
            )
        if constant_everywhere:
            reasons.append("constant_in_every_setting")
        if metric in forced:
            reasons.append("user_excluded_globally")
        keep = not reasons
        if keep:
            selected.append(metric)
        for row in setting_rows:
            diagnostics.append(
                {
                    "metric": metric,
                    **row,
                    "included_version5": keep,
                    "global_exclusion_reason": "; ".join(reasons),
                    "severe_rank_at_least": rank_threshold,
                    "severe_quality_at_most": quality_threshold,
                    "screen_enabled": apply_screen,
                }
            )
    return diagnostics, selected


def _axis(frame, ds, panel, metric, labels, scale):
    rows = [r for r in frame if (r["dataset"], r["panel"], r["metric"]) == (ds, panel, metric)]
    if len(rows) != 23 or not all(math.isfinite(r["display_score"]) for r in rows):
        raise ValueError(f"Incomplete radar axis: {ds}/{panel}/{metric}")
    first = rows[0]
    return {
        "dataset": ds,
        "metric": metric,
        "label": labels.get(metric, first["metric_label"]),
        "direction": first["direction"],
        "display_low": first["display_low"],
        "display_high": first["display_high"],
        "scale": scale,
        "values": {
            r["model_label"]: (23 - r["rank"]) / 22 if scale == "rank" else r["display_score"]
            for r in rows
        },
        "raw_values": {r["model_label"]: r["value"] for r in rows},
        "rank_values": {r["model_label"]: r["rank"] for r in rows},
        "clipped_models": [r["model_label"] for r in rows if r.get("display_clipped")],
    }


def make_jobs(
    prepared,
    config,
    versions=(1, 2, 3, 4, 5),
    panels=PANELS,
    datasets=DATASETS,
    scale="range",
    include_singles=True,
):
    if scale not in {"range", "rank"}:
        raise ValueError("Radar scale must be range or rank.")
    core_metrics = [spec["name"] for spec in config["core_plan"]["metrics"]]
    labels = {spec["name"]: spec.get("label", spec["name"]) for spec in config["candidate_metrics"]}
    labels.update(
        {spec["name"]: spec.get("label", spec["name"]) for spec in config["core_plan"]["metrics"]}
    )
    labels["overall"] = "Equal-aspect average"
    jobs = []
    for panel in panels:
        native = [r for r in prepared["frame"] if r["panel"] == panel]
        models = list(dict.fromkeys(r["model_label"] for r in native))
        for version in versions:
            metric_names = (
                core_metrics
                if version in (1, 3)
                else (
                    core_metrics + ["overall"]
                    if version in (2, 4)
                    else list(prepared["extended_metrics"])
                )
            )
            if version == 5 and config.get("extended_selection", {}).get(
                "include_core_average", True
            ):
                metric_names.append("overall")
            if len(metric_names) < 3:
                raise ValueError(
                    "Version 5 has fewer than three eligible axes; inspect METRIC_SELECTION.csv."
                )
            axes = [
                _axis(native, ds, panel, metric, labels, scale)
                for ds in datasets
                for metric in metric_names
            ]
            variants = [("combined" if version <= 2 else "triptych", tuple(datasets))]
            if version >= 3 and include_singles and len(datasets) > 1:
                variants.extend(("single", (ds,)) for ds in datasets)
            for layout, group in variants:
                sub_axes = [axis for axis in axes if axis["dataset"] in group]
                # This controls draw order and legend, not axis-specific ordering.
                means = {
                    model: sum(
                        r["value"]
                        for r in native
                        if r["model_label"] == model
                        and r["metric"] == "overall"
                        and r["dataset"] in group
                    )
                    / len(group)
                    for model in models
                }
                order = sorted((m for m in models if m != "BLOOM"), key=lambda m: (means[m], m)) + [
                    "BLOOM"
                ]
                id_ = f"{panel}/version{version}/" + (
                    "combined"
                    if layout == "combined"
                    else "three_datasets" if layout == "triptych" and len(group) > 1 else group[0]
                )
                if layout == "triptych" and len(group) == 1:
                    layout = "single"
                    id_ += "_panel"
                jobs.append(
                    {
                        "id": id_,
                        "panel": panel,
                        "version": version,
                        "title": f"BLOOM benchmark · {panel.upper()}",
                        "datasets": list(group),
                        "axes": sub_axes,
                        "models": models,
                        "model_order": order,
                        "model_sort_scores": means,
                        "layout": layout,
                        "scale": scale,
                        "metrics_per_dataset": len(metric_names),
                        "average_definition": "Fixed core: (genePCC + RMSE + (MMD + SW)/2)/3 after empirical all-model minmax normalization",
                        "selection_status": (
                            "post_hoc_global_screen" if version == 5 else "fixed_four_endpoint_plan"
                        ),
                    }
                )
    return jobs
