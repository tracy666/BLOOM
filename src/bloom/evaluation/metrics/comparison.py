"""Global metric comparisons without treating cells or genes as replicates.

The input is the canonical long metric table. Dataset, native gene panel,
target cohort, metric definition, source route, aggregation, weighting, split
and expression normalization define a comparison stratum. A training readout
is model-specific, so it is checked for ambiguity rather than used to separate
models into different strata. No best test readout or best test seed is chosen.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any, Iterable, Mapping


TRANSFORMERS = (
    "Autoformer", "Crossformer", "DLinear", "FEDformer", "FiLM", "FreTS",
    "Informer", "iTransformer", "LightTS", "Mamba", "Nonstationary_Transformer",
    "PatchTST", "PAttn", "Reformer", "SegRNN", "TiDE", "TimeFilter",
    "TimesNet", "TimeXer", "Transformer", "TSMixer", "WPMixer",
)
STRATUM_FIELDS = (
    "dataset", "panel", "cohort_id", "route", "definition_id", "aggregation",
    "weighting", "split", "normalization", "metric",
    "pcc_nan_policy", "pcc_eps", "pcc_min_valid_fraction",
)
_EMPTY = {"", "none", "nan", "not_recorded", "not_applicable", "unknown"}
_BAD_STATUSES = {
    "missing", "excluded", "invalid", "error", "blocked", "no_data", "failed",
    "unsupported", "undefined", "not_available", "ambiguous",
}


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _seed(value: Any) -> str:
    text = _text(value)
    if text.lower() in _EMPTY:
        return ""
    # CSVs sometimes serialize integer seeds as floating-point numbers.
    try:
        numeric = float(text)
        if math.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (ValueError, OverflowError):
        pass
    return text


def _canonical_model(model: Any, family: Any) -> str:
    text = _text(model)
    if _text(family).upper() == "TRANS":
        for suffix in ("_author", "_hvg200", "_hvg", "_200"):
            if text.endswith(suffix):
                text = text[:-len(suffix)]
                break
        if text == "Nonstationary Transformer":
            text = "Nonstationary_Transformer"
    return text


def _display(model: str) -> str:
    if model in {"v6_200_new", "v6_author"}:
        return "unidirectional BLOOM"
    return "Nonstationary Transformer" if model == "Nonstationary_Transformer" else model


def _fm_model(panel: str) -> str:
    return "v6_author" if panel.lower() == "author" else "v6_200_new"


def _meta(name: str, supplied: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    if supplied is not None:
        return dict(supplied.get(name, {}))
    from .registry import get_metric
    return dict(get_metric(name) or {})


def _tie(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def exact_paired_sign_flip(differences: Iterable[float], max_seeds: int = 20) -> float | None:
    """Two-sided sign-flip p value for the mean of paired seed differences.

    This enumerates all 2**n assignments, including the observed assignment.
    It is exact under the sign-exchangeability null, not a biological replicate
    test. Fewer than three seeds or more than ``max_seeds`` yields no p value.
    """
    differences = tuple(float(x) for x in differences)
    if any(not math.isfinite(x) for x in differences):
        return None
    n = len(differences)
    if n < 3 or n > max_seeds:
        return None
    observed = abs(math.fsum(differences))
    tolerance = max(1e-12, math.fsum(abs(value) for value in differences) * 1e-12)
    # Iteratively enumerate sums: O(2**n), rather than re-summing n terms for
    # every assignment. The <=20-seed cap bounds peak scratch memory.
    sums = [0.0]
    for difference in differences:
        sums = [value - difference for value in sums] + [value + difference for value in sums]
    at_least_observed = sum(abs(value) + tolerance >= observed for value in sums)
    return at_least_observed / (2 ** n)


def benjamini_hochberg(pvalues: Iterable[float]) -> list[float]:
    """BH adjusted p values, preserving the input order."""
    pvalues = list(pvalues)
    if any(not math.isfinite(p) or p < 0 or p > 1 for p in pvalues):
        raise ValueError("BH requires finite p values between zero and one")
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [1.0] * len(pvalues)
    previous = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        previous = min(previous, pvalues[index] * len(order) / rank, 1.0)
        adjusted[index] = previous
    return adjusted


def benjamini_yekutieli(pvalues: Iterable[float]) -> list[float]:
    """Conservative FDR adjustment allowing arbitrary test dependence."""
    pvalues = list(pvalues)
    harmonic = math.fsum(1 / rank for rank in range(1, len(pvalues) + 1))
    return [min(1.0, value * harmonic) for value in benjamini_hochberg(pvalues)]


def _attested(value: Any) -> bool:
    return value is True or _text(value).lower() in {"true", "1", "yes"}


def _coverage(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Normalize an optional explicit target-day set for aggregate auditing."""
    for field in ("target_days", "target_times", "day_set", "covered_days"):
        value = row.get(field)
        if value is None or _text(value).lower() in _EMPTY:
            continue
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                value = value.replace(";", ",").split(",")
        if isinstance(value, (tuple, list, set)):
            return tuple(sorted({_seed(day) for day in value}, key=lambda day: (0, float(day)) if _number(day) is not None else (1, day)))
    return None


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def build_comparisons(
    rows: Iterable[Mapping[str, Any]],
    metricmeta: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    alpha: float = 0.05,
) -> dict[str, list[dict[str, Any]]]:
    """Return pairwise, summaries, recommendations, significance and issue tables.

    Only global rows participate. Verified paired seed measurements can support
    inference. Unverified legacy/log values support explicitly descriptive
    ordering. In the seeded case all available models use the same intersection
    of seed sets; excluded seeds are reported. Ambiguous readouts or repeated
    seed/run values block the affected model rather than silently picking one.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    issues: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        if _text(row.get("scope", "global")) != "global":
            continue
        key = tuple(_text(row.get(field)) for field in STRATUM_FIELDS)
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    significance: list[dict[str, Any]] = []
    for key in sorted(grouped):
        context = dict(zip(STRATUM_FIELDS, key))
        metadata = _meta(context["metric"], metricmeta)
        direction = _text(metadata.get("direction"))
        if direction not in {"higher", "lower"} or not metadata.get("benchmark_eligible", False):
            issues.append({**context, "status": "not_benchmark_eligible", "reason": "Metric has no audited ranking direction or is a diagnostic, count, training-selection statistic, or anchor"})
            continue
        orient = 1.0 if direction == "higher" else -1.0
        fm_model = _fm_model(context["panel"])
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in grouped[key]:
            model = _canonical_model(row.get("model"), row.get("family"))
            if model not in {fm_model, *TRANSFORMERS}:
                continue
            number = _number(row.get("value"))
            status = _text(row.get("status", "available")).lower()
            if number is None or status in _BAD_STATUSES or status.startswith(("blocked", "error", "invalid", "excluded")):
                continue
            by_model[model].append({**row, "value": number, "seed": _seed(row.get("seed")), "model": model})

        coverages = {_coverage(row) for observations in by_model.values() for row in observations}
        known_coverages = coverages - {None}
        coverage_conflict = len(known_coverages) > 1
        if coverage_conflict:
            issues.append({**context, "status": "unmatched_target_day_coverage",
                "reason": "Global aggregate rows cover different target-day sets; no aggregate ranking or significance is permitted",
                "target_day_sets_json": json.dumps(sorted(known_coverages))})

        # Preserve one observation per model and seed. Exact repeated source rows
        # are harmless; distinct runs/readouts for a seed are not selected here.
        validated: dict[str, dict[str, dict[str, Any]]] = {}
        blocked_models: dict[str, str] = {}
        for model, observations in by_model.items():
            if coverage_conflict:
                blocked_models[model] = "Global aggregate target-day coverage differs within this comparison stratum"
                continue
            if len({_text(row.get("readout")) for row in observations}) > 1:
                blocked_models[model] = "Multiple readouts in the same comparison stratum; select one using validation evidence before comparison"
                continue
            per_seed: dict[str, dict[str, Any]] = {}
            conflict = ""
            for row in observations:
                seed = row["seed"]
                if seed in per_seed:
                    previous = per_seed[seed]
                    if row["value"] != previous["value"] or _text(row.get("run_id")) != _text(previous.get("run_id")):
                        conflict = "Multiple distinct values or run IDs for the same seed (including an unspecified seed); no best test run was selected"
                        break
                    # If verification differs, retain the less strong evidence.
                    if row.get("comparison_status") != "verified":
                        previous["comparison_status"] = row.get("comparison_status", "unverified")
                else:
                    per_seed[seed] = row
            if conflict:
                blocked_models[model] = conflict
            elif "" in per_seed and len(per_seed) > 1:
                blocked_models[model] = "Mixed identified and unidentified seeds within one model"
            else:
                validated[model] = per_seed
        for model, reason in blocked_models.items():
            issues.append({**context, "model": model, "status": "ambiguous_model", "reason": reason})

        seed_sets = [set(values) for values in validated.values()]
        explicit_seeded = bool(seed_sets) and all("" not in seeds for seeds in seed_sets)
        unseeded = bool(seed_sets) and all(seeds == {""} for seeds in seed_sets)
        common_seeds: set[str] = set.intersection(*seed_sets) if explicit_seeded else set()
        seed_issue = ""
        if explicit_seeded and not common_seeds:
            seed_issue = "No seed shared by all available models; aggregate values are not ranked"
        elif not explicit_seeded and not unseeded and any(len(values) > 1 for values in validated.values()):
            seed_issue = "Mixed known/unknown seeds with multiple runs; aggregate values are not ranked"
        if seed_issue:
            issues.append({**context, "status": "unmatched_seed_sets", "reason": seed_issue})
        values: dict[str, float] = {}
        model_rows: dict[str, list[dict[str, Any]]] = {}
        excluded_seeds: dict[str, list[str]] = {}
        if not seed_issue:
            for model, observations in validated.items():
                chosen_seeds = common_seeds if explicit_seeded else set(observations)
                chosen_rows = [observations[seed] for seed in sorted(chosen_seeds)]
                model_rows[model] = chosen_rows
                values[model] = math.fsum(row["value"] for row in chosen_rows) / len(chosen_rows)
                excluded_seeds[model] = sorted(set(observations) - chosen_seeds)

        bloom_value = values.get(fm_model)
        known_trans = [model for model in TRANSFORMERS if model in values]
        missing_trans = [model for model in TRANSFORMERS if model not in values]
        evidence_statuses = sorted({_text(row.get("comparison_status", "unverified")) for chosen in model_rows.values() for row in chosen})
        group_verified = bool(model_rows) and evidence_statuses == ["verified"]
        evidence = "verified" if group_verified else "descriptive_unverified"
        base = {**context, "direction": direction, "aspect": metadata.get("aspect", ""), "comparison_evidence": evidence,
                "source_comparison_statuses": _json_list(evidence_statuses), "BLOOM_model": fm_model,
                "BLOOM_display_name": "unidirectional BLOOM", "BLOOM_value": bloom_value if bloom_value is not None else "",
                "aggregation_across_runs": "equal_weight_matched_seed_mean" if explicit_seeded else "single_reported_aggregate",
                "matched_seeds": _json_list(sorted(common_seeds)), "n_matched_seeds": len(common_seeds),
                "excluded_seeds_json": json.dumps(excluded_seeds, sort_keys=True), "n_expected_TRANS": len(TRANSFORMERS),
                "n_available_TRANS": len(known_trans), "missing_TRANS": _json_list(_display(model) for model in missing_trans),
                "blocked_models_json": json.dumps(blocked_models, sort_keys=True)}
        better, worse, tied = [], [], []
        for trans in TRANSFORMERS:
            trans_value = values.get(trans)
            if bloom_value is None or trans_value is None:
                label = "unavailable"
                difference: float | str = ""
            elif _tie(bloom_value, trans_value):
                label, difference = "tie", 0.0
                tied.append(trans)
            else:
                difference = orient * (bloom_value - trans_value)
                label = "BLOOM_better" if difference > 0 else "TRANS_better"
                (worse if difference > 0 else better).append(trans)
            pair = {**base, "TRANS_model": trans, "TRANS_display_name": _display(trans),
                    "TRANS_value": trans_value if trans_value is not None else "", "oriented_BLOOM_minus_TRANS": difference,
                    "comparison": label, "status": "available" if label != "unavailable" else "missing",
                    "p_value_two_sided": "", "q_value_BH": "", "q_value_BY": "", "statistically_BLOOM_better": False,
                    "statistically_TRANS_better": False, "inference_status": "not_enough_verified_matched_independent_seed_runs"}
            if label != "unavailable" and explicit_seeded:
                used_fm = validated[fm_model]
                used_trans = validated[trans]
                verified = all(used_fm[seed].get("comparison_status") == "verified" and used_trans[seed].get("comparison_status") == "verified" for seed in common_seeds)
                inference_rows = [observations[seed] for observations in (used_fm, used_trans) for seed in common_seeds]
                inference_units = {_text(row.get("inference_unit")) for row in inference_rows}
                independence_verified = all(_attested(row.get("independence_attested")) for row in inference_rows)
                unit_supported = len(inference_units) == 1 and next(iter(inference_units), "") in {"independent_training_seed", "biological_replicate"}
                if verified and len(common_seeds) >= 3 and not (independence_verified and unit_supported):
                    pair["inference_status"] = "independence_or_inference_unit_not_explicitly_attested"
                if verified and len(common_seeds) >= 3 and independence_verified and unit_supported:
                    differences = [orient * (used_fm[seed]["value"] - used_trans[seed]["value"]) for seed in sorted(common_seeds)]
                    pvalue = exact_paired_sign_flip(differences)
                    if pvalue is not None:
                        inference_unit = next(iter(inference_units))
                        pair["p_value_two_sided"] = pvalue
                        pair["inference_status"] = "tested_explicit_independent_units_exploratory"
                        significance.append({**context, "TRANS_model": trans, "BLOOM_model": fm_model,
                            "oriented_BLOOM_minus_TRANS": difference, "n_independent_units": len(common_seeds),
                            "unit": inference_unit, "seeds": _json_list(sorted(common_seeds)),
                            "p_value_two_sided": pvalue, "q_value_BH": "", "q_value_BY": "", "test": "exact_two_sided_paired_sign_flip_mean",
                            "assumption": "Explicitly attested independent units; paired differences sign-exchangeable under null",
                            "inference_domain": "training_seed_variability_conditional_on_this_dataset; not_biological_generalization" if inference_unit == "independent_training_seed" else "attested_independent_biological_replicates_in_this_cohort",
                            "multiplicity_family": "all_tested_global_metric_model_cohort_comparisons_in_this_export",
                            "exploratory": True, "statistically_BLOOM_better": False, "statistically_TRANS_better": False,
                            "_pair_index": len(pairwise)})
                    else:
                        pair["inference_status"] = "exact_seed_enumeration_limit_exceeded"
            pairwise.append(pair)

        strongest = max(known_trans, key=lambda model: orient * values[model]) if known_trans else ""
        rank = 1 + len(better) if bloom_value is not None else ""
        if bloom_value is None:
            conclusion = "BLOOM_missing_or_ambiguous"
        elif not known_trans:
            conclusion = "no_TRANS_comparison"
        elif not better and worse:
            conclusion = "BLOOM_at_least_tied_for_best_among_available_TRANS"
        elif better:
            conclusion = "some_TRANS_outperform_BLOOM"
        else:
            conclusion = "all_available_models_tied"
        summaries.append({**base, "BLOOM_rank": rank,
            "better_TRANS": _json_list(_display(model) for model in better),
            "worse_TRANS": _json_list(_display(model) for model in worse),
            "tied_TRANS": _json_list(_display(model) for model in tied), "n_better_TRANS": len(better),
            "n_worse_TRANS": len(worse), "n_tied_TRANS": len(tied),
            "strongest_TRANS_model": strongest, "strongest_TRANS_display_name": _display(strongest),
            "strongest_TRANS_value": values.get(strongest, ""),
            "oriented_margin_to_strongest_TRANS": orient * (bloom_value - values[strongest]) if bloom_value is not None and strongest else "",
            "complete_22_TRANS": len(known_trans) == len(TRANSFORMERS), "descriptive_conclusion": conclusion,
            "n_statistically_worse_TRANS": 0, "n_statistically_better_TRANS": 0,
            "statistically_worse_TRANS": "[]", "statistically_better_TRANS": "[]",
            "recommended_by_definition": bool(metadata.get("recommended", False)),
            "note": "Ranking includes only this exact dataset/panel/cohort/metric/route stratum. No day, gene or cell is a significance replicate."})

    # One export-wide family avoids adjusting only favorable tests.
    qvalues = benjamini_hochberg(item["p_value_two_sided"] for item in significance)
    qvalues_by = benjamini_yekutieli(item["p_value_two_sided"] for item in significance)
    for test, qvalue, qvalue_by in zip(significance, qvalues, qvalues_by):
        pair = pairwise[test.pop("_pair_index")]
        difference = float(test["oriented_BLOOM_minus_TRANS"])
        bloom_better = qvalue_by <= alpha and difference > 0
        trans_better = qvalue_by <= alpha and difference < 0
        for output in (test, pair):
            output["q_value_BH"] = qvalue
            output["q_value_BY"] = qvalue_by
            output["selection_adjustment"] = "Benjamini_Yekutieli_arbitrary_dependence"
            output["statistically_BLOOM_better"] = bloom_better
            output["statistically_TRANS_better"] = trans_better

    test_by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for test in significance:
        test_by_stratum[tuple(_text(test[field]) for field in STRATUM_FIELDS)].append(test)
    recommendations: list[dict[str, Any]] = []
    for summary in summaries:
        key = tuple(_text(summary[field]) for field in STRATUM_FIELDS)
        tests = test_by_stratum[key]
        worse = [test["TRANS_model"] for test in tests if test["statistically_BLOOM_better"]]
        better = [test["TRANS_model"] for test in tests if test["statistically_TRANS_better"]]
        summary.update({"n_statistically_worse_TRANS": len(worse), "n_statistically_better_TRANS": len(better),
                        "statistically_worse_TRANS": _json_list(_display(model) for model in worse),
                        "statistically_better_TRANS": _json_list(_display(model) for model in better)})
        positive = summary["n_worse_TRANS"] > 0 and summary["n_better_TRANS"] == 0
        complete = summary["complete_22_TRANS"]
        if worse:
            advantage = "exploratory_seed_statistical_advantage_against_some_TRANS"
        elif positive:
            advantage = "descriptive_global_advantage_only"
        else:
            advantage = "no_global_BLOOM_dominance"
        descriptive_candidate = bool(positive and complete and summary["recommended_by_definition"])
        candidate = descriptive_candidate and summary["comparison_evidence"] == "verified"
        reasons = []
        if not positive:
            reasons.append("At least one TRANS is better, no TRANS is worse, or comparison is unavailable")
        if not complete:
            reasons.append("Fewer than all 22 TRANS are available")
        if not summary["recommended_by_definition"]:
            reasons.append("Audited definition marks this metric as supplementary or redundant")
        if summary["comparison_evidence"] != "verified":
            reasons.append("Unverified cohort or preprocessing: numeric ordering is descriptive")
        if not tests:
            reasons.append("No supported independent-seed inferential comparison; statistical significance is unestablished")
        recommendations.append({**summary, "advantage_evidence": advantage,
            "candidate_for_descriptive_review": descriptive_candidate,
            "candidate_for_global_benchmark": candidate, "aspect_shortlist": False,
            "descriptive_aspect_shortlist": False,
            "selection_reason": "; ".join(reasons) if reasons else "Complete global comparison; BLOOM leads or ties; audited metric represents this evaluation aspect",
            "selection_is_exploratory": True,
            "shortlist_policy": "At most one favorable audited metric per aspect, exact dataset/panel/cohort/route; lexical metric order resolves redundant candidates, never largest test effect"})

    # Preserve all unfavorable summaries. The shortlist is only a transparent
    # optional convenience for the requested future multi-aspect benchmark.
    seen_aspects: set[tuple[str, ...]] = set()
    seen_descriptive_aspects: set[tuple[str, ...]] = set()
    for recommendation in sorted(recommendations, key=lambda row: (
            row["dataset"], row["panel"], row["cohort_id"], row["route"],
            row["aspect"], row["metric"], row["definition_id"])):
        aspect_key = tuple(_text(recommendation[field]) for field in (
            "dataset", "panel", "cohort_id", "route", "split", "normalization", "aspect"))
        if recommendation["candidate_for_global_benchmark"] and aspect_key not in seen_aspects:
            recommendation["aspect_shortlist"] = True
            seen_aspects.add(aspect_key)
        if recommendation["candidate_for_descriptive_review"] and aspect_key not in seen_descriptive_aspects:
            recommendation["descriptive_aspect_shortlist"] = True
            seen_descriptive_aspects.add(aspect_key)
    return {"pairwise": pairwise, "summaries": summaries, "recommendations": recommendations,
            "significance": significance, "issues": issues}


def reported_training_ordering(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order the numbers printed by training, explicitly outside fair ranking.

    Two named views are exported for each PCC axis: the primary numbers exactly
    as reported (FM unique targets versus TRANS trajectory occurrences), and a
    unique-target counterpart (FM primary versus TRANS ``*_unique``). Neither
    view proves matching target populations or supports statistical inference.
    Original metric names, weighting, numeric precision and source provenance
    are retained. This does not alter ``build_comparisons`` or its strict strata.

    A supplied ``reported_group`` can associate separate files for descriptive
    review. Otherwise only records from the same source SHA-256/path are grouped.
    Dataset, native panel, route, seed and explicit target-day set stay separate.
    The return value is long-format: 23 model audit rows per view/axis stratum.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for source_index, source_row in enumerate(rows):
        row = dict(source_row)
        if row.get("scope") != "global" or row.get("route") not in {"training_log", "training_summary"}:
            continue
        metric = _text(row.get("metric"))
        matching_axis = next((axis for axis in ("cell", "gene") if metric in {
            f"train_log_mean_day_pcc_{axis}", f"train_log_mean_day_pcc_{axis}_unique"}), None)
        if matching_axis is None:
            continue
        model = _canonical_model(row.get("model"), row.get("family"))
        fm_model = _fm_model(_text(row.get("panel")))
        if model not in {fm_model, *TRANSFORMERS}:
            continue
        # Unselected diagnostic records cannot enter this helper. Canonical
        # metric rows already represent the explicitly chosen training result.
        if row.get("selected_for_benchmark") is not None and not _attested(row["selected_for_benchmark"]):
            continue
        explicit_group = _text(row.get("reported_group"))
        source_group = (_text(row.get("source_sha256")) or _text(row.get("source_path")) or
                        _text(row.get("run_id")) or f"unbound_source_row:{source_index}")
        group_name = f"explicit:{explicit_group}" if explicit_group else f"source:{source_group}"
        target_days = _coverage(row)
        primary_metric = f"train_log_mean_day_pcc_{matching_axis}"
        views = []
        if metric == primary_metric:
            views.append("primary_reported")
            if model == fm_model:
                views.append("unique_target_counterpart")
        elif model in TRANSFORMERS:
            views.append("unique_target_counterpart")
        for view in views:
            key = (_text(row.get("dataset")), _text(row.get("panel")), _text(row.get("route")),
                   group_name, target_days, _seed(row.get("seed")), matching_axis, view)
            groups[key].append({**row, "_canonical_model": model, "_input_index": source_index})

    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(str(part) for part in value)):
        dataset, panel, route, group_name, target_days, seed, axis, view = key
        fm_model = _fm_model(panel)
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in groups[key]:
            by_model[row["_canonical_model"]].append(row)
        selected: dict[str, dict[str, Any]] = {}
        blocked: dict[str, str] = {}
        for model, candidates in by_model.items():
            # A source identity includes its selected line, not just its value:
            # equal test numbers from distinct runs are still distinct results.
            signatures: dict[tuple[Any, ...], dict[str, Any]] = {}
            for row in candidates:
                source_identity = _text(row.get("source_sha256")) or _text(row.get("source_path"))
                signature = (_text(row.get("readout")), _text(row.get("run_id")),
                             source_identity,
                             _text(row.get("source_line")), _text(row.get("result_tag")),
                             _text(row.get("metric")), _text(row.get("source_key", row.get("source_metric"))),
                             _number(row.get("value")))
                signatures.setdefault(signature, row)
            if len(signatures) > 1:
                blocked[model] = "Multiple selected readouts, result lines or runs for this model/seed/source group; no best test result was chosen"
                continue
            candidate = next(iter(signatures.values()))
            status = _text(candidate.get("status", "available")).lower()
            if _number(candidate.get("value")) is None or status in _BAD_STATUSES or status.startswith(("blocked", "error", "invalid", "excluded")):
                continue
            selected[model] = candidate
        values = {model: float(row["value"]) for model, row in selected.items()}
        # Four-decimal RESULT numbers cannot be ranked beyond their reported
        # precision. Full summary floats retain their saved precision.
        rank_values = {model: round(value, 4) if route == "training_log" or "4_decimals" in _text(selected[model].get("precision")) else value
                       for model, value in values.items()}
        known_trans = [model for model in TRANSFORMERS if model in values]
        missing_trans = [model for model in TRANSFORMERS if model not in values]
        fm_value = values.get(fm_model)
        fm_rank_value = rank_values.get(fm_model)
        better, worse, tied = [], [], []
        if fm_rank_value is not None:
            for trans in known_trans:
                if _tie(rank_values[trans], fm_rank_value):
                    tied.append(trans)
                elif rank_values[trans] > fm_rank_value:
                    better.append(trans)
                else:
                    worse.append(trans)
        strongest = max(known_trans, key=rank_values.__getitem__) if known_trans else ""
        if fm_value is None:
            conclusion = "BLOOM_missing_or_ambiguous"
        elif not known_trans:
            conclusion = "no_TRANS_values_available; no_BLOOM_advantage_claim"
        elif better:
            conclusion = "some_reported_TRANS_numbers_exceed_BLOOM"
        elif worse:
            conclusion = "reported_BLOOM_number_is_at_least_tied_for_best_among_available_TRANS"
        else:
            conclusion = "all_available_reported_numbers_tied"
        warning = (
            "Reported numerical ordering only: BLOOM primary PCC uses unique target cells, while TRANS primary PCC uses trajectory occurrences. Different target weighting and populations may confound this ordering. Not a verified benchmark; no significance inference."
            if view == "primary_reported" else
            "Reported unique-target counterpart only: BLOOM primary PCC is paired with TRANS *_unique PCC. Unique-target labels do not establish identical target cells, subset coverage or preprocessing. Not a verified benchmark; no significance inference."
        )
        common = {"dataset": dataset, "panel": panel, "route": route,
                  "reported_group": group_name, "target_days": json.dumps(list(target_days)) if target_days is not None else "unknown",
                  "seed": seed, "axis": axis, "comparison_name": f"{view}_mean_day_pcc_{axis}",
                  "direction": "higher", "unit": "PCC",
                  "comparison_status": "descriptive_reported_ordering_only", "warning": warning,
                  "grouping_basis": "explicit reported_group" if group_name.startswith("explicit:") else "same source digest/path; separate files are not automatically pooled",
                  "n_available_models": len(values), "n_expected_TRANS": len(TRANSFORMERS),
                  "n_available_TRANS": len(known_trans), "missing_TRANS": _json_list(_display(model) for model in missing_trans),
                  "better_TRANS": _json_list(_display(model) for model in better),
                  "worse_TRANS": _json_list(_display(model) for model in worse),
                  "tied_TRANS": _json_list(_display(model) for model in tied),
                  "n_better_TRANS": len(better), "n_worse_TRANS": len(worse), "n_tied_TRANS": len(tied),
                  "BLOOM_model": fm_model, "BLOOM_value": fm_value if fm_value is not None else "",
                  "BLOOM_numerical_rank": 1 + len(better) if fm_rank_value is not None else "",
                  "strongest_TRANS": _display(strongest), "strongest_TRANS_value": values.get(strongest, ""),
                  "reported_BLOOM_minus_strongest_TRANS": fm_value - values[strongest] if fm_value is not None and strongest else "",
                  "descriptive_conclusion": conclusion, "significance_status": "not_tested; heterogeneous_or_unverified_populations",
                  "blocked_models_json": json.dumps(blocked, sort_keys=True)}
        for model in (fm_model, *TRANSFORMERS):
            source = selected.get(model)
            value = values.get(model)
            if model in blocked:
                status, reason = "ambiguous", blocked[model]
            elif source is None:
                status, reason = "missing", "No finite selected training value supplied in this exact source group, panel, seed and target-day set"
            else:
                status, reason = "available", "Original selected training number retained; this view is descriptive only"
            numerical_rank = 1 + sum(not _tie(other, rank_values[model]) and other > rank_values[model] for other in rank_values.values()) if source is not None else ""
            source_fields = {field: source.get(field, "") if source is not None else "" for field in (
                "source_path", "source_sha256", "source_line", "source_key", "source_metric",
                "readout", "result_tag", "run_id", "definition_id", "pcc_nan_policy", "pcc_eps",
                "pcc_min_valid_fraction", "pcc_backend_reported", "precision", "n_cells", "n_genes",
                "split", "normalization")}
            output.append({**common, "model": model, "model_display_name": _display(model),
                           "family": "FM" if model == fm_model else "TRANS", "value": value if value is not None else "",
                           "numerical_rank": numerical_rank, "status": status, "reason": reason,
                           "metric": source.get("metric", "") if source is not None else "",
                           "source_weighting": source.get("weighting", "") if source is not None else "",
                           "source_aggregation": source.get("aggregation", "") if source is not None else "",
                           "source_comparison_status": source.get("comparison_status", "") if source is not None else "",
                           "candidate_sources_json": json.dumps([{field: candidate.get(field, "") for field in (
                               "metric", "value", "readout", "run_id", "source_path", "source_sha256", "source_line")}
                               for candidate in by_model.get(model, [])], ensure_ascii=False, default=str),
                           **source_fields})
    return output
