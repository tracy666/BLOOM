"""Import saved catalogue metrics without treating them as verified raw benchmarks.

Only known metric columns and exact model slots are accepted. The importer never
turns arbitrary numeric metadata into a score, relabels saved PCCs as training
PCCs, chooses a best readout, or pools author and HVG panels. Stage identifiers
are retained as recorded ordinals: their physical-age mapping is not inferred.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from collections import defaultdict

TRANS = (
    "Autoformer", "Crossformer", "DLinear", "FEDformer", "FiLM", "FreTS",
    "Informer", "iTransformer", "LightTS", "Mamba", "Nonstationary_Transformer",
    "PatchTST", "PAttn", "Reformer", "SegRNN", "TiDE", "TimeFilter", "TimesNet",
    "TimeXer", "Transformer", "TSMixer", "WPMixer",
)
CORE = {
    "pseudobulk_pcc", "pseudobulk_spearman", "pseudobulk_mae", "pseudobulk_rmse",
    "covariance_relative_frobenius", "covariance_edge_pcc", "variance_pcc",
    "variance_rmse", "mmd_u", "sliced_wasserstein", "variogram", "mse", "rmse",
    "mae", "r2", "mean_cell_pcc", "median_cell_pcc", "mean_gene_pcc",
    "median_gene_pcc", "mean_row_cosine",
}
DEVELOPMENT = {
    "de_score", "de_direction", "signed_overlap", "chance_overlap", "n_true_up",
    "n_true_down", "de_score_baseline", "de_score_upper", "de_score_skill",
    "de_direction_baseline", "de_direction_upper", "de_direction_skill",
    "mmd_u_baseline", "mmd_u_upper", "mmd_u_skill", "variogram_baseline",
    "variogram_upper", "variogram_skill", "virtual_embryo_t1_score",
}
KNOWN_METRICS = CORE | DEVELOPMENT
LOWER = {
    "mae", "mse", "rmse", "pseudobulk_mae", "pseudobulk_rmse",
    "covariance_relative_frobenius", "variance_rmse", "mmd_u",
    "sliced_wasserstein", "variogram",
}
NON_PERFORMANCE = {
    "chance_overlap", "n_true_up", "n_true_down", "de_score_baseline",
    "de_score_upper", "de_direction_baseline", "de_direction_upper",
    "mmd_u_baseline", "mmd_u_upper", "variogram_baseline", "variogram_upper",
}
MISSING = {"", "na", "nan", "null", "none", "n/a"}
UNKNOWN = {"", "unknown", "not_recorded", "defined_by_comparison_id"}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sort_day(value: str):
    try:
        return (0, Decimal(value), value)
    except InvalidOperation:
        return (1, Decimal(0), value)


def _panel(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"200", "hvg", "hvg200", "hvg_200"}:
        return "hvg"
    if value == "author":
        return "author"
    raise ValueError("panel must explicitly identify HVG/200 or author")


def _identity(record: dict) -> tuple[str, str, str, str]:
    dataset = record.get("dataset", "").strip().lower()
    family = record.get("family", "").strip().upper().replace("TRANSFORMER", "TRANS")
    model = record.get("model", "").strip()
    slot = record.get("slot", "").strip()
    full = model if "/" in model else slot
    if full:
        parts = full.split("/")
        if len(parts) != 3:
            raise ValueError("full slot must be dataset/family/model")
        ds, fam, name = parts
        fam = fam.upper().replace("TRANSFORMER", "TRANS")
        if dataset and dataset != ds:
            raise ValueError("dataset disagrees with full slot")
        if family and family != fam:
            raise ValueError("family disagrees with full slot")
        if model and "/" not in model and model != name:
            raise ValueError("model disagrees with full slot")
        dataset, family, model = ds, fam, name
    if dataset not in {"big", "small", "sted"}:
        raise ValueError("dataset must be big, small, or sted")
    if model == "v6_200_new":
        expected_panel, expected_family = "hvg", "FM"
    elif model == "v6_author":
        expected_panel, expected_family = "author", "FM"
    elif any(model == name + "_" + suffix for name in TRANS for suffix in ("200", "author")):
        expected_panel = "author" if model.endswith("_author") else "hvg"
        expected_family = "TRANS"
    elif model == "v6_200_old" or model.startswith("v7_"):
        raise ValueError("excluded old or bidirectional BLOOM slot")
    else:
        raise ValueError("unsupported exact model slot")
    panel = _panel(record.get("panel", record.get("gene_panel", "")))
    if panel != expected_panel or family != expected_family:
        raise ValueError("native panel or family disagrees with exact model slot")
    return dataset, panel, model, family


def _unit(metric: str) -> str:
    if metric in {"n_true_up", "n_true_down"}:
        return "genes"
    if metric == "mse":
        return "expression^2"
    if metric == "variance_rmse":
        return "expression^2"
    if metric in {"mae", "rmse", "pseudobulk_mae", "pseudobulk_rmse", "sliced_wasserstein", "variogram"}:
        return "source_expression"
    return "unitless"


def _label(model: str, family: str) -> str:
    if family == "FM":
        return "unidirectional BLOOM"
    return model.rsplit("_", 1)[0].replace("Nonstationary_Transformer", "Nonstationary Transformer")


def _cohort(dataset: str, panel: str, task: str, days: list[str], summary=False) -> str:
    tag = hashlib.sha256(json.dumps(days, ensure_ascii=False).encode()).hexdigest()[:12]
    return f"legacy:{dataset}:{panel}:{task}:{'summary' if summary else 'stages'}:{tag}"


def _provenance(record: dict, path: Path, digest: str, line: int) -> dict:
    # The old exporter put its input-table SHA beside a downstream source path.
    # Preserve that original ledger separately; our SHA always identifies the
    # file whose exact row we actually read here.
    return dict(
        source_path=str(path), source_sha256=digest, source_line=line,
        original_source_path=record.get("source_path", ""),
        original_source_sha256=record.get("source_sha256", ""),
        original_source_line=record.get("source_line", ""),
        original_source_row=record.get("source_row", ""),
        original_provenance_json=record.get("provenance_json", ""),
    )


def import_legacy(path) -> dict:
    """Return ``{'rows': canonical finite observations, 'audit': details}``.

    Actual missing rows are retained in the audit. Synthetic inventory fillers
    are ignored. Equal-stage global means require every retained model to have
    exactly the same full recorded stage coverage and a finite, unambiguous value
    at every stage; no intersections or best-readout selection are used.
    """
    path = Path(path).expanduser().resolve()
    digest = _digest(path)
    audit = dict(path=str(path), sha256=digest, input_rows=0, placeholder_rows=0,
                 missing_rows=[], excluded_rows=[], conflicts=[],
                 incomplete_global_groups=[], deduplicated_rows=0,
                 comparison_status="legacy_unverified")
    observations = []
    roster = defaultdict(set)
    stages_by_model = defaultdict(set)
    blocked = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, delimiter="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError(f"{path}: duplicate column names")
        if not {"metric", "value", "dataset", "family"}.issubset(fields) or not ({"model", "slot"} & set(fields)):
            raise ValueError(f"{path}: need explicit dataset/family/model-or-slot/metric/value columns")
        for source in reader:
            audit["input_rows"] += 1
            line = reader.line_num
            if None in source:
                raise ValueError(f"{path}:{line}: extra fields outside header")
            record = {k: (v or "").strip() for k, v in source.items()}
            provenance = _provenance(record, path, digest, line)
            if record.get("metadata_basis") == "expected_slot_audit_placeholder" or record.get("comparison_id") == "NO_SUPPLIED_COMPARISON":
                audit["placeholder_rows"] += 1
                continue
            try:
                dataset, panel, model, family = _identity(record)
                metric = record["metric"]
                if metric not in KNOWN_METRICS:
                    raise ValueError("not an audited legacy v2 metric; training-log metrics require the log route")
                task = record.get("task", "")
                if not task and record.get("task_direction") == "forward":
                    task = "measured_forward"
                if task != "measured_forward":
                    raise ValueError("only measured_forward belongs to this unidirectional benchmark")
                day = record.get("target_time", "") or record.get("stage", "") or record.get("day", "")
                if not day:
                    if record.get("scope") == "global":
                        day = "all"
                    else:
                        raise ValueError("recorded target stage is missing")
                if record.get("target_time") and record.get("stage") and record["target_time"] != record["stage"]:
                    raise ValueError("stage disagrees with target_time")
                base = (dataset, panel, task)
                if day.lower() != "all":
                    roster[base].add(model)
                    stages_by_model[base + (model,)].add(day)
                token = record["value"]
                value = Decimal(token) if token.lower() not in MISSING else None
                if value is None or not value.is_finite():
                    audit["missing_rows"].append(dict(**provenance, dataset=dataset, panel=panel, model=model,
                                                     metric=metric, day=day, raw_value=token,
                                                     reason=record.get("reason") or record.get("status") or "missing or nonfinite value"))
                    continue
                status = record.get("status", "available").lower()
                if status not in {"available", "complete", "ok", "success", ""}:
                    raise ValueError("finite value carries a failed/nonavailable source status")
                direction = "" if metric in NON_PERFORMANCE else "lower" if metric in LOWER else "higher"
                readout = record.get("readout", "") or "not_recorded"
                row = dict(dataset=dataset, panel=panel, model=model, model_label=_label(model, family), family=family,
                           task=task, metric=metric, value=float(value), raw_value=token,
                           scope="global" if day.lower() == "all" else "day", day="" if day.lower() == "all" else day,
                           day_kind="recorded_ordinal_unknown", readout=readout,
                           route="legacy_reported", comparison_status="legacy_unverified",
                           weighting="source_saved_population", aggregation="source_saved_summary" if day.lower() == "all" else "recorded_per_target_stage",
                           definition_id="legacy_v2:" + metric, definition_policy="finite_skip;source_contract_unknown",
                           direction=direction, unit=_unit(metric), status="available",
                           pairing_status="not_attested", preprocessing=record.get("preprocessing", "") or "not_recorded",
                           feature_space=record.get("feature_space", "") or "not_recorded",
                           split=record.get("split", "") or "not_recorded", replicate=record.get("replicate", ""),
                           seed=record.get("seed", ""), original_direction=record.get("source_direction", record.get("direction", "")),
                           original_unit=record.get("unit", ""), **provenance)
                observations.append(row)
            except (ValueError, InvalidOperation) as exc:
                audit["excluded_rows"].append(dict(**provenance, slot=record.get("slot") or record.get("model", ""),
                                                  metric=record.get("metric", ""), raw_value=record.get("value", ""), reason=str(exc)))
    # A duplicate observation must agree in value AND readout/semantics. Different
    # readouts are neither averaged together nor selected using test performance.
    groups = defaultdict(list)
    for row in observations:
        key = tuple(row[k] for k in ("dataset", "panel", "task", "model", "metric", "scope", "day"))
        groups[key].append(row)
    deduped = []
    semantic = ("raw_value", "readout", "definition_id", "unit", "preprocessing", "feature_space", "split", "replicate", "seed")
    for key, group in groups.items():
        signatures = set()
        for row in group:
            signatures.add((Decimal(row["raw_value"]),) + tuple(row[k] for k in semantic[1:]))
        if len(signatures) > 1:
            audit["conflicts"].append(dict(dataset=key[0], panel=key[1], task=key[2], model=key[3], metric=key[4], scope=key[5], day=key[6], observations=group,
                                           reason="conflicting value/readout/semantics; none was selected"))
            blocked.add(key[:5])
            continue
        row = dict(group[0])
        proofs = [{k: r[k] for k in ("source_path", "source_sha256", "source_line", "original_source_path", "original_source_row")} for r in group]
        row["provenance_json"] = json.dumps(proofs, sort_keys=True)
        audit["deduplicated_rows"] += len(group) - 1
        deduped.append(row)
    output = []
    by_base = defaultdict(list)
    for row in deduped:
        base = tuple(row[k] for k in ("dataset", "panel", "task"))
        if row["scope"] == "global":
            row["cohort_id"] = _cohort(*base, ["all"], summary=True)
            output.append(row)
        else:
            by_base[base].append(row)
    for base, group in by_base.items():
        models = sorted(roster[base])
        coverage = {model: stages_by_model[base + (model,)] for model in models}
        all_stages = sorted(set().union(*coverage.values()), key=_sort_day)
        cohort_id = _cohort(*base, all_stages)
        for row in group:
            row["cohort_id"] = cohort_id
            output.append(row)
        complete_cohort = all(coverage[model] == set(all_stages) for model in models)
        for metric in sorted({r["metric"] for r in group}):
            metric_rows = [r for r in group if r["metric"] == metric]
            buckets = defaultdict(dict)
            for row in metric_rows:
                buckets[row["model"]][row["day"]] = row
            complete = complete_cohort and all(set(buckets[model]) == set(all_stages) and base + (model, metric) not in blocked for model in models)
            if not complete:
                audit["incomplete_global_groups"].append(dict(dataset=base[0], panel=base[1], task=base[2], metric=metric,
                    expected_models=models, recorded_stages=all_stages,
                    stage_coverage={model: sorted(buckets[model], key=_sort_day) for model in models},
                    reason="global summary blocked: every retained model must have the same complete finite stage set and no conflicts"))
                continue
            for model in models:
                model_rows = [buckets[model][day] for day in all_stages]
                row = dict(model_rows[0])
                value = sum(Decimal(r["raw_value"]) for r in model_rows) / Decimal(len(model_rows))
                row.update(scope="global", day="", value=float(value), raw_value=str(value),
                           aggregation="equal_recorded_stage_mean", n_days=len(model_rows),
                           recorded_days="|".join(all_stages), source_line="",
                           definition_policy="finite_skip;source_contract_unknown;equal_stage_mean")
                row["provenance_json"] = json.dumps([json.loads(r["provenance_json"]) for r in model_rows], sort_keys=True)
                output.append(row)
    output.sort(key=lambda row: (row["dataset"], row["panel"], row["cohort_id"], row["metric"], row["scope"], _sort_day(row["day"]), row["model"]))
    audit["day_rows"] = sum(r["scope"] == "day" for r in output)
    audit["global_rows"] = sum(r["scope"] == "global" for r in output)
    audit["verified_rows"] = 0
    return {"rows": output, "audit": audit}
