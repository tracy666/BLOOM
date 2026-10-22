"""Read authoritative RESULT records without importing or running training code.

Printed aggregate values are retained exactly.  Day records are a separate,
rounded export, never a replacement for the printed mean.  Native FM and TRANS
populations are explicitly unverified unless the caller supplies a manifest
contract; equal split names alone do not establish equal target cells.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


TRANS_MODELS = frozenset({
    "Autoformer", "Crossformer", "DLinear", "FEDformer", "FiLM", "FreTS",
    "Informer", "iTransformer", "LightTS", "Mamba", "Nonstationary_Transformer",
    "PatchTST", "PAttn", "Reformer", "SegRNN", "TiDE", "TimeFilter", "TimesNet",
    "TimeXer", "Transformer", "TSMixer", "WPMixer",
})
_RESULT = re.compile(r"\bRESULT\s+(FM|TRANS)\s+(\S+)\s*\|\s*(.*)$")
_DAY_RESULT = re.compile(r"\bRESULT_DAY_METRICS\s+(\S+)\s*\|\s*(.*)$")
_NUMBERS = re.compile(
    r"(?<![\w])([A-Za-z_][A-Za-z_0-9]*)=("
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf))"
    r"(?:s)?(?=\s|\||$)", re.IGNORECASE,
)
_CONFIG = re.compile(r"\|\s*(?:INFO|DEBUG)\s*\|\s*([A-Z][A-Z_0-9]*)\s*:\s*(.*?)\s*$")
_DAY_KEY = re.compile(r"^day(-?\d+)_(cell|gene)_pcc(_unique)?$")
_DAY_SPLIT = re.compile(r"\b(train|val|test)\s+(?:model\s+)?timestamps:\s*(\[[^\]]*\])")


class LogBindingError(ValueError):
    """A log cannot safely be attached to the requested benchmark model."""


def _config_value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _trans_id(tag: str, panel: str, model_map: Mapping[str, str]) -> str | None:
    if tag in model_map:
        model = str(model_map[tag])
        suffix = "_200" if panel == "hvg" else "_author"
        if not model.endswith(suffix):
            raise LogBindingError(f"TRANS mapping {tag!r} -> {model!r} conflicts with panel={panel!r}")
        if model[:-len(suffix)] not in TRANS_MODELS:
            raise LogBindingError(f"TRANS mapping {tag!r} is not one of the 22 benchmark models")
        return model
    base = tag
    if base.startswith("TRANS_"):
        base = base[len("TRANS_"):]
        if base.endswith("_scratch"):
            base = base[:-len("_scratch")]
        else:
            # Finetuning and foundation variants require their exact identity.
            return None
    if base not in TRANS_MODELS:
        return None
    return base + ("_200" if panel == "hvg" else "_author")


def _validate_metadata(entry: Mapping[str, Any], config: Mapping[str, Any], path: Path) -> None:
    panel = str(entry["panel"])
    dataset = str(entry["dataset"])
    if panel not in {"hvg", "author"}:
        raise LogBindingError("Training-log panel must be 'hvg' or 'author'")
    fm_id = entry.get("fm_model")
    if fm_id and fm_id != ("v6_200_new" if panel == "hvg" else "v6_author"):
        raise LogBindingError(f"fm_model={fm_id!r} is not forward unidirectional BLOOM for panel={panel}")
    direction = config.get("FM_TEMPORAL_DIRECTION")
    if direction not in (None, "forward"):
        raise LogBindingError(f"This benchmark excludes temporal direction {direction!r}; use a forward v6 log")
    if str(config.get("EVAL_SPLIT", "test")) != "test":
        raise LogBindingError("This benchmark requires EVAL_SPLIT=test; validation RESULT values cannot rank test performance")
    # Only explicit tokens carry identity.  Do not infer species or assay from
    # the scientific name of the source dataset.
    identity_text = " ".join((path.name, str(config.get("MSG", "")), str(config.get("OUTPUT_DIR", ""))))
    dataset_tokens = set(re.findall(r"(?:^|[/_\-])(?P<dataset>small|big|sted)(?=[/_\-]|$)", identity_text))
    if dataset_tokens and dataset not in dataset_tokens:
        raise LogBindingError(f"Requested dataset={dataset!r} conflicts with explicit log tokens {sorted(dataset_tokens)}")
    var_name = str(config.get("ADATA_VAR_PATH", "")).lower()
    confident_panels = set()
    if re.search(r"hvg[_\-]?\d+", var_name) or re.search(r"(?:^|[_\-])hvg\d*(?=[_\-.]|$)", path.name.lower()):
        confident_panels.add("hvg")
    if re.search(r"(?:^|[/_\-])author(?:[_\-.]|$)", var_name) or re.search(r"(?:^|[_\-])author(?=[_\-.]|$)", path.name.lower()):
        confident_panels.add("author")
    if confident_panels and (len(confident_panels) != 1 or panel not in confident_panels):
        raise LogBindingError(f"Requested panel={panel!r} conflicts with explicit log panel {sorted(confident_panels)}")
    pred_len = config.get("PRED_LEN")
    if pred_len is not None and int(pred_len) != 1:
        raise LogBindingError("PRED_LEN must equal 1: this v6 TRANS evaluator pairs the first target expression with the last target-cell ID otherwise")


def _contract_verified(entry: Mapping[str, Any], days: list[int], weighting: str) -> bool:
    contract = entry.get("comparison_contract", {})
    if not isinstance(contract, Mapping) or contract.get("verified") is not True:
        return False
    required = ("target_manifest_sha256", "gene_manifest_sha256", "expression_units", "split_id")
    if not all(contract.get(key) for key in required):
        raise LogBindingError("Verified log comparison_contract requires target_manifest_sha256, gene_manifest_sha256, expression_units and split_id")
    for key in ("target_manifest_sha256", "gene_manifest_sha256"):
        if not re.fullmatch(r"[a-fA-F0-9]{64}", str(contract[key])):
            raise LogBindingError(f"Verified log comparison_contract {key} must be a 64-character SHA-256 digest")
    if str(contract.get("split", "test")) != "test":
        raise LogBindingError("Verified log comparison contract is not a test split")
    if contract.get("target_days") is not None and sorted(int(x) for x in contract["target_days"]) != days:
        raise LogBindingError("Verified comparison_contract target_days differ from printed RESULT_DAY_METRICS")
    if contract.get("weighting") != weighting:
        raise LogBindingError(f"Verified log contract must attest actual weighting={weighting!r}")
    return True


def parse_log(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return benchmark ``rows``, every numeric RESULT ``records``, and audit.

    Required keys: path, dataset, panel.  FM selection requires fm_model and an
    exact fm_result_tag or unambiguous fm_readout.  ``result_line`` selects one
    occurrence when the same tag has been printed more than once; no last-row
    or best-test-value selection is performed.  Optional ``model_map`` binds
    exact TRANS tags.  Unknown models remain in records but not benchmark rows.
    """
    path = Path(str(entry["path"])).expanduser().resolve()
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    lines = raw.decode("utf-8", errors="replace").splitlines()
    config: dict[str, Any] = {}
    split_days: dict[str, list[int]] = {}
    results: list[dict[str, Any]] = []
    latest_by_tag: dict[str, dict[str, Any]] = {}
    orphan_days: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        cfg_match = _CONFIG.search(line)
        if cfg_match:
            key, value = cfg_match.groups()
            parsed = _config_value(value)
            if key in config and config[key] != parsed:
                raise LogBindingError(f"Conflicting log configuration {key} at line {line_number}; split concatenated logs first")
            config[key] = parsed
        split_match = _DAY_SPLIT.search(line)
        if split_match:
            split_days[split_match.group(1)] = list(ast.literal_eval(split_match.group(2)))
        result_match = _RESULT.search(line)
        if result_match:
            family, tag, payload = result_match.groups()
            result = {"family": family, "tag": tag, "source_line": line_number,
                      "values": {k: float(v) for k, v in _NUMBERS.findall(payload)},
                      "payload": payload, "day_values": {}, "day_lines": {}}
            readout_match = re.search(r"(?:^|\s)readout=([^\s|]+)", payload)
            result["readout"] = readout_match.group(1) if readout_match else (tag.split("__readout_", 1)[1] if "__readout_" in tag else "")
            results.append(result)
            latest_by_tag[tag] = result
            continue
        day_match = _DAY_RESULT.search(line)
        if day_match:
            tag, payload = day_match.groups()
            values = {k: float(v) for k, v in _NUMBERS.findall(payload)}
            result = latest_by_tag.get(tag)
            if result is None:
                orphan_days.append({"tag": tag, "source_line": line_number, "values": values})
                continue
            for key, value in values.items():
                if key in result["day_values"]:
                    previous = result["day_values"][key]
                    if not ((math.isnan(value) and math.isnan(previous)) or value == previous):
                        raise LogBindingError(f"Conflicting {key} for {tag!r} at line {line_number}")
                result["day_values"][key] = value
                result["day_lines"][key] = line_number
    _validate_metadata(entry, config, path)
    if not results:
        raise LogBindingError(f"No authoritative RESULT lines found in {path}; epoch validation logs are not test results")
    fm_results = [r for r in results if r["family"] == "FM"]
    selected_fm: dict[str, Any] | None = None
    requested_tag = entry.get("fm_result_tag")
    requested_readout = entry.get("fm_readout")
    if fm_results and entry.get("fm_model"):
        candidates = fm_results
        if requested_tag:
            candidates = [r for r in candidates if r["tag"] == str(requested_tag)]
        if requested_readout:
            candidates = [r for r in candidates if r["readout"] == str(requested_readout)]
        if entry.get("result_line") is not None:
            candidates = [r for r in candidates if r["source_line"] == int(entry["result_line"])]
        if len(candidates) != 1:
            choices = [f"{r['tag']} (line {r['source_line']})" for r in fm_results]
            raise LogBindingError(f"FM selection matched {len(candidates)} RESULT records. Bind fm_result_tag/fm_readout and, for repeated tags, result_line. Available: {choices}")
        selected_fm = candidates[0]
        if "dir_reverse" in selected_fm["tag"] or re.search(r"(?:^|[_\-])v7(?:[_\-]|$)", path.name):
            raise LogBindingError("Selected FM record is reverse/bidirectional; this benchmark accepts forward v6 only")
    model_map = entry.get("model_map", {})
    if not isinstance(model_map, Mapping):
        raise LogBindingError("model_map must map exact TRANS result tags to canonical model IDs")
    trans_results = [r for r in results if r["family"] == "TRANS"]
    selected_trans: list[tuple[dict[str, Any], str]] = []
    trans_by_id: dict[str, list[dict[str, Any]]] = {}
    for result in trans_results:
        model = _trans_id(result["tag"], str(entry["panel"]), model_map)
        if model is not None:
            trans_by_id.setdefault(model, []).append(result)
    trans_line_selectors = entry.get("trans_result_lines", {})
    for model, candidates in trans_by_id.items():
        wanted_line = trans_line_selectors.get(model) if isinstance(trans_line_selectors, Mapping) else None
        if wanted_line is not None:
            candidates = [r for r in candidates if r["source_line"] == int(wanted_line)]
        elif len(candidates) > 1 and entry.get("result_line") is not None and not fm_results:
            candidates = [r for r in candidates if r["source_line"] == int(entry["result_line"])]
        if len(candidates) != 1:
            raise LogBindingError(f"Repeated TRANS RESULT records for {model}: bind trans_result_lines[{model!r}] to one source line; no last/best row is selected")
        selected_trans.append((candidates[0], model))
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    consistency: list[dict[str, Any]] = []
    selected_ids = {id(result): model for result, model in selected_trans}
    if selected_fm is not None:
        selected_ids[id(selected_fm)] = str(entry["fm_model"])
    base = {
        "dataset": str(entry["dataset"]), "panel": str(entry["panel"]),
        "route": "training_log", "source_path": str(path), "source_sha256": digest,
        "seed": config.get("SEED", ""), "dataset_random_seed": config.get("DATASET_RANDOM_SEED", ""),
        "seq_len": config.get("SEQ_LEN", ""), "split": config.get("EVAL_SPLIT", "unknown"),
        "pcc_nan_policy": config.get("PCC_NAN_POLICY", "unknown"), "pcc_eps": config.get("PCC_EPS", "unknown"),
        "pcc_backend_reported": config.get("PCC_BACKEND", "unknown"),
        "pcc_min_valid_fraction": config.get("PCC_MIN_VALID_FRACTION", "unknown"),
        "timestamp_semantics": "original_biological_ordinal_time_zero_based",
        "n_genes": config.get("N_GENES", ""), "precision": "printed_log_4_decimals_for_pcc",
    }
    dimensions = re.search(r"Loaded X with N_CELLS=(\d+), N_GENES=(\d+)", "\n".join(lines))
    if dimensions:
        base["n_genes"] = int(dimensions.group(2))
    fm_counts = re.search(r"\[FM\] dataset sizes train=(\d+) val=(\d+) test=(\d+)", "\n".join(lines))
    for result in results:
        result_base = dict(base, family=result["family"], result_tag=result["tag"],
                           readout=result["readout"], model=selected_ids.get(id(result), ""),
                           selected_for_benchmark=id(result) in selected_ids)
        for source_key, value in result["values"].items():
            records.append(dict(result_base, source_key=source_key, metric=source_key,
                                value=value, scope="diagnostic", source_line=result["source_line"]))
        for source_key, value in result["day_values"].items():
            records.append(dict(result_base, source_key=source_key, metric=source_key,
                                value=value, scope="diagnostic_day", source_line=result["day_lines"][source_key]))
        for axis in ("cell", "gene"):
            for unique in (False, True):
                if result["family"] == "FM" and unique:
                    continue
                day_keys = [k for k in result["day_values"] if _DAY_KEY.match(k) and
                            _DAY_KEY.match(k).group(2) == axis and bool(_DAY_KEY.match(k).group(3)) == unique]
                if result["family"] == "FM":
                    aggregate_key = f"mean_{axis}_pcc"
                elif unique:
                    aggregate_key = f"unique_{axis}_pcc"
                    if aggregate_key not in result["values"]:
                        aggregate_key = f"mean_{axis}_pcc_unique"
                else:
                    aggregate_key = f"occurrence_{axis}_pcc"
                    if aggregate_key not in result["values"]:
                        aggregate_key = f"mean_{axis}_pcc"
                if aggregate_key in result["values"] and day_keys:
                    aggregate = result["values"][aggregate_key]
                    mean_days = sum(result["day_values"][k] for k in day_keys) / len(day_keys)
                    difference = abs(aggregate - mean_days)
                    consistency.append({"result_tag": result["tag"], "source_line": result["source_line"],
                                        "source_key": aggregate_key, "printed_aggregate": aggregate,
                                        "mean_printed_days": mean_days, "difference": difference,
                                        "within_rounding_tolerance": bool(math.isfinite(difference) and difference <= 1e-4 + 1e-12),
                                        "aggregate_replaced": False})
        if id(result) not in selected_ids:
            continue
        days = sorted({int(_DAY_KEY.match(k).group(1)) for k in result["day_values"] if _DAY_KEY.match(k)})
        primary_weighting = "unique_target" if result["family"] == "FM" else "occurrence"
        for unique in (False, True):
            if result["family"] == "FM" and unique:
                continue
            suffix = "_unique" if unique else ""
            weighting = "unique_target" if unique else primary_weighting
            verified = _contract_verified(entry, days, weighting) if (not unique or entry.get("verify_unique_weighting", False)) else False
            contract = entry.get("comparison_contract", {}) if verified else {}
            if verified:
                cohort_contract = {k: contract[k] for k in ("target_manifest_sha256", "gene_manifest_sha256", "expression_units", "split_id")}
                cohort_contract["weighting"] = weighting
                cohort_contract["target_days"] = days
                contract_digest = hashlib.sha256(json.dumps(cohort_contract, sort_keys=True).encode()).hexdigest()[:16]
                default_cohort = f"verified_log:{contract_digest}"
            else:
                default_cohort = f"log:{digest[:16]}:{result['family']}:{weighting}"
            cohort_id = str(entry.get("cohort_id", default_cohort))
            row_base = dict(result_base, weighting=weighting, cohort_id=cohort_id,
                            comparison_status="verified" if verified else "log_population_unverified",
                            n_days=len(days), target_days=days, day="", gene="",
                            aggregation="arithmetic_mean_of_day_means", benchmark_eligible=True)
            if verified:
                row_base.update({k: contract[k] for k in ("target_manifest_sha256", "gene_manifest_sha256", "expression_units", "split_id")})
            if result["family"] == "FM" and fm_counts:
                row_base["n_cells"] = int(fm_counts.group(3))
            else:
                row_base["n_cells"] = result["values"].get("n_unique" if unique else "n_occurrence", "")
            for axis in ("cell", "gene"):
                candidates = ([f"mean_{axis}_pcc"] if result["family"] == "FM" else
                              [f"unique_{axis}_pcc", f"mean_{axis}_pcc_unique"] if unique else
                              [f"occurrence_{axis}_pcc", f"mean_{axis}_pcc"])
                source_key = next((key for key in candidates if key in result["values"]), None)
                if source_key:
                    value = result["values"][source_key]
                    rows.append(dict(row_base, scope="global", metric=f"train_log_mean_day_pcc_{axis}{suffix}",
                                     source_key=source_key, value=value, status="available" if math.isfinite(value) else "missing",
                                     definition_id=f"training_daymacro_{axis}_pcc", source_line=result["source_line"]))
                for source_key, value in result["day_values"].items():
                    match = _DAY_KEY.match(source_key)
                    if match and match.group(2) == axis and bool(match.group(3)) == unique:
                        rows.append(dict(row_base, scope="day", day=int(match.group(1)),
                                         metric=f"train_log_day_pcc_{axis}{suffix}", source_key=source_key,
                                         value=value, status="available" if math.isfinite(value) else "missing",
                                         aggregation=f"mean_{axis}_pcc_within_day", definition_id=f"training_day_{axis}_pcc",
                                         source_line=result["day_lines"][source_key]))
        if result["family"] == "TRANS":
            for axis in ("cell", "gene"):
                candidates = [f"lower_{axis}_pcc", f"reported_lower_{axis}_pcc"]
                source_key = next((key for key in candidates if key in result["values"]), None)
                if source_key:
                    value = result["values"][source_key]
                    rows.append(dict(result_base, scope="global", day="", gene="",
                                     metric=f"train_log_conservative_lower_pcc_{axis}", source_key=source_key,
                                     value=value, status="available" if math.isfinite(value) else "missing",
                                     weighting="minimum_of_occurrence_and_unique_target", n_days=len(days), target_days=days,
                                     cohort_id=str(entry.get("cohort_id", f"log:{digest[:16]}:TRANS:lower")),
                                     comparison_status="log_population_unverified", benchmark_eligible=False,
                                     aggregation="minimum_of_two_global_means", definition_id=f"training_conservative_lower_{axis}_pcc",
                                     source_line=result["source_line"]))
    audit = {
        "source_path": str(path), "source_sha256": digest, "config": config,
        "split_days": split_days, "result_records": len(results), "numeric_records": len(records),
        "selected_fm_result_tag": selected_fm["tag"] if selected_fm else None,
        "selected_fm_result_line": selected_fm["source_line"] if selected_fm else None,
        "selected_trans_models": [model for _, model in selected_trans],
        "available_fm_readouts": [{"result_tag": r["tag"], "readout": r["readout"], "source_line": r["source_line"]} for r in fm_results],
        "printed_mean_consistency": consistency, "orphan_day_records": orphan_days,
        "per_gene_status": "missing_raw_matched_predictions; per-gene values cannot be recovered from mean_gene_pcc",
        "test_error_status": "RESULT FM val_loss is validation error; test_mse requires the full saved summary CSV or raw test outputs",
        "population_status": "FM unique target cells and TRANS trajectory occurrences/unique subsets need exact target manifests before fair comparison",
        "timestamp_semantics": "zero-based ordinal stage, not a physical biological day label",
        "warnings": ["Streaming multi-readout FM and dual-weighting TRANS use fast PCC regardless of the reported PCC_BACKEND option.",
                     "Printed day values and global means are independently rounded; RESULT aggregate values are preserved."],
    }
    return {"rows": rows, "records": records, "audit": audit}
