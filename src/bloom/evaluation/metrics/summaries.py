"""Parse full saved v6 training summary CSVs, preserving their precision.

The CSV is preferable to the rounded RESULT log when available.  It also
contains numeric diagnostics the log's truncated DataFrame display omits.
Validation losses remain diagnostics; only an explicit ``test_mse`` field is
exported as a test-error benchmark value.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from .logs import LogBindingError, _DAY_KEY, _contract_verified, _trans_id, _validate_metadata


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return None


def _true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _seed(row: Mapping[str, Any], entry: Mapping[str, Any]) -> Any:
    known = next((row[key] for key in ("seed", "SEED", "random_seed", "training_seed") if row.get(key) not in (None, "")), None)
    supplied = entry.get("seed")
    if known is not None and supplied is not None and str(known) != str(supplied):
        try:
            equal = float(known) == float(supplied)
        except (ValueError, TypeError):
            equal = False
        if not equal:
            raise LogBindingError(f"Summary seed={known!r} conflicts with explicit entry seed={supplied!r}; use summary_seed to select a row")
    result = known if known is not None else supplied
    if result is None:
        return ""
    numeric = _number(result)
    return int(numeric) if numeric is not None and math.isfinite(numeric) and numeric.is_integer() else result


def _cohort(entry: Mapping[str, Any], digest: str, family: str, weighting: str, days: list[int]) -> tuple[str, str, dict[str, Any]]:
    verified = _contract_verified(entry, days, weighting)
    metadata = {}
    if verified:
        contract = entry["comparison_contract"]
        metadata = {key: contract[key] for key in ("target_manifest_sha256", "gene_manifest_sha256", "expression_units", "split_id")}
        coherent = dict(metadata, weighting=weighting, target_days=days)
        default = "verified_summary:" + hashlib.sha256(json.dumps(coherent, sort_keys=True).encode()).hexdigest()[:16]
    else:
        default = f"summary:{digest[:16]}:{family}:{weighting}"
    return str(entry.get("cohort_id", default)), "verified" if verified else "summary_population_unverified", metadata


def parse_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical rows, every numeric CSV record, and a provenance audit.

    Entry keys match ``parse_log``.  A ``summary_seed`` filter and optional
    ``summary_row`` (one-based data row) / ``source_line`` selector can resolve
    repeated exports.  Distinct identified seeds are preserved, while multiple
    records for one model/seed are rejected rather than choosing a test winner.
    """
    path = Path(str(entry["path"])).expanduser().resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    text = gzip.decompress(payload).decode("utf-8-sig") if path.suffix == ".gz" else payload.decode("utf-8-sig")
    delimiter = "\t" if path.name.lower().endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")) else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames or "family" not in reader.fieldnames or "model" not in reader.fieldnames:
        raise LogBindingError("Full training summary requires family and model CSV columns; a truncated log DataFrame cannot supply them")
    config = dict(entry.get("metadata", {}))
    if entry.get("split") is not None:
        config["EVAL_SPLIT"] = entry["split"]
    _validate_metadata(entry, config, path)
    all_rows: list[dict[str, Any]] = []
    for data_row, raw in enumerate(reader, 1):
        if None in raw:
            raise LogBindingError(f"Malformed summary CSV at line {reader.line_num}: extra columns")
        row = {key: (value.strip() if isinstance(value, str) else value) for key, value in raw.items()}
        family = str(row.get("family", "")).upper()
        if family not in {"FM", "TRANS"}:
            continue
        tag = str(row["model"])
        readout = str(row.get("fm_readout") or (tag.split("__readout_", 1)[1] if "__readout_" in tag else ""))
        if "__readout_" in tag and readout != tag.split("__readout_", 1)[1]:
            raise LogBindingError(f"Summary fm_readout conflicts with model suffix at data row {data_row}")
        record = {"raw": row, "family": family, "tag": tag, "readout": readout,
                  "source_line": reader.line_num, "summary_row": data_row, "seed": _seed(row, entry),
                  "numeric": {key: number for key, value in row.items() if (number := _number(value)) is not None}}
        all_rows.append(record)
    if not all_rows:
        raise LogBindingError("No FM or TRANS rows found in full training summary")
    candidates_fm = [r for r in all_rows if r["family"] == "FM"]
    selected_fm: list[dict[str, Any]] = []
    if candidates_fm and entry.get("fm_model"):
        selected_fm = candidates_fm
        if entry.get("fm_result_tag"):
            selected_fm = [r for r in selected_fm if r["tag"] == str(entry["fm_result_tag"])]
        if entry.get("fm_readout"):
            selected_fm = [r for r in selected_fm if r["readout"] == str(entry["fm_readout"])]
        if entry.get("summary_seed") is not None:
            selected_fm = [r for r in selected_fm if str(r["seed"]) == str(entry["summary_seed"])]
        if entry.get("summary_row") is not None:
            selected_fm = [r for r in selected_fm if r["summary_row"] == int(entry["summary_row"])]
        if entry.get("source_line") is not None:
            selected_fm = [r for r in selected_fm if r["source_line"] == int(entry["source_line"])]
        if not selected_fm or len({(r["tag"], r["readout"]) for r in selected_fm}) != 1:
            choices = [{"tag": r["tag"], "readout": r["readout"], "source_line": r["source_line"], "seed": r["seed"]} for r in candidates_fm]
            raise LogBindingError(f"FM summary selection is missing or ambiguous. Bind one exact fm_result_tag/fm_readout. Available: {choices}")
        if any("dir_reverse" in r["tag"] for r in selected_fm) or re.search(r"(?:^|[_\-])v7(?:[_\-]|$)", path.name):
            raise LogBindingError("Reverse/bidirectional FM summary is excluded from this forward benchmark")
    model_map = entry.get("model_map", {})
    if not isinstance(model_map, Mapping):
        raise LogBindingError("Summary model_map must map exact tags to canonical TRANS IDs")
    selected: list[tuple[dict[str, Any], str]] = [(r, str(entry["fm_model"])) for r in selected_fm]
    for record in all_rows:
        if record["family"] != "TRANS":
            continue
        model = _trans_id(record["tag"], str(entry["panel"]), model_map)
        if model is None:
            continue
        if entry.get("summary_seed") is not None and str(record["seed"]) != str(entry["summary_seed"]):
            continue
        selector = entry.get("trans_summary_rows", {}).get(model) if isinstance(entry.get("trans_summary_rows", {}), Mapping) else None
        if selector is not None and record["summary_row"] != int(selector):
            continue
        selected.append((record, model))
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for record, model in selected:
        key = (model, str(record["seed"]))
        if key in seen:
            previous = seen[key]
            raise LogBindingError(f"Repeated summary result for model={model}, seed={record['seed']!r} at lines {previous['source_line']} and {record['source_line']}; bind summary_row/trans_summary_rows explicitly")
        seen[key] = record
    selected_ids = {id(record): model for record, model in selected}
    canonical: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    consistency: list[dict[str, Any]] = []
    base = {"dataset": str(entry["dataset"]), "panel": str(entry["panel"]),
            "route": "training_summary", "source_path": str(path), "source_sha256": digest,
            "precision": "full_saved_csv_precision", "split": config.get("EVAL_SPLIT", "unknown"),
            "normalization": entry.get("normalization", "unknown"),
            "dataset_random_seed": config.get("DATASET_RANDOM_SEED", entry.get("dataset_random_seed", "")),
            "seq_len": config.get("SEQ_LEN", entry.get("seq_len", "")),
            "pcc_nan_policy": config.get("PCC_NAN_POLICY", entry.get("pcc_nan_policy", "unknown")),
            "pcc_eps": config.get("PCC_EPS", entry.get("pcc_eps", "unknown")),
            "pcc_backend_reported": config.get("PCC_BACKEND", "unknown"),
            "pcc_min_valid_fraction": config.get("PCC_MIN_VALID_FRACTION", "unknown"),
            "timestamp_semantics": "original_biological_ordinal_time_zero_based"}
    for record in all_rows:
        selected_model = selected_ids.get(id(record), "")
        row_base = dict(base, family=record["family"], model=selected_model,
                        seed=record["seed"], readout=record["readout"], result_tag=record["tag"],
                        source_line=record["source_line"], summary_row=record["summary_row"],
                        selected_for_benchmark=bool(selected_model))
        raw = record["raw"]
        numeric = record["numeric"]
        is_excluded = _true(raw.get("skipped")) or _true(raw.get("crashed"))
        for key, value in numeric.items():
            match = re.match(r"^day(-?\d+)_", key)
            records.append(dict(row_base, source_key=key, metric=key, value=value,
                                day=int(match.group(1)) if match else "", gene="",
                                scope="diagnostic_day" if match else "diagnostic",
                                status="excluded" if is_excluded else "available" if math.isfinite(value) else "missing"))
        if not selected_model:
            continue
        days = sorted({int(_DAY_KEY.match(key).group(1)) for key in numeric if _DAY_KEY.match(key)})
        family = record["family"]
        for unique in (False, True):
            if family == "FM" and unique:
                continue
            suffix = "_unique" if unique else ""
            weighting = "unique_target" if family == "FM" or unique else "occurrence"
            contract_entry = dict(entry)
            if unique and not entry.get("verify_unique_weighting", False):
                contract_entry.pop("comparison_contract", None)
            cohort_id, comparison_status, contract_fields = _cohort(contract_entry, digest, family, weighting, days)
            metric_base = dict(row_base, cohort_id=cohort_id, comparison_status=comparison_status,
                               weighting=weighting, n_days=len(days), target_days=days,
                               n_cells=numeric.get("n_cells_unique" if unique else "n_cells_occurrence", numeric.get("n_cells", entry.get("n_cells", ""))),
                               n_genes=numeric.get("n_genes", entry.get("n_genes", "")),
                               day="", gene="", benchmark_eligible=True,
                               **contract_fields)
            for axis in ("cell", "gene"):
                key = f"mean_{axis}_pcc{suffix}"
                if key not in numeric:
                    key = (f"unique_{axis}_pcc" if unique else f"occurrence_{axis}_pcc")
                if key in numeric:
                    value = numeric[key]
                    status = "excluded" if is_excluded else "available" if math.isfinite(value) else "missing"
                    canonical.append(dict(metric_base, scope="global", metric=f"train_log_mean_day_pcc_{axis}{suffix}",
                                          source_key=key, value=value, status=status,
                                          aggregation="arithmetic_mean_of_day_means", definition_id=f"training_daymacro_{axis}_pcc"))
                    valid_days = [value for source_key, value in numeric.items() if (match := _DAY_KEY.match(source_key)) and match.group(2) == axis and bool(match.group(3)) == unique and math.isfinite(value)]
                    if valid_days and math.isfinite(value):
                        mean_days = math.fsum(valid_days) / len(valid_days)
                        difference = abs(value - mean_days)
                        # A full CSV produced by pandas stores unrounded floats;
                        # epsilon permits floating representation/accumulation.
                        consistency.append({"result_tag": record["tag"], "seed": record["seed"], "source_key": key,
                                            "aggregate": value, "mean_days": mean_days, "difference": difference,
                                            "within_full_precision_tolerance": difference <= 1e-10,
                                            "aggregate_replaced": False})
                for key, value in numeric.items():
                    match = _DAY_KEY.match(key)
                    if match and match.group(2) == axis and bool(match.group(3)) == unique:
                        canonical.append(dict(metric_base, scope="day", day=int(match.group(1)),
                                              metric=f"train_log_day_pcc_{axis}{suffix}", source_key=key, value=value,
                                              status="excluded" if is_excluded else "available" if math.isfinite(value) else "missing",
                                              aggregation=f"mean_{axis}_pcc_within_day", definition_id=f"training_day_{axis}_pcc"))
            if not unique and "test_mse" in numeric:
                value = numeric["test_mse"]
                canonical.append(dict(metric_base, scope="global", metric="train_test_mse", source_key="test_mse", value=value,
                                      status="excluded" if is_excluded else "available" if math.isfinite(value) else "missing",
                                      aggregation="mean_squared_error_over_all_matrix_entries", definition_id="training_test_mse"))
        if family == "TRANS":
            for axis in ("cell", "gene"):
                key = f"reported_lower_{axis}_pcc"
                if key not in numeric:
                    key = f"lower_{axis}_pcc"
                if key in numeric:
                    value = numeric[key]
                    canonical.append(dict(row_base, scope="global", day="", gene="",
                                          metric=f"train_log_conservative_lower_pcc_{axis}", source_key=key, value=value,
                                          status="excluded" if is_excluded else "available" if math.isfinite(value) else "missing",
                                          weighting="minimum_of_occurrence_and_unique_target", benchmark_eligible=False,
                                          cohort_id=str(entry.get("cohort_id", f"summary:{digest[:16]}:TRANS:lower")),
                                          comparison_status="summary_population_unverified", n_days=len(days), target_days=days,
                                          aggregation="minimum_of_two_global_means", definition_id=f"training_conservative_lower_{axis}_pcc"))
    audit = {
        "source_path": str(path), "source_sha256": digest, "columns": reader.fieldnames,
        "data_rows": len(all_rows), "numeric_records": len(records),
        "selected_fm": [{"result_tag": r["tag"], "readout": r["readout"], "seed": r["seed"], "source_line": r["source_line"]} for r in selected_fm],
        "selected_trans_models": sorted({model for r, model in selected if r["family"] == "TRANS"}),
        "available_fm_readouts": [{"result_tag": r["tag"], "readout": r["readout"], "seed": r["seed"], "source_line": r["source_line"]} for r in candidates_fm],
        "mean_consistency": consistency,
        "selected_missing_fields": [{"model": model, "seed": r["seed"], "source_line": r["source_line"],
                                     "fields": [key for key in ("mean_cell_pcc", "mean_gene_pcc", "test_mse") if key not in r["numeric"]]}
                                    for r, model in selected],
        "per_gene_status": "not_present_in_training_summary; exact per-gene metrics require matched expression matrices",
        "diagnostics": "All numeric columns retained, including validation selection fields, test composition, gates, counts and run time. Only explicit test_mse is exported as test error.",
        "timestamp_semantics": "zero-based ordinal stage, not physical biological day",
        "population_status": "Exact target-cell manifests, genes, units, split and weighting must match before comparison",
        "split_status": "explicit_test" if base["split"] == "test" else "not_recorded; verify the originating log EVAL_SPLIT before interpreting fields named test_mse",
    }
    return {"rows": canonical, "records": records, "audit": audit}
