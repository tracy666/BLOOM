"""Load selected global endpoints without combining incompatible definitions."""

from contextlib import contextmanager
from pathlib import Path
import csv
import hashlib
import json
import math
import tempfile
import zipfile

DATASETS = ("big", "small", "sted")
PANELS = ("hvg", "author")
TRANS = (
    "Autoformer",
    "Crossformer",
    "DLinear",
    "FEDformer",
    "FiLM",
    "FreTS",
    "Informer",
    "iTransformer",
    "LightTS",
    "Mamba",
    "Nonstationary_Transformer",
    "PatchTST",
    "PAttn",
    "Reformer",
    "SegRNN",
    "TiDE",
    "TimeFilter",
    "TimesNet",
    "TimeXer",
    "Transformer",
    "TSMixer",
    "WPMixer",
)


def models(panel):
    return [
        "v6_200_new" if panel == "hvg" else "v6_author",
        *[m + ("_200" if panel == "hvg" else "_author") for m in TRANS],
    ]


def label(model):
    if model in ("v6_200_new", "v6_author"):
        return "BLOOM"
    return (
        model.removesuffix("_200")
        .removesuffix("_author")
        .replace("Nonstationary_Transformer", "Nonstationary Transformer")
    )


def write_csv(path, rows, preferred=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys([*preferred, *[k for row in rows for k in row]])) or [
        "status",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    k: (
                        json.dumps(v, ensure_ascii=False, sort_keys=True)
                        if isinstance(v, (list, dict))
                        else "" if isinstance(v, float) and not math.isfinite(v) else v
                    )
                    for k, v in row.items()
                }
            )


def write_json(path, obj):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        if isinstance(x, Path):
            return str(x)
        if hasattr(x, "item"):
            return clean(x.item())
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


@contextmanager
def benchmark_input(path, include_report=True):
    """Support the user's ZIP while skipping enormous duplicated ALL_LONG tables."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if not (path / "GLOBAL_METRICS.csv").is_file():
            candidates = [p.parent for p in path.glob("*/GLOBAL_METRICS.csv")]
            if len(candidates) != 1:
                raise ValueError("Point --input at a benchmark folder with GLOBAL_METRICS.csv")
            path = candidates[0]
        yield path
        return
    if not path.is_file() or not zipfile.is_zipfile(path):
        raise ValueError("Input must be a benchmark directory or ZIP")
    names = {
        "GLOBAL_METRICS.csv",
        "GLOBAL_METRIC_SUMMARY.csv",
        "RUN_STATUS.json",
        "RUN_CONFIG.json",
        "METRIC_SUPPORT.csv",
    }
    if include_report:
        names.update({"PER_DAY_METRICS.csv", "PER_GENE_METRICS.csv", "PER_DAY_GENE_METRICS.csv"})
    with tempfile.TemporaryDirectory(prefix="bloom_benchmark_input_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(path) as z:
            candidates = [
                n
                for n in z.namelist()
                if Path(n).name == "GLOBAL_METRICS.csv" and "__MACOSX" not in Path(n).parts
            ]
            if candidates:
                depth = min(len(Path(n).parts) for n in candidates)
                candidates = [n for n in candidates if len(Path(n).parts) == depth]
            if len(candidates) != 1:
                raise ValueError(
                    "ZIP must contain exactly one top-level benchmark GLOBAL_METRICS.csv"
                )
            prefix = str(Path(candidates[0]).parent)
            for name in sorted(names):
                member = name if prefix == "." else prefix + "/" + name
                if member not in z.namelist():
                    continue
                with z.open(member) as src, (root / name).open("wb") as dst:
                    import shutil

                    shutil.copyfileobj(src, dst, 1024 * 1024)
        yield root


def load_globals(root, plan, dataset="all", panel="all", mode="strict"):
    """One declared source per metric/model; strict comparisons require common GT."""
    root = Path(root)
    with (root / "GLOBAL_METRICS.csv").open(newline="", encoding="utf-8-sig") as f:
        source = list(csv.DictReader(f))
    rows = []
    issues = []
    coverage = []
    fields = (
        "definition_id",
        "aggregation",
        "weighting",
        "split",
        "normalization",
        "pcc_nan_policy",
        "pcc_eps",
        "pcc_min_valid_fraction",
        "variance_ddof",
        "target_days",
    )
    datasets = DATASETS if dataset == "all" else (dataset,)
    panels = PANELS if panel == "all" else (panel,)
    for ds in datasets:
        for assay in panels:
            expected = models(assay)
            for spec in plan["metrics"]:
                metric = spec["name"]
                route = spec.get("route", "recomputed")
                chosen = []
                absent = []
                ambiguous = []
                for model in expected:
                    found = [
                        r
                        for r in source
                        if r.get("dataset") == ds
                        and r.get("panel") == assay
                        and r.get("model") == model
                        and r.get("metric") == metric
                        and r.get("route") == route
                        and r.get("scope") == "global"
                        and r.get("status") == "available"
                    ]
                    # Optional exact per-setting source selections, never a best-test selector.
                    binding = (
                        plan.get("source_bindings", {}).get(f"{ds}/{assay}", {}).get(model, {})
                    )
                    found = [
                        r
                        for r in found
                        if all(str(r.get(k, "")) == str(v) for k, v in binding.items())
                    ]
                    found = [r for r in found if _finite(r.get("value"))]
                    if len(found) != 1:
                        (absent if not found else ambiguous).append(model)
                    else:
                        chosen.append(found[0])
                contracts = {tuple(str(r.get(k, "")) for k in fields) for r in chosen}
                cohorts = {r.get("cohort_id", "") for r in chosen}
                complete = len(chosen) == len(expected)
                trusted = bool(chosen) and all(
                    r.get("comparison_status") == "verified" for r in chosen
                )
                same_contract = len(contracts) == 1
                matched = complete and trusted and same_contract and len(cohorts) == 1
                usable = matched or (
                    mode == "descriptive" and complete and trusted and same_contract
                )
                evidence = (
                    "verified_matched"
                    if matched
                    else "descriptive_different_targets" if usable else "blocked"
                )
                info = {
                    "dataset": ds,
                    "panel": assay,
                    "metric": metric,
                    "route": route,
                    "n_expected_models": len(expected),
                    "n_available_models": len(chosen),
                    "cohort_ids": sorted(cohorts),
                    "missing_models": absent,
                    "ambiguous_models": ambiguous,
                    "same_definition_contract": same_contract,
                    "comparison_status": evidence,
                    "target_counts": {label(r["model"]): r.get("n_cells", "") for r in chosen},
                }
                coverage.append(info)
                if not matched:
                    issues.append(
                        {
                            **info,
                            "reason": (
                                "Native evaluation coverage: BLOOM all cells and TRANS trajectory-covered cells; target counts retained"
                                if usable
                                else "A complete common GT cohort and one consistent definition/source per model are required"
                            ),
                        }
                    )
                if not usable:
                    continue
                for r in chosen:
                    rows.append(
                        {
                            **r,
                            "value": float(r["value"]),
                            "model_label": label(r["model"]),
                            "metric_label": spec.get("label", metric),
                            "aspect": spec["aspect"],
                            "direction": spec["direction"],
                            "comparison_status": evidence,
                            "source_comparison_status": r.get("comparison_status", ""),
                            "descriptive_comparison_authorized": mode == "descriptive",
                            "expected_models": expected,
                        }
                    )
    condition = []
    cfg = root / "RUN_CONFIG.json"
    if cfg.is_file():
        data = json.loads(cfg.read_text())
        for r in data.get("raw_sources", []):
            if r.get("model") not in models(r.get("panel", "hvg")):
                continue
            h = r.get("header_metadata", {})
            condition.append(
                {
                    "dataset": r.get("dataset"),
                    "panel": r.get("panel"),
                    "model": r.get("model"),
                    "model_label": label(r.get("model", "")),
                    "source_path": r.get("path"),
                    "readout": r.get("readout"),
                    "checkpoint_path": h.get("checkpoint_path", ""),
                    "uses_conditioned_target_celltype": h.get(
                        "uses_conditioned_target_celltype", ""
                    ),
                    "uses_conditioned_target_cell_indexed_priors": h.get(
                        "uses_conditioned_target_cell_indexed_priors", ""
                    ),
                    "declared_conditioned_target_inputs": h.get("conditioned_target_inputs", ""),
                    "note": "Header provenance claim; target conditioning is part of the evaluation setup, not an unconditional forecast claim.",
                }
            )
    return {
        "rows": rows,
        "issues": issues,
        "coverage": coverage,
        "conditions": condition,
        "input_sha256": _hash(root / "GLOBAL_METRICS.csv"),
    }


def _finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()
