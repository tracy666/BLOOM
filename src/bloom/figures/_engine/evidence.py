"""Machine-readable evidence indexes, without automated figure/model selection."""

from __future__ import annotations
import csv, json
from pathlib import Path
from .plot import source_panel, model_label

SEMANTICS = {
    "grn": {
        "within_age_residual_correlation": "Association; agreement with matched GT is favorable; sign alone is not quality.",
        "absolute_association_error": "abs(model association - matched GT association); lower is closer to GT.",
        "baseline_error_minus_model_error": "Positive means closer to GT than the named current baseline; no historical ablation is inferred.",
    },
    "density": {
        "density_residual": "Weighted generated probability minus matched GT probability on fixed bins; near zero is local occupancy agreement, conditional on the reference and sample."
    },
    "gene": {
        "mean": "Saved native expression mean; match GT pattern and scale for measured targets. No universal larger-is-better direction.",
        "pct": "Fraction of stored values >0; may not represent RNA detection after scaling.",
        "std": "Cell dispersion, not experimental uncertainty.",
    },
    "program": {
        "weighted_coverage": "Covered absolute weight / full supplied absolute weight; inspect before comparing scores.",
        "score": "Covered normalized supplied membership score; reference scaling is reported separately.",
    },
    "classifier": {
        "heldout_truth_balanced_accuracy": "Reliability on held-out observed truth must support interpreting generated recoverability.",
        "generated_target_label_recoverability": "Agreement with requested/template labels, conditional on classifier reliability; not independent fate truth.",
    },
    "drift": {
        "distance_to_reference_stage": "Descriptive change from the documented boundary; smaller is not automatically better for development.",
        "observed_reference_pc_distance": "Reference support diagnostic; large distance flags extrapolation, not proven failure or discovery.",
    },
    "perturbation": {
        "delta": "Edited prediction minus baseline prediction in native saved units; no intervention truth.",
        "mean_response": "Predicted program response; coverage and direct-target behavior must be inspected.",
    },
}


def _write(path, rows, fields):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_evidence(output, result, cohort, question=None, options=None):
    """Index actual tables and exports. Does not infer superiority from a plot."""
    output = Path(output)
    question = question or {}
    options = options or {}
    table_rows = []
    for p in sorted(output.iterdir()):
        if p.suffix.lower() not in (".csv", ".tsv", ".parquet") or p.name in (
            "FIGURE_EVIDENCE.csv",
            "TABLE_CATALOGUE.csv",
        ):
            continue
        columns = []
        rows = None
        try:
            if p.suffix == ".parquet":
                import pyarrow.parquet as pq

                meta = pq.ParquetFile(p)
                rows = meta.metadata.num_rows
                columns = meta.schema_arrow.names
            else:
                with p.open(newline="") as f:
                    reader = csv.reader(f, delimiter="\t" if p.suffix == ".tsv" else ",")
                    columns = next(reader, [])
                    rows = sum(1 for _ in reader)
        except (OSError, ValueError):
            pass
        table_rows.append(
            dict(
                table=p.name,
                format=p.suffix[1:],
                n_rows=rows,
                columns=json.dumps(columns),
                role=(
                    "selection audit"
                    if "selection" in p.name.lower() or "selected_entities" in p.name
                    else "numerical evidence/provenance"
                ),
            )
        )
    _write(
        output / "TABLE_CATALOGUE.csv", table_rows, ["table", "format", "n_rows", "columns", "role"]
    )
    sources = cohort.get("sources", [])
    model_rows = []
    for s in sources:
        model_rows.append(
            {
                k: s.get(k)
                for k in (
                    "model",
                    "role",
                    "source_id",
                    "path",
                    "artifact_dir",
                    "reference_id",
                    "scale_group",
                    "n_total",
                    "n_sample",
                )
            }
            | {"native_panel": source_panel(s), "display_name": model_label(s, options)}
        )
    question_ids = question.get("question_ids", result.get("question_ids", []))
    scenario = cohort.get("scenario", "unknown")
    has_gt = any(s.get("role") == "GT" for s in sources)
    paired = has_gt and scenario in ("measured", "reverse")
    evidence_rows = []
    for record in result.get("figures", []):
        if not isinstance(record, dict):
            record = {"path": str(record)}
        p = Path(record.get("path", ""))
        if not p.is_file():
            continue
        view = record.get("view", question.get("family", "unknown"))
        metrics = {}
        for word, statements in SEMANTICS.items():
            if word in view or (
                word == "gene"
                and view in ("marker_dotplot", "marker_matrix", "marker_violin", "cascade")
            ):
                metrics.update(statements)
        evidence_rows.append(
            dict(
                figure=p.name,
                view=view,
                entity=record.get("entity", ""),
                model=record.get("model", "comparison"),
                dataset=cohort.get("dataset"),
                native_panel=cohort.get("panel"),
                scenario=scenario,
                questions=";".join(question_ids),
                interpretation=(
                    "matched measured fidelity"
                    if paired
                    else (
                        "saved classifier diagnostic"
                        if scenario == "saved_diagnostics"
                        else "model plausibility/sensitivity; no independent generated-time truth"
                    )
                ),
                ground_truth_present=has_gt,
                model_sources_json=json.dumps(model_rows),
                numerical_tables=";".join(r["table"] for r in table_rows),
                metric_meaning_json=json.dumps(metrics),
                guide="GUIDE.md",
                automatic_winner="not assigned",
            )
        )
    fields = [
        "figure",
        "view",
        "entity",
        "model",
        "dataset",
        "native_panel",
        "scenario",
        "questions",
        "interpretation",
        "ground_truth_present",
        "model_sources_json",
        "numerical_tables",
        "metric_meaning_json",
        "guide",
        "automatic_winner",
    ]
    _write(output / "FIGURE_EVIDENCE.csv", evidence_rows, fields)
    return [str(output / "FIGURE_EVIDENCE.csv"), str(output / "TABLE_CATALOGUE.csv")]
