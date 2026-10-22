"""Redraw actual V5 diagnostic tables without requiring their large H5 files.

This is a read-only, native-assay route. Copied folders require an explicit
identity; generated label recoverability is never promoted to forecast accuracy.
"""

from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from . import plot
from .discovery import (
    _modern_identity,
    _logical_absolute,
    dataset_name,
    panel_name,
    FM_MODELS,
    TRANS_MODELS,
)

FILES = {
    "classifier_confusion": [
        "confusion_heldout_truth.csv",
        "confusion_generated_target_label_recoverability.csv",
    ],
    "classifier_recoverability": ["endpoint_metrics.csv"],
    "classifier_composition": ["state_composition.tsv"],
    "temporal_descriptors": ["temporal_descriptors.tsv"],
    "rank_stability": ["adjacent_rank_stability.tsv"],
    "input_sensitivity": ["input_sensitivity.tsv"],
    "deg_pathways": ["deg_biological_metrics.csv"],
    "sampling_convergence": ["sampling_convergence_summary.csv"],
}
GUIDE = """# Reading saved V5 diagnostics

These figures redraw existing V5 numerical evidence. Every comparison stays
within one native HVG or author assay. Real held-out classifier performance must
be reliable before generated target-label recoverability can be informative.
Similar poor scores are inconclusive; a classifier can recognize imposed target
labels without establishing independent fate prediction. Default V5 same-stage
splits are cell-state separability diagnostics, not held-out-stage forecasting.

Confusion colors divide by the full original row support. Omitted class mass is
audited; selected classes are never renormalized to hide the omitted errors.
State names come from the saved truth vocabulary. Composition compares saved
annotations, real-expression classifications, and generated classifications;
target type/count conditioning can make abundance agreement partly imposed.

Peak time and half-range crossing are descriptive saved-grid features. Crossing
is not validated biological onset, and increasing adjacent-rank stability can
also mean frozen expression. Unmeasured interpolation/future/history has no GT.
Downstream input-sensitivity RMS excludes the directly edited gene. It describes
frozen-model response in exported expression units, not an experimental knockout
or a causal regulatory network. Different export units/conditioning/splits must
be assessed using the saved contracts before interpreting model differences.

Optional temporal DEG tables compare changes in the exported expression scale;
their mean differences are not automatically log fold changes. Saved cell-level
q-values describe the supplied test, not specimen-level replicated significance.
Program-change tables summarize explicitly supplied gene sets; they are not
pathway enrichment unless an enrichment test was actually saved. Sampling curves
use saved repeat means/SD: finite-population estimator stability does not establish
generation beyond the exported cell count or increased biological information.
"""


def _read(path, index=False):
    return pd.read_csv(
        path, sep="\t" if Path(path).suffix == ".tsv" else ",", index_col=0 if index else None
    )


def _attested_identities(directory):
    """Read saved source paths, never reopen removed H5s or infer from genes."""
    identities = []
    for name in ("contract_audit.json", "biological_report_manifest.json"):
        path = directory / name
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        paths = [report.get("prediction_h5")]
        paths.extend(
            row.get("input_path") for row in report.get("inputs", []) if isinstance(row, dict)
        )
        for value in paths:
            if value:
                identity = _modern_identity(Path(value))
                if identity and not identity.get("excluded"):
                    identities.append(identity)
    return identities


def discover_saved_diagnostics(config, return_report=False):
    candidates = {}
    errors = []
    for value in config.get("v5_roots", []):
        root = _logical_absolute(value)
        if not root.is_dir():
            continue
        for name in (
            "07_classifier_forward",
            "07_classifier_reverse",
            "06_biological_questions",
            "03_deg_pathway",
            "11_sampling_convergence",
        ):
            for directory in root.rglob(name):
                if any(p in {"_history", "quarantine", "__MACOSX"} for p in directory.parts):
                    continue
                candidates[str(directory)] = {}
    for record in config.get("saved_diagnostic_sources", []):
        if not isinstance(record, dict) or not record.get("path"):
            continue
        root = _logical_absolute(record["path"])
        if not root.is_dir():
            errors.append(
                {"path": str(root), "reason": "explicit saved diagnostic input directory missing"}
            )
            continue
        names = (
            "07_classifier_forward",
            "07_classifier_reverse",
            "06_biological_questions",
            "03_deg_pathway",
            "11_sampling_convergence",
        )
        dirs = (
            [root]
            if root.name in names
            else [root / n for n in names]
            + [root / "01_v1_evaluation" / n for n in ("03_deg_pathway", "11_sampling_convergence")]
        )
        for directory in dirs:
            if directory.is_dir():
                candidates[str(directory)] = record
    groups = {}
    for path, explicit in sorted(candidates.items()):
        directory = Path(path)
        identity = dict(_modern_identity(directory) or {})
        if identity.get("excluded"):
            continue
        identity.update(
            {k: explicit[k] for k in ("dataset", "panel", "model", "role") if explicit.get(k)}
        )
        identity["panel"] = panel_name(identity.get("panel", "unknown"))
        identity["dataset"] = dataset_name(identity.get("dataset", "unknown"))
        if (
            identity["panel"] not in ("hvg", "author")
            or identity["dataset"] == "unknown"
            or identity.get("role") not in ("FM", "TRANS")
            or not identity.get("model")
        ):
            errors.append(
                {
                    "path": path,
                    "reason": "saved tables need a modern native slot path or explicit saved_diagnostic_sources identity",
                }
            )
            continue
        if plot.source_panel(identity) != identity["panel"]:
            errors.append(
                {"path": path, "reason": "explicit model identity conflicts with native assay"}
            )
            continue
        if identity.get("model") not in (
            FM_MODELS if identity.get("role") == "FM" else TRANS_MODELS
        ):
            errors.append(
                {
                    "path": path,
                    "reason": "saved modern gallery supports canonical current native FM/TRANS identities; legacy ablations belong to archive evidence",
                }
            )
            continue
        try:
            attested = _attested_identities(directory)
        except (ValueError, OSError) as exc:
            errors.append(
                {"path": path, "reason": "saved diagnostic source manifest unreadable: " + str(exc)}
            )
            continue
        if any(
            any(a.get(k) != identity.get(k) for k in ("dataset", "panel", "model", "role"))
            for a in attested
        ):
            errors.append(
                {
                    "path": path,
                    "reason": "native binding conflicts with saved diagnostic input-path identity",
                }
            )
            continue
        kind = (
            "biological_report"
            if directory.name == "06_biological_questions"
            else directory.name.replace("07_", "")
        )
        if directory.name == "03_deg_pathway":
            kind = "deg_pathway"
        if directory.name == "11_sampling_convergence":
            kind = "sampling_convergence"
        available = [
            view
            for view, names in FILES.items()
            if all(
                (directory / name).is_file() and (directory / name).stat().st_size > 1
                for name in names
            )
        ]
        if not available:
            continue
        source = {
            **identity,
            "path": path,
            "run_root": str(directory.parent),
            "source_id": hashlib.sha256(path.encode()).hexdigest()[:20],
            "scenario": "saved_diagnostics",
            "diagnostic_kind": kind,
            "direction": "reverse" if kind.endswith("reverse") else "forward",
            "capabilities": ["saved_diagnostics", *available],
        }
        key = (source["dataset"], source["panel"], kind)
        groups.setdefault(key, []).append(source)
    cohorts = []
    for (dataset, panel, kind), sources in sorted(groups.items()):
        # A copied duplicate table folder is not a second model.
        unique = {}
        for source in sources:
            unique.setdefault(source["model"], source)
        settings = config.get("datasets", {}).get(dataset, {})
        cohorts.append(
            {
                "key": f"{dataset}__saved_diagnostics__{kind}",
                "id": f"{dataset}_{panel}_{kind}",
                "dataset": dataset,
                "panel": panel,
                "scenario": "saved_diagnostics",
                "diagnostic_kind": kind,
                "native_panel_verified": True,
                "sources": list(unique.values()),
                "configuration": settings,
                "discovery_errors": errors,
            }
        )
    return {"cohorts": cohorts, "errors": errors} if return_report else cohorts


def discover_saved_diagnostics_report(config):
    """Inventory identity errors even when no eligible table cohort exists."""
    return discover_saved_diagnostics(config, return_report=True)


def eligibility(question, cohort, config=None):
    requested = set(question.get("views", []))
    available = set(v for s in cohort.get("sources", []) for v in s.get("capabilities", []))
    return (
        bool(requested & available),
        (
            "saved V5 tables available"
            if requested & available
            else "No actual V5 tables for the requested diagnostic family"
        ),
    )


def _cap(options, name, default):
    limit = int(options.get(name, default))
    entities = int(options.get("max_entities", 0) or 0)
    if limit < 0 or entities < 0:
        raise ValueError("Display limits must be nonnegative; zero means uncapped")
    positive = [value for value in (limit, entities) if value > 0]
    return min(positive) if positive else None


def _states(sources, options):
    support = {}
    for source in sources:
        path = Path(source["path"]) / "confusion_heldout_truth.csv"
        if path.is_file():
            frame = _read(path, True)
            for state, n in frame.sum(axis=1).items():
                support[str(state)] = support.get(str(state), 0) + float(n)
    chosen = [
        s
        for s, n in sorted(support.items(), key=lambda x: (-x[1], x[0]))
        if n > 0 and _named_state(s)
    ][: _cap(options, "max_display_states", 12)]
    audit = [
        {
            "entity": state,
            "kind": "state",
            "gt_support": n,
            "selected": state in chosen,
            "reason": (
                "ranked by real held-out truth support; actual saved vocabulary"
                if _named_state(state)
                else "unmapped numeric/unknown state code; no anatomical name invented"
            ),
        }
        for state, n in sorted(support.items())
    ]
    return chosen, audit


def _named_state(value):
    text = str(value).strip()
    return (
        text.lower() not in ("unknown", "nan", "none", "unassigned", "")
        and not text.lower().startswith("type_id:")
        and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text) is None
    )


def _genes(frames, cohort, options):
    frames = [f[f.feature_kind == "gene"].copy() if "feature_kind" in f else f for f in frames]
    author = set()
    path = Path(__file__).resolve().parent / "data/author_membership_from_supplied_code.tsv"
    if path.is_file():
        evidence = _read(path)
        evidence = (
            evidence[evidence.dataset.astype(str).map(dataset_name) == cohort["dataset"]]
            if "dataset" in evidence
            else evidence
        )
        author.update(evidence.gene.astype(str))
    universe = set(str(g) for frame in frames for g in frame.get("feature", pd.Series(dtype=str)))
    if cohort["panel"] == "author":
        author.update(universe)
    signal = {g: 0.0 for g in universe}
    observed = set()
    for frame in frames:
        if not {"feature", "range"}.issubset(frame):
            continue
        local = frame[(frame.get("source", pd.Series("prediction", index=frame.index)) == "truth")]
        if len(local):
            for g, group in local.groupby("feature"):
                signal[str(g)] = max(
                    signal.get(str(g), 0),
                    float(pd.to_numeric(group["range"], errors="coerce").max()),
                )
                observed.add(str(g))
    for frame in frames:
        if {"feature", "range"}.issubset(frame):
            for g, group in frame.groupby("feature"):
                if str(g) not in observed:
                    signal[str(g)] = max(
                        signal.get(str(g), 0),
                        float(pd.to_numeric(group["range"], errors="coerce").max()),
                    )
    eligible = universe if options.get("all_genes") else universe & author
    chosen = [
        g
        for g in sorted(eligible, key=lambda g: (-signal[g], g))
        if np.isfinite(signal[g]) and signal[g] > 0
    ][: _cap(options, "max_display_genes", 12)]
    audit = [
        {
            "entity": g,
            "kind": "gene",
            "selected": g in chosen,
            "eligible": g in eligible,
            "signal": signal[g],
            "basis": (
                "saved truth range"
                if g in observed
                else "descriptive saved prediction range; no truth for this series"
            ),
            "reason": (
                "exact supplied author membership"
                if g in author
                else (
                    "explicit exploratory all-genes option"
                    if options.get("all_genes")
                    else "no supplied biological membership"
                )
            ),
        }
        for g in sorted(universe)
    ]
    return chosen, audit


def render_question(question, cohort, output, options, progress=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plot.style()
    sources = plot.available_models(cohort, options)
    figures = []
    tables = []
    notes = []
    audit = []
    (output / "V5_TABLE_INTERPRETATION.md").write_text(GUIDE)
    interpretation = output / "INTERPRETATION.md"
    existing = interpretation.read_text() if interpretation.is_file() else ""
    if GUIDE not in existing:
        interpretation.write_text(existing + "\n\n" + GUIDE if existing else GUIDE)
    states, state_audit = _states(sources, options)
    audit.extend(state_audit)
    for view in question.get("views", []):
        eligible_sources = [s for s in sources if view in s.get("capabilities", [])]
        if not eligible_sources:
            continue
        evidence = []
        panels = []
        if view == "classifier_confusion":
            for source in eligible_sources:
                for name, endpoint in zip(
                    FILES[view], ["real held-out truth", "generated target-label recoverability"]
                ):
                    original = _read(Path(source["path"]) / name, True).astype(float)
                    denom = original.sum(axis=1).replace(0, np.nan)
                    normalized = original.div(denom, axis=0)
                    selected = [s for s in states if s in original.index and s in original.columns]
                    if not selected:
                        continue
                    visible = normalized.loc[selected, selected]
                    for state in selected:
                        evidence.append(
                            {
                                "model": source["model"],
                                "endpoint": endpoint,
                                "state": state,
                                "full_row_support": denom[state],
                                "displayed_prediction_mass": float(visible.loc[state].sum()),
                                "omitted_prediction_mass": 1 - float(visible.loc[state].sum()),
                            }
                        )

                    def callback(ax, legend=True, frame=visible):
                        art = ax.imshow(
                            frame.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto"
                        )
                        ax.set_xticks(
                            range(len(frame.columns)), frame.columns, rotation=70, fontsize=5
                        )
                        ax.set_yticks(range(len(frame)), frame.index, fontsize=5)
                        ax.set(xlabel="Classifier output", ylabel="Held-out supplied label")
                        if legend:
                            ax.figure.colorbar(art, ax=ax, fraction=0.035, pad=0.02)

                    panels.append(
                        (
                            plot.model_label(source, options) + " | " + endpoint,
                            source["role"],
                            callback,
                        )
                    )
        elif view == "classifier_recoverability":
            for source in eligible_sources:
                frame = _read(Path(source["path"]) / "endpoint_metrics.csv")
                frame = frame[
                    frame.endpoint.isin(
                        [
                            "heldout_truth",
                            "generated_target_label_recoverability",
                            "generated_prediction",
                        ]
                    )
                ].copy()
                if frame.empty:
                    continue
                summary_path = Path(source["path"]) / "diagnostic_summary.json"
                summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
                frame["model_slot"] = source["model"]
                frame["split_semantics"] = summary.get("split_semantics", "undeclared")
                evidence.extend(frame.to_dict("records"))

                def callback(ax, legend=True, frame=frame):
                    ax.barh(
                        frame.endpoint.str.replace("_", " "),
                        frame.balanced_accuracy,
                        color=["#456F9A", "#387F79"][: len(frame)],
                    )
                    ax.set(xlim=(0, 1), xlabel="Balanced accuracy / conditional recoverability")
                    ax.tick_params(axis="y", labelsize=6)
                    ax.text(
                        0.02,
                        0.02,
                        "Check real reliability first",
                        transform=ax.transAxes,
                        fontsize=6,
                    )

                panels.append((plot.model_label(source, options), source["role"], callback))
        elif view == "classifier_composition":
            for source in eligible_sources:
                frame = _read(Path(source["path"]) / "state_composition.tsv")
                selected = [s for s in states if s in set(frame.label.astype(str))]
                if not selected:
                    support = frame.groupby("label").measured_annotation_fraction.sum()
                    support = support[[_named_state(s) for s in support.index]]
                    selected = (
                        support[support > 0]
                        .nlargest(_cap(options, "max_display_states", 12) or len(support))
                        .index.tolist()
                    )
                frame = frame[frame.label.isin(selected)].copy()
                frame["model_slot"] = source["model"]
                evidence.extend(frame.to_dict("records"))
                for state in selected:
                    group = frame[frame.label == state].sort_values("time")

                    def callback(ax, legend=True, group=group):
                        for column, color in [
                            ("measured_annotation_fraction", "#456F9A"),
                            ("real_expression_classifier_fraction", "#946AA2"),
                            ("generated_expression_classifier_fraction", "#387F79"),
                        ]:
                            ax.plot(
                                group.time,
                                group[column],
                                label=column.replace("_", " "),
                                color=color,
                                lw=1.3,
                            )
                        ax.set(xlabel="Saved reporting time", ylabel="State fraction", ylim=(0, 1))
                        if legend:
                            ax.legend(fontsize=5)

                    panels.append(
                        (
                            plot.model_label(source, options) + " | " + state,
                            source["role"],
                            callback,
                        )
                    )
        elif view == "temporal_descriptors":
            frames = [_read(Path(s["path"]) / "temporal_descriptors.tsv") for s in eligible_sources]
            genes, gene_audit = _genes(frames, cohort, options)
            audit.extend(gene_audit)
            for source, frame in zip(eligible_sources, frames):
                frame = frame[frame.feature.astype(str).isin(genes)].copy()
                if "feature_kind" in frame:
                    frame = frame[frame.feature_kind == "gene"]
                if "stratum" in frame:
                    frame = frame[frame.stratum == "all_cells"]
                frame["model_slot"] = source["model"]
                evidence.extend(frame.to_dict("records"))
                for (input_id, population, scenario), group in frame.groupby(
                    ["input_id", "source", "scenario"], sort=True
                ):
                    group = group.sort_values("feature")
                    y = np.arange(len(group))
                    cross = pd.to_numeric(group.crossing_time, errors="coerce")

                    def callback(ax, legend=True, group=group, y=y, cross=cross):
                        ax.scatter(group.peak_time, y, c="#387F79", s=16, label="Saved-grid peak")
                        good = np.isfinite(cross)
                        ax.scatter(
                            cross[good],
                            y[good],
                            c="#BA7146",
                            marker="x",
                            s=18,
                            label="Half-range crossing",
                        )
                        ax.set_yticks(y, group.feature, fontsize=6)
                        ax.set(xlabel="Saved time", ylabel="Named gene")
                        if legend:
                            ax.legend(fontsize=5)

                    panels.append(
                        (
                            plot.model_label(source, options)
                            + " | "
                            + str(scenario)
                            + " | "
                            + str(population),
                            source["role"],
                            callback,
                        )
                    )
        elif view == "rank_stability":
            for source in eligible_sources:
                frame = _read(Path(source["path"]) / "adjacent_rank_stability.tsv")
                if "stratum" in frame:
                    frame = frame[frame.stratum == "all_cells"]
                frame = frame.copy()
                frame["model_slot"] = source["model"]
                evidence.extend(frame.to_dict("records"))
                for (input_id, population, scenario, kind), group in frame.groupby(
                    ["input_id", "source", "scenario", "feature_kind"], sort=True
                ):
                    group = group.sort_values("right_time")

                    def callback(ax, legend=True, group=group):
                        ax.plot(
                            group.right_time,
                            group.adjacent_rank_spearman,
                            "o-",
                            color="#387F79",
                            ms=3,
                        )
                        ax.set(
                            xlabel="Right saved time",
                            ylabel="Adjacent feature-rank correlation",
                            ylim=(-1.05, 1.05),
                        )

                    panels.append(
                        (
                            plot.model_label(source, options)
                            + " | "
                            + str(scenario)
                            + " | "
                            + str(population)
                            + " | "
                            + str(kind),
                            source["role"],
                            callback,
                        )
                    )
        elif view == "input_sensitivity":
            for source in eligible_sources:
                frame = _read(Path(source["path"]) / "input_sensitivity.tsv")
                frame = frame[
                    (frame.feature_kind == "global") & (frame.feature == "all_other_genes")
                ].copy()
                ranking = (
                    frame.groupby("intervention").rms_response.max().sort_values(ascending=False)
                )
                candidates = pd.DataFrame(
                    {"feature": ranking.index, "range": ranking.to_numpy(), "source": "prediction"}
                )
                keep, intervention_audit = _genes([candidates], cohort, options)
                frame = frame[frame.intervention.isin(keep)]
                frame["model_slot"] = source["model"]
                evidence.extend(frame.to_dict("records"))
                for intervention, group in frame.groupby("intervention", sort=True):
                    group = group.sort_values("time")

                    def callback(ax, legend=True, group=group):
                        ax.plot(group.time, group.rms_response, "o-", color="#387F79", ms=3)
                        ax.set(
                            xlabel="Saved ordinal target stage",
                            ylabel="Downstream RMS (direct target excluded)",
                        )

                    panels.append(
                        (
                            plot.model_label(source, options) + " | " + str(intervention),
                            source["role"],
                            callback,
                        )
                    )
                audit.extend(
                    {
                        **row,
                        "kind": "intervention",
                        "basis": "saved downstream RMS; direct target excluded; no perturbation GT",
                    }
                    for row in intervention_audit
                )
        elif view == "deg_pathways":
            for source in eligible_sources:
                directory = Path(source["path"])
                summary = _read(directory / "deg_biological_metrics.csv")
                if "model_tag" in summary and summary.model_tag.nunique() > 1:
                    notes.append(
                        f"{source['model']}: pooled multi-model DEG table skipped; bind a native single-model table explicitly."
                    )
                    continue
                summary = summary.copy()
                summary["model_slot"] = source["model"]
                summary["evidence_kind"] = "saved_temporal_DEG_summary"
                evidence.extend(summary.to_dict("records"))
                for path in sorted((directory / "deg_tables").glob("*__genes.csv")):
                    frame = _read(path)
                    if not {"gene", "true_delta", "predicted_delta", "true_qvalue"}.issubset(frame):
                        continue
                    candidates = pd.DataFrame(
                        {"feature": frame.gene, "range": frame.true_delta.abs(), "source": "truth"}
                    )
                    genes, gene_audit = _genes([candidates], cohort, options)
                    audit.extend(
                        {**row, "basis": "saved observed temporal mean-change magnitude"}
                        for row in gene_audit
                    )
                    frame = frame[frame.gene.isin(genes)].copy()
                    frame["model_slot"] = source["model"]
                    frame["contrast_file"] = path.name
                    frame["evidence_kind"] = "saved_temporal_gene_change"
                    evidence.extend(frame.to_dict("records"))
                    if frame.empty:
                        continue

                    def callback(ax, legend=True, frame=frame):
                        ax.scatter(frame.true_delta, frame.predicted_delta, color="#387F79", s=15)
                        limits = np.r_[
                            frame.true_delta.to_numpy(), frame.predicted_delta.to_numpy()
                        ]
                        lo, hi = np.nanmin(limits), np.nanmax(limits)
                        ax.plot([lo, hi], [lo, hi], color="#969FA7", lw=0.7)
                        for row in frame.itertuples():
                            ax.annotate(
                                str(row.gene),
                                (row.true_delta, row.predicted_delta),
                                fontsize=5,
                                xytext=(2, 2),
                                textcoords="offset points",
                            )
                        ax.set(
                            xlabel="Observed mean-expression change",
                            ylabel="Predicted mean-expression change",
                        )

                    panels.append(
                        (
                            plot.model_label(source, options)
                            + " | "
                            + path.stem.replace("__genes", ""),
                            source["role"],
                            callback,
                        )
                    )

                    def volcano(ax, legend=True, frame=frame):
                        q = pd.to_numeric(frame.true_qvalue, errors="coerce")
                        valid = np.isfinite(q) & (q > 0)
                        ax.scatter(
                            frame.loc[valid, "true_delta"],
                            -np.log10(q[valid]),
                            color="#456F9A",
                            s=15,
                        )
                        for index, row in frame.loc[valid].iterrows():
                            ax.annotate(
                                str(row.gene),
                                (row.true_delta, -np.log10(q[index])),
                                fontsize=5,
                                xytext=(2, 2),
                                textcoords="offset points",
                            )
                        ax.set(
                            xlabel="Observed change in exported expression units",
                            ylabel="−log10 saved cell-level q",
                        )

                    if np.isfinite(pd.to_numeric(frame.true_qvalue, errors="coerce")).any():
                        panels.append(
                            (
                                plot.model_label(source, options)
                                + " | saved observed change evidence",
                                source["role"],
                                volcano,
                            )
                        )
                program_path = directory / "gene_program_metrics.csv"
                if program_path.is_file() and program_path.stat().st_size > 1:
                    frame = _read(program_path)
                    if {
                        "gene_set",
                        "contrast",
                        "true_program_delta",
                        "predicted_program_delta",
                    }.issubset(frame):
                        ranking = frame.groupby("gene_set").true_program_delta.apply(
                            lambda s: s.abs().max()
                        )
                        ranking = ranking.nlargest(
                            _cap(options, "max_display_genes", 12) or len(ranking)
                        )
                        frame = frame[frame.gene_set.isin(ranking.index)].copy()
                        frame["model_slot"] = source["model"]
                        frame["evidence_kind"] = "saved_supplied_program_change"
                        evidence.extend(frame.to_dict("records"))
                        all_values = frame[
                            ["true_program_delta", "predicted_program_delta"]
                        ].to_numpy(float)
                        common_limit = (
                            float(np.nanmax(np.abs(all_values)))
                            if np.isfinite(all_values).any()
                            else 1.0
                        )
                        common_limit = common_limit or 1.0
                        for column in ("true_program_delta", "predicted_program_delta"):
                            wide = frame.pivot_table(
                                index="gene_set", columns="contrast", values=column
                            )

                            def callback(ax, legend=True, wide=wide, lim=common_limit):
                                v = wide.to_numpy(float)
                                art = ax.imshow(
                                    v, aspect="auto", cmap=plot.DIVERGING_CMAP, vmin=-lim, vmax=lim
                                )
                                ax.set_yticks(range(len(wide)), wide.index, fontsize=5)
                                ax.set_xticks(
                                    range(len(wide.columns)), wide.columns, rotation=60, fontsize=5
                                )
                                if legend:
                                    ax.figure.colorbar(
                                        art,
                                        ax=ax,
                                        fraction=0.035,
                                        pad=0.02,
                                        label="Exported mean-change units",
                                    )

                            panels.append(
                                (
                                    plot.model_label(source, options)
                                    + " | "
                                    + column.replace("_", " "),
                                    source["role"],
                                    callback,
                                )
                            )
                if not any(p[0].startswith(plot.model_label(source, options)) for p in panels):
                    fields = [
                        c
                        for c in (
                            "mean_expression_delta_pcc",
                            "mean_expression_delta_spearman",
                            "direction_accuracy_true_deg",
                        )
                        if c in summary
                    ]
                    for metric in fields:

                        def callback(ax, legend=True, frame=summary, metric=metric):
                            ax.plot(np.arange(len(frame)), frame[metric], "o-", color="#387F79")
                            ax.set_xticks(
                                range(len(frame)),
                                frame.get("contrast", pd.Series(range(len(frame)))),
                                rotation=60,
                                fontsize=5,
                            )
                            ax.set(
                                ylabel=metric.replace("_", " "), xlabel="Saved temporal contrast"
                            )

                        panels.append(
                            (
                                plot.model_label(source, options) + " | " + metric,
                                source["role"],
                                callback,
                            )
                        )
        elif view == "sampling_convergence":
            for source in eligible_sources:
                frame = _read(Path(source["path"]) / "sampling_convergence_summary.csv")
                if "sample_size" not in frame:
                    notes.append(f"{source['model']}: saved convergence table lacks sample_size.")
                    continue
                if "model_tag" in frame and frame.model_tag.nunique() > 1:
                    notes.append(
                        f"{source['model']}: pooled convergence table skipped; use a native single-model export."
                    )
                    continue
                frame = frame.copy()
                frame["model_slot"] = source["model"]
                evidence.extend(frame.to_dict("records"))
                metrics = [
                    c
                    for c in (
                        "pseudobulk_rmse_to_full_prediction_mean",
                        "pseudobulk_pcc_to_full_prediction_mean",
                        "mean_gene_variance_mean",
                        "truth_mmd_u_mean",
                    )
                    if c in frame
                ]
                for metric in metrics:

                    def callback(ax, legend=True, frame=frame, metric=metric):
                        groups = (
                            frame.groupby("target_time")
                            if "target_time" in frame
                            else [("saved population", frame)]
                        )
                        for i, (time, group) in enumerate(groups):
                            group = group.sort_values("sample_size")
                            x = group.sample_size.to_numpy(float)
                            y = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
                            ax.plot(
                                x,
                                y,
                                label=str(time),
                                color=plot.PALETTE[i % len(plot.PALETTE)],
                                lw=1.1,
                            )
                            sd = metric[:-5] + "_std" if metric.endswith("_mean") else ""
                            if sd in group:
                                err = pd.to_numeric(group[sd], errors="coerce").to_numpy(float)
                                ax.fill_between(
                                    x,
                                    y - err,
                                    y + err,
                                    color=plot.PALETTE[i % len(plot.PALETTE)],
                                    alpha=0.18,
                                )
                        ax.set_xscale("log")
                        ax.set(
                            xlabel="Subsampled saved cell count", ylabel=metric.replace("_", " ")
                        )
                        if legend:
                            ax.legend(fontsize=5, title="Saved target stage")

                    panels.append(
                        (
                            plot.model_label(source, options) + " | " + metric,
                            source["role"],
                            callback,
                        )
                    )
        if evidence:
            path = output / f"{view}_evidence.csv"
            pd.DataFrame(evidence).to_csv(path, index=False)
            tables.append(str(path))
        if panels:
            paths = plot.compare_mosaics(
                panels,
                output / view,
                options,
                question.get("title", view) + " | " + cohort["panel"],
            )
            figures.extend(
                {"path": str(p), "view": view, "entity": "named subset", "model": "comparison"}
                for p in paths
            )
        if progress:
            progress({"view": view, "step": view, "figures": len(figures), "completed": 1})
    if audit:
        path = output / "DISPLAY_SELECTION.csv"
        pd.DataFrame(audit).drop_duplicates().to_csv(path, index=False)
        tables.append(str(path))
    requested = set(question.get("views", []))
    produced = {f["view"] for f in figures}
    status = "complete" if requested.issubset(produced) else "partial" if figures else "unsupported"
    missing = requested - produced
    if missing:
        notes.append("Saved table views unavailable: " + ", ".join(sorted(missing)))
    notes.append(
        "Existing V5 table evidence, without rerunning inference or classifier fitting. Real classifier reliability is required; no-GT diagnostics are plausibility/sensitivity, not accuracy."
    )
    return {"status": status, "figures": figures, "tables": tables, "notes": notes}
