"""Preserve and genuinely re-render saved biological evaluations.

Historical figures remain historical evidence. Their model identities are never
substituted for new v6/v7 outputs and their statistics never enter new galleries.
Numerical re-renders use saved coordinates/aggregates, never artificial cells.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd

MEDIA_EXTENSIONS = {".png", ".svg", ".pdf", ".gif"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".parquet"}
EXCLUDED_PARTS = {"__MACOSX"}

TABLE_FAMILIES = {
    "reference_mapped_coordinates.parquet": ("F2", "reference_landscape", "coordinates"),
    "shared_truth_fitted_embedding_coordinates.parquet": (
        "F2",
        "reference_landscape",
        "coordinates",
    ),
    "centroid_trajectory_coordinates.csv": ("F2", "centroid_trajectory", "centroids"),
    "developmental_gene_cascade.tsv": ("F2", "developmental_cascade", "timecourse"),
    "all_genes_pseudobulk_timecourse.parquet": ("F2", "developmental_cascade", "timecourse"),
    "pseudobulk_timecourse_long.csv": ("F2", "developmental_cascade", "paired_timecourse"),
    "all_gene_population_statistics.parquet": ("F2", "gene_distributions", "distributions"),
    "all_genes_population_statistics.parquet": ("F2", "gene_distributions", "distributions"),
    "all_datasets_all_models_all_gene_population_statistics.parquet": (
        "F2",
        "gene_distributions",
        "distributions",
    ),
    "truth_defined_marker_dotplot.tsv": ("F3", "marker_identity", "markers"),
    "truth_defined_marker_expression.tsv": ("F3", "marker_identity", "markers"),
    "author_driver_program_activity.tsv": ("F3", "author_programs", "timecourse"),
    "author_driver_program_timecourse.csv": ("F3", "author_programs", "timecourse"),
    "all_models_author_driver_program_timecourse.csv": ("F3", "author_programs", "timecourse"),
    "tf_target_module_activity.tsv": ("F3", "tf_target_modules", "timecourse"),
    "state_graph_edges.tsv": ("F3", "state_connectivity", "network"),
    "author_curated_graph.tsv": ("F3", "author_state_graph", "network"),
    "author_curated_graph_input.csv": ("F3", "author_state_graph", "network"),
    "author_graph_node_mapping_and_occupancy.tsv": ("F3", "author_state_graph", "occupancy"),
    "grn_expression_support_by_population.parquet": ("F3", "grn_expression_support", "network"),
    "grn_prediction_implied_edge_statistics.parquet": ("F3", "grn_expression_support", "network"),
    "all_models_grn_prediction_implied_edge_statistics.parquet": (
        "F3",
        "grn_expression_support",
        "network",
    ),
    "interpolation_reference_coordinates.tsv": ("F4", "interpolation_landscape", "coordinates"),
    "interpolation_driver_program_timecourse.tsv": ("F4", "interpolation_programs", "timecourse"),
    "gene_program_timecourse.csv": ("F4", "interpolation_programs", "timecourse"),
    "future_reference_coordinates.tsv": ("F4", "future_landscape", "coordinates"),
    "future_driver_program_timecourse.tsv": ("F4", "future_programs", "timecourse"),
    "history_reference_coordinates.tsv": ("F4", "history_landscape", "coordinates"),
    "history_dynamic_gene_timecourse.tsv": ("F4", "history_genes", "wide_timecourse"),
    "history_driver_program_timecourse.tsv": ("F4", "history_programs", "timecourse"),
    "exhaustive_compact_perturbation_response.parquet": (
        "F5",
        "compact_perturbation",
        "perturbation",
    ),
    "perturbation_gene_responses.parquet": ("F5", "compact_perturbation", "perturbation"),
    "perturbation_screening_candidates.csv": ("F5", "compact_perturbation", "perturbation"),
    "perturbation_program_response.tsv": ("F5", "perturbation_programs", "perturbation_programs"),
    "perturbation_gene_programs.csv": ("F5", "perturbation_programs", "perturbation_programs"),
    "grn_supported_perturbation_cascades.tsv": ("F5", "prior_supported_cascades", "perturbation"),
    "cell_state_proportions.csv": ("F3", "state_composition", "composition"),
    "celltype_proportion_timecourse.csv": ("F4", "interpolation_composition", "composition"),
}

FAMILY_TITLES = {
    "reference_landscape": "Do predictions occupy the observed cell-state landscape?",
    "density": "Where does generated cell density differ from observed density?",
    "centroid_trajectory": "How do population centroids move across development?",
    "developmental_cascade": "Are developmental gene-expression cascades preserved?",
    "gene_distributions": "Are state and age expression distributions preserved?",
    "gene_feature_atlas": "Where are developmental genes expressed in the cell atlas?",
    "marker_identity": "Are observed state markers expressed in the correct generated states?",
    "author_programs": "Are author-defined developmental programs reproduced?",
    "tf_target_modules": "Are supplied TF target-set activities preserved?",
    "state_connectivity": "Is expression-space state connectivity preserved?",
    "author_state_graph": "How do cell populations occupy the supplied author state graph?",
    "grn_expression_support": "Do supplied GRN edges retain expression-association support?",
    "candidate_evidence": "Which candidate regulators have complementary biological evidence?",
    "interpolation_landscape": "How do generated intermediate ages occupy the reference landscape?",
    "interpolation_programs": "How do developmental programs vary between observed anchors?",
    "interpolation_composition": "How do saved state fractions change at intermediate ages?",
    "future_landscape": "How does autoregressive future occupy the observed reference landscape?",
    "future_programs": "How do developmental programs change during future rollout?",
    "history_landscape": "How does autoregressive history depart from the earliest observed anchor?",
    "history_genes": "How do gene-expression profiles change during reverse-history rollout?",
    "history_programs": "How do author programs change during reverse-history rollout?",
    "compact_perturbation": "Which saved source-history edits alter downstream expression?",
    "perturbation_programs": "Which developmental programs respond to saved source-history edits?",
    "prior_supported_cascades": "Which modeled edit responses overlap supplied prior neighborhoods?",
    "state_composition": "Are observed and generated state fractions comparable?",
    "classification": "How well do saved independent classifiers recognize generated states?",
    "quantitative_support": "Which quantitative summaries support the biological comparison?",
    "ablation_support": "How do historical controlled prior ablations affect biological fidelity?",
    "sampling_support": "How stable are saved evaluations across sampling sizes?",
    "rollout_diagnostics": "How numerically stable are the saved scenario trajectories?",
    "protocol_supplement": "What was the historical model and evaluation protocol?",
    "legacy_supporting": "Historical supporting evaluation figures",
}

# Native question routes are complements, not claims that a historical
# controlled-ablation/classifier ceiling is recreated from unrelated models.
NATIVE_EQUIVALENTS = {
    "reference_landscape": ("F2_Q01;F2_Q02", "measured:atlas_time,atlas_state,stage_storyboard"),
    "density": ("F2_Q04;F3_Q14", "measured:density,density_residual,density_contour"),
    "centroid_trajectory": ("F2_Q05;F3_Q12", "measured:trajectory,state_trajectory"),
    "developmental_cascade": ("F2_Q06;F2_Q08", "measured:gene_timecourse,cascade"),
    "gene_distributions": ("F2_Q07;F3_Q10", "measured:gene_distribution,marker_violin"),
    "gene_feature_atlas": ("F2_Q03", "measured:gene_feature"),
    "marker_identity": ("F2_Q10;F3_Q01", "measured:marker_dotplot,marker_matrix,marker_violin"),
    "author_programs": (
        "F3_Q05;F3_Q06;F3_Q13",
        "measured:program_timecourse,program_heatmap,program_feature",
    ),
    "tf_target_modules": ("F3_Q08", "measured:grn_activity"),
    "state_connectivity": ("F3_Q04;F3_Q12", "measured:state_graph,state_trajectory"),
    "author_state_graph": (
        "F3_Q04;F3_Q03",
        "archive:supplied author graph; measured:expression state graph,occupancy",
    ),
    "grn_expression_support": ("F3_Q08;F3_Q09", "measured:grn_support,grn_network"),
    "candidate_evidence": ("F2_Q12", "measured:candidate_evidence"),
    "interpolation_landscape": (
        "F4_Q01;F4_Q02",
        "scenarios:storyboard,geometry,trajectory,density",
    ),
    "interpolation_programs": ("F4_Q04", "scenarios:programs,gene_heatmap"),
    "interpolation_composition": ("F4_Q05", "archive:saved fractions; scenarios:marker_dotplot"),
    "future_landscape": (
        "F4_Q07;F4_Q10;F4_Q11",
        "scenarios:geometry,storyboard,trajectory,density,drift",
    ),
    "future_programs": ("F4_Q09;F4_Q12", "scenarios:programs,drift,marker_dotplot"),
    "history_landscape": (
        "F5_Q04;F5_Q07;F5_Q08",
        "scenarios:history_geometry,storyboard,trajectory,density",
    ),
    "history_genes": ("F5_Q05;F5_Q09", "scenarios:history_timecourse,history_heatmap"),
    "history_programs": ("F5_Q06", "scenarios:programs,drift,horizon"),
    "compact_perturbation": (
        "F5_Q10;F5_Q11;F5_Q12;F5_Q14;F5_Q15",
        "scenarios:perturbation_heatmap,perturbation_timecourse,perturbation_targets",
    ),
    "perturbation_programs": ("F5_Q13", "scenarios:perturbation_programs"),
    "prior_supported_cascades": ("F5_Q16", "scenarios:perturbation_network"),
    "state_composition": ("F3_Q03;F3_Q11", "measured:state_occupancy"),
    "classification": (
        "F3_Q02;F3_Q03",
        "archive:actual classifier diagnostics; measured:reference state assignment and occupancy",
    ),
    "quantitative_support": (
        "",
        "archive:historical quantitative summaries preserved; new native biology exports its numerical source summaries",
    ),
    "ablation_support": (
        "",
        "archive:historical controlled ablation only; v6/v7 differences are not substituted for ablation",
    ),
    "sampling_support": (
        "",
        "archive:actual saved sampling experiment only; new model outputs are not a sampling experiment",
    ),
    "rollout_diagnostics": ("F4_Q06;F4_Q10;F5_Q07", "scenarios:diversity,horizon,drift"),
    "protocol_supplement": (
        "",
        "archive:historical protocol schematic; new workflow documented in guide",
    ),
    "legacy_supporting": ("", "archive:explicit historical supporting artifact"),
}


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_.")[:100] or "source"


def _key(path: Path) -> str:
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]


def _paths(roots):
    seen = set()
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            continue
        values = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in values:
            if not path.is_file() or any(
                p in EXCLUDED_PARTS or p.startswith("._") for p in path.parts
            ):
                continue
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                yield root, path


def classify_legacy_figure(path: Path) -> tuple[str, str]:
    """Map every original figure to a paper figure and stable question family."""
    text = str(path).lower()
    name = path.stem.lower()
    if any(x in name for x in ("flowchart", "model_and_evaluation", "pipeline", "schematic")):
        return "F2", "protocol_supplement"
    if "history" in text:
        family = (
            "history_genes"
            if "gene_heatmap" in name
            else (
                "history_programs"
                if "program" in name
                else "rollout_diagnostics" if "diagnostic" in name else "history_landscape"
            )
        )
        return "F4", family
    if "perturbation" in text or any(
        x in name
        for x in (
            "intervention",
            "downstream_gene_response",
            "direct_target_response",
            "scenario_response",
        )
    ):
        family = (
            "perturbation_programs"
            if "program" in name
            else (
                "prior_supported_cascades"
                if any(x in name for x in ("cascade", "grn"))
                else "compact_perturbation"
            )
        )
        return "F5", family
    if "interpolation" in text:
        family = (
            "interpolation_programs"
            if "program" in name
            else (
                "interpolation_composition"
                if "composition" in name
                else "rollout_diagnostics" if "diagnostic" in name else "interpolation_landscape"
            )
        )
        return "F4", family
    if any(x in text for x in ("future", "extrapolation")):
        return "F4", (
            "future_programs"
            if "program" in name
            else "rollout_diagnostics" if "diagnostic" in name else "future_landscape"
        )
    if "density" in name:
        return "F2", "density"
    if "reference_embedding" in name:
        return "F2", "gene_feature_atlas"
    if "marker" in name or "deg_dotplot" in name:
        return "F3", "marker_identity"
    if "author_graph" in name:
        return "F3", "author_state_graph"
    if "state_graph" in name:
        return "F3", "state_connectivity"
    if "tf_target" in name:
        return "F3", "tf_target_modules"
    if any(x in name for x in ("grn", "prior_supported_edge", "tf_")):
        return "F3", "grn_expression_support"
    if "candidate_regulator" in name or "evidence_map" in name:
        return "F3", "candidate_evidence"
    if "program" in name:
        return "F3", "author_programs"
    if "cascade" in name or "gene_timecourse" in name or "developmental_change" in text:
        return "F2", "developmental_cascade"
    if any(x in name for x in ("classifier", "confusion", "class_proportion")):
        return "F3", "classification"
    if any(x in name for x in ("cell_state", "composition")):
        return "F3", "state_composition"
    if "centroid" in name:
        return "F2", "centroid_trajectory"
    if any(x in name for x in ("reference_atlas", "trajectory", "embedding", "__pca")):
        return "F2", "reference_landscape"
    if "sampling" in text or "convergence" in name:
        return "F2", "sampling_support"
    if any(x in name for x in ("ablation", "prior_specific", "marginal_improvement", "cumulative")):
        return "F3", "ablation_support"
    if any(
        x in name
        for x in (
            "metric",
            "pareto",
            "error_screen",
            "pseudobulk",
            "per_timepoint_truth_prediction",
            "per_stage_heatmap",
        )
    ):
        return "F2", "quantitative_support"
    if (
        "genes/" in text
        or "all_genes" in name
        or "distribution" in name
        or re.match(r"(gene_\d+|time_\d+)", name)
        or "__time_" in name
    ):
        return "F2", "gene_distributions"
    return "F2", "legacy_supporting"


def inventory_legacy(roots, out) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries, tables = [], []
    for root, path in _paths(roots):
        relative = path.name if root.is_file() else str(path.relative_to(root))
        if path.suffix.lower() in MEDIA_EXTENSIONS:
            figure, family = classify_legacy_figure(path)
            equivalent, route = NATIVE_EQUIVALENTS.get(family, ("", "archive:historical original"))
            entries.append(
                {
                    "source_root": str(root),
                    "source_path": str(path),
                    "relative_path": relative,
                    "extension": path.suffix.lower(),
                    "paper_figure": figure,
                    "family": family,
                    "question_id": f"LEG_{figure}_{family}",
                    "current_questions": equivalent,
                    "new_renderer": route,
                    "coverage_mode": (
                        "original preserved plus native biology counterpart"
                        if equivalent
                        else "historical original preserved; task requires its own historical design"
                    ),
                    "source_status": (
                        "source marks not applicable"
                        if "99_not_applicable_single_model" in path.parts
                        else "saved historical figure"
                    ),
                    "source_identity": "historical original; never relabel as current models",
                }
            )
        elif path.suffix.lower() in TABLE_EXTENSIONS:
            if "99_not_applicable_single_model" in path.parts:
                continue
            info = TABLE_FAMILIES.get(path.name)
            nonempty = None
            if info:
                try:
                    if path.suffix.lower() == ".parquet":
                        from pyarrow.parquet import ParquetFile

                        nonempty = ParquetFile(path).metadata.num_rows > 0
                    else:
                        nonempty = (
                            len(
                                pd.read_csv(
                                    path,
                                    sep="\t" if path.suffix.lower() == ".tsv" else ",",
                                    nrows=1,
                                )
                            )
                            > 0
                        )
                except pd.errors.EmptyDataError:
                    nonempty = False
                except Exception:
                    # Actual parse/schema failures remain visible when the
                    # typed renderer runs; they are not disguised as empties.
                    nonempty = True
            tables.append(
                {
                    "source_root": str(root),
                    "source_path": str(path),
                    "relative_path": relative,
                    "paper_figure": info[0] if info else "",
                    "family": info[1] if info else "",
                    "renderer": info[2] if info else "",
                    "known_biological_table": bool(info),
                    "input_nonempty": nonempty,
                }
            )
    summary = {
        "original_figures": len(entries),
        "original_pngs": sum(e["extension"] == ".png" for e in entries),
        "figure_families": len({(e["paper_figure"], e["family"]) for e in entries}),
        "table_files": len(tables),
        "recognized_biological_tables": sum(e["known_biological_table"] for e in tables),
        "runnable_biological_tables": sum(
            e["known_biological_table"] and e["input_nonempty"] for e in tables
        ),
        "empty_biological_tables": sum(
            e["known_biological_table"] and not e["input_nonempty"] for e in tables
        ),
        "by_paper_figure": dict(Counter(e["paper_figure"] for e in entries)),
        "missing_roots": [str(Path(r)) for r in roots if not Path(r).expanduser().exists()],
    }
    pd.DataFrame(entries).to_csv(out / "LEGACY_FIGURE_INVENTORY.csv", index=False)
    pd.DataFrame(tables).to_csv(out / "LEGACY_TABLE_INVENTORY.csv", index=False)
    family_rows = []
    for family in sorted(
        {e["family"] for e in entries}
        | {e["family"] for e in tables if e["known_biological_table"]}
    ):
        current, route = NATIVE_EQUIVALENTS.get(
            family, ("", "archive:typed saved biological table")
        )
        family_rows.append(
            {
                "family": family,
                "biological_question": FAMILY_TITLES.get(family, family),
                "legacy_paper_figures": ";".join(
                    sorted({e["paper_figure"] for e in entries if e["family"] == family})
                ),
                "current_questions": current,
                "new_renderer": route,
                "original_files": sum(e["family"] == family for e in entries),
                "typed_saved_tables": sum(e["family"] == family for e in tables),
            }
        )
    pd.DataFrame(family_rows).to_csv(out / "LEGACY_FAMILY_MAP.csv", index=False)
    (out / "LEGACY_INVENTORY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "figures": entries, "tables": tables}


def _read(path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", low_memory=False)


def _first(frame, names, default=None):
    return next((name for name in names if name in frame), default)


def _truth(values):
    return next(
        (
            value
            for value in values
            if str(value).strip().lower()
            in {"ground truth", "ground_truth", "truth", "gt", "observed"}
        ),
        None,
    )


def _order(values):
    values = list(dict.fromkeys(map(str, values)))

    def key(value):
        try:
            return (0, float(value))
        except ValueError:
            return (
                1,
                [
                    (0, float(x)) if re.fullmatch(r"\d+(?:\.\d+)?", x) else (1, x.lower())
                    for x in re.split(r"(\d+(?:\.\d+)?)", value)
                ],
            )

    return sorted(values, key=key)


def _mpl():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from .plot import style

    style()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    return plt


def _axes(count, width=4.2, height=3.5):
    plt = _mpl()
    cols = min(4, max(count, 1))
    rows = math.ceil(max(count, 1) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(width * cols, height * rows), squeeze=False, layout="constrained"
    )
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:count])


def _save(fig, base, options, figures, view, entity="all"):
    from .plot import save

    allowed = options.setdefault("_archive_view_types", set())
    maximum = int(options.get("max_views_per_entity", 0) or 0)
    if maximum > 0 and view not in allowed and len(allowed) >= maximum:
        _mpl().close(fig)
        return
    allowed.add(view)
    paths = save(fig, Path(base), options)
    _mpl().close(fig)
    for path in paths:
        figures.append(
            {
                "path": str(path),
                "view": view,
                "entity": entity,
                "model": "actual saved table identities",
                "origin": "rerendered saved biological table",
            }
        )


def _limit(options):
    return int(options.get("max_entities", 0) or 0)


def _select(frame, entity, value, group, options, directory, reason):
    basis = frame
    if "population" in basis:
        gt = _truth(basis.population.unique())
        if gt is not None:
            basis = basis[basis.population == gt]
            reason += "; priority from observed GT only"
    grouped = basis.groupby(entity, observed=True)[value]
    score = (
        grouped.max() - grouped.min()
        if group
        else grouped.apply(lambda x: float(np.nanmean(np.abs(x))))
    )
    ranked = score.sort_values(ascending=False, kind="mergesort")
    limit = _limit(options)
    chosen = set(ranked.index[:limit] if limit > 0 else ranked.index)
    pd.DataFrame(
        {
            "entity": ranked.index.astype(str),
            "score": ranked.values,
            "selected": [x in chosen for x in ranked.index],
            "reason": reason,
            "p_value": np.nan,
            "q_value": np.nan,
        }
    ).to_csv(directory / "selected_entities.csv", index=False)
    return [x for x in ranked.index if x in chosen]


def _heatmap(ax, matrix, title, cmap="viridis", symmetric=False, limits=None):
    data = matrix.to_numpy(float)
    if symmetric:
        bound = max(float(np.nanpercentile(np.abs(data), 98)), 1e-8)
        limits = (-bound, bound)
    kwargs = dict(cmap=cmap, aspect="auto", interpolation="nearest")
    if limits:
        kwargs.update(vmin=limits[0], vmax=limits[1])
    im = ax.imshow(np.ma.masked_invalid(data), **kwargs)
    sx = max(1, math.ceil(len(matrix.columns) / 35))
    sy = max(1, math.ceil(len(matrix.index) / 35))
    ax.set_xticks(
        np.arange(len(matrix.columns))[::sx],
        [str(x) for x in matrix.columns[::sx]],
        rotation=90,
        fontsize=7,
    )
    ax.set_yticks(
        np.arange(len(matrix.index))[::sy], [str(x) for x in matrix.index[::sy]], fontsize=7
    )
    ax.set_title(title, fontsize=10)
    return im


def _coordinates(frame, directory, options, figures):
    x = _first(frame, ["axis_1", "PC1", "x", "centroid_axis_1"])
    y = _first(frame, ["axis_2", "PC2", "y", "centroid_axis_2"])
    if x is None or y is None:
        raise ValueError("Coordinate table lacks a real saved x/y pair")
    frame = frame.copy()
    frame[x] = pd.to_numeric(frame[x], errors="coerce")
    frame[y] = pd.to_numeric(frame[y], errors="coerce")
    frame = frame[np.isfinite(frame[x]) & np.isfinite(frame[y])]
    if frame.empty:
        raise ValueError("Coordinate table has no finite saved coordinates")
    if "embedding" in frame and frame.embedding.nunique() > 1:
        for embedding, part in frame.groupby("embedding", sort=False):
            sub = directory / _safe(embedding)
            sub.mkdir(exist_ok=True)
            _coordinates(part.drop(columns="embedding"), sub, options, figures)
        return
    population = _first(frame, ["population", "model_label", "model"], "_population")
    if population == "_population":
        frame[population] = "Saved population"
    populations = list(frame[population].astype(str).unique())
    state = _first(frame, ["transferred_cell_state", "cell_state", "state", "cell_type_id"])
    age = _first(
        frame, ["developmental_time", "actual_time", "target_time", "history_step", "future_step"]
    )
    bounds = [
        float(frame[x].min()),
        float(frame[x].max()),
        float(frame[y].min()),
        float(frame[y].max()),
    ]
    padx = max((bounds[1] - bounds[0]) * 0.03, 1e-5)
    pady = max((bounds[3] - bounds[2]) * 0.03, 1e-5)
    bounds = [bounds[0] - padx, bounds[1] + padx, bounds[2] - pady, bounds[3] + pady]
    groups = [frame[frame[population].astype(str) == p] for p in populations]
    plt = _mpl()
    for color_column, name in [(state, "saved_state_landscape"), (age, "saved_age_landscape")]:
        if color_column is None:
            continue
        fig, axes = _axes(len(groups))
        numeric = pd.to_numeric(frame[color_column], errors="coerce")
        is_numeric = numeric.notna().all() and color_column == age
        levels = _order(frame[color_column].astype(str))
        lookup = {level: i for i, level in enumerate(levels)}
        for p, part, ax in zip(populations, groups, axes):
            colors = (
                pd.to_numeric(part[color_column], errors="coerce")
                if is_numeric
                else part[color_column].astype(str).map(lookup)
            )
            scatter = ax.scatter(
                part[x],
                part[y],
                c=colors,
                cmap="viridis" if is_numeric else "tab20",
                vmin=numeric.min() if is_numeric else 0,
                vmax=numeric.max() if is_numeric else max(len(levels) - 1, 1),
                s=2.3,
                alpha=0.75,
                linewidths=0,
                rasterized=True,
            )
            ax.set(xlim=bounds[:2], ylim=bounds[2:], title=p, xlabel=x, ylabel=y)
        fig.suptitle(
            f"Saved projection colored by {color_column}; historical model identities", fontsize=12
        )
        if is_numeric:
            fig.colorbar(scatter, ax=axes, label=color_column, shrink=0.75)
        elif len(levels) <= 30:
            from matplotlib.lines import Line2D

            cmap = plt.get_cmap("tab20")
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=4,
                    color=cmap(i / max(len(levels) - 1, 1)),
                    label=level,
                )
                for i, level in enumerate(levels)
            ]
            axes[0].legend(
                handles=handles, fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1)
            )
        if not is_numeric:
            pd.DataFrame({"state": levels, "color_index": range(len(levels))}).to_csv(
                directory / "saved_state_color_key.csv", index=False
            )
        _save(fig, directory / name, options, figures, name)
    histograms = []
    for part in groups:
        hist, xe, ye = np.histogram2d(part[x], part[y], bins=55, range=[bounds[:2], bounds[2:]])
        histograms.append(hist.T / max(len(part), 1))
    vmax = max(float(h.max()) for h in histograms)
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    density_cmap = LinearSegmentedColormap.from_list(
        "bloom_archive_density", ["#F6F4EF", "#BCD1CE", "#729EA4", "#3F6B88", "#69517D"]
    )
    fig, axes = _axes(len(groups))
    for p, hist, ax in zip(populations, histograms, axes):
        ax.imshow(
            hist,
            extent=bounds,
            origin="lower",
            aspect="auto",
            cmap=density_cmap,
            norm=PowerNorm(gamma=0.4, vmin=0, vmax=vmax),
        )
        ax.set(title=p, xlabel=x, ylabel=y)
    fig.suptitle(
        "Common-grid density; probability per bin; shared power color scale (γ = 0.4)", fontsize=12
    )
    fig.colorbar(axes[0].images[0], ax=axes, label="probability per bin", shrink=0.75)
    _save(fig, directory / "common_grid_density", options, figures, "common_grid_density")
    gt = _truth(populations)
    if gt is not None and len(populations) > 1:
        reference = histograms[populations.index(gt)]
        residuals = [(p, hist - reference) for p, hist in zip(populations, histograms) if p != gt]
        bound = max(max(float(np.abs(h).max()) for _, h in residuals), 1e-8)
        fig, axes = _axes(len(residuals))
        for (p, h), ax in zip(residuals, axes):
            ax.imshow(
                h,
                extent=bounds,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
            )
            ax.set(title=f"{p} − {gt}", xlabel=x, ylabel=y)
        fig.suptitle("Signed density residual; generated minus observed", fontsize=12)
        fig.colorbar(
            axes[0].images[0], ax=axes, label="probability difference per bin", shrink=0.75
        )
        _save(
            fig, directory / "signed_density_residual", options, figures, "signed_density_residual"
        )
    if age is not None:
        labels = _order(frame[age].astype(str))
        fig, axes = _axes(len(groups))
        trajectory_rows = []
        for p, part, ax in zip(populations, groups, axes):
            centroids = part.groupby(age, observed=True)[[x, y]].mean()
            centroids.index = centroids.index.astype(str)
            centroids = centroids.reindex(labels).dropna()
            ax.scatter(part[x], part[y], s=0.8, c="#969FA7", alpha=0.3, rasterized=True)
            ax.plot(centroids[x], centroids[y], "-o", color="#356B77", lw=1.6, ms=3)
            for label, row in centroids.iterrows():
                if len(centroids) <= 25:
                    ax.annotate(label, (row[x], row[y]), fontsize=6)
                trajectory_rows.append(
                    {"population": p, "saved_time": label, "x": row[x], "y": row[y]}
                )
            ax.set(xlim=bounds[:2], ylim=bounds[2:], title=p, xlabel=x, ylabel=y)
        pd.DataFrame(trajectory_rows).to_csv(directory / "centroid_trajectory.csv", index=False)
        fig.suptitle(
            "Chronological population centroids; cells are not tracked trajectories", fontsize=12
        )
        _save(
            fig,
            directory / "population_centroid_trajectory",
            options,
            figures,
            "population_centroid_trajectory",
        )
    if state is not None:
        counts = pd.crosstab(frame[population], frame[state]).astype(float)
        fractions = counts.div(counts.sum(axis=1), axis=0)
        fig, axes = _axes(
            1,
            width=max(6, min(14, len(fractions.columns) * 0.32)),
            height=max(3, len(fractions) * 0.35),
        )
        im = _heatmap(
            axes[0],
            fractions,
            "Saved transferred-state fractions",
            "viridis",
            limits=(0, float(fractions.max().max())),
        )
        fig.colorbar(im, ax=axes[0], label="fraction")
        fractions.to_csv(directory / "state_fractions.csv")
        _save(fig, directory / "state_fraction_heatmap", options, figures, "state_fraction_heatmap")


def _centroids(frame, directory, options, figures):
    """Plot saved means as means; their rows are never used as cell density."""
    if not {"axis_1", "axis_2", "population", "target_time"}.issubset(frame):
        raise ValueError("Centroid table lacks actual saved population/time/mean coordinates")
    if "embedding" in frame and frame.embedding.nunique() > 1:
        for embedding, part in frame.groupby("embedding", sort=False):
            sub = directory / _safe(embedding)
            sub.mkdir(exist_ok=True)
            _centroids(part, sub, dict(options, _archive_view_types=set()), figures)
        return
    frame = frame.copy()
    frame.target_time = frame.target_time.astype(str)
    times = _order(frame.target_time)
    populations = list(frame.population.astype(str).unique())
    fig, axes = _axes(1, width=8, height=6)
    for population, part in frame.groupby("population", sort=False):
        curve = (
            part.groupby("target_time", observed=True)[["axis_1", "axis_2"]].mean().reindex(times)
        )
        axes[0].plot(curve.axis_1, curve.axis_2, "-o", ms=3, lw=1.2, label=str(population))
    axes[0].set(
        xlabel="saved mean axis 1",
        ylabel="saved mean axis 2",
        title="Saved population-centroid trajectories; no individual cell paths",
    )
    if len(populations) <= 15:
        axes[0].legend(fontsize=7)
    _save(
        fig, directory / "saved_centroid_comparison", options, figures, "saved_centroid_comparison"
    )
    fig, axes = _axes(len(populations), width=4, height=3.5)
    for population, ax in zip(populations, axes):
        curve = (
            frame[frame.population.astype(str) == population]
            .groupby("target_time", observed=True)[["axis_1", "axis_2"]]
            .mean()
            .reindex(times)
        )
        ax.plot(curve.axis_1, curve.axis_2, "-o", ms=3, lw=1.5, color="#477F91")
        for label, row in curve.dropna().iterrows():
            ax.annotate(label, (row.axis_1, row.axis_2), fontsize=6)
        ax.set(xlabel="saved mean axis 1", ylabel="saved mean axis 2", title=population)
    fig.suptitle(
        "Chronological saved centroid positions; aggregate coordinates are not cells", fontsize=12
    )
    _save(fig, directory / "saved_centroid_panels", options, figures, "saved_centroid_panels")


def _markers(frame, directory, options, figures):
    state = _first(frame, ["transferred_cell_state", "cell_state", "state"])
    value = _first(frame, ["mean_expression", "mean"])
    fraction = _first(frame, ["fraction_expressing", "fraction_nonzero", "pct"])
    if state is None or value is None or "gene" not in frame:
        raise ValueError("Marker table lacks gene, saved state, or expression mean")
    frame = frame.copy()
    if "population" not in frame:
        frame["population"] = "Saved population"
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    genes = _select(
        frame,
        "gene",
        value,
        state,
        options,
        directory,
        "Saved biological marker-table members; observed state expression range, descriptive selection",
    )
    frame = frame[frame.gene.isin(genes)]
    populations = list(frame.population.astype(str).unique())
    states = _order(frame[state].astype(str))
    plt = _mpl()
    means = frame.pivot_table(
        index=["population", state], columns="gene", values=value, aggfunc="mean"
    ).reindex(columns=genes)
    vmin, vmax = float(np.nanmin(means)), float(np.nanmax(means))
    for start in range(0, len(genes), 35):
        page = genes[start : start + 35]
        fig, axes = _axes(
            len(populations), width=max(5, len(page) * 0.23), height=max(3, len(states) * 0.23)
        )
        for population, ax in zip(populations, axes):
            part = frame[frame.population.astype(str) == population]
            matrix = part.pivot_table(
                index=state, columns="gene", values=value, aggfunc="mean"
            ).reindex(index=states, columns=page)
            im = _heatmap(ax, matrix, population, "viridis", limits=(vmin, vmax))
        fig.suptitle("Observed-selected biological markers; shared expression scale", fontsize=12)
        fig.colorbar(im, ax=axes, label="saved mean expression", shrink=0.75)
        _save(
            fig,
            directory / f"marker_mean_matrix_{start//35+1:03d}",
            options,
            figures,
            "marker_mean_matrix",
        )
        if fraction is not None:
            fig, axes = _axes(
                len(populations), width=max(5, len(page) * 0.23), height=max(3, len(states) * 0.23)
            )
            for population, ax in zip(populations, axes):
                part = frame[frame.population.astype(str) == population]
                mean = part.pivot_table(
                    index=state, columns="gene", values=value, aggfunc="mean"
                ).reindex(index=states, columns=page)
                pct = part.pivot_table(
                    index=state, columns="gene", values=fraction, aggfunc="mean"
                ).reindex(index=states, columns=page)
                xx, yy = np.meshgrid(np.arange(len(page)), np.arange(len(states)))
                dots = ax.scatter(
                    xx.ravel(),
                    yy.ravel(),
                    s=np.clip(pct.to_numpy(float), 0, 1).ravel() * 130,
                    c=mean.to_numpy(float).ravel(),
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    linewidths=0.15,
                    edgecolors="#4A4A4A",
                )
                ax.set_xticks(range(len(page)), page, rotation=90, fontsize=7)
                ax.set_yticks(range(len(states)), states, fontsize=7)
                ax.invert_yaxis()
                ax.set_title(population)
            fig.suptitle(
                "Marker dotplot: size = saved expressing fraction; color = saved mean", fontsize=12
            )
            fig.colorbar(dots, ax=axes, label="saved mean expression", shrink=0.75)
            _save(
                fig,
                directory / f"marker_dotplot_{start//35+1:03d}",
                options,
                figures,
                "marker_dotplot",
            )
            fig, axes = _axes(
                len(populations), width=max(5, len(page) * 0.23), height=max(3, len(states) * 0.23)
            )
            for population, ax in zip(populations, axes):
                part = frame[frame.population.astype(str) == population]
                matrix = part.pivot_table(
                    index=state, columns="gene", values=fraction, aggfunc="mean"
                ).reindex(index=states, columns=page)
                im = _heatmap(ax, matrix, population, "magma", limits=(0, 1))
            fig.suptitle("Marker detection fraction; shared 0–1 scale", fontsize=12)
            fig.colorbar(im, ax=axes, label="saved expressing fraction", shrink=0.75)
            _save(
                fig,
                directory / f"marker_fraction_matrix_{start//35+1:03d}",
                options,
                figures,
                "marker_fraction_matrix",
            )


def _timecourse(frame, directory, options, figures):
    frame = frame.copy()
    entity = _first(frame, ["gene", "program", "gene_set", "TF"])
    time = _first(
        frame,
        [
            "actual_time",
            "target_time",
            "time_value",
            "time",
            "future_step",
            "stage",
            "history_step",
        ],
    )
    value = _first(
        frame,
        [
            "mean_expression",
            "pseudobulk_expression",
            "program_score_mean",
            "module_activity_mean",
            "program_mean",
            "program_score",
            "mean",
        ],
    )
    if entity is None or time is None or value is None:
        raise ValueError("Biological timecourse lacks entity, saved time/stage, or mean activity")
    if "population" not in frame:
        frame["population"] = "Saved population"
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    frame = frame[np.isfinite(frame[value])]
    if frame.empty:
        raise ValueError("No finite biological timecourse values")
    selected = _select(
        frame,
        entity,
        value,
        time,
        options,
        directory,
        "Saved biological entity; observed temporal activity range, descriptive selection",
    )
    frame = frame[frame[entity].isin(selected)]
    state = _first(frame, ["cell_state", "transferred_cell_state", "state"])
    if state is not None and frame[state].nunique() > 1:
        populations = list(frame.population.astype(str).unique())
        if "n_cells" in frame:
            weight = pd.to_numeric(frame.n_cells, errors="coerce").fillna(0).clip(lower=0)
            frame["_weighted_value"] = frame[value] * weight
            frame["_weight"] = weight
            grouped = frame.groupby(["population", entity, state], observed=True)[
                ["_weighted_value", "_weight"]
            ].sum()
            grouped["_activity"] = grouped["_weighted_value"] / grouped["_weight"].replace(
                0, np.nan
            )
            state_table = grouped.reset_index()
        else:
            state_table = (
                frame.groupby(["population", entity, state], observed=True)[value]
                .mean()
                .reset_index()
                .rename(columns={value: "_activity"})
            )
        states = _order(state_table[state].astype(str))
        state_table[state] = state_table[state].astype(str)
        lo = float(state_table._activity.min())
        hi = float(state_table._activity.max())
        for start in range(0, len(selected), 45):
            page = selected[start : start + 45]
            fig, axes = _axes(
                len(populations),
                width=max(4, min(10, len(states) * 0.3)),
                height=max(3, len(page) * 0.18),
            )
            for population, ax in zip(populations, axes):
                matrix = (
                    state_table[state_table.population.astype(str) == population]
                    .pivot_table(index=entity, columns=state, values="_activity", aggfunc="mean")
                    .reindex(index=page, columns=states)
                )
                im = _heatmap(ax, matrix, population, "viridis", limits=(lo, hi))
            fig.colorbar(im, ax=axes, label="saved activity", shrink=0.75)
            fig.suptitle(
                "Biological program/target-set activity across saved cell states", fontsize=12
            )
            _save(
                fig,
                directory / f"state_activity_matrix_{start//45+1:03d}",
                options,
                figures,
                "state_activity_matrix",
            )
        # Cell-state means can be pooled only with their saved counts. Without
        # counts, these remain explicitly equal-weight summaries of strata.
        if "n_cells" in frame:
            pooled = (
                frame.groupby(["population", entity, time], observed=True)[
                    ["_weighted_value", "_weight"]
                ]
                .sum()
                .reset_index()
            )
            pooled[value] = pooled["_weighted_value"] / pooled["_weight"].replace(0, np.nan)
            frame = pooled
    times = (
        list(dict.fromkeys(frame[time].astype(str)))
        if time == "stage"
        else _order(frame[time].astype(str))
    )
    populations = list(frame.population.astype(str).unique())
    frame[time] = frame[time].astype(str)
    gt = _truth(populations)
    means = frame.groupby(entity, observed=True)[value].mean()
    stds = frame.groupby(entity, observed=True)[value].std().fillna(0).clip(lower=1e-6)
    if gt is not None:
        baseline = frame[frame.population.astype(str) == gt].groupby(entity, observed=True)[value]
        means = baseline.mean()
        stds = baseline.std().fillna(0).clip(lower=1e-6)
    frame["_z"] = (frame[value] - frame[entity].map(means)) / frame[entity].map(stds)
    for start in range(0, len(selected), 45):
        page = selected[start : start + 45]
        fig, axes = _axes(
            len(populations),
            width=max(4, min(11, len(times) * 0.27)),
            height=max(3, len(page) * 0.18),
        )
        for population, ax in zip(populations, axes):
            part = frame[frame.population.astype(str) == population]
            matrix = part.pivot_table(
                index=entity, columns=time, values="_z", aggfunc="mean"
            ).reindex(index=page, columns=times)
            _heatmap(ax, matrix, population, "RdBu_r", limits=(-2.5, 2.5))
            ax.set_xlabel(time)
        fig.suptitle(
            (
                "Saved developmental cascade; GT-centered gene/activity scale"
                if gt
                else "Saved developmental cascade; descriptive standardized activity"
            ),
            fontsize=12,
        )
        _save(
            fig,
            directory / f"activity_cascade_{start//45+1:03d}",
            options,
            figures,
            "activity_cascade",
        )
    # Every selected entity has a genuine line panel, paginated for readability.
    for start in range(0, len(selected), 12):
        page = selected[start : start + 12]
        fig, axes = _axes(len(page), width=4, height=3)
        for identity, ax in zip(page, axes):
            for population, part in frame[frame[entity] == identity].groupby(
                "population", sort=False
            ):
                values = part.groupby(time, observed=True)[value].mean().reindex(times)
                ax.plot(range(len(times)), values, "-o", ms=2, lw=1.1, label=str(population))
            stride = max(1, math.ceil(len(times) / 8))
            ax.set_xticks(np.arange(len(times))[::stride], times[::stride], rotation=45, fontsize=7)
            ax.set_title(str(identity), fontsize=10)
            ax.set_xlabel(time)
            ax.set_ylabel("saved activity")
        if len(populations) <= 14 and axes:
            axes[0].legend(fontsize=6)
        fig.suptitle(
            "Saved biological trajectories; means describe populations, not tracked cells",
            fontsize=12,
        )
        _save(
            fig, directory / f"activity_lines_{start//12+1:03d}", options, figures, "activity_lines"
        )


def _network(frame, directory, options, figures):
    if not {"source", "target"}.issubset(frame):
        raise ValueError("Network table lacks source and target")
    value = _first(
        frame,
        [
            "_association",
            "time_residualized_correlation",
            "connectivity",
            "prediction_correlation",
            "prediction_residualized_correlation",
            "prediction_pearson",
            "prediction_corr",
            "prior_weight",
        ],
    )
    if value is None:
        numeric = [
            c
            for c in frame
            if c not in {"source", "target", "dataset_id", "population"}
            and pd.api.types.is_numeric_dtype(frame[c])
        ]
        value = numeric[0] if numeric else "_weight"
        frame = frame.copy()
        frame[value] = 1.0
    if "population" not in frame:
        frame = frame.copy()
        frame["population"] = "Supplied prior"
    nodes = _order(pd.concat([frame.source.astype(str), frame.target.astype(str)]))
    limit = _limit(options)
    if limit > 0 and len(nodes) > limit:
        degree = pd.concat([frame.source.astype(str), frame.target.astype(str)]).value_counts()
        # Picking only highest-degree sources can remove every edge in a
        # bipartite author graph. Start with one actual strongest-supported
        # endpoint pair, then retain connected neighbors in degree order.
        pairs = frame[["source", "target"]].astype(str).drop_duplicates()
        pairs["_degree"] = pairs.source.map(degree) + pairs.target.map(degree)
        first = pairs.sort_values("_degree", ascending=False, kind="mergesort").iloc[0]
        nodes = list(dict.fromkeys([first.source, first.target]))[: max(2, limit)]
        while len(nodes) < max(2, limit):
            adjacent = pairs[pairs.source.isin(nodes) | pairs.target.isin(nodes)]
            choices = set(adjacent.source) | set(adjacent.target)
            choices -= set(nodes)
            if not choices:
                choices = set(degree.index) - set(nodes)
            if not choices:
                break
            nodes.append(max(choices, key=lambda n: (degree.get(n, 0), str(n))))
        frame = frame[frame.source.astype(str).isin(nodes) & frame.target.astype(str).isin(nodes)]
    if frame.empty:
        raise ValueError("No saved edges connect selected biological nodes")
    pd.DataFrame(
        {
            "entity": nodes,
            "selected": True,
            "reason": "Supplied biological network node; high saved prior/connectivity degree if capped; association is not causality",
        }
    ).to_csv(directory / "selected_entities.csv", index=False)
    populations = list(frame.population.astype(str).unique())
    lookup = {node: i for i, node in enumerate(nodes)}
    theta = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    xy = np.c_[np.cos(theta), np.sin(theta)]
    fig, axes = _axes(len(populations), width=5, height=5)
    bound = max(float(np.nanmax(np.abs(pd.to_numeric(frame[value], errors="coerce")))), 1e-8)
    for population, ax in zip(populations, axes):
        part = frame[frame.population.astype(str) == population]
        for source, target, weight in zip(
            part.source.astype(str),
            part.target.astype(str),
            pd.to_numeric(part[value], errors="coerce"),
        ):
            if source not in lookup or target not in lookup or not np.isfinite(weight):
                continue
            a, b = xy[lookup[source]], xy[lookup[target]]
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="#477F91" if weight >= 0 else "#AB6577",
                alpha=0.2 + 0.5 * abs(weight) / bound,
                lw=0.3 + 1.3 * abs(weight) / bound,
            )
        ax.scatter(xy[:, 0], xy[:, 1], s=40, color="#477F91", edgecolor="white", lw=0.5)
        if len(nodes) <= 45:
            for node, (x, y) in zip(nodes, xy):
                ax.text(x * 1.09, y * 1.09, node, ha="center", va="center", fontsize=6)
        ax.set_title(population)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(
        f"Fixed saved biological network; {value}; topology/association, not causal inference",
        fontsize=11,
    )
    _save(fig, directory / "fixed_saved_network", options, figures, "fixed_saved_network")
    fig, axes = _axes(len(populations), width=5, height=4)
    for population, ax in zip(populations, axes):
        part = frame[frame.population.astype(str) == population]
        matrix = part.pivot_table(
            index="source", columns="target", values=value, aggfunc="mean"
        ).reindex(index=nodes, columns=nodes)
        _heatmap(ax, matrix, population, "RdBu_r", limits=(-bound, bound))
    fig.suptitle("Saved edge weights/associations; absent prior edges remain missing", fontsize=12)
    _save(fig, directory / "edge_weight_matrix", options, figures, "edge_weight_matrix")


def _perturbation(frame, directory, options, figures, program=False):
    frame = frame.copy()
    intervention = _first(frame, ["intervention_gene", "scenario_id", "edit_id"])
    entity = _first(frame, ["program", "gene_set"] if program else ["response_gene", "gene"])
    value = _first(
        frame,
        (
            ["signed_program_response", "predicted_program_delta"]
            if program
            else ["standardized_response", "predicted_delta", "delta"]
        ),
    )
    time = _first(frame, ["target_time", "target_age", "time"])
    if value is None and {"baseline_mean", "perturbed_mean"}.issubset(frame):
        value = "saved_mean_delta"
        frame[value] = frame.perturbed_mean - frame.baseline_mean
    if intervention is None or entity is None or value is None:
        raise ValueError("Saved sensitivity table lacks actual intervention/readout/response")
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    frame = frame[np.isfinite(frame[value])]
    if frame.empty:
        raise ValueError("No finite saved sensitivity effects")
    selected = _select(
        frame,
        intervention,
        value,
        None,
        options,
        directory,
        "Saved source-history edit; mean absolute modeled response, hypothesis ranking; no experimental perturbation truth",
    )
    frame = frame[frame[intervention].isin(selected)]
    matrix = frame.pivot_table(
        index=intervention, columns=entity, values=value, aggfunc="mean"
    ).reindex(index=selected)
    fig, axes = _axes(1, width=13, height=max(4, min(15, len(selected) * 0.2)))
    im = _heatmap(
        axes[0], matrix, "Saved edit × biological readout response", "RdBu_r", symmetric=True
    )
    axes[0].set_xlabel(entity)
    axes[0].set_ylabel(intervention)
    fig.colorbar(im, ax=axes[0], label=value)
    _save(
        fig,
        directory / "saved_edit_response_matrix",
        options,
        figures,
        "saved_edit_response_matrix",
    )
    matrix.to_csv(directory / "saved_edit_response_matrix.csv")
    plt = _mpl()
    fig, axes = _axes(2, width=6, height=4)
    totals = (
        frame.groupby(intervention, observed=True)[value]
        .apply(lambda x: float(np.mean(np.abs(x))))
        .reindex(selected)
    )
    axes[0].bar(range(len(totals)), totals, color="#477F91")
    stride = max(1, math.ceil(len(totals) / 25))
    axes[0].set_xticks(
        np.arange(len(totals))[::stride],
        [str(x) for x in totals.index[::stride]],
        rotation=90,
        fontsize=7,
    )
    axes[0].set(title="Mean absolute saved response", ylabel=value, xlabel=intervention)
    axes[1].hist(frame[value], bins=60, color="#AB6577", alpha=0.85)
    axes[1].axvline(0, color="#666666", lw=0.8)
    axes[1].set(
        title="Signed biological readout response distribution",
        xlabel=value,
        ylabel="saved aggregate rows",
    )
    fig.suptitle(
        "Frozen-model source-history sensitivity; not experimental knockout truth", fontsize=12
    )
    _save(
        fig,
        directory / "response_strength_and_distribution",
        options,
        figures,
        "response_strength_and_distribution",
    )
    if time is not None:
        fig, axes = _axes(1, width=8, height=4)
        times = _order(frame[time].astype(str))
        frame[time] = frame[time].astype(str)
        profiles = (
            frame.groupby([intervention, time], observed=True)[value]
            .apply(lambda x: float(np.mean(np.abs(x))))
            .unstack(time)
            .reindex(index=selected, columns=times)
        )
        for identity, row in profiles.iterrows():
            axes[0].plot(range(len(times)), row, "-o", ms=2, lw=1, label=str(identity))
        axes[0].set_xticks(range(len(times)), times, rotation=45, fontsize=7)
        axes[0].set(
            xlabel=time,
            ylabel="mean absolute response",
            title="Saved age dependence of modeled sensitivity",
        )
        if len(selected) <= 12:
            axes[0].legend(fontsize=6)
        _save(fig, directory / "response_age_profiles", options, figures, "response_age_profiles")


def _distributions(frame, directory, options, figures):
    # Saved quantiles remain quantiles; they are never sampled into artificial cells.
    frame = frame.copy()
    if "population" not in frame:
        frame["population"] = "Saved population"
    time = _first(frame, ["target_time", "time", "actual_time"])
    if time is None or "gene" not in frame or "mean" not in frame:
        raise ValueError("Population-statistics table lacks gene, time, or actual means")
    chosen = _select(
        frame,
        "gene",
        "mean",
        time,
        options,
        directory,
        "Saved biological gene; observed temporal mean range, with actual cell distribution quantiles",
    )
    times = _order(frame[time].astype(str))
    frame[time] = frame[time].astype(str)
    for start in range(0, len(chosen), 12):
        page = chosen[start : start + 12]
        fig, axes = _axes(len(page), width=4, height=3)
        for gene, ax in zip(page, axes):
            for population, part in frame[frame.gene == gene].groupby("population", sort=False):
                grouped = part.groupby(time, observed=True).mean(numeric_only=True).reindex(times)
                line = ax.plot(
                    range(len(times)),
                    grouped["median"] if "median" in grouped else grouped["mean"],
                    "-o",
                    ms=2,
                    label=str(population),
                )[0]
                if {"q25", "q75"}.issubset(grouped):
                    ax.fill_between(
                        range(len(times)),
                        grouped.q25,
                        grouped.q75,
                        color=line.get_color(),
                        alpha=0.16,
                    )
            stride = max(1, math.ceil(len(times) / 8))
            ax.set_xticks(np.arange(len(times))[::stride], times[::stride], rotation=45, fontsize=7)
            ax.set_title(str(gene), fontsize=10)
        if len(frame.population.unique()) <= 14 and axes:
            axes[0].legend(fontsize=6)
        fig.suptitle(
            "Saved expression medians and cell IQR; bands are distributions, not replicate uncertainty",
            fontsize=11,
        )
        _save(
            fig,
            directory / f"gene_distribution_ribbons_{start//12+1:03d}",
            options,
            figures,
            "gene_distribution_ribbons",
        )


def _composition(frame, directory, options, figures):
    frame = frame.copy()
    state = _first(frame, ["celltype", "cell_type", "cell_state", "label", "state", "type_label"])
    time = _first(frame, ["time_value", "target_time", "time", "time_label", "day"])
    value = _first(
        frame,
        [
            "predicted_proportion",
            "prediction_proportion",
            "proportion",
            "fraction",
            "truth_proportion",
        ],
    )
    population = _first(frame, ["model_tag", "model", "population", "model_label"])
    if state is None or value is None:
        raise ValueError("State composition table lacks saved state/fraction columns")
    if population is None:
        frame["population"] = "Saved population"
        population = "population"
    if time is None:
        frame["_time"] = "all_saved"
        time = "_time"
    frame[time] = frame[time].astype(str)
    times = _order(frame[time])
    states = _order(frame[state])
    populations = list(frame[population].astype(str).unique())
    fig, axes = _axes(len(populations), width=5, height=4)
    for model, ax in zip(populations, axes):
        matrix = (
            frame[frame[population].astype(str) == model]
            .pivot_table(index=state, columns=time, values=value, aggfunc="mean")
            .reindex(index=states, columns=times)
        )
        _heatmap(ax, matrix, model, "viridis", limits=(0, 1))
    fig.suptitle(
        "Saved state fractions; input-conditioned labels are not autonomous composition predictions",
        fontsize=11,
    )
    _save(
        fig,
        directory / "saved_state_fraction_heatmap",
        options,
        figures,
        "saved_state_fraction_heatmap",
    )


def _occupancy(frame, directory, options, figures):
    if not {"author_node", "n_cells", "population"}.issubset(frame):
        raise ValueError("Author graph occupancy lacks real node/count/population")
    matrix = frame.pivot_table(
        index="author_node", columns="population", values="n_cells", aggfunc="sum"
    )
    fig, axes = _axes(
        1, width=max(6, len(matrix.columns) * 0.6), height=max(4, min(15, len(matrix) * 0.2))
    )
    im = _heatmap(
        axes[0], np.log1p(matrix), "Supplied author nodes: log1p(saved occupancy)", "viridis"
    )
    fig.colorbar(im, ax=axes[0], label="log1p count")
    _save(fig, directory / "author_node_occupancy", options, figures, "author_node_occupancy")


def _render_frame(frame, directory, options, figures, renderer):
    if frame.empty:
        raise ValueError("Saved table is empty")
    # Aggregate all-dataset tables must never pool different tissues. Generic
    # truth/prediction rows from different model evaluations retain their own
    # model/cohort truth instead of silently averaging into one prediction.
    if "dataset_id" in frame and frame.dataset_id.nunique() > 1:
        total = 0
        for identity, part in frame.groupby("dataset_id", sort=False):
            sub = directory / _safe(identity)
            sub.mkdir(exist_ok=True)
            total += _render_frame(
                part, sub, dict(options, _archive_view_types=set()), figures, renderer
            )
        return total
    if "model_label" in frame and frame.model_label.nunique() > 1:
        if ("population" in frame and frame.population.nunique() <= 2) or (
            renderer == "network"
            and {"truth_correlation", "prediction_correlation"}.issubset(frame)
        ):
            total = 0
            for identity, part in frame.groupby("model_label", sort=False):
                sub = directory / _safe(identity)
                sub.mkdir(exist_ok=True)
                total += _render_frame(
                    part, sub, dict(options, _archive_view_types=set()), figures, renderer
                )
            return total
        if "population" not in frame:
            frame = frame.copy()
            frame["population"] = frame.model_label.astype(str)
    if renderer == "network" and {"truth_correlation", "prediction_correlation"}.issubset(frame):
        common = [
            c
            for c in frame
            if c not in {"truth_correlation", "prediction_correlation", "population"}
        ]
        truth = frame[common + ["truth_correlation"]].rename(
            columns={"truth_correlation": "_association"}
        )
        truth["population"] = "Ground truth"
        predicted = frame[common + ["prediction_correlation"]].rename(
            columns={"prediction_correlation": "_association"}
        )
        predicted["population"] = (
            predicted["model_label"].astype(str)
            if "model_label" in predicted
            else "Saved prediction"
        )
        frame = pd.concat([truth, predicted], ignore_index=True)
    if renderer == "wide_timecourse":
        numeric = [c for c in frame if c != "stage" and pd.api.types.is_numeric_dtype(frame[c])]
        if not numeric:
            raise ValueError("Wide history table contains no actual numeric gene columns")
        frame = frame.melt(
            id_vars=["stage"], value_vars=numeric, var_name="gene", value_name="mean_expression"
        )
        renderer = "timecourse"
    if renderer == "paired_timecourse":
        common = [
            c for c in frame if c not in {"truth_mean", "prediction_mean", "prediction_minus_truth"}
        ]
        if not {"truth_mean", "prediction_mean"}.issubset(frame):
            raise ValueError("Paired pseudobulk table lacks saved truth and prediction means")
        truth = frame[common + ["truth_mean"]].rename(columns={"truth_mean": "mean_expression"})
        truth["population"] = "Ground truth"
        predicted = frame[common + ["prediction_mean"]].rename(
            columns={"prediction_mean": "mean_expression"}
        )
        predicted["population"] = (
            predicted["model_tag"].astype(str) if "model_tag" in predicted else "Saved prediction"
        )
        frame = pd.concat([truth, predicted], ignore_index=True)
        renderer = "timecourse"
    if renderer == "coordinates":
        _coordinates(frame, directory, options, figures)
    elif renderer == "centroids":
        _centroids(frame, directory, options, figures)
    elif renderer == "markers":
        _markers(frame, directory, options, figures)
    elif renderer == "timecourse":
        _timecourse(frame, directory, options, figures)
    elif renderer == "network":
        _network(frame, directory, options, figures)
    elif renderer == "perturbation":
        _perturbation(frame, directory, options, figures)
    elif renderer == "perturbation_programs":
        _perturbation(frame, directory, options, figures, program=True)
    elif renderer == "distributions":
        _distributions(frame, directory, options, figures)
    elif renderer == "composition":
        _composition(frame, directory, options, figures)
    elif renderer == "occupancy":
        _occupancy(frame, directory, options, figures)
    else:
        raise ValueError(f"Unknown biological table renderer: {renderer}")
    return len(frame)


def _render_table(path, directory, options, figures, renderer):
    return _render_frame(
        _read(path), directory, dict(options, _archive_view_types=set()), figures, renderer
    )


def _archive_table_job(task):
    item, directory, options = task
    path = Path(item["source_path"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    figures = []
    qid = f"LEG_{item['paper_figure']}_{item['family']}"
    row = {**item, "question_id": qid, "rerendered_paths": "", "rows": 0}
    try:
        row["rows"] = _render_table(path, directory, options, figures, item["renderer"])
        if not figures:
            raise ValueError("Biological renderer produced no figures")
        row["status"] = "BIOLOGICAL_RERENDERED"
        row["rerendered_paths"] = ";".join(f["path"] for f in figures)
        (directory / "SOURCE.json").write_text(
            json.dumps(
                {
                    "source_table": str(path),
                    "renderer": item["renderer"],
                    "rows": row["rows"],
                    "interpretation": "Authentic saved biological numerical data; historical identities retained; no artificial cell reconstruction",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        row["status"] = "PARTIAL_TABLE_RERENDER" if figures else "UNSUPPORTED_TABLE"
        row["reason"] = f"{type(exc).__name__}: {exc}"
        row["rerendered_paths"] = ";".join(f["path"] for f in figures)
        _mpl().close("all")
    return row, figures


def _gallery(directory, title, figures, note):
    items = []
    for record in figures:
        path = Path(record["path"])
        try:
            relative = path.relative_to(directory).as_posix()
        except ValueError:
            continue
        label = html.escape(record.get("view", path.name))
        url = html.escape(relative, quote=True)
        if path.suffix.lower() in {".png", ".gif", ".svg"}:
            items.append(
                f'<figure><a href="{url}"><img src="{url}" loading="lazy" alt="{label}"></a><figcaption>{label}</figcaption></figure>'
            )
        else:
            items.append(f'<p><a href="{url}">{label}</a></p>')
    page = (
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title><style>body{{font:15px system-ui;margin:30px;background:#faf9f5}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}figure{{margin:0;background:white;padding:12px}}img{{width:100%;height:320px;object-fit:contain}}figcaption{{font-size:12px}}</style><h1>{html.escape(title)}</h1><p>{html.escape(note)}</p><div class="grid">'
        + "".join(items)
        + "</div>"
    )
    (directory / "INDEX.html").write_text(page, encoding="utf-8")


def render_archive(roots, out, options, progress: Callable | None = None) -> dict:
    """Render typed saved biology and preserve complete original-figure coverage."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory = inventory_legacy(roots, out)
    ledger = []
    table_ledger = []
    all_figures = []
    questions = {}
    errors = []
    original_items = inventory["figures"]
    typed_tables = [
        t for t in inventory["tables"] if t["known_biological_table"] and t["input_nonempty"]
    ]
    for item in inventory["tables"]:
        if item["known_biological_table"] and not item["input_nonempty"]:
            table_ledger.append(
                {
                    **item,
                    "question_id": f"LEG_{item['paper_figure']}_{item['family']}",
                    "status": "EMPTY_INPUT",
                    "rows": 0,
                    "rerendered_paths": "",
                    "reason": "Saved table has no data rows; excluded from drawing jobs",
                }
            )
    total = len(original_items) + len(typed_tables)
    completed = 0
    for item in original_items:
        path = Path(item["source_path"])
        qid = item["question_id"]
        qdir = out / qid
        destination = (
            qdir / "legacy_originals" / (_key(path) + "__" + _safe(path.stem) + path.suffix.lower())
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        record = {
            "path": str(destination),
            "view": "Historical original: " + path.name,
            "entity": path.stem,
            "model": "actual historical identity",
            "origin": "legacy original",
        }
        all_figures.append(record)
        questions.setdefault(
            qid, {"figure": item["paper_figure"], "family": item["family"], "figures": []}
        )["figures"].append(record)
        ledger.append(
            {
                **item,
                "preserved_path": str(destination),
                "status": "PRESERVED_ORIGINAL",
                "rerendered_paths": "",
            }
        )
        completed += 1
        if progress:
            progress(
                {
                    "completed": completed,
                    "total": total,
                    "stage": "preserve_legacy_original",
                    "source": str(path),
                }
            )
    tasks = []
    for item in typed_tables:
        path = Path(item["source_path"])
        qid = f"LEG_{item['paper_figure']}_{item['family']}"
        directory = out / qid / "rerendered" / (_safe(path.parent.parent.name) + "__" + _key(path))
        tasks.append((item, str(directory), options))
    from .util import resources

    workers, _ = resources(
        options.get("workers", "auto"),
        options.get("cpu_reserve", 0),
        float(options.get("archive_worker_memory_gb", 0.8)),
    )
    workers = min(workers, max(1, len(tasks)))
    print(
        f"Preserved {len(ledger)} historical figure files; re-rendering {len(tasks)} biological tables with {workers} CPU processes.",
        flush=True,
    )
    if workers == 1:
        iterator = map(_archive_table_job, tasks)
        executor = None
    else:
        import concurrent.futures as futures
        import multiprocessing as mp

        executor = futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        )
        pending = [executor.submit(_archive_table_job, task) for task in tasks]
        iterator = (future.result() for future in futures.as_completed(pending))
    from tqdm import tqdm

    try:
        with tqdm(
            total=len(tasks),
            unit="biology table",
            dynamic_ncols=True,
            mininterval=1,
            disable=progress is not None,
        ) as bar:
            for row, figures in iterator:
                qid = row["question_id"]
                if figures:
                    all_figures.extend(figures)
                    questions.setdefault(
                        qid, {"figure": row["paper_figure"], "family": row["family"], "figures": []}
                    )["figures"].extend(figures)
                if row["status"] != "BIOLOGICAL_RERENDERED":
                    errors.append(
                        {
                            "source_table": row["source_path"],
                            "reason": row.get("reason", "No complete biological re-render"),
                        }
                    )
                table_ledger.append(row)
                completed += 1
                bar.update(1)
                bar.set_postfix_str(
                    f"{row['family']}: {row['status']}; new exports={sum(f['origin']=='rerendered saved biological table' for f in all_figures)}"
                )
                if progress:
                    progress(
                        {
                            "completed": completed,
                            "total": total,
                            "stage": "rerender_saved_biology",
                            "source": row["source_path"],
                            "status": row["status"],
                        }
                    )
    except BaseException:
        if executor is not None:
            processes = list((getattr(executor, "_processes", None) or {}).values())
            executor.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=0.5)
            executor = None
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    for qid, question in questions.items():
        family = question["family"]
        qdir = out / qid
        note = "Historical originals and genuine re-renders of saved source tables. These retain their actual historical/current-table model identities and are separate from newly computed v5 all-model comparisons. Density/centroid paths are descriptive population geometry; saved compact means are not per-cell data."
        _gallery(qdir, FAMILY_TITLES.get(family, family), question["figures"], note)
        (qdir / "QUESTION.json").write_text(
            json.dumps(
                {
                    "id": qid,
                    "paper_figure": question["figure"],
                    "family": family,
                    "title": FAMILY_TITLES.get(family, family),
                    "origin": "legacy/table biology",
                    "interpretation": note,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    pd.DataFrame(ledger).to_csv(out / "LEGACY_COVERAGE_LEDGER.csv", index=False)
    pd.DataFrame(ledger).to_csv(out / "LEGACY_COVERAGE.csv", index=False)
    pd.DataFrame(table_ledger).to_csv(out / "BIOLOGICAL_TABLE_RERENDER_LEDGER.csv", index=False)
    summary = {
        **inventory["summary"],
        "preserved_originals": len(ledger),
        "preserved_pngs": sum(row["extension"] == ".png" for row in ledger),
        "coverage_complete": len(ledger) == len(original_items)
        and not inventory["summary"]["missing_roots"],
        "biological_tables_rerendered": sum(
            t["status"] == "BIOLOGICAL_RERENDERED" for t in table_ledger
        ),
        "rerendered_biological_figures": sum(
            f["origin"] == "rerendered saved biological table" for f in all_figures
        ),
        "question_folders": len(questions),
        "unsupported_tables": len(errors),
        "errors": errors,
    }
    (out / "ARCHIVE_REPORT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    links = "".join(
        f'<li><a href="{html.escape(qid)}/INDEX.html">{html.escape(qid)}: {html.escape(FAMILY_TITLES.get(q["family"],q["family"]))}</a> ({len(q["figures"])} files)</li>'
        for qid, q in sorted(questions.items())
    )
    (out / "INDEX.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Saved biological evaluation gallery</title><h1>Saved biological evaluation gallery</h1><p>Historical originals and numerical biological re-renders; current v5 comparisons are separate.</p><ul>'
        + links
        + "</ul>",
        encoding="utf-8",
    )
    return {
        "status": "complete" if summary["coverage_complete"] and not errors else "partial",
        "summary": summary,
        "figures": all_figures,
        "tables": [
            str(out / "LEGACY_COVERAGE_LEDGER.csv"),
            str(out / "BIOLOGICAL_TABLE_RERENDER_LEDGER.csv"),
        ],
        "notes": [
            "Legacy numerical tables are re-rendered with their actual identities; original biological families are completely preserved."
        ],
        "errors": errors,
    }
