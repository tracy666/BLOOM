#!/usr/bin/env python3
"""Convert official STED-EC author files into validated FM v5 priors.

The converter mirrors the contract of build_author_based_priors_for_fm_v1.py:
author sources are never edited, model-aligned outputs are accompanied by
all-gene provenance tables, mappings use exact cell IDs, and every conversion
decision is recorded in JSON reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

VERSION = "1.0.0"
NODE_SEPARATOR = "::"


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "na", "n/a"} else text


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).lower())


def canonical_state(value: Any) -> str:
    state = " ".join(clean_text(value).split())
    aliases = {
        "Endocardial Ecs": "Endocardial ECs",
        "Undifferentiated": "UOD ECs",
    }
    return aliases.get(state, state)


def canonical_node(state: Any, period: Any) -> str:
    state_text = canonical_state(state)
    period_text = clean_text(period)
    if not state_text or not period_text:
        return ""
    # Match the previous converter's cluster_label::age convention.
    return f"{state_text}{NODE_SEPARATOR}{period_text}"


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    return value


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(data), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_table(dataframe: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix.lower() == ".parquet":
        dataframe.to_parquet(temporary, index=False)
    elif path.name.endswith(".tsv.gz"):
        dataframe.to_csv(temporary, sep="\t", index=False, compression="gzip")
    else:
        dataframe.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)
    log(f"[write] {path} rows={len(dataframe):,}")


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def infer_source(explicit: str, author_dir: Path, relative: str, label: str) -> Path:
    path = Path(explicit).expanduser().resolve() if clean_text(explicit) else author_dir / relative
    return require_file(path, label).resolve()


def read_delimited(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)


def resolve_column(
    dataframe: pd.DataFrame,
    requested: str,
    candidates: Sequence[str],
    *,
    label: str,
    required: bool = True,
) -> Optional[str]:
    lookup = {normalize_name(column): str(column) for column in dataframe.columns}
    if clean_text(requested):
        result = lookup.get(normalize_name(requested))
        if result is not None:
            return result
        if required:
            raise KeyError(
                f"Requested {label} column {requested!r} not found; columns={list(dataframe.columns)}"
            )
        return None
    for candidate in candidates:
        result = lookup.get(normalize_name(candidate))
        if result is not None:
            return result
    if required:
        raise KeyError(
            f"Could not infer {label} column from {list(candidates)}; columns={list(dataframe.columns)}"
        )
    return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def dataframe_preview(dataframe: pd.DataFrame, rows: int = 5) -> Dict[str, Any]:
    preview = dataframe.head(rows).astype(object).where(pd.notna(dataframe.head(rows)), "")
    return {
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "columns": [str(column) for column in dataframe.columns],
        "head": preview.astype(str).to_dict(orient="records"),
    }


def read_model_genes(
    path: Path, requested_gene_col: str, hvg_col: str
) -> Tuple[Set[str], Dict[str, Any]]:
    dataframe = read_delimited(path)
    original_rows = len(dataframe)
    hvg_mask_used = False
    resolved_hvg = resolve_column(
        dataframe, hvg_col, ["highly_variable", "hvg"], label="HVG flag", required=False
    )
    if resolved_hvg is not None:
        mask = dataframe[resolved_hvg].fillna(False).astype(bool)
        if int(mask.sum()) > 0:
            dataframe = dataframe.loc[mask].copy()
            hvg_mask_used = True
    gene_col = resolve_column(
        dataframe,
        requested_gene_col,
        ["gene_short_name", "gene_name", "gene_symbol", "symbol", "gene", "feature_name"],
        label="model gene",
        required=False,
    )
    if gene_col is None:
        if isinstance(dataframe.index, pd.RangeIndex):
            raise KeyError(f"No model-gene column and no informative index in {path}")
        genes = [clean_text(value) for value in dataframe.index]
        gene_source = "__index__"
    else:
        genes = [clean_text(value) for value in dataframe[gene_col]]
        gene_source = gene_col
    genes = [gene for gene in genes if gene]
    duplicates = [gene for gene, count in Counter(genes).items() if count > 1]
    if duplicates:
        raise ValueError(f"Model genes are not unique; examples={sorted(duplicates)[:20]}")
    if not genes:
        raise ValueError("Model var table selected zero genes")
    return set(genes), {
        "path": str(path),
        "rows_original": int(original_rows),
        "hvg_mask_used": bool(hvg_mask_used),
        "hvg_column": resolved_hvg,
        "gene_column": gene_source,
        "n_model_genes": int(len(genes)),
        "examples": sorted(genes)[:20],
    }


def day_to_period_map(obs: pd.DataFrame, day_col: str, period_col: str) -> Dict[float, str]:
    pairs = obs[[day_col, period_col]].dropna().drop_duplicates().copy()
    pairs[day_col] = pd.to_numeric(pairs[day_col], errors="raise")
    ambiguous = pairs.groupby(day_col, observed=True)[period_col].nunique()
    if int((ambiguous != 1).sum()) > 0:
        raise ValueError("Each numeric day must map to exactly one embryonic period")
    return {float(row[day_col]): clean_text(row[period_col]) for _, row in pairs.iterrows()}


def assert_no_bad_edges(graph: pd.DataFrame) -> None:
    if graph.empty:
        raise ValueError("STED-EC curated graph contains zero edges")
    if bool(graph["source"].eq(graph["target"]).any()):
        raise ValueError("STED-EC curated graph contains self-loops")
    if bool(graph.duplicated(["source", "target"]).any()):
        raise ValueError("STED-EC curated graph contains duplicate edges")
    if bool((graph["source_day"] >= graph["target_day"]).any()):
        raise ValueError("STED-EC curated graph contains non-forward developmental edges")


def graph_is_dag(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> bool:
    children: Dict[str, Set[str]] = defaultdict(set)
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        if target not in children[source]:
            children[source].add(target)
            indegree[target] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in children[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(nodes)


def build_graph(
    lineage_path: Path,
    obs: pd.DataFrame,
    day_col: str,
    period_col: str,
    report: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lineage = pd.read_csv(lineage_path, low_memory=False)
    unnamed = [column for column in lineage.columns if normalize_name(column).startswith("unnamed")]
    if unnamed:
        lineage_name_col = unnamed[0]
        lineage_names = lineage[lineage_name_col].map(clean_text).tolist()
        lineage = lineage.drop(columns=unnamed)
    else:
        lineage_name_col = "__row_number__"
        lineage_names = [f"lineage_{index}" for index in range(len(lineage))]
    if lineage.empty or lineage.shape[1] < 2:
        raise ValueError("lineages_new.csv must contain lineage rows and at least two time columns")

    time_columns = list(lineage.columns)
    try:
        days = [float(column) for column in time_columns]
    except ValueError as error:
        raise ValueError(
            f"All lineage columns after the first must be numeric days: {time_columns}"
        ) from error
    if any(right <= left for left, right in zip(days[:-1], days[1:])):
        raise ValueError("Author lineage time columns must be strictly increasing")
    day_period = day_to_period_map(obs, day_col, period_col)
    missing_days = [day for day in days if day not in day_period]
    if missing_days:
        raise ValueError(f"Author lineage days absent from obs: {missing_days}")

    edge_support: Counter[Tuple[str, str]] = Counter()
    edge_lineages: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    node_meta: Dict[str, Dict[str, Any]] = {}
    node_lineages: Dict[str, Set[str]] = defaultdict(set)
    for row_number, (_, row) in enumerate(lineage.iterrows()):
        name = lineage_names[row_number] or f"lineage_{row_number}"
        row_nodes = []
        for column, day in zip(time_columns, days):
            state = canonical_state(row[column])
            if not state:
                raise ValueError(f"Missing lineage state in row {row_number}, day {day}")
            period = day_period[day]
            node = canonical_node(state, period)
            row_nodes.append(node)
            node_meta[node] = {"label": state, "embryonic_period": period, "day": day}
            node_lineages[node].add(name)
        for source, target in zip(row_nodes[:-1], row_nodes[1:]):
            edge_support[(source, target)] += 1
            edge_lineages[(source, target)].add(name)

    ordered_edges = sorted(
        edge_support, key=lambda edge: (node_meta[edge[0]]["day"], edge[0], edge[1])
    )
    graph = pd.DataFrame(
        {
            "source": [edge[0] for edge in ordered_edges],
            "target": [edge[1] for edge in ordered_edges],
            "weight": np.ones(len(ordered_edges), dtype=np.float32),
            "supporting_lineages": [int(edge_support[edge]) for edge in ordered_edges],
            "lineage_names": ["|".join(sorted(edge_lineages[edge])) for edge in ordered_edges],
            "source_label": [node_meta[edge[0]]["label"] for edge in ordered_edges],
            "target_label": [node_meta[edge[1]]["label"] for edge in ordered_edges],
            "source_period": [node_meta[edge[0]]["embryonic_period"] for edge in ordered_edges],
            "target_period": [node_meta[edge[1]]["embryonic_period"] for edge in ordered_edges],
            "source_day": [node_meta[edge[0]]["day"] for edge in ordered_edges],
            "target_day": [node_meta[edge[1]]["day"] for edge in ordered_edges],
            "author_source": "ot_pairs/lineages_new.csv",
        }
    )
    ordered_nodes = sorted(
        node_meta, key=lambda node: (node_meta[node]["day"], node_meta[node]["label"])
    )
    nodes = pd.DataFrame(
        {
            "node": ordered_nodes,
            "label": [node_meta[node]["label"] for node in ordered_nodes],
            "embryonic_period": [node_meta[node]["embryonic_period"] for node in ordered_nodes],
            "day": [node_meta[node]["day"] for node in ordered_nodes],
            "author_lineages": ["|".join(sorted(node_lineages[node])) for node in ordered_nodes],
            "author_source": "ot_pairs/lineages_new.csv",
        }
    )
    assert_no_bad_edges(graph)
    dag = graph_is_dag(
        nodes["node"].tolist(), list(graph[["source", "target"]].itertuples(index=False, name=None))
    )
    if not dag:
        raise ValueError("Converted STED-EC author graph is not acyclic")
    report["sted_graph"] = {
        "lineage_path": str(lineage_path),
        "lineage_name_column": lineage_name_col,
        "time_columns": time_columns,
        "n_author_lineages": int(len(lineage)),
        "n_edges": int(len(graph)),
        "n_nodes": int(len(nodes)),
        "n_roots": int(len(set(graph["source"]) - set(graph["target"]))),
        "n_terminals": int(len(set(graph["target"]) - set(graph["source"]))),
        "is_dag": bool(dag),
        "n_backward_or_same_day_edges": int((graph["source_day"] >= graph["target_day"]).sum()),
        "node_definition": "label::embryonic_period",
        "primary_weighting": "uniform topology weight=1.0",
        "supporting_lineages_is_provenance_not_weight": True,
    }
    return graph, nodes


def build_cell_map(
    args: argparse.Namespace,
    obs: pd.DataFrame,
    nodes: pd.DataFrame,
    report: Dict[str, Any],
) -> Path:
    label_col = resolve_column(
        obs, args.obs_label_col, ["label", "celltype", "cell_type"], label="obs label"
    )
    period_col = resolve_column(
        obs,
        args.obs_period_col,
        ["embryonic_period", "age", "stage", "time"],
        label="obs embryonic period",
    )
    cell_id_col = resolve_column(
        obs,
        args.cell_id_col,
        ["cell_id", "cell", "barcode", "cell_barcode"],
        label="cell ID",
        required=False,
    )
    if cell_id_col is None:
        if not obs.index.is_unique:
            raise ValueError("obs.index is not unique and no unique --cell-id-col was supplied")
        cell_ids = pd.Series(obs.index.astype(str), index=obs.index)
        cell_id_source = "obs.index"
    else:
        cell_ids = obs[cell_id_col].map(clean_text)
        if bool(cell_ids.eq("").any()):
            raise ValueError(f"Cell-ID column {cell_id_col!r} contains missing values")
        if not cell_ids.is_unique:
            examples = cell_ids[cell_ids.duplicated()].unique()[:20].tolist()
            raise ValueError(f"Cell-ID column {cell_id_col!r} is not unique; examples={examples}")
        cell_id_source = cell_id_col

    graph_nodes = pd.Series(
        [canonical_node(label, period) for label, period in zip(obs[label_col], obs[period_col])],
        index=obs.index,
    )
    valid_nodes = set(nodes["node"].astype(str))
    mapped = graph_nodes.isin(valid_nodes)
    mapping = pd.DataFrame(
        {
            "cell_id": cell_ids.to_numpy(),
            "graph_node": graph_nodes.to_numpy(),
            "label": obs[label_col].map(canonical_state).to_numpy(),
            "embryonic_period": obs[period_col].map(clean_text).to_numpy(),
            "mapped": mapped.to_numpy(dtype=bool),
        }
    )
    try:
        import pyarrow  # type: ignore  # noqa: F401

        output_path = args.output_dir / "sted_cell_to_graph_node.parquet"
    except Exception:
        output_path = args.output_dir / "sted_cell_to_graph_node.tsv.gz"
        log("[sted][cell-map] pyarrow unavailable; writing compressed TSV")
    coverage = float(mapping["mapped"].mean()) if len(mapping) else 0.0
    unmapped = mapping.loc[~mapping["mapped"]]
    report["sted_cell_map"] = {
        "created": True,
        "path": str(output_path),
        "resolved_columns": {
            "cell_id": cell_id_source,
            "label": label_col,
            "embryonic_period": period_col,
        },
        "n_cells": int(len(mapping)),
        "n_mapped_cells": int(mapping["mapped"].sum()),
        "n_unmapped_cells": int((~mapping["mapped"]).sum()),
        "cell_coverage": coverage,
        "n_distinct_mapped_nodes": int(mapping.loc[mapping["mapped"], "graph_node"].nunique()),
        "unmapped_label_counts": unmapped["label"].value_counts().to_dict(),
        "examples_unmapped": (
            unmapped[["label", "embryonic_period", "graph_node"]]
            .drop_duplicates()
            .head(40)
            .to_dict(orient="records")
        ),
    }
    if args.strict and coverage < float(args.minimum_cell_coverage):
        raise ValueError(
            f"STED-EC graph cell coverage {coverage:.4f} is below "
            f"--minimum-cell-coverage={args.minimum_cell_coverage}"
        )
    write_table(mapping, output_path, overwrite=args.overwrite)
    return output_path


def build_drivers(
    args: argparse.Namespace,
    marker_path: Path,
    nodes: pd.DataFrame,
    model_genes: Set[str],
    report: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    markers = pd.read_csv(marker_path, low_memory=False)
    gene_col = resolve_column(
        markers, "", ["names", "gene", "symbol", "gene_name"], label="marker gene"
    )
    state_col = resolve_column(
        markers, "", ["state", "celltype", "cell_type"], label="marker state"
    )
    logfc_col = resolve_column(
        markers, "", ["logfoldchanges", "logfc", "log2fc"], label="marker logFC", required=False
    )
    statistic_col = resolve_column(
        markers, "", ["scores", "score", "statistic"], label="marker statistic", required=False
    )
    group_col = resolve_column(
        markers, "", ["group", "maturity_group"], label="marker group", required=False
    )
    markers = markers.copy()
    markers["gene"] = markers[gene_col].map(clean_text)
    markers["program_state"] = markers[state_col].map(canonical_state)
    markers["author_logfoldchange"] = (
        pd.to_numeric(markers[logfc_col], errors="coerce") if logfc_col else np.nan
    )
    markers["author_marker_statistic"] = (
        pd.to_numeric(markers[statistic_col], errors="coerce") if statistic_col else np.nan
    )
    markers["author_group"] = markers[group_col].map(clean_text) if group_col else ""
    valid = markers["gene"].str.len().gt(0) & markers["program_state"].str.len().gt(0)
    markers = markers.loc[valid].drop_duplicates(["program_state", "gene"]).copy()
    markers["logfc_sensitivity_score"] = markers.groupby("program_state", observed=True)[
        "author_logfoldchange"
    ].transform(
        lambda values: (
            values.abs() / values.abs().max()
            if values.notna().any() and float(values.abs().max()) > 0
            else 1.0
        )
    )

    expanded_parts = []
    for state, state_nodes in nodes.groupby("label", observed=True):
        local = markers.loc[markers["program_state"].eq(state)].copy()
        if local.empty:
            continue
        for node in state_nodes["node"].astype(str):
            part = local.copy()
            part.insert(0, "node", node)
            expanded_parts.append(part)
    if not expanded_parts:
        raise ValueError("No author marker states could be mapped to graph nodes")
    expanded = pd.concat(expanded_parts, ignore_index=True)
    primary = pd.DataFrame(
        {
            "node": expanded["node"],
            "gene": expanded["gene"],
            # The author maturity score is a mean across markers; equal weights
            # are therefore the least transformed primary representation.
            "score": np.ones(len(expanded), dtype=np.float32),
            "program_state": expanded["program_state"],
            "driver_type": "author_state_or_maturity_marker",
            "author_group": expanded["author_group"],
            "author_logfoldchange": expanded["author_logfoldchange"],
            "author_marker_statistic": expanded["author_marker_statistic"],
            "author_source": "DATA/Top15_marker_score_states.csv",
        }
    )
    sensitivity = primary.copy()
    sensitivity["score"] = (
        expanded["logfc_sensitivity_score"].fillna(1.0).to_numpy(dtype=np.float32)
    )
    sensitivity["driver_type"] = "author_marker_logfc_sensitivity"
    model_mask = primary["gene"].isin(model_genes)
    model_primary = primary.loc[model_mask].reset_index(drop=True)
    model_sensitivity = sensitivity.loc[model_mask].reset_index(drop=True)
    states_in_graph = set(nodes["label"].astype(str))
    states_in_markers = set(markers["program_state"].astype(str))
    states_in_model = set(model_primary["program_state"].astype(str))
    report["sted_drivers"] = {
        "marker_path": str(marker_path),
        "resolved_columns": {
            "gene": gene_col,
            "state": state_col,
            "logfoldchange": logfc_col,
            "statistic": statistic_col,
            "group": group_col,
        },
        "n_author_marker_rows": int(len(markers)),
        "n_author_marker_genes": int(markers["gene"].nunique()),
        "author_marker_states": sorted(states_in_markers),
        "graph_states": sorted(states_in_graph),
        "unused_marker_states_absent_from_graph": sorted(states_in_markers - states_in_graph),
        "graph_states_without_author_markers": sorted(states_in_graph - states_in_markers),
        "all_gene_node_rows": int(len(primary)),
        "model_gene_node_rows": int(len(model_primary)),
        "unique_author_marker_genes_in_model": int(model_primary["gene"].nunique()),
        "nodes_with_model_markers": int(model_primary["node"].nunique()),
        "nodes_without_model_markers": int(
            nodes["node"].nunique() - model_primary["node"].nunique()
        ),
        "graph_states_without_model_markers": sorted(states_in_graph - states_in_model),
        "primary_score_definition": "uniform score=1 because the author maturity score averages marker expression",
        "sensitivity_score_definition": "absolute author logfoldchange normalized within marker state",
        "scientific_interpretation": "state/maturity program, not a claim of causal transition drivers",
    }
    if args.strict and model_primary.empty:
        raise ValueError("STED-EC author program has zero overlap with model genes")
    return model_primary, primary, model_sensitivity


def build_schema_report(
    args: argparse.Namespace,
    lineage_path: Path,
    marker_path: Path,
    obs_path: Path,
    model_var_path: Path,
) -> Dict[str, Any]:
    lineage = pd.read_csv(lineage_path, low_memory=False)
    markers = pd.read_csv(marker_path, low_memory=False)
    obs = read_delimited(obs_path)
    var = read_delimited(model_var_path)
    return {
        "converter_version": VERSION,
        "dataset": "sted_ec",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "author_prior_dir": str(args.author_prior_dir),
        "sources": {
            "lineages_new": {**file_record(lineage_path), **dataframe_preview(lineage)},
            "top15_marker_states": {**file_record(marker_path), **dataframe_preview(markers)},
            "obs": {**file_record(obs_path), **dataframe_preview(obs)},
            "model_var": {**file_record(model_var_path), **dataframe_preview(var)},
        },
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert official STED-EC author lineage and marker tables into FM v5 priors.",
    )
    parser.add_argument("--author-prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--obs-path", type=Path, required=True)
    parser.add_argument("--model-var-path", type=Path, required=True)
    parser.add_argument("--lineages-path", default="")
    parser.add_argument("--marker-path", default="")
    parser.add_argument("--obs-label-col", default="label")
    parser.add_argument("--obs-period-col", default="embryonic_period")
    parser.add_argument("--obs-day-col", default="day")
    parser.add_argument("--cell-id-col", default="")
    parser.add_argument("--model-gene-column", default="")
    parser.add_argument("--model-hvg-column", default="highly_variable")
    parser.add_argument("--minimum-cell-coverage", type=float, default=0.80)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inspect-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.author_prior_dir = args.author_prior_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.obs_path = args.obs_path.expanduser().resolve()
    args.model_var_path = args.model_var_path.expanduser().resolve()
    if not args.author_prior_dir.is_dir():
        raise NotADirectoryError(f"Author-prior directory not found: {args.author_prior_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lineage_path = infer_source(
        args.lineages_path,
        args.author_prior_dir,
        "ot_pairs/lineages_new.csv",
        "adjusted lineage table",
    )
    marker_path = infer_source(
        args.marker_path,
        args.author_prior_dir,
        "DATA/Top15_marker_score_states.csv",
        "author marker table",
    )
    obs_path = require_file(args.obs_path, "obs table")
    model_var_path = require_file(args.model_var_path, "model var table")
    schema = build_schema_report(args, lineage_path, marker_path, obs_path, model_var_path)
    schema_path = args.output_dir / "input_schema_report.json"
    if schema_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {schema_path}")
    write_json(schema, schema_path)
    log(f"[inspect] wrote {schema_path}")
    if args.inspect_only:
        log("[done] inspect-only mode completed")
        return

    report: Dict[str, Any] = {
        "status": "running",
        "converter_version": VERSION,
        "dataset": "sted_ec",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "author_prior_dir": str(args.author_prior_dir),
        "output_dir": str(args.output_dir),
        "strict": bool(args.strict),
        "minimum_cell_coverage": float(args.minimum_cell_coverage),
        "inputs": {
            "lineages_new": file_record(lineage_path),
            "top15_marker_states": file_record(marker_path),
            "obs": file_record(obs_path),
            "model_var": file_record(model_var_path),
        },
    }
    try:
        obs = read_delimited(obs_path)
        day_col = resolve_column(
            obs, args.obs_day_col, ["day", "age_numeric"], label="obs numeric day"
        )
        period_col = resolve_column(
            obs,
            args.obs_period_col,
            ["embryonic_period", "age", "stage", "time"],
            label="obs embryonic period",
        )
        graph, nodes = build_graph(lineage_path, obs, day_col, period_col, report)
        write_table(graph, args.output_dir / "sted_curated_graph.tsv", overwrite=args.overwrite)
        write_table(nodes, args.output_dir / "sted_graph_nodes.tsv", overwrite=args.overwrite)

        model_genes, model_report = read_model_genes(
            model_var_path,
            args.model_gene_column,
            args.model_hvg_column,
        )
        report["model_genes"] = model_report
        model_program, all_program, sensitivity_program = build_drivers(
            args,
            marker_path,
            nodes,
            model_genes,
            report,
        )
        write_table(
            model_program, args.output_dir / "sted_driver_program.tsv", overwrite=args.overwrite
        )
        write_table(
            all_program,
            args.output_dir / "sted_driver_program_all_genes.tsv",
            overwrite=args.overwrite,
        )
        write_table(
            sensitivity_program,
            args.output_dir / "sted_driver_program_logfc_sensitivity.tsv",
            overwrite=args.overwrite,
        )
        build_cell_map(args, obs, nodes, report)
        report["status"] = "success"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(report, args.output_dir / "conversion_report.json")
        raise
    write_json(report, args.output_dir / "conversion_report.json")
    log(f"[done] success; report={args.output_dir / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
