#!/usr/bin/env python3
"""Convert released developmental graphs and gene programmes into BLOOM priors.

The original converter supports visual cortex (``--dataset small``) and prenatal
mouse development (``--dataset big``). It preserves the publication-derived
sheet selection, filtering, edge weights, cell mappings and conversion reports.
Expression matrices and transport couplings are not modified by this step.

Use ``python -m bloom.preprocessing.author_priors --help`` for all options.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

VERSION = "1.1.1"
NODE_SEPARATOR = "::"
KNOWN_SMALL_LINEAGES = ("Glut", "Glia", "MGE", "CGE")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def normalize_name(value: Any) -> str:
    text = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "na", "n/a"} else text


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    return value


def write_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(data), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def write_table(dataframe: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    temporary_path = path.with_name(path.name + ".tmp")
    if path.suffix.lower() == ".parquet":
        dataframe.to_parquet(temporary_path, index=False)
    elif path.name.endswith(".tsv.gz"):
        dataframe.to_csv(temporary_path, sep="\t", index=False, compression="gzip")
    else:
        dataframe.to_csv(temporary_path, sep="\t", index=False)
    os.replace(temporary_path, path)
    log(f"[write] {path} rows={len(dataframe):,}")


def file_record(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    result: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        result.update({"size_bytes": int(stat.st_size), "modified_ns": int(stat.st_mtime_ns)})
    return result


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def infer_existing_path(
    explicit: str, directory: Path, candidates: Sequence[str], label: str
) -> Path:
    if clean_text(explicit):
        return require_file(Path(explicit).expanduser().resolve(), label)
    for candidate in candidates:
        path = directory / candidate
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Could not locate {label} under {directory}. Tried: {list(candidates)}. "
        "Provide the corresponding explicit command-line option."
    )


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
    original_by_normalized = {normalize_name(column): str(column) for column in dataframe.columns}
    if clean_text(requested):
        normalized = normalize_name(requested)
        if normalized in original_by_normalized:
            return original_by_normalized[normalized]
        if required:
            raise KeyError(
                f"Requested {label} column {requested!r} not found; columns={list(dataframe.columns)}"
            )
        return None
    for candidate in candidates:
        normalized = normalize_name(candidate)
        if normalized in original_by_normalized:
            return original_by_normalized[normalized]
    if required:
        raise KeyError(
            f"Could not infer {label} column from {list(candidates)}; columns={list(dataframe.columns)}"
        )
    return None


def canonical_yes(value: Any) -> bool:
    return normalize_name(value) in {
        "yes",
        "y",
        "true",
        "t",
        "1",
        "included",
        "include",
        "selected",
    }


def canonical_edge_type(value: Any) -> str:
    normalized = normalize_name(value)
    if "development" in normalized or "progression" in normalized:
        return "developmental_progression"
    if "spatial" in normalized or "continuity" in normalized:
        return "spatial_continuity"
    if (
        "equiv" in normalized
        or "alias" in normalized
        or "dataset" in normalized
        or "same" in normalized
    ):
        return "dataset_equivalence"
    return normalized or "unknown"


def canonical_small_node(cluster_label: Any, age: Any) -> str:
    cluster = clean_text(cluster_label)
    age_text = clean_text(age)
    if not cluster or not age_text:
        return ""
    return f"{cluster}{NODE_SEPARATOR}{age_text}"


def developmental_age_order(value: Any) -> float:
    text = clean_text(value).upper().replace(" ", "")
    match = re.search(r"([EP])\s*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        number = float(match.group(2))
        return number if match.group(1) == "E" else 1000.0 + number
    match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", text)
    return float(match.group(0)) if match else float("nan")


def split_gene_tokens(value: Any) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    text = re.sub(r"[\[\]\{\}\(\)\"']", " ", text)
    tokens = [token.strip() for token in re.split(r"[,;|\n]+", text) if token.strip()]
    result: List[str] = []
    for token in tokens:
        if " " in token:
            whitespace_tokens = [part for part in token.split() if part]
            if len(whitespace_tokens) > 1 and all(
                re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", part) for part in whitespace_tokens
            ):
                result.extend(whitespace_tokens)
                continue
        result.append(token)
    return result


def looks_like_gene_symbol(value: Any) -> bool:
    text = clean_text(value)
    if not text or len(text) > 80 or " " in text:
        return False
    if normalize_name(text) in {
        "gene",
        "genes",
        "module",
        "class",
        "subclass",
        "cluster",
        "age",
        "score",
        "weight",
        "yes",
        "no",
        "true",
        "false",
        "glut",
        "glia",
        "mge",
        "cge",
    }:
        return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*", text)) and not text.isdigit()


def dataframe_preview(dataframe: pd.DataFrame, rows: int = 3) -> Dict[str, Any]:
    preview = dataframe.head(rows).astype(object)
    preview = preview.where(pd.notna(preview), "")
    return {
        "shape": [int(dataframe.shape[0]), int(dataframe.shape[1])],
        "columns": [str(column) for column in dataframe.columns],
        "head": preview.astype(str).to_dict(orient="records"),
    }


def detect_excel_header(
    path: Path, sheet_name: str, expected_tokens: Sequence[str]
) -> Tuple[int, List[str], int]:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=40, engine="openpyxl")
    expected = {normalize_name(token) for token in expected_tokens}
    best_row = 0
    best_score = -1
    best_values: List[str] = []
    for row_index in range(len(preview)):
        values = [clean_text(value) for value in preview.iloc[row_index].tolist()]
        normalized_values = {normalize_name(value) for value in values if value}
        score = len(normalized_values & expected)
        if score > best_score:
            best_score = score
            best_row = int(row_index)
            best_values = values
    return best_row, best_values, best_score


def read_excel_sheet_detected(
    path: Path,
    sheet_name: str,
    expected_tokens: Sequence[str],
    *,
    minimum_score: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    header_row, header_values, score = detect_excel_header(path, sheet_name, expected_tokens)
    if score < minimum_score:
        header_row = 0
    dataframe = pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    dataframe = dataframe.dropna(axis=0, how="all").dropna(axis=1, how="all")
    dataframe.columns = [
        clean_text(column) or f"unnamed_{index}" for index, column in enumerate(dataframe.columns)
    ]
    return dataframe, {
        "sheet": str(sheet_name),
        "header_row_zero_based": int(header_row),
        "header_detection_score": int(score),
        "detected_header_values": header_values,
        **dataframe_preview(dataframe),
    }


def workbook_schema(path: Path, expected_tokens: Sequence[str]) -> Dict[str, Any]:
    excel_file = pd.ExcelFile(path, engine="openpyxl")
    result: Dict[str, Any] = {"path": str(path), "sheets": []}
    for sheet_name in excel_file.sheet_names:
        try:
            _, information = read_excel_sheet_detected(
                path, sheet_name, expected_tokens, minimum_score=0
            )
            result["sheets"].append(information)
        except Exception as error:
            result["sheets"].append({"sheet": str(sheet_name), "error": repr(error)})
    return result


def choose_sheets_by_number_or_columns(
    path: Path,
    requested_sheet: str,
    table_number: int,
    required_column_groups: Sequence[Sequence[str]],
) -> List[str]:
    excel_file = pd.ExcelFile(path, engine="openpyxl")
    if clean_text(requested_sheet):
        if requested_sheet not in excel_file.sheet_names:
            raise KeyError(
                f"Requested sheet {requested_sheet!r} not found in {path}; sheets={excel_file.sheet_names}"
            )
        return [requested_sheet]

    numbered = [
        sheet
        for sheet in excel_file.sheet_names
        if re.search(rf"(^|[^0-9]){int(table_number)}([^0-9]|$)", str(sheet), flags=re.IGNORECASE)
    ]
    if numbered:
        return numbered

    matches: List[str] = []
    expected_tokens = sorted({candidate for group in required_column_groups for candidate in group})
    for sheet in excel_file.sheet_names:
        try:
            dataframe, _ = read_excel_sheet_detected(path, sheet, expected_tokens, minimum_score=1)
            normalized_columns = {normalize_name(column) for column in dataframe.columns}
            if all(
                any(normalize_name(candidate) in normalized_columns for candidate in group)
                for group in required_column_groups
            ):
                matches.append(sheet)
        except Exception:
            continue
    if not matches:
        raise RuntimeError(
            f"Could not identify Supplementary Table {table_number} in {path}. "
            f"Available sheets={excel_file.sheet_names}"
        )
    return matches


def read_model_genes(
    path_text: str, requested_column: str, hvg_column: str
) -> Tuple[Optional[Set[str]], Dict[str, Any]]:
    if not clean_text(path_text):
        return None, {"provided": False}
    path = require_file(Path(path_text).expanduser().resolve(), "model var table")
    dataframe = read_delimited(path)
    original_rows = len(dataframe)
    mask_used = False
    if clean_text(hvg_column) and hvg_column in dataframe.columns:
        mask = dataframe[hvg_column].fillna(False).astype(bool)
        if int(mask.sum()) > 0:
            dataframe = dataframe.loc[mask].copy()
            mask_used = True
    gene_column = resolve_column(
        dataframe,
        requested_column,
        [
            "gene_short_name",
            "gene_name",
            "gene_symbol",
            "symbol",
            "gene",
            "feature_name",
            "gene_id",
        ],
        label="model gene name",
        required=False,
    )
    if gene_column is None:
        if isinstance(dataframe.index, pd.RangeIndex):
            raise KeyError(
                f"No usable model gene column in {path}; columns={list(dataframe.columns)}"
            )
        genes = [clean_text(value) for value in dataframe.index]
        gene_column = "__index__"
    else:
        genes = [clean_text(value) for value in dataframe[gene_column]]
    genes = [gene for gene in genes if gene]
    duplicates = sorted([gene for gene, count in Counter(genes).items() if count > 1])
    if duplicates:
        raise ValueError(f"Model gene names are not unique; examples={duplicates[:20]}")
    gene_set = set(genes)
    if not gene_set:
        raise ValueError("Model var table selected zero genes")
    return gene_set, {
        "provided": True,
        "path": str(path),
        "rows_original": int(original_rows),
        "hvg_mask_used": bool(mask_used),
        "gene_column": gene_column,
        "n_model_genes": int(len(gene_set)),
        "examples": sorted(gene_set)[:20],
    }


def filter_and_report_model_genes(
    dataframe: pd.DataFrame,
    model_genes: Optional[Set[str]],
    *,
    gene_column: str = "gene",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if model_genes is None:
        return dataframe.copy(), {
            "model_genes_provided": False,
            "rows_retained": int(len(dataframe)),
        }
    matched = dataframe[gene_column].astype(str).isin(model_genes)
    filtered = dataframe.loc[matched].copy()
    return filtered, {
        "model_genes_provided": True,
        "rows_before": int(len(dataframe)),
        "rows_retained": int(len(filtered)),
        "unique_driver_genes_before": int(dataframe[gene_column].nunique()),
        "unique_driver_genes_retained": int(filtered[gene_column].nunique()),
        "fraction_rows_retained": float(matched.mean()) if len(matched) else 0.0,
    }


def assert_no_bad_edges(dataframe: pd.DataFrame, name: str) -> None:
    if dataframe.empty:
        raise ValueError(f"{name} contains zero edges")
    missing = dataframe["source"].astype(str).str.len().eq(0) | dataframe["target"].astype(
        str
    ).str.len().eq(0)
    self_loop = dataframe["source"].astype(str).eq(dataframe["target"].astype(str))
    duplicated = dataframe.duplicated(["source", "target"])
    if bool(missing.any()):
        raise ValueError(f"{name} contains {int(missing.sum())} edges with missing endpoints")
    if bool(self_loop.any()):
        raise ValueError(f"{name} contains {int(self_loop.sum())} self-loops")
    if bool(duplicated.any()):
        raise ValueError(f"{name} contains {int(duplicated.sum())} duplicate source-target pairs")


def infer_small_lineage(text: str) -> str:
    normalized = normalize_name(text)
    if "mge" in normalized or "pvalb" in normalized or "sst" in normalized:
        return "MGE"
    if "cge" in normalized or "vip" in normalized or "sncg" in normalized or "lamp5" in normalized:
        return "CGE"
    if any(
        token in normalized
        for token in ("glia", "astro", "oligo", "opc", "radialglia", "glioblast")
    ):
        return "Glia"
    if any(
        token in normalized
        for token in ("glut", "excit", "nonit", "imnit", "ip", "crcell", "cajalretzius")
    ):
        return "Glut"
    return ""


def build_small_graph(
    args: argparse.Namespace, report: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    workbook = infer_existing_path(
        args.small_trajectory_workbook,
        args.author_prior_dir,
        ["Supplementary_table_5_-_Trajectory_tree_update.xlsx"],
        "small trajectory workbook",
    )
    excel_file = pd.ExcelFile(workbook, engine="openpyxl")
    requested_sheets = [sheet for sheet in args.small_trajectory_sheets if clean_text(sheet)]
    sheets = [sheet for sheet in requested_sheets if sheet in excel_file.sheet_names]
    missing_requested = sorted(set(requested_sheets) - set(sheets))
    if missing_requested:
        raise KeyError(
            f"Small trajectory sheets missing: {missing_requested}; available={excel_file.sheet_names}"
        )
    if not sheets:
        sheets = list(excel_file.sheet_names)

    all_rows: List[pd.DataFrame] = []
    sheet_reports: List[Dict[str, Any]] = []
    for sheet in sheets:
        dataframe, sheet_report = read_excel_sheet_detected(
            workbook,
            sheet,
            ["pre_cluster_label", "nex_cluster_label", "prob", "pre_time", "nex_time", "included"],
            minimum_score=3,
        )
        pre_cluster_col = resolve_column(
            dataframe,
            "",
            ["pre_cluster_label", "previous_cluster_label", "antecedent_cluster_label", "pre"],
            label="antecedent cluster",
        )
        next_cluster_col = resolve_column(
            dataframe,
            "",
            ["nex_cluster_label", "next_cluster_label", "descendant_cluster_label", "nex"],
            label="descendant cluster",
        )
        pre_time_col = resolve_column(
            dataframe,
            "",
            ["pre_time", "previous_time", "antecedent_time", "source_time"],
            label="antecedent age",
        )
        next_time_col = resolve_column(
            dataframe,
            "",
            ["nex_time", "next_time", "descendant_time", "target_time"],
            label="descendant age",
        )
        included_col = resolve_column(
            dataframe, "", ["included", "include", "selected", "chosen"], label="included flag"
        )
        probability_col = resolve_column(
            dataframe,
            "",
            ["prob", "probability", "edge_weight", "weight"],
            label="author probability",
            required=False,
        )
        pre_author_col = resolve_column(
            dataframe,
            "",
            ["pre", "antecedent", "source"],
            label="author antecedent label",
            required=False,
        )
        next_author_col = resolve_column(
            dataframe,
            "",
            ["nex", "next", "descendant", "target"],
            label="author descendant label",
            required=False,
        )

        included = dataframe[included_col].map(canonical_yes)
        selected = dataframe.loc[included].copy()
        converted = pd.DataFrame(
            {
                "source": [
                    canonical_small_node(cluster, age)
                    for cluster, age in zip(selected[pre_cluster_col], selected[pre_time_col])
                ],
                "target": [
                    canonical_small_node(cluster, age)
                    for cluster, age in zip(selected[next_cluster_col], selected[next_time_col])
                ],
                "weight": np.ones(len(selected), dtype=np.float32),
                "author_probability": (
                    pd.to_numeric(selected[probability_col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32)
                    if probability_col is not None
                    else np.ones(len(selected), dtype=np.float32)
                ),
                "lineage": str(sheet),
                "source_cluster": selected[pre_cluster_col].map(clean_text).to_numpy(),
                "target_cluster": selected[next_cluster_col].map(clean_text).to_numpy(),
                "source_age": selected[pre_time_col].map(clean_text).to_numpy(),
                "target_age": selected[next_time_col].map(clean_text).to_numpy(),
                "author_source_label": (
                    selected[pre_author_col].map(clean_text).to_numpy() if pre_author_col else ""
                ),
                "author_target_label": (
                    selected[next_author_col].map(clean_text).to_numpy() if next_author_col else ""
                ),
                "included": "yes",
            }
        )
        converted = converted.loc[
            converted["source"].str.len().gt(0) & converted["target"].str.len().gt(0)
        ].copy()
        all_rows.append(converted)
        sheet_report.update(
            {
                "resolved_columns": {
                    "pre_cluster": pre_cluster_col,
                    "next_cluster": next_cluster_col,
                    "pre_time": pre_time_col,
                    "next_time": next_time_col,
                    "included": included_col,
                    "probability": probability_col,
                },
                "rows_included": int(len(converted)),
                "rows_excluded": int((~included).sum()),
            }
        )
        sheet_reports.append(sheet_report)

    graph = pd.concat(all_rows, ignore_index=True)
    graph = graph.sort_values(
        ["source", "target", "author_probability"], ascending=[True, True, False]
    )
    graph = graph.drop_duplicates(["source", "target"], keep="first").reset_index(drop=True)
    assert_no_bad_edges(graph, "small curated graph")

    probability_graph = graph.copy()
    probability_graph["weight"] = probability_graph["author_probability"].astype(np.float32)

    source_nodes = graph[["source", "source_cluster", "source_age", "lineage"]].rename(
        columns={"source": "node", "source_cluster": "cluster_label", "source_age": "age"}
    )
    target_nodes = graph[["target", "target_cluster", "target_age", "lineage"]].rename(
        columns={"target": "node", "target_cluster": "cluster_label", "target_age": "age"}
    )
    nodes = (
        pd.concat([source_nodes, target_nodes], ignore_index=True)
        .drop_duplicates("node")
        .sort_values("node")
        .reset_index(drop=True)
    )
    nodes["age_order"] = nodes["age"].map(developmental_age_order)
    node_set = set(nodes["node"].astype(str))
    backwards = np.asarray(
        [
            developmental_age_order(source_age) > developmental_age_order(target_age)
            for source_age, target_age in zip(graph["source_age"], graph["target_age"])
        ],
        dtype=bool,
    )
    report["small_graph"] = {
        "source_workbook": str(workbook),
        "sheets": sheet_reports,
        "n_edges": int(len(graph)),
        "n_nodes": int(len(node_set)),
        "n_roots": int(len(set(graph["source"]) - set(graph["target"]))),
        "n_terminals": int(len(set(graph["target"]) - set(graph["source"]))),
        "n_backward_age_edges": int(backwards.sum()),
        "author_probability_summary": graph["author_probability"].describe().to_dict(),
        "primary_weighting": "uniform topology weight=1.0",
        "sensitivity_weighting": "author probability",
    }
    return graph, probability_graph, nodes


def extract_module_rows_from_workbook(
    path: Path, modality: str, report: Dict[str, Any]
) -> pd.DataFrame:
    excel_file = pd.ExcelFile(path, engine="openpyxl")
    output_rows: List[Dict[str, Any]] = []
    sheet_reports: List[Dict[str, Any]] = []
    expected = [
        "gene",
        "gene_name",
        "gene_short_name",
        "symbol",
        "genes",
        "module_genes",
        "age",
        "age_bin",
        "subclass",
        "class",
        "cluster",
        "module",
        "score",
        "correlation",
    ]
    for sheet in excel_file.sheet_names:
        dataframe, sheet_report = read_excel_sheet_detected(path, sheet, expected, minimum_score=1)
        gene_column = resolve_column(
            dataframe,
            "",
            [
                "gene_short_name",
                "gene_name",
                "gene_symbol",
                "symbol",
                "gene",
                "genes",
                "module_genes",
            ],
            label="module gene",
            required=False,
        )
        age_column = resolve_column(
            dataframe,
            "",
            ["age", "age_bin", "time", "stage", "developmental_age"],
            label="module age",
            required=False,
        )
        class_column = resolve_column(
            dataframe,
            "",
            ["class", "cell_class", "major_class"],
            label="module class",
            required=False,
        )
        subclass_column = resolve_column(
            dataframe,
            "",
            ["subclass", "cell_subclass", "sub_class"],
            label="module subclass",
            required=False,
        )
        cluster_column = resolve_column(
            dataframe,
            "",
            ["cluster_label", "cluster", "cell_type", "celltype"],
            label="module cluster",
            required=False,
        )
        module_column = resolve_column(
            dataframe,
            "",
            ["module", "module_id", "gene_module", "module_name"],
            label="module id",
            required=False,
        )
        score_column = resolve_column(
            dataframe,
            "",
            ["correlation", "corr", "score", "weight", "loading", "importance"],
            label="module score",
            required=False,
        )

        rows_before = len(output_rows)
        if gene_column is not None:
            for row_index, row in dataframe.iterrows():
                genes = split_gene_tokens(row[gene_column])
                for position, gene in enumerate(genes, start=1):
                    if not looks_like_gene_symbol(gene):
                        continue
                    numeric_score = (
                        pd.to_numeric(pd.Series([row[score_column]]), errors="coerce").iloc[0]
                        if score_column
                        else np.nan
                    )
                    output_rows.append(
                        {
                            "gene": gene,
                            "age": clean_text(row[age_column]) if age_column else "",
                            "class": clean_text(row[class_column]) if class_column else "",
                            "subclass": clean_text(row[subclass_column]) if subclass_column else "",
                            "cluster_label": (
                                clean_text(row[cluster_column]) if cluster_column else ""
                            ),
                            "module": (
                                clean_text(row[module_column]) if module_column else str(sheet)
                            ),
                            "author_score": (
                                float(numeric_score) if pd.notna(numeric_score) else np.nan
                            ),
                            "within_cell_gene_position": int(position),
                            "source_sheet": str(sheet),
                            "source_row_zero_based": int(row_index),
                            "modality": modality,
                        }
                    )
        else:
            context_columns = {
                column
                for column in (
                    age_column,
                    class_column,
                    subclass_column,
                    cluster_column,
                    module_column,
                    score_column,
                )
                if column
            }
            for column in dataframe.columns:
                if column in context_columns or normalize_name(column).startswith("unnamed"):
                    continue
                rank = 0
                for row_index, value in dataframe[column].items():
                    for gene in split_gene_tokens(value):
                        if not looks_like_gene_symbol(gene):
                            continue
                        rank += 1
                        output_rows.append(
                            {
                                "gene": gene,
                                "age": (
                                    clean_text(dataframe.at[row_index, age_column])
                                    if age_column
                                    else ""
                                ),
                                "class": (
                                    clean_text(dataframe.at[row_index, class_column])
                                    if class_column
                                    else ""
                                ),
                                "subclass": (
                                    clean_text(dataframe.at[row_index, subclass_column])
                                    if subclass_column
                                    else ""
                                ),
                                "cluster_label": (
                                    clean_text(dataframe.at[row_index, cluster_column])
                                    if cluster_column
                                    else ""
                                ),
                                "module": str(column),
                                "author_score": np.nan,
                                "within_cell_gene_position": int(rank),
                                "source_sheet": str(sheet),
                                "source_row_zero_based": int(row_index),
                                "modality": modality,
                            }
                        )
        sheet_report.update(
            {
                "resolved_columns": {
                    "gene": gene_column,
                    "age": age_column,
                    "class": class_column,
                    "subclass": subclass_column,
                    "cluster": cluster_column,
                    "module": module_column,
                    "score": score_column,
                },
                "module_gene_rows_extracted": int(len(output_rows) - rows_before),
            }
        )
        sheet_reports.append(sheet_report)

    report.setdefault("small_module_workbooks", {})[modality] = {
        "path": str(path),
        "sheets": sheet_reports,
        "rows_extracted": int(len(output_rows)),
    }
    return pd.DataFrame(output_rows)


def score_module_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.assign(source_score=pd.Series(dtype=np.float32))
    rows = rows.copy()
    context_columns = [
        "age",
        "class",
        "subclass",
        "cluster_label",
        "module",
        "source_sheet",
        "modality",
    ]
    rows["context_key"] = rows[context_columns].fillna("").astype(str).agg("|".join, axis=1)
    rows["rank"] = rows.groupby("context_key", observed=True).cumcount() + 1
    author_absolute = pd.to_numeric(rows["author_score"], errors="coerce").abs()
    author_normalized = author_absolute.groupby(rows["context_key"]).transform(
        lambda values: (
            values / values.max()
            if bool(values.notna().any()) and float(values.max()) > 0
            else pd.Series(np.nan, index=values.index)
        )
    )
    rank_decay = 1.0 / np.sqrt(rows["rank"].astype(np.float32))
    rows["source_score"] = (
        author_normalized.fillna(rank_decay).clip(lower=0.0, upper=1.0).astype(np.float32)
    )
    return rows


def map_small_context_to_nodes(row: pd.Series, nodes: pd.DataFrame) -> Tuple[List[str], str]:
    candidates = nodes.copy()
    age = clean_text(row.get("age", ""))
    lineage_text = " ".join(
        clean_text(row.get(column, ""))
        for column in ("source_sheet", "class", "subclass", "cluster_label", "module")
    )
    lineage = infer_small_lineage(lineage_text)
    cluster = clean_text(row.get("cluster_label", ""))
    subclass = clean_text(row.get("subclass", ""))
    class_name = clean_text(row.get("class", ""))

    quality_parts: List[str] = []
    if age:
        age_norm = normalize_name(age)
        filtered = candidates[candidates["age"].map(normalize_name).eq(age_norm)]
        if not filtered.empty:
            candidates = filtered
            quality_parts.append("age")
    if lineage:
        filtered = candidates[candidates["lineage"].astype(str).str.lower().eq(lineage.lower())]
        if not filtered.empty:
            candidates = filtered
            quality_parts.append("lineage")
    if cluster:
        cluster_norm = normalize_name(cluster)
        exact = candidates[candidates["cluster_label"].map(normalize_name).eq(cluster_norm)]
        if not exact.empty:
            candidates = exact
            quality_parts.append("cluster_exact")
    for label, value in (("subclass", subclass), ("class", class_name)):
        value_norm = normalize_name(value)
        if not value_norm:
            continue
        match = candidates[
            candidates["cluster_label"]
            .map(normalize_name)
            .map(lambda cluster_norm: value_norm in cluster_norm or cluster_norm in value_norm)
        ]
        if not match.empty:
            candidates = match
            quality_parts.append(label)

    if candidates.empty:
        return [], "unmapped"
    if not age and not lineage and not cluster and not subclass and not class_name:
        return [], "unmapped_no_context"
    quality = "+".join(quality_parts) if quality_parts else "broad_context"
    if len(candidates) > 250:
        return [], "unmapped_too_broad"
    return sorted(candidates["node"].astype(str).unique().tolist()), quality


def build_small_drivers(
    args: argparse.Namespace,
    graph: pd.DataFrame,
    nodes: pd.DataFrame,
    model_genes: Optional[Set[str]],
    report: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expression_workbook = infer_existing_path(
        args.small_expression_module_workbook,
        args.author_prior_dir,
        ["Supplementary_table_6_-_Gene_module.xlsx"],
        "small expression module workbook",
    )
    chromatin_workbook = infer_existing_path(
        args.small_chromatin_module_workbook,
        args.author_prior_dir,
        ["Supplementary_table_9_-_correlated_peak_gene_module.xlsx"],
        "small chromatin-supported module workbook",
    )
    expression = score_module_rows(
        extract_module_rows_from_workbook(expression_workbook, "expression", report)
    )
    chromatin = score_module_rows(
        extract_module_rows_from_workbook(chromatin_workbook, "chromatin", report)
    )
    raw = pd.concat([expression, chromatin], ignore_index=True)
    if raw.empty:
        raise ValueError("No small-dataset module genes could be extracted from Tables 6 and 9")

    incoming_sources: Dict[str, List[str]] = defaultdict(list)
    for source, target in graph[["source", "target"]].itertuples(index=False, name=None):
        incoming_sources[str(target)].append(str(source))

    mapped_rows: List[Dict[str, Any]] = []
    unmapped_rows: List[Dict[str, Any]] = []
    for _, row in raw.iterrows():
        mapped_nodes, mapping_quality = map_small_context_to_nodes(row, nodes)
        base = row.to_dict()
        base["mapping_quality"] = mapping_quality
        if not mapped_nodes:
            unmapped_rows.append(base)
            continue
        for node in mapped_nodes:
            sources = incoming_sources.get(node, [""])
            for source in sources:
                converted = dict(base)
                converted.update({"node": node, "source": source, "target": node})
                mapped_rows.append(converted)

    mapped = pd.DataFrame(mapped_rows)
    unmapped = pd.DataFrame(unmapped_rows)
    if mapped.empty:
        raise ValueError(
            "Small driver modules were read, but none mapped to cluster+age graph nodes. "
            "Inspect small_driver_context_unmapped.tsv and set explicit workbook/column conventions before training."
        )

    per_modality = mapped.groupby(
        ["node", "source", "target", "gene", "modality"], observed=True, as_index=False
    ).agg(
        modality_score=("source_score", "max"),
        module=("module", lambda values: "|".join(sorted(set(map(str, values))))),
        mapping_quality=("mapping_quality", lambda values: "|".join(sorted(set(map(str, values))))),
        source_sheet=("source_sheet", lambda values: "|".join(sorted(set(map(str, values))))),
    )
    pivot = per_modality.pivot_table(
        index=["node", "source", "target", "gene"],
        columns="modality",
        values="modality_score",
        aggfunc="max",
        fill_value=0.0,
    ).reset_index()
    pivot.columns.name = None
    if "expression" not in pivot.columns:
        pivot["expression"] = 0.0
    if "chromatin" not in pivot.columns:
        pivot["chromatin"] = 0.0
    pivot = pivot.rename(columns={"expression": "expression_score", "chromatin": "chromatin_score"})
    pivot["support_count"] = (pivot["expression_score"] > 0).astype(int) + (
        pivot["chromatin_score"] > 0
    ).astype(int)
    pivot["score"] = (0.5 * pivot["expression_score"] + 0.5 * pivot["chromatin_score"]).astype(
        np.float32
    )
    pivot["driver_type"] = np.where(
        pivot["support_count"] == 2,
        "expression_and_chromatin",
        np.where(pivot["expression_score"] > 0, "expression_only", "chromatin_only"),
    )
    pivot = pivot[
        [
            "node",
            "gene",
            "score",
            "source",
            "target",
            "driver_type",
            "support_count",
            "expression_score",
            "chromatin_score",
        ]
    ]
    pivot = pivot.sort_values(["node", "score", "gene"], ascending=[True, False, True]).reset_index(
        drop=True
    )

    if int(args.small_top_genes_per_node) > 0:
        pivot = (
            pivot.groupby("node", observed=True, group_keys=False)
            .head(int(args.small_top_genes_per_node))
            .reset_index(drop=True)
        )

    model_aligned, overlap = filter_and_report_model_genes(pivot, model_genes)
    report["small_drivers"] = {
        "expression_workbook": str(expression_workbook),
        "chromatin_workbook": str(chromatin_workbook),
        "raw_module_gene_rows": int(len(raw)),
        "mapped_expanded_rows": int(len(mapped)),
        "unmapped_context_rows": int(len(unmapped)),
        "final_all_gene_rows": int(len(pivot)),
        "final_model_gene_rows": int(len(model_aligned)),
        "nodes_with_drivers_all_genes": int(pivot["node"].nunique()),
        "nodes_with_drivers_model_genes": (
            int(model_aligned["node"].nunique()) if not model_aligned.empty else 0
        ),
        "model_gene_overlap": overlap,
        "score_definition": "0.5*expression rank/author score + 0.5*chromatin rank/author score; consensus receives support from both modalities",
    }
    return model_aligned, pivot, unmapped


def infer_obs_column(
    dataframe: pd.DataFrame, requested: str, candidates: Sequence[str], label: str
) -> str:
    column = resolve_column(dataframe, requested, candidates, label=label, required=False)
    if column is None:
        raise KeyError(f"Could not infer {label} from obs columns={list(dataframe.columns)}")
    return column


def build_small_cell_map(
    args: argparse.Namespace, nodes: pd.DataFrame, report: Dict[str, Any]
) -> Optional[Path]:
    if not clean_text(args.obs_path):
        report["small_cell_map"] = {"created": False, "reason": "--obs-path was not provided"}
        return None
    obs_path = require_file(Path(args.obs_path).expanduser().resolve(), "small obs table")
    obs = read_delimited(obs_path)
    cluster_col = infer_obs_column(
        obs,
        args.small_obs_cluster_col,
        ["cluster_label", "cluster", "celltype", "cell_type"],
        "small obs cluster",
    )
    age_col = infer_obs_column(
        obs,
        args.small_obs_age_col,
        ["age", "age_bin", "developmental_age", "stage", "day", "time"],
        "small obs age",
    )
    cell_id_col = resolve_column(
        obs,
        args.cell_id_col,
        ["cell_id", "cell", "barcode", "cell_barcode"],
        label="cell ID",
        required=False,
    )
    cell_ids = (
        obs[cell_id_col].map(clean_text)
        if cell_id_col
        else pd.Series(obs.index.astype(str), index=obs.index)
    )
    graph_nodes = pd.Series(
        [
            canonical_small_node(cluster, age)
            for cluster, age in zip(obs[cluster_col], obs[age_col])
        ],
        index=obs.index,
    )
    valid_nodes = set(nodes["node"].astype(str))
    matched = graph_nodes.isin(valid_nodes)
    mapping = pd.DataFrame(
        {
            "cell_id": cell_ids.to_numpy(),
            "graph_node": graph_nodes.to_numpy(),
            "cluster_label": obs[cluster_col].map(clean_text).to_numpy(),
            "age": obs[age_col].map(clean_text).to_numpy(),
            "mapped": matched.to_numpy(dtype=bool),
        }
    )
    try:
        import pyarrow  # type: ignore  # noqa: F401

        output_path = args.output_dir / "small_cell_to_graph_node.parquet"
    except Exception:
        output_path = args.output_dir / "small_cell_to_graph_node.tsv.gz"
        log("[small][cell-map] pyarrow unavailable; writing compressed TSV instead of Parquet")
    # `obs` loaded from Parquet can retain a non-consecutive, string, or named
    # index. `mapping` is intentionally constructed positionally and therefore
    # has a fresh RangeIndex. Use its own Boolean column for all downstream QC
    # instead of applying the index-bearing `matched` Series to `mapping`.
    mapping_matched = mapping["mapped"].astype(bool)
    coverage = float(mapping_matched.mean()) if len(mapping_matched) else 0.0
    report["small_cell_map"] = {
        "created": True,
        "path": str(output_path),
        "obs_path": str(obs_path),
        "resolved_columns": {
            "cell_id": cell_id_col or "__index__",
            "cluster": cluster_col,
            "age": age_col,
        },
        "n_cells": int(len(mapping)),
        "n_mapped_cells": int(mapping_matched.sum()),
        "cell_coverage": coverage,
        "n_distinct_mapped_nodes": int(mapping.loc[mapping_matched, "graph_node"].nunique()),
        "examples_unmapped": (
            mapping.loc[~mapping_matched, ["cluster_label", "age", "graph_node"]]
            .drop_duplicates()
            .head(30)
            .to_dict(orient="records")
        ),
    }
    if args.strict and coverage < float(args.minimum_cell_coverage):
        raise ValueError(
            f"Small graph cell coverage {coverage:.4f} is below --minimum-cell-coverage={args.minimum_cell_coverage}. "
            "Do not train until the obs age/cluster mapping is corrected."
        )
    write_table(mapping, output_path, overwrite=args.overwrite)
    return output_path


def load_big_edges(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    dataframe = pd.read_csv(path, sep="\t", low_memory=False)
    if not {normalize_name(column) for column in dataframe.columns} & {
        "system",
        "x",
        "y",
        "xname",
        "yname",
        "edgetype",
    }:
        dataframe = pd.read_csv(path, sep="\t", header=None, low_memory=False)
        if dataframe.shape[1] < 6:
            raise ValueError(
                f"Big edges file must have at least 6 columns; found {dataframe.shape[1]}"
            )
        dataframe = dataframe.iloc[:, :6].copy()
        dataframe.columns = ["system", "x", "y", "x_name", "y_name", "edge_type"]
    columns = {
        "system": resolve_column(dataframe, "", ["system"], label="edge system"),
        "source": resolve_column(
            dataframe, "", ["x", "source", "from", "node_a"], label="edge source"
        ),
        "target": resolve_column(
            dataframe, "", ["y", "target", "to", "node_b"], label="edge target"
        ),
        "source_name": resolve_column(
            dataframe,
            "",
            ["x_name", "source_name", "celltype_a"],
            label="source state name",
            required=False,
        ),
        "target_name": resolve_column(
            dataframe,
            "",
            ["y_name", "target_name", "celltype_b"],
            label="target state name",
            required=False,
        ),
        "edge_type": resolve_column(
            dataframe, "", ["edge_type", "type", "comments"], label="edge type"
        ),
    }
    return dataframe, {key: value or "" for key, value in columns.items()}


def build_big_graph(
    args: argparse.Namespace, report: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges_path = infer_existing_path(
        args.big_edges_path, args.author_prior_dir, ["edges.txt"], "big edges table"
    )
    nodes_path = infer_existing_path(
        args.big_nodes_path, args.author_prior_dir, ["nodes.txt"], "big nodes table"
    )
    raw_edges, columns = load_big_edges(edges_path)
    converted = pd.DataFrame(
        {
            "source": raw_edges[columns["source"]].map(clean_text),
            "target": raw_edges[columns["target"]].map(clean_text),
            "weight": np.ones(len(raw_edges), dtype=np.float32),
            "edge_type": raw_edges[columns["edge_type"]].map(canonical_edge_type),
            "author_edge_type": raw_edges[columns["edge_type"]].map(clean_text),
            "system": raw_edges[columns["system"]].map(clean_text),
            "source_name": (
                raw_edges[columns["source_name"]].map(clean_text) if columns["source_name"] else ""
            ),
            "target_name": (
                raw_edges[columns["target_name"]].map(clean_text) if columns["target_name"] else ""
            ),
        }
    )
    converted = converted.loc[
        converted["source"].str.len().gt(0) & converted["target"].str.len().gt(0)
    ].copy()
    converted = converted.loc[~converted["source"].eq(converted["target"])].copy()
    converted = converted.drop_duplicates(["source", "target", "edge_type"]).reset_index(drop=True)

    developmental = converted.loc[converted["edge_type"].eq("developmental_progression")].copy()
    spatial = converted.loc[converted["edge_type"].eq("spatial_continuity")].copy()
    alias = converted.loc[
        ~converted["edge_type"].isin(["developmental_progression", "spatial_continuity"])
    ].copy()
    assert_no_bad_edges(developmental, "big developmental graph")

    nodes = pd.read_csv(nodes_path, sep="\t", low_memory=False)
    node_col = resolve_column(
        nodes, "", ["meta_group", "node", "node_id", "id"], label="big graph node ID"
    )
    nodes = nodes.copy()
    nodes["node"] = nodes[node_col].map(clean_text)
    nodes = nodes.loc[nodes["node"].str.len().gt(0)].drop_duplicates("node").reset_index(drop=True)
    graph_node_set = set(nodes["node"])
    endpoint_set = set(converted["source"]) | set(converted["target"])
    missing_nodes = sorted(endpoint_set - graph_node_set)
    if missing_nodes:
        raise ValueError(
            f"Big graph contains {len(missing_nodes)} endpoints absent from nodes.txt; examples={missing_nodes[:20]}"
        )

    report["big_graph"] = {
        "edges_path": str(edges_path),
        "nodes_path": str(nodes_path),
        "resolved_edge_columns": columns,
        "resolved_node_column": node_col,
        "n_raw_edges": int(len(raw_edges)),
        "n_developmental_edges": int(len(developmental)),
        "n_spatial_edges": int(len(spatial)),
        "n_alias_or_other_edges": int(len(alias)),
        "edge_type_counts": converted["edge_type"].value_counts().to_dict(),
        "n_nodes": int(len(nodes)),
        "n_graph_endpoint_nodes": int(len(endpoint_set)),
        "primary_graph_rule": "edge_type == Developmental progression; uniform weight=1.0",
        "spatial_rule": "kept separately for anatomy sensitivity; excluded from developmental graph",
        "alias_rule": "kept separately; never treated as a developmental transition",
    }
    return developmental, spatial, alias, nodes


def standardize_big_driver_table(
    dataframe: pd.DataFrame,
    driver_type: str,
    args: argparse.Namespace,
    source_sheet: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    source_col = resolve_column(
        dataframe, "", ["node_A", "source", "x", "from"], label="driver source node"
    )
    target_col = resolve_column(
        dataframe, "", ["node_B", "target", "y", "to"], label="driver target node"
    )
    gene_col = resolve_column(
        dataframe,
        "",
        ["gene_short_name", "gene_name", "gene", "symbol"],
        label="driver gene",
        required=False,
    )
    summary_gene_candidates = (
        ["keyTF", "key_TF", "key_TFs", "keyTFs", "top_TF", "top_TFs"]
        if driver_type == "TF"
        else ["keyGene", "key_gene", "key_genes", "keyGenes", "topGene", "top_genes"]
    )
    summary_gene_col = resolve_column(
        dataframe,
        "",
        summary_gene_candidates,
        label=f"summarized {driver_type} list",
        required=False,
    )
    if gene_col is None and summary_gene_col is None:
        raise KeyError(
            f"Could not infer a long-format gene column or summarized {driver_type} list. "
            f"Tried long columns=['gene_short_name', 'gene_name', 'gene', 'symbol'] and "
            f"summary columns={summary_gene_candidates}; columns={list(dataframe.columns)}"
        )
    adjusted_p_col = resolve_column(
        dataframe,
        "",
        ["p_val_adj", "p_adj", "padj", "adjusted_p_value", "fdr"],
        label="adjusted P value",
        required=False,
    )
    logfc_col = resolve_column(
        dataframe,
        "",
        ["avg_logFC", "avg_log2FC", "logfc", "log2fc", "fold_change"],
        label="driver log fold-change",
        required=False,
    )
    direction_col = resolve_column(
        dataframe,
        "",
        ["high_in_which", "direction", "transition_direction"],
        label="driver direction",
    )
    edge_type_col = resolve_column(dataframe, "", ["edge_type", "type"], label="driver edge type")
    source_name_col = resolve_column(
        dataframe,
        "",
        ["celltype_A", "source_name", "x_name"],
        label="driver source state name",
        required=False,
    )
    target_name_col = resolve_column(
        dataframe,
        "",
        ["celltype_B", "target_name", "y_name"],
        label="driver target state name",
        required=False,
    )
    system_col = resolve_column(
        dataframe, "", ["system", "subsystem"], label="driver system", required=False
    )
    comparing_col = resolve_column(
        dataframe, "", ["comparing", "comparison"], label="driver comparison", required=False
    )

    standardized = pd.DataFrame(
        {
            "source": dataframe[source_col].map(clean_text),
            "target": dataframe[target_col].map(clean_text),
            "node": dataframe[target_col].map(clean_text),
            "p_val_adj": (
                pd.to_numeric(dataframe[adjusted_p_col], errors="coerce")
                if adjusted_p_col
                else np.nan
            ),
            "avg_logFC": (
                pd.to_numeric(dataframe[logfc_col], errors="coerce") if logfc_col else np.nan
            ),
            "high_in_which": dataframe[direction_col].map(clean_text),
            "edge_type": dataframe[edge_type_col].map(clean_text),
            "source_name": dataframe[source_name_col].map(clean_text) if source_name_col else "",
            "target_name": dataframe[target_name_col].map(clean_text) if target_name_col else "",
            "system": dataframe[system_col].map(clean_text) if system_col else "",
            "comparing": dataframe[comparing_col].map(clean_text) if comparing_col else "",
            "driver_type": driver_type,
            "source_sheet": source_sheet,
            "_author_row": np.arange(len(dataframe), dtype=np.int64),
        }
    )

    if gene_col is not None:
        input_format = "long_per_gene"
        standardized["gene"] = dataframe[gene_col].map(clean_text)
        standardized["author_list_rank"] = 1
    else:
        input_format = "summarized_gene_list"
        standardized["gene"] = dataframe[summary_gene_col].map(split_gene_tokens)
        standardized = standardized.explode("gene", ignore_index=False)
        standardized["gene"] = standardized["gene"].map(clean_text)
        standardized["author_list_rank"] = (
            standardized.groupby("_author_row", observed=True).cumcount() + 1
        )

    nonmissing = (
        standardized["source"].str.len().gt(0)
        & standardized["target"].str.len().gt(0)
        & standardized["gene"].str.len().gt(0)
    )
    developmental = (
        standardized["edge_type"].map(canonical_edge_type).eq("developmental_progression")
    )
    direction = (
        standardized["high_in_which"]
        .map(normalize_name)
        .eq(normalize_name(args.big_driver_direction))
    )
    if adjusted_p_col is None:
        # keyTF/keyGene are already author-curated summarized lists.  Their
        # source workbook does not contain per-gene P values, so applying a
        # fabricated significance threshold would be scientifically invalid.
        significant = pd.Series(True, index=standardized.index, dtype=bool)
    else:
        significant = standardized["p_val_adj"].notna() & standardized["p_val_adj"].lt(
            float(args.big_adjusted_p_max)
        )
    filtered = standardized.loc[nonmissing & developmental & direction & significant].copy()
    type_weight = 1.0 if driver_type == "TF" else float(args.big_gene_driver_weight)
    if adjusted_p_col is not None:
        score_mode = "pvalue_effect_rank_decay"
        filtered["effect_abs"] = filtered["avg_logFC"].abs().fillna(1.0).clip(lower=1e-6)
        filtered["minus_log10_p"] = -np.log10(
            filtered["p_val_adj"].clip(lower=np.finfo(float).tiny)
        )
        filtered["raw_score"] = filtered["effect_abs"] * filtered["minus_log10_p"]
        filtered = filtered.sort_values(
            ["source", "target", "p_val_adj", "effect_abs", "gene"],
            ascending=[True, True, True, False, True],
        )
        filtered = filtered.drop_duplicates(["source", "target", "gene"], keep="first")
        filtered["rank_within_edge"] = (
            filtered.groupby(["source", "target"], observed=True).cumcount() + 1
        )
        max_raw = (
            filtered.groupby(["source", "target"], observed=True)["raw_score"]
            .transform("max")
            .replace(0.0, 1.0)
        )
        normalized_evidence = (filtered["raw_score"] / max_raw).clip(0.0, 1.0)
        rank_decay = 1.0 / np.sqrt(filtered["rank_within_edge"].astype(np.float32))
        filtered["score"] = (type_weight * normalized_evidence * rank_decay).astype(np.float32)
    else:
        score_mode = "author_list_rank_decay"
        filtered = filtered.sort_values(
            ["source", "target", "_author_row", "author_list_rank", "gene"],
            ascending=[True, True, True, True, True],
        )
        filtered = filtered.drop_duplicates(["source", "target", "gene"], keep="first")
        filtered["rank_within_edge"] = (
            filtered.groupby(["source", "target"], observed=True).cumcount() + 1
        )
        filtered["score"] = (
            type_weight / np.sqrt(filtered["rank_within_edge"].astype(np.float32))
        ).astype(np.float32)
    top_k = int(args.big_top_tf_per_edge if driver_type == "TF" else args.big_top_gene_per_edge)
    if top_k > 0:
        filtered = filtered.groupby(["source", "target"], observed=True, group_keys=False).head(
            top_k
        )
    resolved = {
        "source": source_col,
        "target": target_col,
        "gene": gene_col,
        "summary_gene_list": summary_gene_col,
        "p_val_adj": adjusted_p_col,
        "avg_logFC": logfc_col,
        "direction": direction_col,
        "edge_type": edge_type_col,
        "source_name": source_name_col,
        "target_name": target_name_col,
        "system": system_col,
        "comparing": comparing_col,
    }
    stats = {
        "driver_type": driver_type,
        "source_sheet": source_sheet,
        "input_format": input_format,
        "score_mode": score_mode,
        "resolved_columns": resolved,
        "rows_source_input": int(len(dataframe)),
        "rows_after_gene_list_expansion": int(len(standardized)),
        "rows_nonmissing": int(nonmissing.sum()),
        "rows_developmental": int(developmental.sum()),
        "rows_matching_direction": int(direction.sum()),
        "rows_significant": int(significant.sum()),
        "significance_filter_applied": bool(adjusted_p_col is not None),
        "rows_after_all_filters_and_topk": int(len(filtered)),
        "top_k_per_edge": int(top_k),
    }
    return filtered, stats


def build_big_drivers(
    args: argparse.Namespace,
    developmental_graph: pd.DataFrame,
    model_genes: Optional[Set[str]],
    report: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    workbook = infer_existing_path(
        args.big_supplementary_workbook,
        args.author_prior_dir,
        ["supplementary_tables.xlsx"],
        "big supplementary workbook",
    )
    groups = [
        ["node_A", "source", "x"],
        ["node_B", "target", "y"],
        ["gene_short_name", "gene", "symbol", "keyTF", "keyGene", "key_TF", "key_gene"],
        ["high_in_which", "direction"],
        ["edge_type", "type"],
    ]
    table23_sheets = choose_sheets_by_number_or_columns(
        workbook, args.big_table23_sheet, 23, groups
    )
    table24_sheets = choose_sheets_by_number_or_columns(
        workbook, args.big_table24_sheet, 24, groups
    )
    expected = sorted(
        {candidate for group in groups for candidate in group}
        | {
            "p_val_adj",
            "avg_logFC",
            "celltype_A",
            "celltype_B",
            "system",
            "Subsystem",
            "keyTF",
            "keyGene",
        }
    )

    tf_parts: List[pd.DataFrame] = []
    gene_parts: List[pd.DataFrame] = []
    table_reports: List[Dict[str, Any]] = []
    for sheet in table23_sheets:
        dataframe, schema = read_excel_sheet_detected(workbook, sheet, expected, minimum_score=4)
        standardized, stats = standardize_big_driver_table(dataframe, "TF", args, sheet)
        tf_parts.append(standardized)
        table_reports.append({"table": 23, "schema": schema, "conversion": stats})
    for sheet in table24_sheets:
        dataframe, schema = read_excel_sheet_detected(workbook, sheet, expected, minimum_score=4)
        standardized, stats = standardize_big_driver_table(dataframe, "gene", args, sheet)
        gene_parts.append(standardized)
        table_reports.append({"table": 24, "schema": schema, "conversion": stats})

    tf = pd.concat(tf_parts, ignore_index=True) if tf_parts else pd.DataFrame()
    genes = pd.concat(gene_parts, ignore_index=True) if gene_parts else pd.DataFrame()
    if tf.empty:
        raise ValueError(
            "Supplementary Table 23 produced zero TF drivers after the primary filters"
        )
    combined = pd.concat([tf, genes], ignore_index=True)
    combined = combined.sort_values(
        ["source", "target", "score", "gene"], ascending=[True, True, False, True]
    )
    combined = combined.drop_duplicates(
        ["source", "target", "gene", "driver_type"], keep="first"
    ).reset_index(drop=True)
    tf = tf.sort_values(["source", "target", "score", "gene"], ascending=[True, True, False, True])
    tf = tf.drop_duplicates(["source", "target", "gene"], keep="first").reset_index(drop=True)

    valid_edges = set(
        zip(developmental_graph["source"].astype(str), developmental_graph["target"].astype(str))
    )
    tf_edge_valid = np.asarray(
        [
            (source, target) in valid_edges
            for source, target in tf[["source", "target"]].itertuples(index=False, name=None)
        ],
        dtype=bool,
    )
    combined_edge_valid = np.asarray(
        [
            (source, target) in valid_edges
            for source, target in combined[["source", "target"]].itertuples(index=False, name=None)
        ],
        dtype=bool,
    )
    tf = tf.loc[tf_edge_valid].copy()
    combined = combined.loc[combined_edge_valid].copy()
    tf_model, tf_overlap = filter_and_report_model_genes(tf, model_genes)
    combined_model, combined_overlap = filter_and_report_model_genes(combined, model_genes)

    output_columns = [
        "node",
        "gene",
        "score",
        "source",
        "target",
        "driver_type",
        "rank_within_edge",
        "p_val_adj",
        "avg_logFC",
        "high_in_which",
        "edge_type",
        "system",
        "source_name",
        "target_name",
        "comparing",
        "source_sheet",
    ]
    tf_model = tf_model[output_columns].reset_index(drop=True)
    combined_model = combined_model[output_columns].reset_index(drop=True)
    report["big_drivers"] = {
        "workbook": str(workbook),
        "table23_sheets": table23_sheets,
        "table24_sheets": table24_sheets,
        "table_reports": table_reports,
        "primary_filters": {
            "edge_type": "Developmental progression",
            "direction": args.big_driver_direction,
            "p_val_adj_less_than": float(args.big_adjusted_p_max),
            "top_tf_per_edge": int(args.big_top_tf_per_edge),
            "top_gene_per_edge": int(args.big_top_gene_per_edge),
        },
        "score_definition": (
            "Long per-gene tables: type_weight * within-edge normalized(abs(avg_logFC)*-log10(p_adj)) "
            "/ sqrt(rank). Summarized keyTF/keyGene tables: type_weight / sqrt(author list rank). TF weight=1."
        ),
        "tf_rows_before_model_filter": int(len(tf)),
        "combined_rows_before_model_filter": int(len(combined)),
        "tf_model_gene_overlap": tf_overlap,
        "combined_model_gene_overlap": combined_overlap,
        "tf_edges_covered": int(tf_model[["source", "target"]].drop_duplicates().shape[0]),
        "combined_edges_covered": int(
            combined_model[["source", "target"]].drop_duplicates().shape[0]
        ),
    }
    return tf_model, combined_model


def validate_big_cell_map(
    path: Path, graph_nodes: Set[str], chunksize: int = 1_000_000
) -> Dict[str, Any]:
    total = 0
    mapped = 0
    graph_node_counts: Counter[str] = Counter()
    missing_node_counts: Counter[str] = Counter()
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            import pyarrow.parquet as parquet  # type: ignore
        except Exception as error:
            raise RuntimeError("Validating a large Parquet cell map requires pyarrow") from error
        parquet_file = parquet.ParquetFile(path)
        iterator = (
            batch.to_pandas()
            for batch in parquet_file.iter_batches(
                batch_size=chunksize, columns=["cell_id", "graph_node"]
            )
        )
    else:
        iterator = pd.read_csv(
            path, sep="\t", chunksize=chunksize, compression="infer", low_memory=False
        )
    for chunk in iterator:
        if "graph_node" not in chunk.columns:
            if "meta_group" in chunk.columns:
                chunk = chunk.rename(columns={"meta_group": "graph_node"})
            else:
                raise KeyError(f"Cell map lacks graph_node/meta_group column: {path}")
        nodes = chunk["graph_node"].map(clean_text)
        valid = nodes.isin(graph_nodes)
        total += len(chunk)
        mapped += int(valid.sum())
        graph_node_counts.update(nodes[valid].value_counts().to_dict())
        missing_node_counts.update(nodes[~valid].value_counts().head(1000).to_dict())
    return {
        "n_rows": int(total),
        "n_rows_matching_graph_nodes": int(mapped),
        "cell_coverage": float(mapped / total) if total else 0.0,
        "n_distinct_graph_nodes_observed": int(len(graph_node_counts)),
        "top_unmatched_node_values": missing_node_counts.most_common(30),
    }


def build_big_cell_map(
    args: argparse.Namespace, nodes: pd.DataFrame, report: Dict[str, Any]
) -> Optional[Path]:
    graph_nodes = set(nodes["node"].astype(str))
    if clean_text(args.big_cell_map_path):
        output_path = require_file(
            Path(args.big_cell_map_path).expanduser().resolve(), "pre-extracted big cell map"
        )
        created = False
    elif clean_text(args.big_cell_graph_rds):
        rds_path = require_file(
            Path(args.big_cell_graph_rds).expanduser().resolve(), "big df_cell_graph.rds"
        )
        helper_path = require_file(
            Path(args.rds_helper_path).expanduser().resolve(), "RDS extraction helper"
        )
        rscript = shutil.which(args.rscript_executable)
        if rscript is None:
            raise RuntimeError(
                f"Could not find {args.rscript_executable!r}. The .rds source inherently requires R (or pyreadr). "
                "Install/activate R, or provide --big-cell-map-path from a previous extraction."
            )
        output_path = args.output_dir / "big_cell_to_graph_node.tsv.gz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        command = [rscript, str(helper_path), str(rds_path), str(output_path)]
        log(f"[big][cell-map] invoking isolated R extraction: {' '.join(command)}")
        subprocess.run(command, check=True)
        created = True
    else:
        report["big_cell_map"] = {
            "created": False,
            "reason": "Neither --big-cell-graph-rds nor --big-cell-map-path was provided",
        }
        return None

    validation = validate_big_cell_map(output_path, graph_nodes)
    report["big_cell_map"] = {"created": created, "path": str(output_path), **validation}
    if args.strict and float(validation["cell_coverage"]) < float(args.minimum_cell_coverage):
        raise ValueError(
            f"Big graph cell coverage {validation['cell_coverage']:.4f} is below "
            f"--minimum-cell-coverage={args.minimum_cell_coverage}."
        )
    return output_path


def build_schema_report(args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "converter_version": VERSION,
        "dataset": args.dataset,
        "author_prior_dir": str(args.author_prior_dir),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.dataset == "small":
        paths = {
            "trajectory": infer_existing_path(
                args.small_trajectory_workbook,
                args.author_prior_dir,
                ["Supplementary_table_5_-_Trajectory_tree_update.xlsx"],
                "small trajectory workbook",
            ),
            "expression_modules": infer_existing_path(
                args.small_expression_module_workbook,
                args.author_prior_dir,
                ["Supplementary_table_6_-_Gene_module.xlsx"],
                "small expression module workbook",
            ),
            "chromatin_modules": infer_existing_path(
                args.small_chromatin_module_workbook,
                args.author_prior_dir,
                ["Supplementary_table_9_-_correlated_peak_gene_module.xlsx"],
                "small chromatin module workbook",
            ),
        }
        for label, path in paths.items():
            result[label] = workbook_schema(
                path,
                ["pre_cluster_label", "nex_cluster_label", "gene", "age", "subclass", "module"],
            )
    else:
        workbook = infer_existing_path(
            args.big_supplementary_workbook,
            args.author_prior_dir,
            ["supplementary_tables.xlsx"],
            "big supplementary workbook",
        )
        result["supplementary_workbook"] = workbook_schema(
            workbook,
            ["node_A", "node_B", "gene_short_name", "p_val_adj", "high_in_which", "edge_type"],
        )
        edges_path = infer_existing_path(
            args.big_edges_path, args.author_prior_dir, ["edges.txt"], "big edges table"
        )
        nodes_path = infer_existing_path(
            args.big_nodes_path, args.author_prior_dir, ["nodes.txt"], "big nodes table"
        )
        result["edges"] = dataframe_preview(pd.read_csv(edges_path, sep="\t", low_memory=False))
        result["nodes"] = dataframe_preview(pd.read_csv(nodes_path, sep="\t", low_memory=False))
    return result


def run_small(args: argparse.Namespace, report: Dict[str, Any]) -> None:
    graph, probability_graph, nodes = build_small_graph(args, report)
    write_table(graph, args.output_dir / "small_curated_graph.tsv", overwrite=args.overwrite)
    write_table(
        probability_graph,
        args.output_dir / "small_curated_graph_probability_sensitivity.tsv",
        overwrite=args.overwrite,
    )
    write_table(nodes, args.output_dir / "small_graph_nodes.tsv", overwrite=args.overwrite)

    model_genes, model_report = read_model_genes(
        args.model_var_path, args.model_gene_column, args.model_hvg_column
    )
    report["model_genes"] = model_report
    model_drivers, all_drivers, unmapped = build_small_drivers(
        args, graph, nodes, model_genes, report
    )
    write_table(
        model_drivers, args.output_dir / "small_driver_program.tsv", overwrite=args.overwrite
    )
    write_table(
        all_drivers,
        args.output_dir / "small_driver_program_all_genes.tsv",
        overwrite=args.overwrite,
    )
    write_table(
        unmapped, args.output_dir / "small_driver_context_unmapped.tsv", overwrite=args.overwrite
    )
    build_small_cell_map(args, nodes, report)

    if args.strict and model_genes is not None and model_drivers.empty:
        raise ValueError("Small driver program has zero overlap with model genes")


def run_big(args: argparse.Namespace, report: Dict[str, Any]) -> None:
    developmental, spatial, alias, nodes = build_big_graph(args, report)
    write_table(developmental, args.output_dir / "big_curated_graph.tsv", overwrite=args.overwrite)
    write_table(spatial, args.output_dir / "big_spatial_graph.tsv", overwrite=args.overwrite)
    write_table(
        alias, args.output_dir / "big_alias_or_equivalence_edges.tsv", overwrite=args.overwrite
    )
    write_table(nodes, args.output_dir / "big_graph_nodes.tsv", overwrite=args.overwrite)

    model_genes, model_report = read_model_genes(
        args.model_var_path, args.model_gene_column, args.model_hvg_column
    )
    report["model_genes"] = model_report
    tf_drivers, all_drivers = build_big_drivers(args, developmental, model_genes, report)
    write_table(tf_drivers, args.output_dir / "big_driver_program_tf.tsv", overwrite=args.overwrite)
    write_table(
        all_drivers, args.output_dir / "big_driver_program_all.tsv", overwrite=args.overwrite
    )
    build_big_cell_map(args, nodes, report)

    if args.strict and model_genes is not None and tf_drivers.empty:
        raise ValueError("Big Table-23 TF driver program has zero overlap with model genes")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert released author graph/module tables into validated FM biological-prior inputs.",
    )
    parser.add_argument("--dataset", choices=["small", "big"], required=True)
    parser.add_argument("--author-prior-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-var-path", default="", help="Optional exact FM var table for driver-gene alignment"
    )
    parser.add_argument(
        "--model-gene-column",
        default="",
        help="Explicit gene-symbol column; auto prioritizes gene_short_name",
    )
    parser.add_argument("--model-hvg-column", default="highly_variable")
    parser.add_argument(
        "--obs-path", default="", help="Optional obs table; used for small exact cell mapping"
    )
    parser.add_argument("--cell-id-col", default="")
    parser.add_argument("--minimum-cell-coverage", type=float, default=0.01)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inspect-only", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--small-trajectory-workbook", default="")
    parser.add_argument("--small-expression-module-workbook", default="")
    parser.add_argument("--small-chromatin-module-workbook", default="")
    parser.add_argument("--small-trajectory-sheets", nargs="*", default=list(KNOWN_SMALL_LINEAGES))
    parser.add_argument("--small-obs-cluster-col", default="")
    parser.add_argument("--small-obs-age-col", default="")
    parser.add_argument("--small-top-genes-per-node", type=int, default=100)

    parser.add_argument("--big-edges-path", default="")
    parser.add_argument("--big-nodes-path", default="")
    parser.add_argument("--big-cell-graph-rds", default="")
    parser.add_argument("--big-cell-map-path", default="")
    parser.add_argument("--big-supplementary-workbook", default="")
    parser.add_argument("--big-table23-sheet", default="")
    parser.add_argument("--big-table24-sheet", default="")
    parser.add_argument("--big-driver-direction", default="up in early transition")
    parser.add_argument("--big-adjusted-p-max", type=float, default=0.05)
    parser.add_argument("--big-top-tf-per-edge", type=int, default=10)
    parser.add_argument("--big-top-gene-per-edge", type=int, default=25)
    parser.add_argument("--big-gene-driver-weight", type=float, default=0.5)
    parser.add_argument("--rscript-executable", default="Rscript")
    parser.add_argument(
        "--rds-helper-path",
        default=str(Path(__file__).with_name("extract_big_cell_graph_rds_for_fm.R")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.author_prior_dir = args.author_prior_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.author_prior_dir.is_dir():
        raise NotADirectoryError(f"Author-prior directory not found: {args.author_prior_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schema_report = build_schema_report(args)
    schema_report_path = args.output_dir / "input_schema_report.json"
    if schema_report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {schema_report_path}")
    write_json(schema_report, schema_report_path)
    log(f"[inspect] wrote {schema_report_path}")
    if args.inspect_only:
        log("[done] inspect-only mode completed")
        return

    report: Dict[str, Any] = {
        "status": "running",
        "converter_version": VERSION,
        "dataset": args.dataset,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "author_prior_dir": str(args.author_prior_dir),
        "output_dir": str(args.output_dir),
        "strict": bool(args.strict),
        "inputs": {
            "model_var": (
                file_record(Path(args.model_var_path).expanduser().resolve())
                if clean_text(args.model_var_path)
                else file_record(None)
            ),
            "obs": (
                file_record(Path(args.obs_path).expanduser().resolve())
                if clean_text(args.obs_path)
                else file_record(None)
            ),
        },
    }
    try:
        if args.dataset == "small":
            run_small(args, report)
        else:
            run_big(args, report)
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
