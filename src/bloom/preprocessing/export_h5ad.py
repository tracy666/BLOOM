"""Export a prepared AnnData object to BLOOM's aligned sparse-table inputs.

This release utility preserves the selected expression layer exactly. Dataset
quality control and normalization belong to the dataset preparation notebooks.
Gene panels are full-length Boolean masks; expression columns are never silently
reordered or reduced when exporting the model-panel metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


def read_gene_list(path: str | Path, column: str = "gene") -> list[str]:
    """Read a plain one-gene-per-line file or a named CSV/TSV column."""
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t" if path.suffix == ".tsv" else ",")
            if column not in (reader.fieldnames or []):
                raise ValueError(f"{path} has no gene column {column!r}")
            names = [str(row[column]).strip() for row in reader]
    else:
        names = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return list(dict.fromkeys(name for name in names if name))


def panel_mask(
    var,
    *,
    gene_column: str,
    gene_list: str | None = None,
    mask_column: str | None = None,
    allow_missing: bool = False,
):
    """Return a full-column mask and an explicit alignment report."""
    import numpy as np

    if bool(gene_list) == bool(mask_column):
        raise ValueError("Specify exactly one gene-list file or existing mask column per panel")
    names = (
        var[gene_column].astype(str).to_numpy()
        if gene_column in var.columns
        else var.index.astype(str).to_numpy()
    )
    if gene_list:
        requested = read_gene_list(gene_list)
        available = set(names)
        missing = [name for name in requested if name not in available]
        if missing and not allow_missing:
            raise ValueError(
                f"{len(missing)} requested genes are absent; first examples: {missing[:10]}"
            )
        mask = np.isin(names, requested)
        report = {
            "source": str(gene_list),
            "requested_genes": len(requested),
            "missing_genes": missing,
        }
    else:
        if mask_column not in var.columns:
            raise ValueError(f"Prepared AnnData.var has no panel mask {mask_column!r}")
        values = var[mask_column]
        if not set(values.dropna().unique()).issubset({True, False, 0, 1}):
            raise ValueError(f"{mask_column!r} must contain Boolean or 0/1 values")
        mask = values.fillna(False).to_numpy(dtype=bool)
        report = {"source_column": mask_column}
    if not bool(mask.any()):
        raise ValueError("The selected panel contains zero expression columns")
    return mask, {
        **report,
        "selected_columns": int(mask.sum()),
        "total_expression_columns": len(mask),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path, help="Prepared .h5ad file")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--layer", default=None, help="Expression layer to export; default preserves AnnData.X"
    )
    parser.add_argument("--gene-column", default="gene_short_name")
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--celltype-column", default="class_label")
    parser.add_argument("--author-gene-list", default=None)
    parser.add_argument("--author-mask-column", default=None)
    parser.add_argument("--hvg-gene-list", default=None)
    parser.add_argument("--hvg-mask-column", default=None)
    parser.add_argument(
        "--hvg-size",
        type=int,
        default=200,
        help="Expected model-HVG panel size and output filename suffix",
    )
    parser.add_argument("--transport-gene-list", default=None)
    parser.add_argument("--transport-mask-column", default=None)
    parser.add_argument(
        "--transport-hvg-size",
        type=int,
        default=2000,
        help="Expected transport/GRN panel size and output filename suffix",
    )
    parser.add_argument("--allow-missing-genes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import anndata as ad
        import scipy.sparse as sp
    except ImportError as exc:
        raise SystemExit(
            "H5AD export requires the preprocessing dependencies; install the repository with pip install -e '.[preprocess]'."
        ) from exc
    adata = ad.read_h5ad(args.input)
    for column in [args.time_column, args.celltype_column]:
        if column not in adata.obs:
            raise ValueError(f"AnnData.obs is missing {column!r}")
    if not adata.obs_names.is_unique:
        raise ValueError("AnnData.obs_names must uniquely identify rows before exporting")
    if not adata.var_names.is_unique:
        raise ValueError("AnnData.var_names must uniquely identify columns before exporting")
    matrix = adata.X if args.layer is None else adata.layers[args.layer]
    matrix = sp.csr_matrix(matrix)
    var = adata.var.copy()
    if args.gene_column not in var:
        var[args.gene_column] = var.index.astype(str)
    panels = []
    for name, gene_list, mask_column in [
        ("author", args.author_gene_list, args.author_mask_column),
        (f"hvg_{args.hvg_size}", args.hvg_gene_list, args.hvg_mask_column),
        (
            f"transport_hvg_{args.transport_hvg_size}",
            args.transport_gene_list,
            args.transport_mask_column,
        ),
    ]:
        if gene_list or mask_column:
            mask, report = panel_mask(
                var,
                gene_column=args.gene_column,
                gene_list=gene_list,
                mask_column=mask_column,
                allow_missing=args.allow_missing_genes,
            )
            expected = (
                args.hvg_size
                if name.startswith("hvg_")
                else args.transport_hvg_size if name.startswith("transport_hvg_") else None
            )
            if expected is not None and int(mask.sum()) != expected:
                raise ValueError(
                    f"Panel {name!r} selected {int(mask.sum())} columns, expected {expected}; correct the list/mask or set its explicit size option"
                )
            panels.append((name, mask, report))
    if not panels:
        raise ValueError(
            "Supply at least one explicit author/HVG/transport gene list or mask column"
        )
    out = args.output_dir
    files = [out / "X.npz", out / "obs.parquet", out / "export_manifest.json"] + [
        out / f"genes_{name}.parquet" for name, _, _ in panels
    ]
    if not args.overwrite and any(path.exists() for path in files):
        raise FileExistsError(
            "Export outputs already exist; choose another output directory or pass --overwrite"
        )
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / "X.npz", matrix)
    adata.obs.to_parquet(out / "obs.parquet")
    for name, mask, _ in panels:
        table = var.copy()
        table["highly_variable"] = mask
        table.to_parquet(out / f"genes_{name}.parquet")
    manifest = {
        "input": str(args.input),
        "expression_layer": args.layer or "X",
        "shape": list(matrix.shape),
        "expression_scale_modified": False,
        "row_order": "unchanged AnnData.obs order",
        "column_order": "unchanged AnnData.var order",
        "gene_column": args.gene_column,
        "time_column": args.time_column,
        "celltype_column": args.celltype_column,
        "panels": {name: report for name, _, report in panels},
        "utility_scope": "Export of prepared data; no quality control, normalization, HVG ranking or transport inference is performed.",
    }
    (out / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"output_dir": str(out), "shape": list(matrix.shape), "panels": manifest["panels"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
