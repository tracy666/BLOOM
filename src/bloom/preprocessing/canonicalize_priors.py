"""Publish stable filenames for the unmodified author-converter outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

FILES = {
    "cortex": {
        "graph_edges.tsv": "small_curated_graph.tsv",
        "graph_nodes.tsv": "small_graph_nodes.tsv",
        "programmes.tsv": "small_driver_program.tsv",
        "cell_map.parquet": "small_cell_to_graph_node.parquet",
    },
    "prenatal": {
        "graph_edges.tsv": "big_curated_graph.tsv",
        "graph_nodes.tsv": "big_graph_nodes.tsv",
        "programmes.tsv": "big_driver_program_tf.tsv",
        "cell_map.parquet": "big_cell_to_graph_node.tsv.gz",
    },
    "sted": {
        "graph_edges.tsv": "sted_curated_graph.tsv",
        "graph_nodes.tsv": "sted_graph_nodes.tsv",
        "programmes.tsv": "sted_driver_program.tsv",
        "cell_map.parquet": "sted_cell_to_graph_node.parquet",
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(FILES), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for target, source in FILES[args.dataset].items():
        source = args.input_dir / source
        if not source.exists() and source.name.endswith(".parquet"):
            source = source.with_suffix(".tsv.gz")
        if not source.is_file():
            raise FileNotFoundError(f"Required converter output is missing: {source}")
        destination = args.output_dir / target
        if destination.exists() and not args.overwrite:
            raise FileExistsError(destination)
        if target.endswith(".parquet") and not source.name.endswith(".parquet"):
            pd.read_csv(source, sep="\t").to_parquet(destination, index=False)
        else:
            shutil.copyfile(source, destination)
        report[target] = source.name
    programmes = pd.read_csv(args.output_dir / "programmes.tsv", sep="\t")
    context = "node" if "node" in programmes else "target"
    programme_table = programmes[[context, "gene", "score"]].rename(
        columns={context: "program", "score": "weight"}
    )
    programme_table.to_csv(args.output_dir / "analysis_programmes.tsv", sep="\t", index=False)
    genes = list(dict.fromkeys(programmes["gene"].astype(str)))
    (args.output_dir / "author_gene_list.txt").write_text("\n".join(genes) + "\n")
    (args.output_dir / "canonical_prior_manifest.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "native_files": report,
                "analysis_programmes": {"context_column": context, "weight_column": "score"},
                "author_genes": len(genes),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
