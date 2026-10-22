"""Solve sequential moscot problems and export row-aligned temporal couplings.

The producer follows the supplied notebook's TemporalProblem.prepare/solve
workflow. Solver settings are explicit release recipes: the notebook records
several trials and does not identify one successful final historical export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Prepared AnnData .h5ad in the same row order as prepared/obs.parquet",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Base output directory; couplings are written below the PCA-key subdirectory",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        help="Numeric measured-stage column; existing time values are retained",
    )
    parser.add_argument("--pca-key", default="X_pca", help="Existing AnnData.obsm representation")
    parser.add_argument(
        "--compute-pca",
        action="store_true",
        help="Compute missing/replacement PCA from the explicitly selected layer and mask",
    )
    parser.add_argument(
        "--pca-layer", default=None, help="Default uses AnnData.X without changing its scale"
    )
    parser.add_argument(
        "--pca-mask-column",
        default=None,
        help="Optional full-length var Boolean mask, normally the broad transport HVG mask",
    )
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--tau-a", type=float, default=1.0)
    parser.add_argument("--tau-b", type=float, default=1.0)
    parser.add_argument(
        "--rank",
        type=int,
        default=-1,
        help="-1 uses full rank; positive values request moscot's low-rank solver",
    )
    parser.add_argument("--scale-cost", default="mean")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.001)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "gpu", "tpu"], default="cpu")
    parser.add_argument(
        "--seed", type=int, default=0, help="PCA random state when --compute-pca is used"
    )
    parser.add_argument(
        "--allow-nonconverged",
        action="store_true",
        help="Export an explicitly marked exploratory result despite a false convergence flag",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def prepare_numeric_stages(obs, column):
    """Keep numeric times and their original string labels for coupling filenames."""
    import numpy as np
    import pandas as pd

    if column not in obs:
        raise ValueError(f"Missing temporal column {column!r}")
    values = pd.to_numeric(obs[column], errors="raise")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Temporal values must be finite numeric measured-stage coordinates")
    labels = obs[column].astype(str)
    mapping = {}
    for number, label in zip(values, labels):
        number = float(number)
        if number in mapping and mapping[number] != label:
            raise ValueError(
                f"Numeric stage {number} has conflicting text labels; standardize the temporal column first"
            )
        mapping[number] = label
    if len(mapping) < 2:
        raise ValueError("At least two measured stages are required for temporal OT")
    return values.to_numpy(dtype=float), mapping


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        import anndata as ad
        import numpy as np
        from moscot.problems.time import TemporalProblem
    except ImportError as exc:
        raise SystemExit(
            f"Temporal transport dependencies could not be imported ({exc}); install the transport extra/environment described in docs/preprocessing.md."
        ) from exc
    adata = ad.read_h5ad(args.input)
    if not adata.obs_names.is_unique:
        raise ValueError("AnnData.obs_names must uniquely identify the expression rows")
    numeric, labels = prepare_numeric_stages(adata.obs, args.time_column)
    transport_time = "_bloom_transport_time"
    adata.obs[transport_time] = numeric
    if args.compute_pca:
        try:
            import scanpy as sc
        except ImportError as exc:
            raise SystemExit(
                "Computing PCA requires scanpy; otherwise provide an existing --pca-key."
            ) from exc
        view = (
            adata.copy()
            if args.pca_mask_column is None
            else adata[:, adata.var[args.pca_mask_column].fillna(False).to_numpy(dtype=bool)].copy()
        )
        if args.pca_layer is not None:
            view.X = view.layers[args.pca_layer]
        components = min(args.pca_components, view.n_vars - 1, view.n_obs - 1)
        if components < 1:
            raise ValueError("Insufficient rows/genes to compute PCA")
        sc.pp.pca(view, n_comps=components, random_state=args.seed, svd_solver="arpack")
        adata.obsm[args.pca_key] = view.obsm["X_pca"]
    if args.pca_key not in adata.obsm:
        raise ValueError(
            f"No AnnData.obsm[{args.pca_key!r}]; provide a prepared PCA representation or pass --compute-pca"
        )
    output = args.output_dir / args.pca_key
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "Transport output directory is not empty; choose another directory or pass --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    problem = TemporalProblem(adata).prepare(
        time_key=transport_time, joint_attr=args.pca_key, policy="sequential", cost="sq_euclidean"
    )
    solve = {
        "epsilon": args.epsilon,
        "tau_a": args.tau_a,
        "tau_b": args.tau_b,
        "rank": args.rank,
        "scale_cost": args.scale_cost,
        "batch_size": args.batch_size,
        "threshold": args.threshold,
        "device": args.device,
    }
    if args.max_iterations is not None:
        solve["max_iterations"] = args.max_iterations
    problem = problem.solve(**solve)
    records = []
    for (source_time, target_time), solution in problem.solutions.items():
        source_time, target_time = float(source_time), float(target_time)
        source_label, target_label = labels[source_time], labels[target_time]
        converged_value = getattr(solution, "converged", None)
        converged = None if converged_value is None else bool(converged_value)
        if converged is False and not args.allow_nonconverged:
            raise RuntimeError(
                f"OT solution {source_label}->{target_label} did not converge; revise the solver recipe before generating trajectories"
            )
        matrix = np.asarray(solution.transport_matrix)
        expected = (int((numeric == source_time).sum()), int((numeric == target_time).sum()))
        if (
            matrix.shape != expected
            or not np.isfinite(matrix).all()
            or (matrix < 0).any()
            or float(matrix.sum()) <= 0
        ):
            raise ValueError(
                f"Invalid transport matrix for {source_label}->{target_label}: shape={matrix.shape}, expected={expected}"
            )
        name = f"start_{source_label}_end_{target_label}_transport_matrix.npy"
        np.save(output / name, matrix)
        records.append(
            {
                "source": source_label,
                "target": target_label,
                "shape": list(matrix.shape),
                "converged": converged,
                "path": name,
            }
        )
    if len(records) != len(labels) - 1:
        raise RuntimeError(
            "Sequential solve did not provide exactly one coupling for each adjacent-stage pair"
        )
    manifest = {
        "input": str(args.input),
        "time_column": args.time_column,
        "pca_key": args.pca_key,
        "computed_pca": args.compute_pca,
        "pca_layer": args.pca_layer or "X",
        "pca_mask_column": args.pca_mask_column,
        "pca_random_state": args.seed,
        "policy": "sequential",
        "cost": "sq_euclidean",
        "solve": solve,
        "row_order": "Original AnnData row order within each stage",
        "couplings": records,
        "provenance": "Portable producer derived from the supplied notebook; solver settings are explicit release recipes, not an attestation of the final historical run.",
    }
    (output / "transport_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    adata.obs.drop(columns=[transport_time]).to_parquet(output / "obs.parquet")
    print(json.dumps({"output": str(output), "couplings": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
