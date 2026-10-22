"""Resolve portable data paths and launch the shared trainer.

The public CLI deliberately separates lightweight recipe inspection from heavy
scientific imports. Native uppercase arguments remain available as overrides;
``--native-help`` prints the complete parser for the selected model family.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
from datetime import datetime, timezone

from . import arguments

_RECIPE_PATH = Path(__file__).with_name("recipes.json")


def load_recipes() -> dict:
    """Read the source-derived release defaults shipped with the package."""
    return json.loads(_RECIPE_PATH.read_text())


def build_parser() -> argparse.ArgumentParser:
    recipes = load_recipes()
    parser = argparse.ArgumentParser(
        description="Train BLOOM or one paper forecasting baseline on a prepared dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Append native uppercase arguments to override individual settings, for example "
            "--FM_EPOCHS 200 or --TRANS_BATCH 512. Use --native-help to list all native options."
        ),
    )
    parser.add_argument(
        "--model",
        choices=["bloom-unidirectional", "bloom-bidirectional", *recipes["paper_baselines"]],
        default="bloom-unidirectional",
    )
    parser.add_argument("--dataset", choices=sorted(recipes["datasets"]), default="cortex")
    parser.add_argument("--panel", choices=["author", "hvg"], default="author")
    parser.add_argument("--hvg-genes", type=int, default=200)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset-root", type=Path, help="Override data-root/dataset.")
    parser.add_argument("--prepared-dir", type=Path, help="Override dataset-root/prepared.")
    parser.add_argument("--priors-dir", type=Path, help="Override dataset-root/priors.")
    parser.add_argument("--grn-path", type=Path, help="Override dataset-root/grn/adjacencies.tsv.")
    parser.add_argument("--expression", type=Path, help="Override prepared/X.npz.")
    parser.add_argument("--obs", type=Path, help="Override prepared/obs.parquet.")
    parser.add_argument(
        "--genes",
        type=Path,
        help="Full-length var table containing the selected highly_variable mask.",
    )
    parser.add_argument("--trajectories", type=Path, help="Override prepared/trajectories.npy.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/training"))
    parser.add_argument(
        "--tsl-root",
        type=Path,
        default=Path(os.environ.get("BLOOM_TSL_ROOT", "external/Time-Series-Library")),
    )
    parser.add_argument(
        "--direction",
        choices=["forward", "reverse", "bidirectional"],
        help="Unidirectional models default forward; bidirectional BLOOM defaults bidirectional.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON file with a native argument-name/value mapping under 'native'.",
    )
    parser.add_argument(
        "--native-help",
        action="store_true",
        help="Print the selected original parser without loading data or PyTorch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved native invocation and settings without loading data or writing outputs.",
    )
    parser.add_argument(
        "--write-config",
        type=Path,
        help="Optionally save the resolved settings to this JSON path, including with --dry-run.",
    )
    return parser


def model_family(model: str) -> str:
    if model == "bloom-unidirectional":
        return "unidirectional"
    if model == "bloom-bidirectional":
        return "bidirectional"
    return "baseline"


def _argument_vector(settings: dict) -> list[str]:
    result = []
    for name, value in settings.items():
        if value is None:
            continue
        if isinstance(value, bool):
            result.append(f"--{'' if value else 'no-'}{name}")
        else:
            result.append(f"--{name}")
            (
                result.extend(str(item) for item in value)
                if isinstance(value, list)
                else result.append(str(value))
            )
    return result


def resolve_settings(
    options: argparse.Namespace, native_overrides: list[str]
) -> tuple[str, argparse.Namespace, list[str]]:
    """Merge source defaults, dataset paths, optional JSON and explicit arguments."""
    recipe = load_recipes()
    family = model_family(options.model)
    native_parser = arguments.build_parser(family)
    if options.native_help:
        native_parser.print_help()
        raise SystemExit(0)
    dataset_root = options.dataset_root or options.data_root / options.dataset
    prepared = options.prepared_dir or dataset_root / "prepared"
    priors = options.priors_dir or dataset_root / "priors"
    genes = options.genes or prepared / (
        "genes_author.parquet"
        if options.panel == "author"
        else f"genes_hvg_{options.hvg_genes}.parquet"
    )
    recipe_family = "baseline" if family == "baseline" else "bloom"
    settings = dict(recipe["shared"])
    settings.update(recipe[recipe_family])
    settings.update(recipe["datasets"][options.dataset]["native"])
    settings.update(
        {
            "ADATA_X_PATH": str(options.expression or prepared / "X.npz"),
            "ADATA_OBS_PATH": str(options.obs or prepared / "obs.parquet"),
            "ADATA_VAR_PATH": str(genes),
            "TRAJ_PATH": str(options.trajectories or prepared / "trajectories.npy"),
            "REPO_DIR": str(options.tsl_root),
            "OUTPUT_DIR": str(
                options.output_root / options.dataset / options.model / options.panel
            ),
            "MSG": f"{options.dataset}_{options.panel}_{options.model}",
            "RUN_TRANS": family == "baseline",
            "RUN_FM": family != "baseline",
        }
    )
    if family == "baseline":
        settings["TRANS_BASE_MODELS"] = [options.model]
    else:
        settings.update(
            {
                "CURATED_GRAPH_PATH": str(priors / "graph_edges.tsv"),
                "CURATED_GRAPH_NODES_PATH": str(priors / "graph_nodes.tsv"),
                "CURATED_GRAPH_CELL_MAP_PATH": str(priors / "cell_map.parquet"),
                "DRIVER_PROGRAM_PATH": str(priors / "programmes.tsv"),
                "GRN_PATH": str(options.grn_path or dataset_root / "grn" / "adjacencies.tsv"),
                "FM_TEMPORAL_DIRECTION": options.direction
                or ("bidirectional" if family == "bidirectional" else "forward"),
            }
        )
        settings["MSG"] += f"_{settings['FM_TEMPORAL_DIRECTION']}"
        if family == "bidirectional":
            settings.update(
                FM_BIDIRECTIONAL_CONDITIONING="embedding_adapter",
                FM_BIDIRECTIONAL_ADAPTER_HIDDEN=64,
                FM_BIDIRECTIONAL_FORWARD_FRACTION=0.5,
                FM_BIDIRECTIONAL_SPLIT_POLICY="global_biological_targets",
            )
    if options.config:
        payload = json.loads(options.config.read_text())
        overrides = payload.get("native", {})
        if not isinstance(overrides, dict):
            raise ValueError("The JSON 'native' entry must be an argument-name/value mapping.")
        settings.update(overrides)
    native_argv = _argument_vector(settings) + native_overrides
    parsed = native_parser.parse_args(native_argv)
    if parsed.model_family != family:
        raise ValueError("--model-family cannot contradict the selected --model.")
    if family == "baseline" and (parsed.RUN_FM or not parsed.RUN_TRANS):
        raise ValueError(
            "A baseline invocation requires RUN_TRANS and disables RUN_FM; select a BLOOM model for flow-matching training."
        )
    if family != "baseline" and (parsed.RUN_TRANS or not parsed.RUN_FM):
        raise ValueError(
            "A BLOOM invocation requires RUN_FM and disables RUN_TRANS; select a forecasting model for baseline training."
        )
    if family == "unidirectional" and parsed.FM_TEMPORAL_DIRECTION == "bidirectional":
        raise ValueError(
            "Select --model bloom-bidirectional for a shared bidirectional checkpoint."
        )
    return family, parsed, native_argv


def _validate_inputs(args: argparse.Namespace, family: str) -> None:
    files = [args.ADATA_X_PATH, args.ADATA_OBS_PATH, args.ADATA_VAR_PATH, args.TRAJ_PATH]
    if family != "baseline":
        ablations = set(args.FM_BIO_ABLATIONS)
        if ablations - {
            "bio0_baseline",
            "bio1_actual_time",
            "bio2_hierarchy",
            "bio_grn_only",
            "bio2_hierarchy_grn",
        }:
            files.extend(
                [
                    args.CURATED_GRAPH_PATH,
                    args.CURATED_GRAPH_NODES_PATH,
                    args.CURATED_GRAPH_CELL_MAP_PATH,
                ]
            )
        if ablations & {"bio5_drivers", "bio6_grn", "bio7_adaptive_full"}:
            files.append(args.DRIVER_PROGRAM_PATH)
        if ablations & {"bio6_grn", "bio_grn_only", "bio2_hierarchy_grn", "bio7_adaptive_full"}:
            files.append(args.GRN_PATH)
    missing = [str(path) for path in files if not path or not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Required prepared inputs are missing:\n  " + "\n  ".join(missing))
    if family == "baseline" and not (Path(args.REPO_DIR) / "models").is_dir():
        raise FileNotFoundError(
            f"Time-Series-Library model implementations were not found at {args.REPO_DIR}/models. Install the external dependency and pass --tsl-root."
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    options, overrides = parser.parse_known_args(argv)
    try:
        family, args, native_argv = resolve_settings(options, overrides)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    payload = {
        "model": options.model,
        "dataset": options.dataset,
        "panel": options.panel,
        "model_family": family,
        "native_arguments": vars(args),
        "recipe_provenance": load_recipes()["provenance"][
            "bloom" if family != "baseline" else "baseline"
        ],
    }
    if options.write_config:
        options.write_config.parent.mkdir(parents=True, exist_ok=True)
        options.write_config.write_text(json.dumps(payload, indent=2) + "\n")
    if options.dry_run:
        print(
            shlex.join(
                [
                    sys.executable,
                    "-m",
                    "bloom.training.trainer",
                    "--model-family",
                    family,
                    *native_argv,
                ]
            )
        )
        print(json.dumps(payload, indent=2))
        return 0
    try:
        _validate_inputs(args, family)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    output = Path(args.OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    record = output / f"launch_{stamp}.json"
    record.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Resolved launch settings: {record}", flush=True)
    from . import trainer

    result = trainer.run(args)
    rows = result.get("RESULT_ROWS", []) if isinstance(result, dict) else (result or [])
    failures = [row for row in rows if row.get("crashed") or row.get("skipped")]
    if failures:
        for row in failures:
            reason = (
                row.get("crash_reason")
                or row.get("skip_reason")
                or "trainer did not complete this model"
            )
            print(
                f"Training failed for {row.get('model', options.model)}: {reason}", file=sys.stderr
            )
        return 1
    return 0
