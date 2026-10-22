"""Portable orchestration around the preserved checkpoint-export adapters."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TASK_NAMES = {
    "measured": "normal",
    "reverse": "normal",
    "interpolation": "interpolation",
    "history": "reverse_history",
    "future": "extrapolation",
    "perturbation": "perturbation",
}


def repository_root() -> Path:
    """Locate the source checkout containing the versioned export backend."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "third_party" / "downstream" / "run_v4_single.py").is_file():
        raise FileNotFoundError(
            "The export backend is missing. Install BLOOM from its complete source checkout with `pip install -e .`."
        )
    return root


def _environment(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def native_help() -> int:
    root = repository_root()
    return subprocess.call(
        [sys.executable, str(root / "third_party/downstream/run_v4_single.py"), "--help"],
        env=_environment(root),
    )


def resolved_dataset_config(args: Any) -> dict[str, Any]:
    """Bind paths to the input JSON before creating a temporary run config."""
    path = args.dataset_config.expanduser().resolve()
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must be a JSON object")
    for key, value in list(config.items()):
        if key.endswith("_path") and isinstance(value, str) and value:
            p = Path(os.path.expandvars(value)).expanduser()
            config[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
    fractions = (
        sorted(set(args.interpolation_fractions))
        if args.interpolation_fractions is not None
        else None
    )
    if fractions is not None and (not fractions or any(not 0 < f < 1 for f in fractions)):
        raise ValueError("Interpolation fractions must lie strictly between zero and one")
    if args.future_steps < 0 or args.history_steps < 0 or args.max_cells < 0:
        raise ValueError("Rollout counts and maximum cell counts must be nonnegative")
    scenarios = dict(config.get("scenarios", {}))
    interpolation = dict(scenarios.get("interpolation", {}))
    interpolation.update(max_cells_per_stage=args.max_cells)
    if fractions is not None:
        interpolation["fractions"] = fractions
    scenarios["interpolation"] = interpolation
    perturbation = dict(scenarios.get("perturbation", {}))
    perturbation.update(
        set_value=args.set_value,
        history_scope=args.history_scope,
        max_target_cells=args.perturbation_max_cells,
    )
    if args.genes:
        perturbation["genes"] = list(dict.fromkeys(args.genes))
    scenarios["perturbation"] = perturbation
    future = dict(scenarios.get("extrapolation", {}))
    future.update(
        future_rollout_steps=args.future_steps,
        future_max_cells=args.max_cells,
        future_physical_time_step=args.future_time_step,
    )
    scenarios["extrapolation"] = future
    history = dict(scenarios.get("reverse_history", {}))
    history.update(
        history_rollout_steps=args.history_steps,
        history_max_cells=args.max_cells,
        history_physical_time_step=args.history_time_step,
    )
    scenarios["reverse_history"] = history
    config["scenarios"] = scenarios
    # The retained unidirectional adapter uses flat names and dyadic grids;
    # the bidirectional adapter uses nested scenario options.
    flat = dict(config.get("release_scenario_options", {}))
    flat.update(
        interpolation_max_cells_per_stage=args.max_cells,
        perturbation_history_scope=args.history_scope,
        perturbation_set_value=args.set_value,
    )
    if args.genes:
        flat["perturbation_genes"] = list(dict.fromkeys(args.genes))
    if args.model == "bloom-unidirectional" and fractions is not None:
        levels = [
            level
            for level in range(1, 9)
            if len(fractions) == 2**level - 1
            and all(abs(f - (i + 1) / 2**level) < 1e-12 for i, f in enumerate(fractions))
        ]
        if not levels:
            raise ValueError(
                "Unidirectional interpolation uses complete dyadic grids (e.g. 0.25 0.5 0.75); use bidirectional BLOOM for arbitrary fractions"
            )
        flat["interpolation_dyadic_level"] = levels[0]
    config["release_scenario_options"] = flat
    return config


def build_export_command(args: Any, dataset_config: Path) -> list[str]:
    root = repository_root()
    tasks = args.task or ["measured"]
    if args.model == "transformer" and any(t != "measured" for t in tasks):
        raise ValueError(
            "The supplied forecasting baseline adapter supports measured forward prediction; select a BLOOM checkpoint for developmental queries"
        )
    native_tasks = list(dict.fromkeys(["normal", *(TASK_NAMES[t] for t in tasks)]))
    command = [
        sys.executable,
        str(root / "third_party/downstream/run_v4_single.py"),
        "--dataset-config",
        str(dataset_config),
        "--output-root",
        str(args.output.expanduser().resolve()),
        "--tasks",
        ",".join(native_tasks),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--skip-v1",
        "--skip-downstream",
    ]
    if args.model == "bloom-unidirectional":
        if not args.reverse_checkpoint:
            raise ValueError(
                "The preserved unidirectional exporter requires a forward/reverse checkpoint pair. Train the reverse direction with the same unidirectional launcher and supply --reverse-checkpoint."
            )
        command += [
            "--fm-v6-forward-checkpoint",
            str(args.checkpoint.resolve()),
            "--fm-v6-reverse-checkpoint",
            str(args.reverse_checkpoint.resolve()),
        ]
        optional = {
            "--forward-metadata": args.metadata,
            "--reverse-metadata": args.reverse_metadata,
            "--forward-training-log": args.training_log,
            "--reverse-training-log": args.reverse_training_log,
        }
    else:
        checkpoint_flag = (
            "--fm-v7-checkpoint"
            if args.model == "bloom-bidirectional"
            else "--transformer-v5-checkpoint"
        )
        command += [checkpoint_flag, str(args.checkpoint.resolve())]
        optional = {"--metadata": args.metadata, "--training-log": args.training_log}
    optional.update(
        {
            "--training-var-path": args.training_var,
            "--training-obs-path": args.training_obs,
            "--model-source": args.model_source,
            "--timeseries-library-root": args.timeseries_library_root,
            "--grn-tsv": args.grn,
            "--driver-tsv": args.programmes,
            "--curated-graph-tsv": args.graph,
            "--curated-graph-nodes-path": args.graph_nodes,
            "--curated-graph-cell-map-path": args.graph_cell_map,
        }
    )
    for flag, value in optional.items():
        if value is not None:
            command += [flag, str(value.expanduser().resolve())]
    for name in ("checkpoint_gene_order_sha256", "checkpoint_celltype_order_sha256"):
        if getattr(args, name):
            command += ["--" + name.replace("_", "-"), getattr(args, name)]
    if args.model != "transformer":
        command += ["--model-repo-root", str(root)]
    if args.batch_size:
        command += ["--batch-size", str(args.batch_size)]
    if args.readout:
        command += ["--readout", args.readout]
    command += [
        "--future-rollout-steps",
        str(args.future_steps),
        "--history-rollout-steps",
        str(args.history_steps),
        "--future-max-cells",
        str(args.max_cells),
        "--history-max-cells",
        str(args.max_cells),
        "--future-physical-time-step",
        str(args.future_time_step),
        "--history-physical-time-step",
        str(args.history_time_step),
    ]
    for attr, flag in [
        ("resume", "--resume"),
        ("preflight_only", "--preflight-only"),
        ("allow_pickled_trajectory", "--allow-pickled-trajectory"),
        ("allow_unsafe_checkpoint_load", "--allow-unsafe-checkpoint-load"),
    ]:
        if getattr(args, attr):
            command.append(flag)
    command += args.native_arg
    return command


def export_predictions(args: Any) -> int:
    root = repository_root()
    config = resolved_dataset_config(args)
    if args.dry_run:
        command = build_export_command(args, Path("<resolved-dataset-config.json>"))
        print(json.dumps({"command": command, "configuration": config}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    # Runtime-generated configuration is retained with the output for reuse.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", prefix="bloom-dataset-", dir=args.output, delete=False
    ) as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
        path = Path(stream.name).resolve()
    command = build_export_command(args, path)
    print(shlex.join(command), flush=True)
    return subprocess.call(command, cwd=root, env=_environment(root))


def validate_h5(path: Path, direction: str | None = None) -> dict[str, Any]:
    root = repository_root()
    sys.path.insert(0, str(root / "third_party/downstream"))
    try:
        from v4lib.contract import validate_prediction_h5

        return validate_prediction_h5(path.expanduser().resolve(), expected_direction=direction)
    finally:
        sys.path.pop(0)
