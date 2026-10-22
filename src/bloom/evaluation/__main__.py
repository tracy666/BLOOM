"""Command-line entry point for paper-ordered evaluation workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BLOOM evaluation: checkpoint export, measured-stage metrics and validation."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Generate measured and developmental-query HDF5 files")
    export.add_argument(
        "--model",
        choices=("bloom-unidirectional", "bloom-bidirectional", "transformer"),
        required=True,
    )
    export.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Forward BLOOM, shared bidirectional BLOOM, or forecasting checkpoint",
    )
    export.add_argument(
        "--reverse-checkpoint",
        type=Path,
        help="Separate reverse-trained unidirectional checkpoint; required for its paired exporter",
    )
    export.add_argument("--metadata", type=Path)
    export.add_argument("--reverse-metadata", type=Path)
    export.add_argument("--training-log", type=Path)
    export.add_argument("--reverse-training-log", type=Path)
    export.add_argument("--dataset-config", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--task",
        action="append",
        choices=("measured", "reverse", "interpolation", "history", "future", "perturbation"),
        help="Repeat for several queries; measured predictions are always exported",
    )
    export.add_argument("--device", default="cpu")
    export.add_argument("--seed", type=int, default=42)
    export.add_argument("--batch-size", type=int)
    export.add_argument(
        "--readout", help="For BLOOM, e.g. blend_0.85; otherwise use the checkpoint record"
    )
    export.add_argument("--future-steps", type=int, default=8)
    export.add_argument("--history-steps", type=int, default=8)
    export.add_argument("--max-cells", type=int, default=10000)
    export.add_argument(
        "--future-time-step",
        type=float,
        default=0.0,
        help="Checkpoint reporting-coordinate increment; zero uses the boundary interval",
    )
    export.add_argument("--history-time-step", type=float, default=0.0)
    export.add_argument("--interpolation-fractions", type=float, nargs="+", default=None)
    export.add_argument(
        "--genes",
        nargs="+",
        help="Genes for separate single-gene input edits; default is source-expressed panel genes",
    )
    export.add_argument("--set-value", type=float, default=0.0)
    export.add_argument("--history-scope", choices=("last", "all"), default="last")
    export.add_argument(
        "--perturbation-max-cells",
        type=int,
        default=2048,
        help="Bidirectional perturbation sampling cap; unidirectional retains its native target cohort",
    )
    export.add_argument(
        "--training-var",
        type=Path,
        help="Original training gene-order parquet when the recorded path was relocated",
    )
    export.add_argument(
        "--training-obs",
        type=Path,
        help="Original training cell-type-order parquet when its recorded path was relocated",
    )
    export.add_argument("--checkpoint-gene-order-sha256")
    export.add_argument("--checkpoint-celltype-order-sha256")
    export.add_argument(
        "--timeseries-library-root",
        type=Path,
        help="Exact Time-Series-Library checkout used by forecasting baselines",
    )
    export.add_argument(
        "--model-source",
        type=Path,
        help="Advanced override for the reorganized matching training runtime",
    )
    export.add_argument("--grn", type=Path)
    export.add_argument("--programmes", type=Path)
    export.add_argument("--graph", type=Path)
    export.add_argument("--graph-nodes", type=Path)
    export.add_argument("--graph-cell-map", type=Path)
    export.add_argument("--preflight-only", action="store_true")
    export.add_argument("--resume", action="store_true")
    export.add_argument(
        "--allow-pickled-trajectory",
        action="store_true",
        help="Allow a trusted legacy object-array/pickle trajectory",
    )
    export.add_argument(
        "--allow-unsafe-checkpoint-load",
        action="store_true",
        help="Allow a trusted legacy checkpoint requiring pickle",
    )
    export.add_argument(
        "--native-arg",
        action="append",
        default=[],
        help="Additional exporter argument; use --native-arg=--flag and --native-arg=value",
    )
    export.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command/configuration without loading a checkpoint",
    )
    sub.add_parser("metrics", help="Run the preserved numerical benchmark CLI", add_help=False)
    sub.add_parser(
        "classifier", help="Generate measured-expression classifier diagnostics", add_help=False
    )
    sub.add_parser(
        "biology-report", help="Generate programme and regulatory summaries", add_help=False
    )
    validate = sub.add_parser(
        "validate-h5", help="Validate the canonical measured-stage prediction contract"
    )
    validate.add_argument("path", type=Path)
    validate.add_argument("--direction", choices=("forward", "reverse"))
    checkpoints = sub.add_parser(
        "validate-checkpoints",
        help="Check future checkpoint manifests, paths and optional SHA-256 values",
    )
    checkpoints.add_argument("--manifest", type=Path, required=True)
    checkpoints.add_argument(
        "--allow-missing", action="store_true", help="Report not-yet-uploaded files as pending"
    )
    checkpoints.add_argument("--output", type=Path)
    sub.add_parser("native-help", help="Show the full internal exporter's supported options")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "metrics":
        from .metrics_cli import main as metrics_main

        return metrics_main(argv[1:])
    if argv and argv[0] == "classifier":
        from .diagnostics.classifier import main as classifier_main

        return classifier_main(argv[1:])
    if argv and argv[0] == "biology-report":
        from .diagnostics.biology import main as biology_main

        return biology_main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "export":
        from .export import export_predictions

        return export_predictions(args)
    if args.command == "native-help":
        from .export import native_help

        return native_help()
    if args.command == "validate-checkpoints":
        from .checkpoints import validate_manifest

        report = validate_manifest(args.manifest, allow_missing=args.allow_missing)
        text = json.dumps(report, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        return 0 if report["valid"] else 2
    if args.command == "validate-h5":
        from .export import validate_h5

        print(json.dumps(validate_h5(args.path, args.direction), indent=2, default=str))
        return 0
    parser.error("Unknown evaluation command")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
