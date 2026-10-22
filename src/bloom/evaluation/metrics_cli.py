"""Public CLI for the preserved BLOOM numerical benchmark implementation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measured forward-prediction numerical benchmark; separate native panels/cohorts"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Discover canonical forward H5 files and bind training logs")
    init.add_argument(
        "--prediction-root",
        "--v5-root",
        dest="v5_root",
        action="append",
        default=[],
        help="Export tree; searches immutable run folders recursively using canonical H5 metadata",
    )
    init.add_argument("--training-log-root", action="append", default=[])
    init.add_argument("--legacy-input", action="append", default=[])
    init.add_argument("--fm-readout", default="blend_0.85")
    init.add_argument("--fm-result-tag")
    init.add_argument("--normalization")
    init.add_argument("--split")
    init.add_argument("--out", required=True)
    audit = sub.add_parser("audit", help="Validate saved source/log bindings")
    audit.add_argument("--config", required=True)
    audit.add_argument("--out", required=True)
    run_parser = sub.add_parser("run", help="Write global, stage and gene metrics")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--out-root", required=True)
    run_parser.add_argument("--tag")
    run_parser.add_argument("--workers", default="auto")
    run_parser.add_argument("--cpu-reserve", type=int, default=1)
    run_parser.add_argument("--mode", choices=("core", "full"), default="full")
    run_parser.add_argument(
        "--dataset",
        choices=("all", "prenatal", "cortex", "sted", "big", "small"),
        default="all",
        help="Public dataset names; big/small remain accepted export aliases",
    )
    run_parser.add_argument("--panel", choices=("all", "hvg", "200", "author"), default="all")
    args = parser.parse_args(argv)
    # Preserve the original runner's one-thread-per-worker numerical policy.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    from .metrics.config import discover, load_config
    from .metrics.runner import audit_config, run, write_csv, write_json

    if args.command == "init":
        path = Path(args.out).expanduser().resolve()
        if path.exists():
            raise ValueError(f"Configuration already exists: {path}; choose a new filename")
        config, report = discover(
            args.v5_root,
            args.training_log_root,
            args.legacy_input,
            args.fm_readout,
            args.fm_result_tag,
            args.normalization,
            args.split,
        )
        from .discovery import add_recursive_exports

        add_recursive_exports(
            config, args.v5_root, report, normalization=args.normalization, split=args.split
        )
        write_json(path, config)
        write_json(path.with_suffix(".discovery.json"), report)
        write_csv(path.with_suffix(".discovery.csv"), report)
        print(f"Configuration: {path}; paired sources: {len(config['raw_sources'])}")
        return 0
    if args.command == "audit":
        audit_config(load_config(args.config), args.out)
        return 0
    if args.cpu_reserve < 0 or (args.workers != "auto" and int(args.workers) < 1):
        raise ValueError("Worker count must be positive (or auto), and cpu-reserve nonnegative")
    dataset = {"prenatal": "big", "cortex": "small"}.get(args.dataset, args.dataset)
    _, status = run(
        load_config(args.config),
        args.out_root,
        args.tag,
        args.workers,
        args.cpu_reserve,
        args.mode,
        dataset,
        "hvg" if args.panel == "200" else args.panel,
    )
    return 2 if status["blocked_inputs"] else 0
