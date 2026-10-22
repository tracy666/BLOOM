"""Dispatch preprocessing stages without importing optional numerical packages."""

from __future__ import annotations

import argparse
import importlib
import sys

COMMANDS = {
    "export-h5ad": "export_h5ad",
    "transport": "transport",
    "author-priors": "author_priors",
    "grn": "grn",
    "trajectories": "trajectories",
    "sted-priors": "sted_priors",
    "canonicalize-priors": "canonicalize_priors",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Prepare BLOOM expression inputs, temporal correspondences and biological priors."
    )
    parser.add_argument("stage", choices=list(COMMANDS))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    stage = parser.parse_args(argv[:1]).stage
    module = importlib.import_module(f"bloom.preprocessing.{COMMANDS[stage]}")
    sys.argv = [f"bloom.preprocessing {stage}", *argv[1:]]
    result = module.main()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
