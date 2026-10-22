#!/usr/bin/env bash
# Render the perturbation figure group using saved evaluation outputs.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${BLOOM_PYTHON:-python3}" -m bloom.figures render \
  --group perturbation \
  --config "${BLOOM_FIGURE_CONFIG:-configs/figures/config.json}" \
  --panel author "$@"
