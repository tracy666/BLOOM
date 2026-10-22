#!/usr/bin/env bash
# Fetch the verified external baseline source; keep its upstream license intact.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
tsl_dir="${BLOOM_TSL_ROOT:-$repo_root/external/Time-Series-Library}"
python_bin="${BLOOM_PYTHON:-python3}"
skip_install=0
upstream_url="https://github.com/thuml/Time-Series-Library.git"
verified_commit="4e938a1767106324dd753b2a44832bf870a0252e"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_timeseries_library.sh [options]

  --dest DIR       Checkout location (default: external/Time-Series-Library).
  --python PYTHON  Python environment for the optional dependency install.
  --skip-install   Fetch and verify source only; do not invoke pip.
  -h, --help       Show this help.

Install BLOOM's core environment and the appropriate PyTorch build first.
The pinned commit is a release compatibility reference. The historical
upstream commit used for the paper's training runs was not supplied.
EOF
}

while (($#)); do
  case "$1" in
    --dest)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      tsl_dir="$2"; shift 2 ;;
    --python)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      python_bin="$2"; shift 2 ;;
    --skip-install) skip_install=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v git >/dev/null || { printf 'git is required.\n' >&2; exit 1; }
command -v "$python_bin" >/dev/null || { printf 'Python not found: %s\n' "$python_bin" >&2; exit 1; }

if [[ -e "$tsl_dir" && ! -d "$tsl_dir/.git" ]]; then
  printf 'Destination already exists and is not a git checkout: %s\n' "$tsl_dir" >&2
  exit 1
fi
if [[ ! -d "$tsl_dir/.git" ]]; then
  mkdir -p -- "$tsl_dir"
  git -C "$tsl_dir" init --quiet
  git -C "$tsl_dir" remote add origin "$upstream_url"
fi
origin="$(git -C "$tsl_dir" remote get-url origin)"
if [[ "$origin" != "$upstream_url" && "$origin" != "${upstream_url%.git}" ]]; then
  printf 'Existing checkout has a different origin: %s\n' "$origin" >&2
  exit 1
fi
if [[ -n "$(git -C "$tsl_dir" status --porcelain)" ]]; then
  printf 'Checkout has local changes. Preserve them or choose another --dest.\n' >&2
  exit 1
fi
current_commit="$(git -C "$tsl_dir" rev-parse --verify HEAD 2>/dev/null || true)"
if [[ "$current_commit" != "$verified_commit" ]]; then
  git -C "$tsl_dir" fetch --depth 1 origin "$verified_commit"
  git -C "$tsl_dir" checkout --detach "$verified_commit"
fi
tsl_dir="$(cd -- "$tsl_dir" && pwd)"

"$python_bin" - "$tsl_dir" "$verified_commit" <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
expected = sys.argv[2]
models = (
    "Autoformer", "Crossformer", "DLinear", "FEDformer", "FiLM", "FreTS",
    "Informer", "LightTS", "Mamba", "Nonstationary_Transformer", "PAttn",
    "PatchTST", "Reformer", "SegRNN", "TSMixer", "TiDE", "TimeFilter",
    "TimeXer", "TimesNet", "Transformer", "WPMixer", "iTransformer",
)
actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
if actual != expected:
    raise SystemExit(f"Unexpected checkout commit: {actual}")
missing = [name for name in models if not (root / "models" / f"{name}.py").is_file()]
if missing:
    raise SystemExit(f"Missing paper baseline model sources: {missing}")
if not (root / "LICENSE").is_file():
    raise SystemExit("Upstream LICENSE is missing.")
print(f"Verified all {len(models)} paper model files at {actual}.")
PY

if ((skip_install == 0)); then
  "$python_bin" -c 'import torch' || {
    printf 'Install PyTorch for your CPU/CUDA environment first, or use --skip-install.\n' >&2
    exit 1
  }
  "$python_bin" -m pip install einops==0.8.1 reformer-pytorch==1.4.4 PyWavelets==1.9.0
fi

printf '\nUse this environment variable for BLOOM training:\n'
printf 'export BLOOM_TSL_ROOT=%q\n' "$tsl_dir"
printf '\nMamba additionally requires a compatible CUDA environment and mamba_ssm.\n'
