"""Forecasting baseline registry and adapters for single-cell expression series.

The benchmark uses the models supplied by THUML's Time-Series-Library. This
module centralizes their import paths, the paper's short-sequence configuration,
and the optional cell-type output adapter. The upstream architectures retain
their own implementations and licenses; they were not included in the source
archive supplied for this release.

Use ``repo_dir`` or ``BLOOM_TSL_ROOT`` to select an external Time-Series-Library
checkout. Importing this module does not parse training arguments or load data.
"""

from __future__ import annotations

import copy
import importlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import nn

TRANS_MODEL_REGISTRY = {
    "Autoformer": ("models.Autoformer", "Model"),
    "Chronos": ("models.Chronos", "Model"),
    "Chronos2": ("models.Chronos2", "Model"),
    "Crossformer": ("models.Crossformer", "Model"),
    "DLinear": ("models.DLinear", "Model"),
    "FEDformer": ("models.FEDformer", "Model"),
    "FiLM": ("models.FiLM", "Model"),
    "FreTS": ("models.FreTS", "Model"),
    "Informer": ("models.Informer", "Model"),
    "LightTS": ("models.LightTS", "Model"),
    "MICN": ("models.MICN", "Model"),
    "Nonstationary_Transformer": ("models.Nonstationary_Transformer", "Model"),
    "PAttn": ("models.PAttn", "Model"),
    "PatchTST": ("models.PatchTST", "Model"),
    "Reformer": ("models.Reformer", "Model"),
    "SCINet": ("models.SCINet", "Model"),
    "SegRNN": ("models.SegRNN", "Model"),
    "Sundial": ("models.Sundial", "Model"),
    "TSMixer": ("models.TSMixer", "Model"),
    "TiDE": ("models.TiDE", "Model"),
    "TiRex": ("models.TiRex", "Model"),
    "TimeFilter": ("models.TimeFilter", "Model"),
    "TimeMoE": ("models.TimeMoE", "Model"),
    "TimeXer": ("models.TimeXer", "Model"),
    "TimesNet": ("models.TimesNet", "Model"),
    "Transformer": ("models.Transformer", "Model"),
    "WPMixer": ("models.WPMixer", "Model"),
    "iTransformer": ("models.iTransformer", "Model"),
    "Mamba": ("models.Mamba", "Model"),
}

# The complete source registry above also retains historical experiments so
# existing native run configurations remain readable. These are the 22 models
# in the paper's numerical benchmark, verified against its exported tables.
PAPER_BASELINES = (
    "Autoformer",
    "Crossformer",
    "DLinear",
    "FEDformer",
    "FiLM",
    "FreTS",
    "Informer",
    "LightTS",
    "Mamba",
    "Nonstationary_Transformer",
    "PAttn",
    "PatchTST",
    "Reformer",
    "SegRNN",
    "TSMixer",
    "TiDE",
    "TimeFilter",
    "TimeXer",
    "TimesNet",
    "Transformer",
    "WPMixer",
    "iTransformer",
)
BASELINE_REGISTRY = {name: TRANS_MODEL_REGISTRY[name] for name in PAPER_BASELINES}


def list_models() -> tuple[str, ...]:
    """Return the 22 forecasting baselines included in the paper."""
    return PAPER_BASELINES


TRANS_THU_DEFAULTS = dict(
    task_name="long_term_forecast",
    is_training=True,
    model_id="gene",
    features="M",
    target="0",
    freq="h",
    inverse=False,
    seq_len=None,
    label_len=None,
    pred_len=None,
    enc_in=None,
    dec_in=None,
    c_out=None,
    d_model=64,
    n_heads=4,
    e_layers=1,
    d_layers=1,
    d_ff=32,
    factor=3,
    distil=True,
    embed="timeF",
    dropout=0.1,
    attn="prob",
    activation="gelu",
    output_attention=False,
    individual=False,
    mix=True,
    root_path="./",
    data_path="none",
    num_workers=4,
    moving_avg=25,
    top_k=5,
    num_kernels=6,
)


class CellTypeConditionWrapper(nn.Module):

    def __init__(
        self, base: nn.Module, *, n_celltypes: int, n_genes: int, init_std: float, scale: float
    ):
        super().__init__()
        self.base = base
        self.scale = float(scale)
        self.emb = nn.Embedding(int(n_celltypes), int(n_genes))
        if float(init_std) <= 0:
            nn.init.zeros_(self.emb.weight)
        else:
            nn.init.normal_(self.emb.weight, mean=0.0, std=float(init_std))

    def forward(self, enc, enc_mark, dec, dec_mark, *, enc_types=None, dec_types=None):
        out = self.base(enc, enc_mark, dec, dec_mark)
        if enc_types is None or out is None or (not torch.is_tensor(out)):
            return out
        last_type = enc_types[:, -1].to(self.emb.weight.device)
        bias = self.emb(last_type) * self.scale
        bias = bias.to(out.device)
        if out.ndim == 3:
            prefix = out[:, :-1, :]
            last = out[:, -1:, :] + bias.unsqueeze(1)
            return torch.cat([prefix, last], dim=1)
        if out.ndim == 2:
            return out + bias
        return out


def configure_time_series_library(repo_dir: str | Path | None = None) -> Path | None:
    """Make a user-provided upstream checkout available to its absolute imports.

    Time-Series-Library imports its own ``models`` and ``layers`` packages.
    A checkout is therefore a runtime dependency, rather than a Python model
    copied into this release. Existing modules from another checkout are not
    silently replaced.
    """
    selected = repo_dir or os.environ.get("BLOOM_TSL_ROOT")
    if selected is None:
        return None
    path = Path(selected).expanduser().resolve()
    if not (path / "models").is_dir() or not (path / "layers").is_dir():
        raise FileNotFoundError(
            f"{path} must contain the upstream models/ and layers/ directories."
        )
    for package_name in ("models", "layers"):
        cached = sys.modules.get(package_name)
        origins = getattr(cached, "__path__", ()) if cached is not None else ()
        if origins and not any(Path(origin).resolve().is_relative_to(path) for origin in origins):
            raise ImportError(
                f"An unrelated {package_name!r} package is already imported. "
                "Start a fresh process with the intended Time-Series-Library checkout."
            )
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
    importlib.invalidate_caches()
    return path


def canonical_model_name(model_name: str) -> str:
    """Resolve source names and the readable Nonstationary Transformer label."""
    normalized = model_name.strip().replace(" ", "_").casefold()
    for name in TRANS_MODEL_REGISTRY:
        if name.casefold() == normalized:
            return name
    raise ValueError(
        f"Unknown forecasting model {model_name!r}. "
        f"Available names: {', '.join(TRANS_MODEL_REGISTRY)}"
    )


def build_transformer_config(
    model_name: str,
    *,
    n_genes: int,
    seq_len: int = 8,
    label_len: int = 4,
    pred_len: int = 1,
    d_model: int = 64,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
    use_amp: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
) -> SimpleNamespace:
    """Return the v5 short-series configuration used by the baseline adapter.

    This preserves the supplied builder's model-specific settings, including
    its effective ``d_ff=16`` override. ``config_overrides`` is applied last for
    deliberate new experiments; it is not used by the default benchmark recipe.
    Native v6/v7 runtime configurations remain in their respective trainers.
    """
    name = canonical_model_name(model_name)
    if min(n_genes, seq_len, pred_len, d_model, batch_size) <= 0:
        raise ValueError(
            "Gene count, sequence length, horizon, width and batch size must be positive."
        )
    if not 0 <= label_len <= seq_len:
        raise ValueError("label_len must be between zero and seq_len.")
    hp = copy.deepcopy(TRANS_THU_DEFAULTS)
    hp.update(
        seq_len=int(seq_len),
        label_len=int(label_len),
        pred_len=int(pred_len),
        enc_in=int(n_genes),
        dec_in=int(n_genes),
        c_out=int(n_genes),
        d_model=int(d_model),
    )
    resolved_device = torch.device(device)
    hp["device"] = resolved_device if name == "WPMixer" else str(resolved_device)
    hp.setdefault("expand", 2)
    hp.setdefault("dt_rank", math.ceil(int(d_model) / 16))
    hp.setdefault("d_conv", 4)
    hp["d_ff"] = 16
    if name == "FreTS":
        hp.setdefault("channel_independence", 1)
    if name == "Nonstationary_Transformer":
        hp.setdefault("p_hidden_dims", [64])
        hp.setdefault("p_hidden_layers", 1)
    if name == "SegRNN":
        hp.setdefault("seg_len", 1)
    if name in {"PAttn", "PatchTST", "TimeXer", "WPMixer", "TimeFilter"}:
        hp["patch_len"] = int(max(1, min(int(seq_len), 16)))
        hp.setdefault("stride", 1)
    if name == "TimeFilter":
        hp.setdefault("alpha", 0.1)
        hp.setdefault("top_p", 0.5)
        hp.setdefault("pos", 1)
    if name == "TimeXer":
        hp.setdefault("use_norm", 1)
        hp.setdefault("patch_len", 1)
    if name == "TimesNet":
        hp["top_k"] = 1
    if name == "WPMixer":
        hp["batch_size"] = int(batch_size)
        hp.setdefault("patch_len", 1)
        hp.setdefault("use_amp", bool(use_amp))
    if config_overrides:
        hp.update(config_overrides)
    return SimpleNamespace(**hp)


def build_transformer(
    model_name: str,
    *,
    n_genes: int,
    seq_len: int = 8,
    label_len: int = 4,
    pred_len: int = 1,
    d_model: int = 64,
    device: str | torch.device = "cpu",
    repo_dir: str | Path | None = None,
    batch_size: int = 256,
    use_amp: bool = False,
    config_overrides: Mapping[str, Any] | None = None,
    condition_cell_types: bool = False,
    n_celltypes: int | None = None,
    condition_init_std: float = 0.0,
    condition_scale: float = 1.0,
) -> nn.Module:
    """Instantiate one upstream forecasting model for cell-by-gene sequences.

    The forward protocol is the Time-Series-Library four-tensor call:
    ``model(enc, enc_mark, dec, dec_mark)``. Optional cell-type conditioning
    preserves the source wrapper's ``base`` and ``emb`` checkpoint key prefixes.
    """
    name = canonical_model_name(model_name)
    path = configure_time_series_library(repo_dir)
    module_name, class_name = TRANS_MODEL_REGISTRY[name]
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"Cannot import {module_name}. Supply a compatible Time-Series-Library "
            "checkout through repo_dir/BLOOM_TSL_ROOT and install the dependencies "
            f"required by {name}. The upstream model source is not bundled."
        ) from error
    if path is not None:
        origin = getattr(module, "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(path):
            raise ImportError(f"{module_name} was resolved outside the selected checkout {path}.")
    config = build_transformer_config(
        name,
        n_genes=n_genes,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        d_model=d_model,
        device=device,
        batch_size=batch_size,
        use_amp=use_amp,
        config_overrides=config_overrides,
    )
    model = getattr(module, class_name)(config).to(torch.device(device))
    if condition_cell_types:
        if n_celltypes is None or n_celltypes <= 0:
            raise ValueError("A positive n_celltypes is required for cell-type conditioning.")
        model = CellTypeConditionWrapper(
            model,
            n_celltypes=n_celltypes,
            n_genes=n_genes,
            init_std=condition_init_std,
            scale=condition_scale,
        ).to(torch.device(device))
    return model
