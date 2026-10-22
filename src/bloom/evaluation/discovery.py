"""Discover immutable measured exports using their HDF5 identity metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

DATASET_ALIASES = {
    "big": "big",
    "prenatal": "big",
    "big_dataset": "big",
    "big_prenatal": "big",
    "big_prenatal_scalea": "big",
    "small": "small",
    "cortex": "small",
    "small_dataset": "small",
    "small_visual_cortex": "small",
    "sted": "sted",
    "sted_dataset": "sted",
    "sted-ec": "sted",
    "sted_ec": "sted",
    "sted_ec_new": "sted",
}


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def add_recursive_exports(
    config: dict[str, Any],
    roots: list[str],
    audit: list[dict[str, Any]],
    *,
    normalization: str | None = None,
    split: str | None = None,
) -> None:
    """Add new public exporter trees without inferring identity from folders.

    Existing numerical-registry rules are retained: measured forward v6 BLOOM
    and the 22 forecasting baselines. Bidirectional v7 displays remain separate.
    """
    import h5py
    from .metrics.config import inspect_header
    from .metrics.registry import canonical_identity

    seen = {str(Path(row["path"]).expanduser().resolve()) for row in config.get("raw_sources", [])}
    for value in roots:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("measured_forward.h5")):
            path = path.resolve()
            if str(path) in seen:
                continue
            seen.add(str(path))
            try:
                with h5py.File(path, "r") as handle:
                    attrs = {str(k): _text(v) for k, v in handle.attrs.items()}
                if attrs.get("schema_version") != "v4.canonical_prediction.1":
                    raise ValueError("Not a canonical measured prediction H5")
                if attrs.get("temporal_direction") != "forward":
                    raise ValueError(
                        "Only explicitly forward measured predictions enter this benchmark"
                    )
                if attrs.get("model_version") == "v7":
                    audit.append(
                        {
                            "path": str(path),
                            "status": "excluded_bidirectional",
                            "reason": "v7 is retained for biological displays, outside the v6/22-baseline numerical registry",
                        }
                    )
                    continue
                dataset = DATASET_ALIASES.get(attrs.get("dataset_id", "").casefold())
                if dataset is None:
                    raise ValueError("Unresolved dataset_id in canonical H5 metadata")
                panel = attrs.get("gene_panel", "").casefold()
                panel = "hvg" if panel in {"200", "hvg", "hvg200", "hvg_200"} else panel
                family = attrs.get("model_family", "").upper()
                if family == "FM":
                    if attrs.get("model_version") != "v6":
                        raise ValueError(
                            "FM benchmark source must explicitly attest model_version=v6"
                        )
                    model = "v6_author" if panel == "author" else "v6_200_new"
                elif family == "TRANS":
                    model = attrs.get("model_name", "")
                    if not model:
                        raise ValueError("Forecasting export lacks model_name metadata")
                else:
                    raise ValueError("Unresolved model_family in canonical H5 metadata")
                dataset, panel, model, family = canonical_identity(dataset, panel, model, family)
                source = {
                    "path": str(path),
                    "run_root": str(path.parent.parent),
                    "dataset": dataset,
                    "panel": panel,
                    "model": model,
                    "family": family,
                }
                if normalization:
                    source["normalization"] = normalization
                if split:
                    source["split"] = split
                source = inspect_header(source)
                config.setdefault("raw_sources", []).append(source)
                audit.append(
                    {
                        "path": str(path),
                        "status": "header_ready_recursive",
                        "dataset": dataset,
                        "panel": panel,
                        "model": model,
                        "identity_source": "canonical H5 attributes, validated by the preserved metrics header contract",
                    }
                )
            except (ValueError, KeyError, OSError) as error:
                audit.append(
                    {"path": str(path), "status": "blocked_header_recursive", "reason": str(error)}
                )
