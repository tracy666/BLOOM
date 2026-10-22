"""Cheap, explicit discovery of saved expression outputs; never performs inference."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

FM_MODELS = ("v6_200_new", "v6_author", "v7_200", "v7_author")
TRANS_MODELS = (
    "Autoformer",
    "Crossformer",
    "DLinear",
    "FEDformer",
    "FiLM",
    "FreTS",
    "Informer",
    "iTransformer",
    "LightTS",
    "Mamba",
    "Nonstationary_Transformer",
    "PatchTST",
    "PAttn",
    "Reformer",
    "SegRNN",
    "TiDE",
    "TimeFilter",
    "TimesNet",
    "TimeXer",
    "Transformer",
    "TSMixer",
    "WPMixer",
)
SCENARIO_SCHEMAS = {
    "v5_dense_developmental_time_h5_v1": "interpolation",
    "v5_autoregressive_future_h5_v1": "future",
    "v4_autoregressive_history_h5_v1": "history",
    "fm_compact_history_perturbation_v1": "perturbation",
}


def _logical_absolute(path: str | Path) -> Path:
    """Normalize user syntax without erasing a symlink's modern slot name."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _iter_saved_files(root: Path):
    """Follow intentional exported-run links, while pruning ancestor cycles."""
    ignored = {"_history", "quarantine", ".venv", "__MACOSX", ".git", "__pycache__"}
    for directory, names, files in os.walk(root, followlinks=True):
        current = Path(directory)
        kept = []
        for name in names:
            if name in ignored:
                continue
            child = current / name
            if child.is_symlink():
                try:
                    target = child.stat()
                    inode = (target.st_dev, target.st_ino)
                    ancestors = [current, *current.parents]
                    if any(
                        (value.st_dev, value.st_ino) == inode
                        for value in (parent.stat() for parent in ancestors)
                    ):
                        continue
                except (OSError, RuntimeError):
                    continue
            kept.append(name)
        names[:] = kept
        for name in files:
            if Path(name).suffix.lower() in {".h5", ".hdf5", ".h5ad"}:
                yield current / name


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(x) for x in value]
    return value


def dataset_name(value: Any) -> str:
    name = str(value).lower()
    if name == "big" or "prenatal" in name:
        return "big"
    if name == "small" or "cortex" in name:
        return "small"
    if "sted" in name:
        return "sted"
    return "unknown"


def panel_name(value: Any) -> str:
    text = str(value).lower()
    return (
        "hvg"
        if text in {"hvg", "200", "200.0", "hvg200"}
        else "author" if text == "author" else "unknown"
    )


def _modern_identity(path: Path) -> dict[str, Any] | None:
    # Only exact supported slot layouts confer modern canonical identities.
    parts = path.parts
    for i in range(len(parts) - 2):
        dataset, family, slot = parts[i : i + 3]
        if dataset not in {"big", "small", "sted"} or family not in {"FM", "TRANS"}:
            continue
        if family == "FM":
            if slot == "v6_200_old":
                return {"excluded": True}
            if slot not in FM_MODELS:
                return None
            model, panel = slot, "author" if "author" in slot else "hvg"
        else:
            architecture, _, raw_panel = slot.rpartition("_")
            if architecture not in TRANS_MODELS or raw_panel not in {"200", "author"}:
                return {"excluded": True}
            model, panel = architecture, "hvg" if raw_panel == "200" else "author"
        return {
            "dataset": dataset,
            "role": family,
            "model": model,
            "panel": panel,
            "run_root": str(Path(*parts[: i + 3])),
        }
    return None


def _run_metadata(path: Path) -> tuple[dict[str, Any], str]:
    for directory in [path.parent, *path.parents]:
        provenance = directory / "00_provenance"
        spec_path = provenance / "run_spec.json"
        if spec_path.is_file():
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            resolved = provenance / "resolved_dataset_config.json"
            if resolved.is_file():
                try:
                    spec["_dataset_config"] = json.loads(resolved.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
            spec["_dataset_config_path"] = str(resolved) if resolved.is_file() else str(spec_path)
            return spec, str(directory)
    return {}, ""


def inspect_source(
    path: str | Path, explicit: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Read headers and row vectors only. Unrelated or excluded files return None."""
    import h5py

    path = _logical_absolute(path)
    explicit = dict(explicit or {})
    modern = _modern_identity(path)
    if modern and modern.get("excluded"):
        return None
    with h5py.File(path, "r") as handle:
        is_h5ad = str(_plain(handle.attrs.get("encoding-type", ""))) == "anndata" or (
            "obs" in handle and "var" in handle and "X" in handle
        )
        attrs = {str(k): _plain(v) for k, v in handle.attrs.items()}
        if is_h5ad:
            # Prepared H5AD is an explicit input path or self-labelled artifact;
            # a filename such as FM_projected.h5ad is insufficient identity.
            if not explicit.get("model") and not attrs.get("model_name"):
                return None
            import anndata as ad

            backed = ad.read_h5ad(path, backed="r")
            try:
                n_cells, n_genes = backed.shape
            finally:
                backed.file.close()
            matrix_key, genes_key, time_key = "X", "var", "obs"
            scenario = explicit.get("scenario", "measured")
        else:
            schema = str(attrs.get("schema_version", ""))
            scenario = SCENARIO_SCHEMAS.get(schema)
            if scenario is None:
                if "prediction_expression" not in handle or "gene_names" not in handle:
                    return None
                scenario = (
                    "reverse"
                    if str(attrs.get("temporal_direction", "")).lower() == "reverse"
                    or path.name == "measured_reverse.h5"
                    else "measured"
                )
            matrix_key = (
                "baseline_mean"
                if scenario == "perturbation"
                else (
                    "prediction_expression" if scenario in {"measured", "reverse"} else "prediction"
                )
            )
            genes_key = "gene_names"
            if matrix_key not in handle or genes_key not in handle:
                raise ValueError(f"{scenario} missing {matrix_key} or {genes_key}")
            matrix = handle[matrix_key]
            if (
                matrix.ndim != 2
                or handle[genes_key].ndim != 1
                or matrix.shape[1] != len(handle[genes_key])
            ):
                raise ValueError("expression and gene axes are inconsistent")
            n_cells, n_genes = map(int, matrix.shape)
            if n_cells == 0 or n_genes == 0:
                raise ValueError("expression matrix is empty")
            time_key = (
                "target_times"
                if scenario == "perturbation"
                else (
                    "prediction_reporting_time"
                    if "prediction_reporting_time" in handle
                    else (
                        "prediction_time"
                        if scenario in {"measured", "reverse"}
                        else (
                            "prediction_actual_time"
                            if "prediction_actual_time" in handle
                            else "prediction_time"
                        )
                    )
                )
            )
            if time_key not in handle or handle[time_key].shape != (n_cells,):
                raise ValueError(f"missing or invalid row time vector {time_key}")
            if scenario == "perturbation":
                if "perturbed_mean" not in handle or "intervention_gene_index" not in handle:
                    raise ValueError("compact perturbation lacks edited mean or intervention axis")
                if handle["perturbed_mean"].shape != (
                    len(handle["intervention_gene_index"]),
                    n_cells,
                    n_genes,
                ):
                    raise ValueError("compact perturbation axes are inconsistent")
        spec, run_root = _run_metadata(path)
        identity = dict(modern or {})
        identity.update(
            {k: explicit[k] for k in ("dataset", "role", "model", "panel") if explicit.get(k)}
        )
        attr_dataset = dataset_name(attrs.get("dataset_id", "unknown"))
        attr_panel = panel_name(attrs.get("gene_panel", "unknown"))
        if (
            identity.get("dataset")
            and attr_dataset != "unknown"
            and identity["dataset"] != attr_dataset
        ):
            raise ValueError(
                f"dataset identity conflict: path/config={identity['dataset']}, H5={attr_dataset}"
            )
        if (
            identity.get("panel")
            and attr_panel != "unknown"
            and panel_name(identity["panel"]) != attr_panel
        ):
            raise ValueError(
                f"gene-panel identity conflict: path/config={identity['panel']}, H5={attr_panel}"
            )
        if not identity:
            # Legacy V6 pair metadata does not distinguish old/new trained models.
            # Preserve a truthful legacy model label, never call it v6_200_new.
            dataset = attr_dataset
            if dataset == "unknown":
                dataset = dataset_name(
                    spec.get("dataset_id", spec.get("dataset", {}).get("dataset_id", "unknown"))
                )
            if dataset == "unknown":
                dataset = dataset_name(str(path))
                if dataset != "unknown":
                    attrs["dataset_identity_basis"] = (
                        "legacy dataset name in saved source path; no modern FM model version inferred"
                    )
            family = str(attrs.get("model_family", "legacy"))
            named = str(attrs.get("model_name", ""))
            if family == "TRANS" and named in TRANS_MODELS:
                identity = {
                    "dataset": dataset,
                    "role": "TRANS",
                    "model": named,
                    "panel": attr_panel,
                }
            else:
                tag = str(attrs.get("model_tag") or attrs.get("model_version") or "unknown_model")
                identity = {
                    "dataset": dataset,
                    "role": "legacy",
                    "model": "legacy_" + tag,
                    "panel": attr_panel,
                }
        identity["panel"] = panel_name(identity.get("panel", attr_panel))
        if identity.get("role") == "FM" and identity.get("model") in FM_MODELS:
            version = str(attrs.get("model_version", ""))
            expected_version = "v7" if identity["model"].startswith("v7") else "v6"
            if version and version != expected_version:
                raise ValueError(
                    f"FM identity conflict: slot {identity['model']}, H5 version {version}"
                )
        family_attr = str(attrs.get("model_family", ""))
        if (
            identity.get("role") in {"FM", "TRANS"}
            and family_attr
            and family_attr.upper() != identity["role"]
        ):
            raise ValueError(
                f"model-family identity conflict: slot {identity['role']}, H5 {family_attr}"
            )
        architecture = str(attrs.get("model_name", ""))
        if (
            identity.get("role") == "TRANS"
            and architecture in TRANS_MODELS
            and identity.get("model") in TRANS_MODELS
            and architecture != identity["model"]
        ):
            raise ValueError(
                f"TRANS architecture identity conflict: slot {identity['model']}, H5 {architecture}"
            )
    source_id = hashlib.sha256(str(path).encode()).hexdigest()[:20]
    source = {
        **identity,
        "source_id": source_id,
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "scenario": scenario,
        "matrix_key": matrix_key,
        "genes_key": genes_key,
        "time_key": time_key,
        "n_cells": n_cells,
        "n_genes": n_genes,
        "time_kind": (
            "reporting"
            if "reporting" in time_key
            else (
                "ordinal"
                if scenario in {"measured", "reverse", "perturbation"}
                else "scenario_reporting"
            )
        ),
        "metadata": attrs,
        "format": "h5ad" if is_h5ad else "h5",
        "run_root": identity.get("run_root", run_root),
        "run_spec": spec,
    }
    source["dataset_config"] = spec.get("_dataset_config") or spec.get("dataset", {})
    source["dataset_config_path"] = spec.get("_dataset_config_path", "")
    priors = spec.get("priors", {}) if isinstance(spec.get("priors"), dict) else {}
    source["annotations"] = {
        key: priors[name]
        for key, name in (
            ("programs_path", "driver_tsv"),
            ("edges_path", "grn_tsv"),
            ("grn_path", "grn_tsv"),
            ("author_graph_path", "curated_graph_tsv"),
            ("graph_nodes_path", "curated_graph_nodes_path"),
            ("graph_cell_map_path", "curated_graph_cell_map_path"),
        )
        if priors.get(name)
    }
    source["dataset_config"] = {**source["dataset_config"], **source["annotations"]}
    # A dataset nickname or gene capitalization does not establish species.
    # These exact source-path phrases are provenance evidence supplied by V5.
    provenance_text = json.dumps(
        {
            "configuration": source["dataset_config"],
            "identity_attestation": attrs.get("identity_attestation", ""),
        },
        default=str,
    ).lower()
    if (
        not source["dataset_config"].get("species")
        and "mouse_prenatal_development" in provenance_text
    ):
        source["dataset_config"]["species"] = "mouse"
        source["dataset_config"][
            "species_provenance"
        ] = "supplied V5 source path explicitly names mouse_prenatal_development"
    source.update({k: v for k, v in explicit.items() if k not in source})
    return source


def discover(config: dict[str, Any]) -> dict[str, Any]:
    """Return {sources, errors, excluded}; errors are visible per file/root."""
    candidates: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    # Legacy evaluation folders are preserved by archive.py; their ablations
    # must not enter current four-FM/22-TRANS statistics by default.
    roots = list(config.get("v5_roots", []))
    if config.get("include_legacy_sources", False):
        roots.extend(config.get("legacy_roots", []))
    for root_value in roots:
        root = _logical_absolute(root_value)
        if not root.is_dir():
            errors.append(
                {"path": str(root), "error": "input root is missing or is not a directory"}
            )
            continue
        for path in _iter_saved_files(root):
            candidates.setdefault(str(_logical_absolute(path)), {})
    for source in config.get("sources", []):
        if not isinstance(source, dict) or not source.get("path"):
            errors.append({"path": "config.sources", "error": "explicit source requires a path"})
            continue
        candidates[str(_logical_absolute(source["path"]))] = source
    sources = []
    fingerprints = set()
    excluded = 0
    exclusions = []

    def order(item):
        modern = _modern_identity(Path(item[0]))
        return (
            not bool(item[1]),
            not bool(modern and not modern.get("excluded")),
            "01_predictions" not in item[0],
            item[0],
        )

    for path, explicit in sorted(candidates.items(), key=order):
        try:
            source = inspect_source(path, explicit)
            if source is None:
                excluded += 1
                continue
            # Numerical-guard views are analysis alternatives to the source
            # rollout, not independent model predictions. Select one explicitly
            # in config.sources when desired; never silently mix source and view.
            if source.get("metadata", {}).get("evaluation_view_schema") and not explicit:
                excluded += 1
                exclusions.append(
                    {
                        "path": path,
                        "reason": "numerical-guard evaluation view requires explicit config.sources selection",
                    }
                )
                continue
            stat = Path(path).stat()
            file_key = (stat.st_dev, stat.st_ino)
            # Hard-linked materialized scenarios and bridge copies are one input.
            identity_key = (
                source["dataset"],
                source["role"],
                source["model"],
                source["panel"],
                source["scenario"],
            )
            if file_key in fingerprints or identity_key in fingerprints:
                excluded += 1
                exclusions.append(
                    {
                        "path": path,
                        "reason": "duplicate physical source or native model/scenario input",
                    }
                )
                continue
            fingerprints.update((file_key, identity_key))
            sources.append(source)
        except Exception as exc:
            errors.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
    return {"sources": sources, "errors": errors, "excluded": excluded, "exclusions": exclusions}


discover_sources = discover
