"""Portable biological plot helpers; expression matrices stay memory mapped.

All model panels use the biological renderer's common limits. Missing native
panel genes are displayed as N/A by the renderer, never imputed as zero.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import inspect
import textwrap
import json
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

PALETTE = [
    "#387F79",
    "#456F9A",
    "#946AA2",
    "#C67C58",
    "#A68A3E",
    "#5C9172",
    "#9C6176",
    "#627A8C",
    "#7A734C",
    "#B36B4E",
    "#588CA0",
    "#80679E",
]
BACKGROUND = "#969FA7"
EXPRESSION_CMAP = LinearSegmentedColormap.from_list(
    "bloom_expression", [BACKGROUND, "#E1D5B4", "#C48756", "#925740", "#613E62"]
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "bloom_difference", ["#3C718E", "#FAF5EC", "#A35568"]
)
FM_ORDER = ["v6_200_new", "v6_author", "v7_200", "v7_author"]
TRANS_ORDER = [
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
]
DEFAULT_ALIASES = {
    "v6_200_new": "unidirectional BLOOM",
    "v6_author": "unidirectional BLOOM",
    "v7_200": "bidirectional BLOOM",
    "v7_author": "bidirectional BLOOM",
    "Nonstationary_Transformer": "Nonstationary Transformer",
    "GT_hvg": "GT · HVG",
    "GT_author": "GT · author",
}


def style():
    """Dark muted colors with legible grey cells and clean export typography."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#7B858C",
            "axes.labelcolor": "#293D46",
            "text.color": "#293D46",
            "xtick.color": "#53636D",
            "ytick.color": "#53636D",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 150,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "unnamed"


def model_label(source, options=None):
    options = options or {}
    aliases = dict(DEFAULT_ALIASES)
    aliases.update(options.get("aliases", {}))
    model = str(source.get("model", source.get("source_id", "unknown")))
    label = aliases.get(model, model)
    if source.get("role") == "REFERENCE" and model not in aliases:
        label = "Observed boundary · " + str(source.get("panel", "unknown"))
    if source.get("role") == "FM" and options.get("label_native_panel", True):
        panel = _panel(source.get("panel", "unknown"))
        if not str(label).lower().endswith(("_hvg", "_author", " · hvg", " · author")):
            label += " · " + ("HVG" if panel == "hvg" else panel)
    if source.get("role") == "TRANS" and options.get("label_trans_panel", False):
        panel = _panel(source.get("panel", "unknown"))
        label += " · " + ("HVG" if panel == "hvg" else panel)
    return label


def _panel(value):
    return (
        "hvg"
        if str(value).strip().lower() in ("200", "hvg", "hvg200")
        else str(value).strip().lower()
    )


def source_panel(source):
    """Resolve the native assay without using a display alias or gene overlap.

    Canonical FM/GT identities provide assay evidence. Conflicting explicit
    metadata invalidates that source rather than transferring it to an assay.
    Generic TRANS names, paths and a partially overlapping gene list do not
    establish a native panel.
    """
    explicit = {
        _panel(source[key])
        for key in ("panel", "native_panel", "gene_panel")
        if source.get(key) is not None and _panel(source[key]) in ("hvg", "author")
    }
    if len(explicit) > 1:
        return None
    model = str(source.get("model", ""))
    identity = {
        "v6_200_new": "hvg",
        "v6_200_old": "hvg",
        "v7_200": "hvg",
        "v6_author": "author",
        "v7_author": "author",
        "GT_hvg": "hvg",
        "GT_200": "hvg",
        "GT_author": "author",
        "Observed_boundary_hvg": "hvg",
        "Observed_boundary_200": "hvg",
        "Observed_boundary_author": "author",
    }.get(model)
    if identity and explicit and identity not in explicit:
        return None
    return identity or next(iter(explicit), None)


def cohort_panel(cohort, options=None):
    """Choose one native assay; a single known assay supports old sparse cohorts."""
    options = options or {}
    requested = _panel(options.get("panel", cohort.get("panel", "unknown")))
    if requested in ("hvg", "author"):
        return requested
    known = {source_panel(s) for s in cohort.get("sources", [])}
    known.discard(None)
    if len(known) == 1:
        return next(iter(known))
    return "hvg"


def native_sources(cohort, options=None, sources=None):
    """Return sources from precisely one native assay, including hidden GT.

    An unresolved panel is not a shared reference. An explicitly verified
    parent-assay binding may be inherited only when the cohort declares one
    native panel; this requires provenance flags, not a plotting preference.
    """
    options = options or {}
    requested = cohort_panel(cohort, options)
    candidates = list(cohort.get("sources", []) if sources is None else sources)
    inherited = (
        bool(cohort.get("native_panel_verified")) and _panel(cohort.get("panel")) == requested
    )
    return [
        dict(s)
        for s in candidates
        if s.get("model") != "v6_200_old"
        and (
            source_panel(s) == requested
            or (
                source_panel(s) is None
                and inherited
                and not any(
                    _panel(s.get(k)) in ("hvg", "author")
                    for k in ("panel", "native_panel", "gene_panel")
                )
                and bool(s.get("panel_inherited_from_verified_parent"))
            )
        )
    ]


def _chosen_models(value, dataset, panel):
    """Resolve canonical per-assay selection keys before dataset fallbacks."""
    if isinstance(value, dict):
        value = value.get(f"{dataset}_{panel}", value.get(dataset, value.get("default", [])))
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    return set(value or [])


def _selection_labels(chosen, options):
    aliases = dict(DEFAULT_ALIASES)
    aliases.update(options.get("aliases", {}))
    canonical = set(chosen) | {model for model, label in aliases.items() if label in chosen}
    labels = canonical | {aliases.get(str(m), str(m)) for m in canonical}
    # A custom FM alias may have an assay suffix added for a figure title.
    # Keep canonical identities available for routing and title forms for the
    # comparison helper, whose inputs are already panel-filtered callbacks.
    for model in canonical:
        if model in FM_ORDER:
            labels.add(
                model_label(
                    {"model": model, "role": "FM", "panel": source_panel({"model": model})}, options
                )
            )
        elif options.get("label_trans_panel"):
            labels.add(
                model_label(
                    {"model": model, "role": "TRANS", "panel": options.get("panel", "hvg")}, options
                )
            )
    return labels


def available_models(cohort, options=None):
    """Pick GT, FMs and TRANS trained on the same native gene assay.

    A full gallery can include sources with partial gene coverage. The source's
    actual native axis remains available to each biological renderer.
    """
    options = options or {}
    sources = [x for x in native_sources(cohort, options) if x.get("model") != "v6_200_old"]
    sources = [x for x in sources if x.get("role") in ("GT", "REFERENCE", "FM", "TRANS", "legacy")]
    requested_panel = cohort_panel(cohort, options)
    sources.sort(
        key=lambda x: (
            x.get("role") != "GT",
            x.get("role") != "FM",
            FM_ORDER.index(x["model"]) if x.get("model") in FM_ORDER else 100,
            str(x.get("model", "")),
            str(x.get("source_id", "")),
        )
    )
    gt = [x for x in sources if x.get("role") == "GT"]
    if gt and options.get("gt_panels") != "both":
        gt.sort(key=lambda x: (_panel(x.get("panel")) != requested_panel, -len(x.get("genes", []))))
        gt = gt[:1]
    references = [x for x in sources if x.get("role") == "REFERENCE"]
    if references and options.get("gt_panels") != "both":
        references.sort(
            key=lambda x: (_panel(x.get("panel")) != requested_panel, -len(x.get("genes", [])))
        )
        references = references[:1]
    fms = OrderedDict()
    for x in sources:
        if x.get("role") == "FM":
            fms.setdefault(x.get("model"), x)
    trans = OrderedDict()
    for x in sorted(
        [x for x in sources if x.get("role") == "TRANS"], key=lambda s: str(s.get("source_id", ""))
    ):
        trans.setdefault(x.get("model"), x)
    legacy = [x for x in sources if x.get("role") == "legacy"]
    out = gt + references + list(fms.values()) + list(trans.values()) + legacy
    if not out:
        out = sources
    if options.get("model_policy") == "selected":
        chosen = _chosen_models(
            options.get("selected_models", []), cohort.get("dataset", ""), requested_panel
        )
        chosen = _selection_labels(chosen, options)
        selected_fms = options.get("selected_fms")
        if selected_fms is not None:
            selected_fms = _selection_labels(
                _chosen_models(selected_fms, cohort.get("dataset", ""), requested_panel), options
            )

        def include(x):
            if x.get("role") in ("GT", "REFERENCE", "legacy"):
                return True
            if x.get("role") == "FM":
                return (
                    selected_fms is None
                    or x.get("model") in selected_fms
                    or model_label(x, options) in selected_fms
                )
            return x.get("model") in chosen or model_label(x, options) in chosen

        out = [x for x in out if include(x)]
    return out


def _read_table(path):
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if path.suffix in (".tsv", ".txt") else ",")


def load_artifact(source):
    """Read one small artifact, memory-map expression, cache up to 40 sources."""
    directory = Path(source.get("artifact_dir") or source.get("path", "."))
    if directory.is_file():
        directory = directory.parent
    stat = (directory / "metadata.json").stat() if (directory / "metadata.json").exists() else None
    cache_key = (str(directory.resolve()), stat.st_mtime_ns if stat else None)
    if cache_key in _ARTIFACTS:
        _ARTIFACTS.move_to_end(cache_key)
        return _ARTIFACTS[cache_key]
    meta = dict(source)
    if (directory / "metadata.json").exists():
        meta.update(json.loads((directory / "metadata.json").read_text()))

    def table(name):
        for suffix in (".parquet", ".csv", ".tsv"):
            path = directory / (name + suffix)
            if path.exists():
                return _read_table(path)
        return pd.DataFrame()

    expression_path = directory / "expression.npy"
    expression = np.load(expression_path, mmap_mode="r") if expression_path.exists() else None
    genes = list(meta.get("genes", source.get("genes", [])))
    artifact = {
        "meta": meta,
        "source": source,
        "genes": genes,
        "cells": table("cells"),
        "expression": expression,
        "stage": table("stage_gene"),
        "state": table("state_gene"),
        "perturbation": table("perturbation"),
    }
    _ARTIFACTS[cache_key] = artifact
    while len(_ARTIFACTS) > 40:
        _ARTIFACTS.popitem(last=False)
    return artifact


_ARTIFACTS = OrderedDict()


def save(fig, base, options=None):
    """Save legend variants of one real figure and restore original visibility."""
    options = options or {}
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    formats = options.get("formats", ["png"])
    if isinstance(formats, str):
        formats = formats.split(",")
    variants = options.get("variants", ["legend", "no_legend"])
    if isinstance(variants, str):
        variants = variants.split(",")
    legends = [a.get_legend() for a in fig.axes if a.get_legend() is not None] + list(fig.legends)
    coloraxes = [
        a for a in fig.axes if a.get_label() == "<colorbar>" or getattr(a, "_bloom_colorbar", False)
    ]
    visibility = [(a, a.get_visible()) for a in legends + coloraxes]
    outputs = []
    try:
        for variant in variants:
            visible = variant not in ("no_legend", "without_legend")
            for artist, original in visibility:
                artist.set_visible(original and visible)
            for fmt in formats:
                fmt = str(fmt).lower().lstrip(".")
                if fmt not in ("png", "pdf", "svg"):
                    raise ValueError(f"Unsupported plot format: {fmt}")
                path = Path(str(base) + "__" + slug(variant) + "." + fmt)
                fig.savefig(
                    path, dpi=int(options.get("dpi", 150)), bbox_inches="tight", pad_inches=0.12
                )
                outputs.append(path)
    finally:
        for artist, original in visibility:
            artist.set_visible(original)
    return outputs


def _draw(callback, ax, legend):
    try:
        signature = inspect.signature(callback)
    except (ValueError, TypeError):
        callback(ax, legend)
        return
    if "legend" in signature.parameters:
        callback(ax, legend=legend)
    elif len(signature.parameters) >= 2:
        callback(ax, legend)
    else:
        callback(ax)


def mosaic(panels, base, options=None, title=""):
    """Draw the supplied model callbacks as a true comparison mosaic."""
    options = options or {}
    if not panels:
        return []
    style()
    ncols = min(int(options.get("mosaic_columns", 5)), len(panels))
    ncols = max(1, ncols)
    nrows = math.ceil(len(panels) / ncols)
    width = float(options.get("panel_width", 3.15))
    height = float(options.get("panel_height", 2.75))
    wrapped_title = (
        "\n".join(textwrap.wrap(str(title), max(26, int(width * ncols * 8.5)))) if title else ""
    )
    title_height = 0.18 + 0.22 * len(wrapped_title.splitlines()) if title else 0
    total_height = height * nrows + title_height + 0.18 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, total_height), squeeze=False)
    try:
        for ax, (label, callback) in zip(axes.ravel(), panels):
            _draw(callback, ax, True)
            ax.set_title("\n".join(textwrap.wrap(str(label), 26, break_on_hyphens=False)), pad=8)
        for ax in axes.ravel()[len(panels) :]:
            ax.set_visible(False)
        if title:
            fig.suptitle(wrapped_title, fontsize=12, y=0.995)
        # Reserve the wrapped heading once. tight_layout otherwise also
        # reserves the suptitle automatically and compresses heatmap rows.
        if title:
            fig._suptitle.set_in_layout(False)
        fig.tight_layout(rect=(0, 0, 1, 1 - title_height / total_height if title else 1))
        if title:
            fig._suptitle.set_in_layout(True)
        return save(fig, base, options)
    finally:
        plt.close(fig)


def compare_mosaics(panels, base, options=None, title=""):
    """All-model, GT+FM, GT+FM+one TRANS and optional individual exports.

    Inputs are (label, role, callback). The all-model layout always retains all
    actual models; GT+FM+TRANS is a compact representative comparison whose
    selected TRANS label is printed and may be configured explicitly.
    """
    options = options or {}
    panels = list(panels)
    if not panels:
        return []
    layouts = options.get("layouts", ["all", "gt_fm", "gt_fm_trans"])
    if isinstance(layouts, str):
        layouts = layouts.split(",")
    outputs = []
    rendered_subsets = {}
    layout_map = {}

    def emit(layout, sub, suffix):
        if not sub:
            layout_map[layout] = {"status": "empty", "files": []}
            return
        # Callbacks identify actual source data when custom aliases coincide.
        # The same native source subset has identical axes and content across
        # these comparison layout labels; draw it only once.
        identity = tuple((str(p[0]), str(p[1]), id(p[2])) for p in sub)
        if identity in rendered_subsets:
            original, paths = rendered_subsets[identity]
            layout_map[layout] = {
                "status": "alias",
                "canonical_layout": original,
                "files": [Path(p).name for p in paths],
            }
        else:
            paths = mosaic([(p[0], p[2]) for p in sub], Path(str(base) + suffix), options, title)
            outputs.extend(paths)
            rendered_subsets[identity] = (layout, paths)
            layout_map[layout] = {"status": "rendered", "files": [Path(p).name for p in paths]}
        layout_map[layout]["models"] = [{"label": str(p[0]), "role": str(p[1])} for p in sub]

    for layout in layouts:
        if layout in ("all", "all_models"):
            sub = panels
        elif layout == "gt_fm":
            sub = [p for p in panels if p[1] in ("GT", "REFERENCE", "FM", "legacy")]
        elif layout == "gt_fm_trans":
            sub = [p for p in panels if p[1] in ("GT", "REFERENCE", "FM", "legacy")]
            trans = [p for p in panels if p[1] == "TRANS"]
            preferred = options.get("compact_trans")
            if preferred:
                chosen = [p for p in trans if p[0] == preferred]
                trans = chosen or trans
            sub += trans[:1]
        elif layout in ("individual", "single"):
            for label, role, callback in panels:
                emit(
                    "individual:" + str(label),
                    [(label, role, callback)],
                    "__individual_" + slug(label),
                )
            continue
        else:
            raise ValueError(f"Unknown comparison layout: {layout}")
        emit(layout, sub, "__" + slug(layout))
    if options.get("model_policy") == "both" and options.get("selected_models"):
        dataset = str(options.get("dataset", ""))
        panel = _panel(options.get("panel", "hvg"))
        labels = _selection_labels(
            _chosen_models(options.get("selected_models", []), dataset, panel), options
        )
        selected_fms = options.get("selected_fms")
        fm_labels = (
            _selection_labels(_chosen_models(selected_fms, dataset, panel), options)
            if selected_fms is not None
            else None
        )
        selected = [
            p
            for p in panels
            if (
                p[1] in ("GT", "REFERENCE", "legacy")
                or (p[1] == "FM" and (fm_labels is None or p[0] in fm_labels))
                or (p[1] != "FM" and p[0] in labels)
            )
        ]
        if any(p[1] not in ("GT", "REFERENCE") for p in selected):
            emit("selected_models", selected, "__selected_models")
    if any(record["status"] == "alias" for record in layout_map.values()):
        path = Path(str(base) + "__COMPARISON_LAYOUT_MAP.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dataset": options.get("dataset"),
                    "native_panel": options.get("panel"),
                    "layout_mapping": layout_map,
                },
                indent=2,
            )
            + "\n"
        )
    return outputs


def missing_panel(ax, message="Gene absent from native panel"):
    ax.text(
        0.5,
        0.5,
        "N/A\n" + str(message),
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#737F87",
        fontsize=9,
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])


style()
