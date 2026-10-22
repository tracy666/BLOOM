"""Reference-style radar artwork from independently scaled benchmark axes.

Every polygon uses the supplied per-model values. Sorting determines only the
paint order; it never changes scores or makes polygons artificially nested.
Raw scale annotations are inverted on lower-is-better axes. Source values and
paint order are included in the JSON sidecar for every exported figure.
"""

from __future__ import annotations

import colorsys
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

_DEFAULT_DATASETS = {
    "big": {"label": "Big prenatal", "fill": "#FBE3D6", "border": "#80340E", "text": "#80340E"},
    "small": {"label": "Small cortex", "fill": "#DDE9F7", "border": "#163E64", "text": "#163E64"},
    "sted": {"label": "STED-EC", "fill": "#C1F1C8", "border": "#0D3512", "text": "#0D3512"},
}


def _label(model: str) -> str:
    label = str(model).replace("Nonstationary_Transformer", "Nonstationary Transformer")
    if "bloom" in label.casefold():
        return "BLOOM"
    return label


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "radar"


def _number(value: Any) -> str:
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 10000 or abs(value) < 0.001:
        return f"{value:.2g}"
    if abs(value) >= 10:
        return f"{value:.3g}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _lookup(mapping: dict, model: str):
    if model in mapping:
        return mapping[model]
    clean = _label(model)
    if clean in mapping:
        return mapping[clean]
    return next((v for k, v in mapping.items() if _label(k) == clean), None)


def _model_colors(models: list[str], config: dict) -> dict[str, str]:
    """Stable model-specific colors independent of score/paint order."""
    configured = config.get("model_colors", {})
    trans = sorted((m for m in models if _label(m) != "BLOOM"), key=lambda m: _label(m).casefold())
    colors = {}
    for i, model in enumerate(trans):
        # The golden-angle sequence separates nearby legend entries. Muted
        # saturation keeps 22 simultaneous trajectories softer than BLOOM.
        hue = (0.57 + i * 0.38196601125) % 1
        rgb = colorsys.hls_to_rgb(hue, 0.55, 0.30)
        default = matplotlib.colors.to_hex(rgb)
        colors[model] = _lookup(configured, model) or default
    for model in models:
        if _label(model) == "BLOOM":
            colors[model] = _lookup(configured, model) or "#714366"
    return colors


def _dataset(dataset: str, config: dict) -> dict:
    default = _DEFAULT_DATASETS.get(
        dataset, {"label": dataset, "fill": "#E9EEF2", "border": "#A0ADB9", "text": "#394954"}
    )
    return {**default, **config.get("datasets", {}).get(dataset, {})}


def _wrap_label(label: str, metric: str) -> str:
    # Short conventional names keep label boxes legible without opaque IDs.
    substitutions = {
        "mean_day_pcc_gene": "Mean-day\ngene PCC",
        "mean_day_pcc_cell": "Mean-day\ncell PCC",
        "rmse": "RMSE",
        "mmd_u": "MMD-U",
        "sliced_wasserstein": "Sliced\nWasserstein",
        "overall": "Equal-aspect\naverage",
        "covariance_edge_pcc": "Covariance\nPCC",
    }
    if metric in substitutions:
        return substitutions[metric]
    text = str(label).replace("Mean day", "Mean-day")
    if len(text) < 24:
        return text
    words = text.split()
    lines, current = [], []
    for word in words:
        if current and len(" ".join(current + [word])) > 23:
            lines.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _raw_tick(axis: dict, radius: float) -> float:
    if axis.get("scale") == "rank":
        return 23 - 22 * radius
    low, high = float(axis["display_low"]), float(axis["display_high"])
    lower = str(axis.get("direction", "higher")).casefold() in {
        "lower",
        "min",
        "minimize",
        "lower_is_better",
        "lower-is-better",
    }
    return high - radius * (high - low) if lower else low + radius * (high - low)


def _paint_order(job: dict) -> list[str]:
    models = list(job["models"])
    supplied = list(job.get("model_order", models))
    if len(set(models)) != len(models):
        raise ValueError("Radar models must have distinct identities")
    if set(supplied) != set(models) or len(supplied) != len(models):
        raise ValueError("model_order must contain every model exactly once")
    bloom = [m for m in supplied if _label(m) == "BLOOM"]
    if len(bloom) != 1:
        raise ValueError("A comparison radar requires exactly one BLOOM model")
    return [m for m in supplied if _label(m) != "BLOOM"] + bloom


def _scores(axes: list[dict], model: str) -> np.ndarray:
    values = []
    for axis in axes:
        value = _lookup(axis.get("values", {}), model)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"Missing finite display score for {model}: {axis.get('dataset')}/{axis.get('metric')}"
            )
        value = float(value)
        if value < -1e-9 or value > 1 + 1e-9:
            raise ValueError(f"Display scores must lie within [0, 1], got {value}")
        values.append(np.clip(value, 0, 1))
    return np.asarray(values, dtype=float)


def _draw_radar(
    ax,
    axes: list[dict],
    models: list[str],
    colors: dict,
    config: dict,
    combined: bool,
    font_scale: float = 1,
):
    if len(axes) < 3:
        raise ValueError("Radar charts require at least three axes")
    n = len(axes)
    angle = np.arange(n) * 2 * np.pi / n
    unit = np.column_stack((np.sin(angle), np.cos(angle)))
    ax.set_aspect("equal")
    ax.axis("off")
    # Labels use fixed horizontal boxes, as in Fig. 1c, rather than rotating
    # text around a circle. White outside space is reserved for those boxes.
    ax.set_xlim(-1.68, 1.68)
    ax.set_ylim(-1.52, 1.53)
    for radius in (0.25, 0.5, 0.75, 1):
        ax.add_patch(
            Circle(
                (0, 0),
                radius,
                fill=False,
                edgecolor="#C7CFD5" if radius == 1 else "#E0E5E8",
                linewidth=0.75 if radius == 1 else 0.55,
                zorder=1,
            )
        )
    for i, axis in enumerate(axes):
        x, y = unit[i]
        ax.plot([0, x], [0, y], color="#C8D1D7", linewidth=0.65, zorder=1)
    style = config.get("style", {})
    trans_alpha = float(style.get("transformer_line_alpha", 0.54))
    trans_width = float(style.get("transformer_line_width", 0.82))
    trans_fill = float(style.get("transformer_fill_alpha", 0.012))
    bloom_width = float(style.get("bloom_line_width", 2.5))
    bloom_fill = float(style.get("bloom_fill_alpha", 0.11))
    for index, model in enumerate(models):
        radial = _scores(axes, model)
        points = radial[:, None] * unit
        closed = np.vstack((points, points[0]))
        bloom = _label(model) == "BLOOM"
        z = 10 + index
        ax.fill(
            closed[:, 0],
            closed[:, 1],
            color=colors[model],
            alpha=bloom_fill if bloom else trans_fill,
            linewidth=0,
            zorder=z,
        )
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=colors[model],
            alpha=1 if bloom else trans_alpha,
            linewidth=bloom_width if bloom else trans_width,
            zorder=z + 0.2,
            solid_joinstyle="round",
            solid_capstyle="round",
        )
        if bloom:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                color=colors[model],
                s=11 * font_scale,
                zorder=z + 0.5,
                edgecolor="white",
                linewidth=0.5,
            )
    tick_radii = [0.4, 0.7, 1] if n > 18 else [0.25, 0.5, 0.75, 1]
    show_ticks = bool(style.get("show_scale_ticks", config.get("show_scale_ticks", True)))
    tick_font = max(4.8, min(7.5, 8.5 - max(0, n - 5) * 0.11)) * font_scale
    if show_ticks:
        # On each spoke, raw units at a shared radius can differ. Lower-is-
        # better axes reverse the raw tick labels while retaining outward gain.
        for i, axis in enumerate(axes):
            x, y = unit[i]
            tangent = np.array((y, -x))
            side = 1 if x >= 0 else -1
            for radius in tick_radii:
                position = unit[i] * radius + tangent * (0.014 * side)
                ax.text(
                    *position,
                    _number(_raw_tick(axis, radius)),
                    ha="left" if side > 0 else "right",
                    va="center",
                    fontsize=tick_font,
                    color="#5B646A",
                    zorder=60,
                    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.3},
                )
    box_font = max(7, 10.2 - max(0, n - 8) * 0.13) * font_scale
    # Above 18 axes, adjacent boxes need additional room. Radial leaders keep
    # the association visible when their vertical coordinates are spread.
    positions = unit * 1.20
    positions[:, 1] *= 1.09
    if n > 18:
        for i, (x, y) in enumerate(unit):
            if abs(x) < 0.12:
                positions[i, 1] = math.copysign(1.43, y)
            elif abs(x) < 0.36 and abs(y) > 0.87:
                positions[i, 0] = math.copysign(0.57, x)
    if n > 18:
        for sign in (-1, 1):
            indexes = [i for i in range(n) if sign * unit[i, 0] > 0.08]
            indexes.sort(key=lambda i: positions[i, 1])
            if indexes:
                min_sep = 0.175 if combined else 0.145
                spread = np.array([positions[i, 1] for i in indexes])
                for k in range(1, len(spread)):
                    spread[k] = max(spread[k], spread[k - 1] + min_sep)
                if spread[-1] > 1.32:
                    spread -= spread[-1] - 1.32
                for k in range(len(spread) - 2, -1, -1):
                    spread[k] = min(spread[k], spread[k + 1] - min_sep)
                for i, new_y in zip(indexes, spread):
                    positions[i, 1] = new_y
    for i, axis in enumerate(axes):
        dataset = _dataset(axis["dataset"], config)
        x, y = unit[i]
        anchor = positions[i].copy()
        centered = abs(x) < 0.12
        ha = "center" if centered else ("left" if x > 0 else "right")
        if not centered:
            anchor[0] += 0.015 if x > 0 else -0.015
        label = _wrap_label(axis.get("label", axis["metric"]), axis["metric"])
        if combined:
            label += f"\n({dataset['label']})"
        ax.plot(
            [x * 1.025, anchor[0]],
            [y * 1.025, anchor[1]],
            color=dataset["border"],
            linewidth=0.7,
            zorder=65,
            alpha=0.75,
        )
        ax.text(
            *anchor,
            label,
            ha=ha,
            va="center",
            fontsize=box_font,
            linespacing=1.15,
            color=dataset["text"],
            zorder=70,
            bbox={
                "boxstyle": "round,pad=0.32,rounding_size=0.16",
                "facecolor": dataset["fill"],
                "edgecolor": dataset["border"],
                "linewidth": 1.15,
            },
        )
    return {
        "axis_count": n,
        "spoke_angles_degrees": (angle * 180 / np.pi).tolist(),
        "tick_radii": tick_radii,
        "raw_ticks": [[_raw_tick(axis, r) for r in tick_radii] for axis in axes],
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def render_figure(job: dict, out_dir, config: dict, formats=("png", "pdf", "svg")) -> dict:
    """Render combined, three-dataset, or single-dataset overlaid radar charts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(formats)
    if not formats or any(fmt not in {"png", "pdf", "svg"} for fmt in formats):
        raise ValueError("formats must contain png, pdf, and/or svg")
    axes = list(job["axes"])
    if len(axes) < 3:
        raise ValueError("Radar charts require at least three axes")
    models = _paint_order(job)
    colors = _model_colors(models, config)
    layout = job.get("layout", "combined")
    if layout not in {"combined", "triptych", "single"}:
        raise ValueError(f"Unsupported radar layout: {layout}")
    style = config.get("style", {})
    legend = bool(style.get("legend", config.get("legend", True)))
    dpi = int(style.get("dpi", config.get("dpi", 300)))
    if dpi < 50 or dpi > 1200:
        raise ValueError("dpi must be between 50 and 1200")
    dataset_keys = list(job.get("datasets") or dict.fromkeys(axis["dataset"] for axis in axes))
    if layout == "single" and len(set(a["dataset"] for a in axes)) != 1:
        raise ValueError("A single radar must contain one dataset")
    if layout == "triptych":
        max_axes = max(sum(a["dataset"] == ds for a in axes) for ds in dataset_keys)
        sizes = style.get("triptych_figsize", (26, 11.5) if max_axes > 18 else (20.5, 9.0))
    elif layout == "single":
        sizes = style.get("single_figsize", (14, 14) if len(axes) > 18 else (10.5, 10.6))
    else:
        sizes = style.get("combined_figsize", (13, 13))
    theme = {
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 10,
    }
    with plt.rc_context(theme):
        fig = plt.figure(figsize=sizes)
        radar_metadata = []
        if layout == "triptych":
            count = len(dataset_keys)
            if count < 1:
                raise ValueError("A triptych requires datasets")
            left, total_width = 0.027, 0.946
            width = total_width / count
            for i, ds in enumerate(dataset_keys):
                subset = [a for a in axes if a["dataset"] == ds]
                if len(subset) < 3:
                    raise ValueError(f"Dataset {ds} has fewer than three radar axes")
                ax = fig.add_axes(
                    (left + i * width, 0.245 if legend else 0.13, width, 0.615 if legend else 0.74)
                )
                info = _draw_radar(ax, subset, models, colors, config, combined=False, font_scale=1)
                if len(subset) > 18:
                    # Long external labels need a larger horizontal gutter
                    # between adjacent dense radars.
                    ax.set_xlim(-2.08, 2.08)
                dsinfo = _dataset(ds, config)
                ax.text(
                    0.5,
                    1.06,
                    dsinfo["label"],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=15,
                    fontweight="bold",
                    color=dsinfo["text"],
                    bbox={
                        "boxstyle": "round,pad=.35",
                        "facecolor": dsinfo["fill"],
                        "edgecolor": dsinfo["border"],
                        "linewidth": 1.2,
                    },
                )
                info["dataset"] = ds
                radar_metadata.append(info)
        else:
            ax = fig.add_axes((0.035, 0.17 if legend else 0.045, 0.93, 0.75 if legend else 0.86))
            radar_metadata.append(
                _draw_radar(ax, axes, models, colors, config, combined=layout == "combined")
            )
        title = str(job.get("title", "BLOOM benchmark"))
        title = title.replace("unidirectional BLOOM", "BLOOM").replace(
            "Unidirectional BLOOM", "BLOOM"
        )
        fig.suptitle(
            title,
            x=0.5,
            y=0.975,
            fontsize=17 if layout == "triptych" else 16,
            fontweight="bold",
            color="#343C43",
        )
        fig.text(
            0.5,
            0.944 if layout != "triptych" else 0.932,
            "Larger radius indicates better performance",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#646E75",
        )
        if legend:
            # BLOOM is easy to find; transformer legend order is best to worst.
            legend_models = [models[-1]] + list(reversed(models[:-1]))
            handles = [
                Line2D(
                    [0],
                    [0],
                    color=colors[m],
                    linewidth=2.5 if _label(m) == "BLOOM" else 1.6,
                    label=_label(m),
                )
                for m in legend_models
            ]
            lg = fig.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.022),
                ncol=6 if layout == "triptych" else 4,
                frameon=False,
                fontsize=9 if layout == "triptych" else 9.2,
                handlelength=2.3,
                handletextpad=0.55,
                columnspacing=1.6,
                labelspacing=0.62,
            )
            for text in lg.get_texts():
                if text.get_text() == "BLOOM":
                    text.set_fontweight("bold")
                    text.set_color(colors[models[-1]])
        relative_id = Path(str(job["id"]))
        if relative_id.is_absolute() or ".." in relative_id.parts:
            plt.close(fig)
            raise ValueError("Figure id must be a relative path without parent traversal")
        filename = Path(*[_slug(part) for part in relative_id.parts])
        (out_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        files = {}
        for fmt in formats:
            path = out_dir / f"{filename.as_posix()}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches=None)
            files[fmt] = path.relative_to(out_dir).as_posix()
        plt.close(fig)
    metadata = {
        "schema": "bloom_radar_figure/v1",
        "id": job["id"],
        "panel": job.get("panel"),
        "version": job.get("version"),
        "title": title,
        "layout": layout,
        "axis_count": len(axes),
        "axes": axes,
        "datasets": {ds: _dataset(ds, config) for ds in dataset_keys},
        "model_draw_order": models,
        "legend_order": [models[-1]] + list(reversed(models[:-1])),
        "model_colors": colors,
        "score_transformation_by_renderer": "none",
        "radial_direction": "outward is better",
        "radars": radar_metadata,
        "coverage_text_in_artwork": False,
        "source_notes": job.get("notes", []),
        "files": files,
    }
    sidecar = out_dir / f"{filename.as_posix()}.json"
    sidecar.write_text(
        json.dumps(_json_safe(metadata), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "id": job["id"],
        "panel": job.get("panel"),
        "version": job.get("version"),
        "title": title,
        "axis_count": len(axes),
        "layout": layout,
        "files": files,
        "sidecar": sidecar.relative_to(out_dir).as_posix(),
    }
