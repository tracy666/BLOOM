"""Biological views of SAVED interpolation, rollout, history and intervention data.

All plots consume prepared artifacts. Future/history populations remain predictions
without invented truth; compact interventions remain stage/gene summaries rather
than fabricated edited cells. Coordinates are never refitted in this module.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from . import plot
from .selection import select_genes, select_programs


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unnamed"


def _label(source: dict, options: dict) -> str:
    return plot.model_label(source, options)


def _notify(progress, view: str, entity: str = "", figures: int = 0) -> None:
    if progress:
        try:
            progress(
                {"view": view, "step": view, "entity": entity, "completed": 1, "figures": figures}
            )
        except TypeError:
            try:
                progress(1)
            except TypeError:
                progress()


def _record(paths, view, entity="", model="comparison"):
    return [{"path": str(p), "view": view, "model": model, "entity": str(entity)} for p in paths]


def _read_table(path: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    )


def _artifacts(cohort, options):
    loaded, errors = [], []
    for source in plot.available_models(cohort, options):
        try:
            artifact = plot.load_artifact(source)
            artifact["source"] = source
            if "genes" not in artifact:
                artifact["genes"] = source.get("genes", artifact.get("meta", {}).get("genes", []))
            loaded.append(artifact)
            warning = artifact.get("meta", {}).get("warning", [])
            if isinstance(warning, str):
                warning = [warning]
            errors.extend(f'{source.get("model")}: {w}' for w in warning)
        except Exception as exc:
            errors.append(
                f"{source.get('model', source.get('source_id'))}: {type(exc).__name__}: {exc}"
            )
    return loaded, errors


def _native_cohort_options(cohort, options):
    # Scenario artifacts may lack GT. Let selection explicitly report descriptive
    # model variation rather than treating missing truth as a failed metric route.
    opts = dict(options)
    opts["scenario"] = cohort.get("scenario")
    return opts


def _gene_choices(cohort, artifacts, options, output):
    try:
        choices = select_genes(cohort, _native_cohort_options(cohort, options))
    except Exception as exc:
        # Selection errors must be visible. Never bypass author/developmental
        # evidence by silently selecting arbitrary high-variance native genes.
        choices = pd.DataFrame(
            columns=["gene", "score", "effect", "p_value", "q_value", "reason", "selected"]
        )
        (output / "GENE_SELECTION_ERROR.txt").write_text(f"{type(exc).__name__}: {exc}")
    if choices is None:
        choices = pd.DataFrame(
            columns=["gene", "score", "effect", "p_value", "q_value", "reason", "selected"]
        )
    if "selected" not in choices:
        choices["selected"] = True
    cap = int(options.get("max_entities", 0) or 0)
    if cap:
        indices = choices.index[choices["selected"].astype(bool)]
        choices.loc[indices[cap:], "selected"] = False
    choices.to_csv(output / "selected_entities.csv", index=False)
    return choices, choices.loc[choices["selected"].astype(bool), "gene"].astype(str).tolist()


def _stage(item, gene=None):
    stage = item.get("stage", pd.DataFrame()).copy()
    if stage.empty or "time" not in stage:
        return pd.DataFrame()
    stage["time"] = pd.to_numeric(stage["time"], errors="coerce")
    if gene is not None:
        stage = stage[stage["gene"].astype(str) == str(gene)] if "gene" in stage else pd.DataFrame()
    return stage.sort_values("time")


def _cells(item):
    cells = item.get("cells", pd.DataFrame()).copy()
    if not {"x", "y", "time"}.issubset(cells.columns):
        return pd.DataFrame()
    for key in ["x", "y", "time"]:
        cells[key] = pd.to_numeric(cells[key], errors="coerce")
    return cells.dropna(subset=["x", "y", "time"])


def _scenario(cohort):
    return str(cohort.get("scenario", "temporal"))


def _geometry_key(item):
    meta = {**item.get("meta", {}), **item.get("source", {})}
    return str(
        meta.get("reference_id")
        or "unverified_" + str(meta.get("source_id", meta.get("model", "unknown")))
    )


def _geometry_limits(items, padding=0.04):
    groups = {}
    for item in items:
        cells = _cells(item)
        if not cells.empty:
            groups.setdefault(_geometry_key(item), []).append(cells[["x", "y"]].to_numpy(float))
    limits = {}
    for key, arrays in groups.items():
        xy = np.concatenate(arrays)
        span = np.maximum(np.ptp(xy, axis=0), 1e-8)
        limits[key] = [
            (float(xy[:, j].min() - padding * span[j]), float(xy[:, j].max() + padding * span[j]))
            for j in (0, 1)
        ]
    return limits


def _geometry_panel_title(item, options, multiple=False):
    label = _label(item["source"], options)
    if multiple:
        method = item.get("meta", {}).get(
            "embedding_method", item["source"].get("embedding_method", "Reference")
        )
        label += f"\n{method} | axis: {_geometry_key(item)}"
    return label


def _view_status(figures, views, notes):
    produced = {ALIASES.get(row.get("view"), row.get("view")) for row in figures}
    missing = list(dict.fromkeys(view for view in views if ALIASES.get(view, view) not in produced))
    if missing:
        notes.append("Requested views without drawable output: " + ", ".join(missing) + ".")
    return ("partial" if missing else "complete") if figures else "unsupported"


def _model_rows(cohort, options, output, items=None, references=None):
    """Record every actual gallery source plus hidden diagnostic boundaries."""
    primary = list(items or [])
    if not primary:
        for source in plot.available_models(cohort, options):
            meta = dict(source)
            path = Path(source.get("artifact_dir", "")) / "metadata.json"
            if path.is_file():
                import json

                meta.update(json.loads(path.read_text()))
            primary.append({"source": source, "meta": meta})

    def key(item):
        meta = _reference_metadata(item)
        return str(meta.get("source_id") or meta.get("artifact_dir"))

    gallery = {key(item) for item in primary}
    seen = set()
    rows = []
    for item in primary + list(references or []):
        identity = key(item)
        if identity in seen:
            continue
        seen.add(identity)
        meta = _reference_metadata(item)
        cells = item.get("cells", pd.DataFrame())
        native = list(item.get("genes", meta.get("genes", [])))
        metadata = meta.get("source_metadata", {})
        rows.append(
            {
                "model": meta.get("model"),
                "label": _label(meta, options),
                "role": meta.get("role"),
                "dataset": meta.get("dataset", cohort.get("dataset")),
                "scenario": meta.get("scenario", cohort.get("scenario")),
                "panel": meta.get("panel"),
                "source_id": meta.get("source_id"),
                "source_path": meta.get("path"),
                "artifact_dir": meta.get("artifact_dir"),
                "gallery_included": identity in gallery,
                "n_native_genes": len(native),
                "native_genes": ";".join(map(str, native)),
                "n_saved_cells": meta.get("n_total", meta.get("n_cells")),
                "n_displayed_cells": len(cells) if not cells.empty else meta.get("n_sample"),
                "time_kind": meta.get("time_kind"),
                "embedding_method": meta.get("embedding_method"),
                "reference_id": meta.get("reference_id"),
                "scale_group": meta.get("scale_group"),
                "scale_policy": meta.get("scale_policy"),
                "reference_policy": meta.get("reference_policy"),
                "state_method": meta.get("state_method"),
                "state_summary_policy": meta.get("state_summary_policy"),
                "detection_units": meta.get("detection_units"),
                "normalization": (
                    metadata.get("normalization") if isinstance(metadata, dict) else None
                ),
                "parent_source_id": meta.get("parent_source_id"),
                "matching_parent_sources": ";".join(
                    map(str, meta.get("matching_parent_sources", []))
                ),
                "display_sample_policy": meta.get("sample_policy", meta.get("sampling_policy")),
            }
        )
    pd.DataFrame(rows).to_csv(output / "MODEL_ROWS.csv", index=False)


def _geometry(item, options, mode="geometry"):
    cells = _cells(item)

    def draw(ax, legend=True):
        if cells.empty:
            ax.text(
                0.5,
                0.5,
                "No saved common-reference coordinates",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        if mode == "density":
            artist = ax.hexbin(
                cells.x, cells.y, gridsize=45, mincnt=1, cmap="viridis", linewidths=0
            )
            if legend:
                plt.colorbar(
                    artist, ax=ax, label="Displayed cells per hexagon", fraction=0.05, pad=0.03
                )
        else:
            tr = options.get(
                "_global_time_range", (float(cells.time.min()), float(cells.time.max()))
            )
            artist = ax.scatter(
                cells.x,
                cells.y,
                c=cells.time,
                s=4,
                alpha=0.75,
                cmap="viridis",
                vmin=tr[0],
                vmax=tr[1],
                rasterized=True,
                linewidths=0,
            )
            if legend:
                plt.colorbar(artist, ax=ax, label="Saved time", fraction=0.05, pad=0.03)
        if mode == "trajectory":
            centers = (
                cells.groupby("time")[["x", "y"]]
                .mean()
                .sort_index(ascending=item["source"].get("scenario") != "history")
            )
            ax.plot(centers.x, centers.y, color="#7C3453", lw=1.5, marker="o", ms=3)
            if len(centers) > 1:
                xy = centers.to_numpy()
                ax.quiver(
                    xy[:-1, 0],
                    xy[:-1, 1],
                    np.diff(xy[:, 0]),
                    np.diff(xy[:, 1]),
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    color="#7C3453",
                    width=0.003,
                )
            for t, row in centers.iloc[
                np.linspace(0, len(centers) - 1, min(8, len(centers))).astype(int)
            ].iterrows():
                ax.annotate(f"{t:g}", (row.x, row.y), fontsize=6)
        if "is_generated" in cells and (~cells.is_generated.astype(bool)).any():
            anchors = cells[~cells.is_generated.astype(bool)].groupby("time")[["x", "y"]].mean()
            ax.scatter(
                anchors.x,
                anchors.y,
                s=40,
                facecolors="none",
                edgecolors="#20252C",
                marker="D",
                lw=1,
                label="Observed anchors",
            )
            if legend:
                ax.legend(fontsize=6, loc="best")
        method = item.get("meta", {}).get(
            "embedding_method", item["source"].get("embedding_method", "Reference")
        )
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")

    return draw


def _comparisons(items, callback_factory, base, options, title, view, entity=""):
    callbacks = [callback_factory(a) for a in items]
    if view in ("geometry", "trajectory", "density", "storyboard"):
        limits = _geometry_limits(items, padding=0.03)
        if limits:

            def aligned(callback, item):
                def draw(ax, legend=True):
                    callback(ax, legend)
                    if _geometry_key(item) in limits:
                        xlim, ylim = limits[_geometry_key(item)]
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)

                return draw

            callbacks = [aligned(c, a) for c, a in zip(callbacks, items)]
    multiple = len({_geometry_key(a) for a in items if not _cells(a).empty}) > 1 and view in (
        "geometry",
        "trajectory",
        "density",
        "storyboard",
    )
    panels = [
        (_geometry_panel_title(a, options, multiple), a["source"].get("role", "FM"), callback)
        for a, callback in zip(items, callbacks)
    ]
    if not panels:
        return []
    if hasattr(plot, "compare_mosaics"):
        paths = plot.compare_mosaics(panels, base, options, title=title)
    else:
        paths = plot.mosaic(
            [(label, callback) for label, role, callback in panels], base, options, title=title
        )
    return _record(paths, view, entity)


def _snapshot_times(items, options):
    times = sorted(
        {
            float(t)
            for item in items
            for t in _cells(item).get("time", pd.Series(dtype=float)).unique()
        }
    )
    if any(
        item["source"].get("scenario") == "history"
        and item["source"].get("role") not in ("GT", "REFERENCE")
        for item in items
    ):
        times = times[::-1]
    cap = int(options.get("storyboard_timepoints", 12) or 0)
    if cap and len(times) > cap:
        times = [
            times[i] for i in np.unique(np.linspace(0, len(times) - 1, cap).round().astype(int))
        ]
    return times


def _storyboard(items, output, options, title, progress=None, start_figures=0):
    times = _snapshot_times(items, options)
    pd.DataFrame(
        {
            "time": times,
            "reason": "Evenly spaced saved timepoints including earliest/latest; exact stored coordinates",
        }
    ).to_csv(output / "storyboard_times.csv", index=False)
    figures = []
    for time in times:

        def factory(item, t=time):
            cells = _cells(item)

            def draw(ax, legend=True):
                if cells.empty:
                    ax.set_axis_off()
                    return
                ax.scatter(
                    cells.x,
                    cells.y,
                    s=2,
                    color=options.get("background_color", "#969FA7"),
                    alpha=float(options.get("background_alpha", 0.8)),
                    rasterized=True,
                    linewidths=0,
                )
                near = cells[np.isclose(cells.time, t, rtol=0, atol=1e-6)]
                if near.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "Time unavailable",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                else:
                    ax.scatter(
                        near.x,
                        near.y,
                        s=7,
                        color="#357F87",
                        alpha=0.85,
                        rasterized=True,
                        linewidths=0,
                    )
                ax.set_xlabel("Reference coordinate 1")
                ax.set_ylabel("Reference coordinate 2")

            return draw

        figures.extend(
            _comparisons(
                items,
                factory,
                output / f"storyboard_time_{_safe(time)}",
                options,
                f"{title} | saved time {time:g}",
                "storyboard",
                str(time),
            )
        )
        _notify(progress, "storyboard", str(time), start_figures + len(figures))
    return figures


def _animation(items, output, options, title, progress=None, start_figures=0):
    """Chronological saved-coordinate GIF; no new trajectory is inferred."""
    if not options.get("animation", True):
        return []
    from PIL import Image

    times = _snapshot_times(items, options)
    if len(times) < 2:
        return []
    coords = [_cells(item) for item in items]
    if not any(not cells.empty for cells in coords):
        return []
    columns = min(5, max(1, len(items)))
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.65 * rows), squeeze=False)
    limits = _geometry_limits(items)
    multiple = len(limits) > 1
    all_times = np.concatenate([cells.time.to_numpy(float) for cells in coords if not cells.empty])
    norm = Normalize(float(all_times.min()), float(all_times.max()))
    moving = []
    for ax, item, cells in zip(axes.flat, items, coords):
        ax.set_title(_geometry_panel_title(item, options, multiple), fontsize=7 if multiple else 8)
        if not cells.empty:
            ax.scatter(
                cells.x,
                cells.y,
                s=2,
                color=options.get("background_color", "#969FA7"),
                alpha=float(options.get("background_alpha", 0.8)),
                linewidths=0,
                rasterized=True,
            )
        point = ax.scatter([], [], c=[], s=7, cmap="viridis", norm=norm, alpha=0.9, linewidths=0)
        moving.append(point)
        if _geometry_key(item) in limits:
            xlim, ylim = limits[_geometry_key(item)]
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        method = item.get("meta", {}).get(
            "embedding_method", item["source"].get("embedding_method", "Reference")
        )
        ax.set_xlabel(f"{method} 1", fontsize=7)
        ax.set_ylabel(f"{method} 2", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(items) :]:
        ax.axis("off")
    caption = fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.86, 0.93))
    coloraxis = fig.add_axes([0.90, 0.20, 0.018, 0.56])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=coloraxis)
    colorbar.set_label("Saved time (shared)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    frames = []
    rows_log = []
    try:
        for frame_id, time in enumerate(times):
            for item, cells, point in zip(items, coords, moving):
                subset = (
                    cells[np.isclose(cells.time, time, rtol=0, atol=1e-6)]
                    if not cells.empty
                    else cells
                )
                # Boundary references remain visible as an explicitly observed
                # context, including times at which no new observation exists.
                if item["source"].get("role") == "REFERENCE" and subset.empty:
                    subset = cells
                point.set_offsets(
                    subset[["x", "y"]].to_numpy() if not subset.empty else np.empty((0, 2))
                )
                point.set_array(subset.time.to_numpy(float) if not subset.empty else np.array([]))
                local_limits = limits.get(_geometry_key(item), [(np.nan, np.nan), (np.nan, np.nan)])
                rows_log.append(
                    {
                        "frame": frame_id,
                        "time": time,
                        "model": item["source"].get("model"),
                        "role": item["source"].get("role"),
                        "n_displayed": len(subset),
                        "reference_policy": item.get("meta", {}).get("reference_policy", ""),
                        "reference_id": _geometry_key(item),
                        "embedding_method": item.get("meta", {}).get(
                            "embedding_method", item["source"].get("embedding_method", "Reference")
                        ),
                        "scale_group": _reference_metadata(item).get("scale_group"),
                        "x_min": local_limits[0][0],
                        "x_max": local_limits[0][1],
                        "y_min": local_limits[1][0],
                        "y_max": local_limits[1][1],
                    }
                )
            caption.set_text(f"{title} | saved time {time:g}")
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(
                Image.fromarray(rgba[:, :, :3].copy()).convert("P", palette=Image.Palette.ADAPTIVE)
            )
            if progress:
                progress(
                    {
                        "view": "animation",
                        "step": f"animation frame {frame_id+1}/{len(times)}",
                        "figures": start_figures,
                    }
                )
        path = output / "temporal_animation__all_models.gif"
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=int(options.get("animation_frame_ms", 650)),
            loop=0,
            optimize=False,
        )
    finally:
        plt.close(fig)
    pd.DataFrame(rows_log).to_csv(output / "animation_frames.csv", index=False)
    return _record([path], "animation")


def _gene_line(item, gene):
    stage = _stage(item, gene)

    def draw(ax, legend=True):
        if stage.empty:
            ax.text(
                0.5,
                0.5,
                f"{gene}: outside native gene panel",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        ax.plot(
            stage.time,
            stage["mean"],
            color="#397F86",
            marker="o",
            ms=2.5,
            lw=1.5,
            label="Saved population mean",
        )
        # Population SD is not a standard error/CI. Display it only when requested.
        if "std" in stage and bool(item.get("_show_sd", False)):
            ax.fill_between(
                stage.time,
                stage["mean"] - stage["std"],
                stage["mean"] + stage["std"],
                color="#397F86",
                alpha=0.15,
                label="Population SD",
            )
        ax.set_xlabel("Saved time")
        ax.set_ylabel("Stored expression mean")
        if legend:
            ax.legend(fontsize=6)

    return draw


def _gene_distribution(item, gene):
    cells = item.get("cells", pd.DataFrame())
    genes = list(map(str, item.get("genes", [])))

    def draw(ax, legend=True):
        expression = item.get("expression")
        if gene not in genes or expression is None or cells.empty or "time" not in cells:
            ax.text(
                0.5,
                0.5,
                "No cell-level expression for native gene",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        idx = genes.index(gene)
        times = pd.to_numeric(cells.time, errors="coerce").to_numpy()
        values = np.asarray(expression[:, idx], float)
        if "expression_row" in cells:
            positions = pd.to_numeric(cells.expression_row, errors="raise").to_numpy(dtype=int)
            values = np.asarray(expression[positions, idx], float)
        levels = np.unique(times[np.isfinite(times)])
        cap = 12
        if len(levels) > cap:
            levels = levels[np.unique(np.linspace(0, len(levels) - 1, cap).round().astype(int))]
        vectors = [values[np.isclose(times, t, atol=1e-6, rtol=0)] for t in levels]
        pos = np.arange(len(levels))
        usable = [(i, v[np.isfinite(v)]) for i, v in enumerate(vectors) if np.isfinite(v).any()]
        for i, v in usable:
            if len(v) > 1 and np.ptp(v) > 1e-12:
                violin = ax.violinplot(
                    [v], positions=[i], widths=0.7, showextrema=False, showmedians=False
                )
                for body in violin["bodies"]:
                    body.set_facecolor("#397F86")
                    body.set_alpha(0.55)
                    body.set_edgecolor("none")
            ax.boxplot(
                [v],
                positions=[i],
                widths=0.15,
                showfliers=False,
                medianprops={"color": "#222222"},
                patch_artist=True,
                boxprops={"facecolor": "white"},
            )
        ax.set_xticks(pos, [f"{t:g}" for t in levels], rotation=45, fontsize=6)
        ax.set_xlabel("Saved time (displayed sample)")
        ax.set_ylabel("Stored expression")

    return draw


def _expression_heatmap(item, genes, mode="gene_heatmap", limits=None):
    stage = _stage(item)
    data = stage[stage.gene.astype(str).isin(genes)] if "gene" in stage else pd.DataFrame()

    def draw(ax, legend=True):
        if data.empty:
            ax.text(
                0.5,
                0.5,
                "No covered selected genes",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        wide = data.pivot_table(index="gene", columns="time", values="mean").reindex(
            [g for g in genes if g in set(data.gene.astype(str))]
        )
        if mode == "marker_dotplot":
            pct = (
                data.pivot_table(index="gene", columns="time", values="pct").reindex(
                    index=wide.index, columns=wide.columns
                )
                if "pct" in data
                else wide.notna().astype(float)
            )
            fraction = pct.to_numpy(float)
            fraction = np.clip(fraction / 100 if np.nanmax(fraction) > 1.00001 else fraction, 0, 1)
            xx, yy = np.meshgrid(np.arange(wide.shape[1]), np.arange(wide.shape[0]))
            artist = ax.scatter(
                xx.ravel(),
                yy.ravel(),
                s=12 + 110 * fraction.ravel(),
                c=wide.to_numpy().ravel(),
                cmap="viridis",
                vmin=limits[0] if limits else None,
                vmax=limits[1] if limits else None,
                rasterized=True,
            )
        else:
            artist = ax.imshow(
                wide.to_numpy(),
                aspect="auto",
                cmap="viridis",
                vmin=limits[0] if limits else None,
                vmax=limits[1] if limits else None,
                interpolation="nearest",
            )
        ax.set_yticks(range(len(wide)), wide.index, fontsize=6)
        step = max(1, math.ceil(wide.shape[1] / 10))
        ax.set_xticks(
            np.arange(0, wide.shape[1], step),
            [f"{t:g}" for t in wide.columns[::step]],
            rotation=45,
            fontsize=6,
        )
        ax.set_xlabel("Saved time")
        ax.set_ylabel("Native selected gene")
        if legend:
            plt.colorbar(artist, ax=ax, fraction=0.05, pad=0.03, label="Mean stored expression")

    return draw


def _program_records(cohort, items, options):
    try:
        programs = select_programs(cohort, options)
    except Exception:
        programs = []
    if isinstance(programs, pd.DataFrame):
        programs = programs.to_dict("records")
    if isinstance(programs, dict):
        programs = [{"program": k, "genes": v} for k, v in programs.items()]
    return programs or []


def _weighted_program_rows(frame, program, gene_column, value_column, group_columns):
    """Preserve the legacy scenario score: normalized absolute prior weights."""
    genes = program.get("genes", [])
    if isinstance(genes, str):
        genes = [g for g in re.split("[,;| ]+", genes) if g]
    supplied = program.get("weights")
    weights = (
        {str(g): abs(float(w)) for g, w in supplied.items() if np.isfinite(float(w))}
        if isinstance(supplied, dict)
        else {str(g): 1.0 for g in genes}
    )
    weights = {g: w for g, w in weights.items() if w > 0}
    full_mass = float(program.get("full_weight_mass", sum(weights.values())))
    basis = program.get(
        "scoring_basis",
        (
            "normalized absolute supplied weights"
            if supplied is not None
            else "uniform marker weights"
        ),
    )
    formula = "sum_g abs(w_g)*x_g / sum_g abs(w_g), over covered native genes"
    if frame.empty or not {gene_column, value_column, *group_columns}.issubset(frame):
        return pd.DataFrame()
    data = frame[frame[gene_column].astype(str).isin(weights)].copy()
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data = data[np.isfinite(data[value_column])]
    rows = []
    for key, group in data.groupby(group_columns, sort=True, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        # Prepared rows are unique per gene/stratum. Averaging exact duplicate
        # member rows prevents duplicated program membership from double scoring.
        values = group.groupby(gene_column, sort=False)[value_column].mean()
        mass = sum(weights[str(g)] for g in values.index)
        if mass <= 0:
            continue
        score = sum(float(value) * weights[str(g)] for g, value in values.items()) / mass
        rows.append(
            {
                **dict(zip(group_columns, key)),
                "weighted_score": score,
                "n_covered_genes": len(values),
                "total_genes": int(program.get("total", len(genes))),
                "covered_weight_mass": mass,
                "full_weight_mass": full_mass,
                "weighted_coverage": mass / full_mass if full_mass > 0 else np.nan,
                "scoring_formula": formula,
                "scoring_basis": basis,
            }
        )
    return pd.DataFrame(rows)


def _program_views(cohort, items, output, options, title):
    programs = _program_records(cohort, items, options)
    figures = []
    rows = []
    for program in programs:
        name = str(program.get("program", program.get("name", "program")))
        genes = program.get("genes", [])
        if isinstance(genes, str):
            genes = [g for g in re.split("[,;| ]+", genes) if g]

        def factory(item):
            stage = _stage(item)
            covered = (
                stage[stage.gene.astype(str).isin(genes)] if "gene" in stage else pd.DataFrame()
            )
            summary = _weighted_program_rows(covered, program, "gene", "mean", ["time"])
            means = (
                summary.set_index("time")["weighted_score"]
                if not summary.empty
                else pd.Series(dtype=float)
            )
            ng = int(summary.n_covered_genes.max()) if not summary.empty else 0
            for row in summary.to_dict("records"):
                score = row.pop("weighted_score")
                rows.append(
                    {
                        "model": item["source"].get("model"),
                        "program": name,
                        "mean_expression_score": score,
                        **row,
                    }
                )

            def draw(ax, legend=True):
                if means.empty:
                    ax.text(
                        0.5,
                        0.5,
                        "Program genes outside native panel",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                    return
                ax.plot(means.index, means.values, color="#8B5674", lw=1.5, marker="o", ms=2.5)
                ax.set_xlabel("Saved time")
                ax.set_ylabel(
                    f'Weighted mean expression ({ng}/{int(program.get("total",len(genes)))} genes)'
                )

            return draw

        figures.extend(
            _comparisons(
                items,
                factory,
                output / f"program_{_safe(name)}",
                options,
                f"{title} | {name}",
                "programs",
                name,
            )
        )
    if rows:
        pd.DataFrame(rows).drop_duplicates().to_csv(output / "program_timecourses.csv", index=False)
    return figures


def _diversity_table(items):
    records = []
    for item in items:
        stage = _stage(item)
        cells = _cells(item)
        model = item["source"].get("model", "unknown")
        for t, group in stage.groupby("time") if not stage.empty else []:
            variances = (
                pd.to_numeric(
                    group.get("std", pd.Series(index=group.index, dtype=float)), errors="coerce"
                )
                ** 2
            )
            detection = pd.to_numeric(
                group.get("pct", pd.Series(index=group.index, dtype=float)), errors="coerce"
            )
            mean = pd.to_numeric(group["mean"], errors="coerce")
            records.append(
                {
                    "model": model,
                    "time": float(t),
                    "mean_gene_variance": float(variances.mean()),
                    "mean_detection": float(detection.mean()),
                    "global_mean_expression": float(mean.mean()),
                    "pseudobulk_gene_variance": float(mean.var(ddof=0)),
                    "n_genes": int(len(group)),
                }
            )
    return pd.DataFrame(records)


def _reference_metadata(item):
    return {**item.get("meta", {}), **item.get("source", {})}


def _boundary_references(cohort, options, items):
    """Load every saved boundary for diagnostics; leave gallery choices alone."""
    references = [a for a in items if _reference_metadata(a).get("role") == "REFERENCE"]
    seen = {
        str(_reference_metadata(a).get("source_id") or _reference_metadata(a).get("artifact_dir"))
        for a in references
    }
    notes = []
    sources = (
        list(cohort.get("reference_sources", []))
        + list(options.get("reference_sources", []))
        + list(cohort.get("sources", []))
    )
    sources = plot.native_sources(cohort, options, sources=sources)
    for source in sources:
        if source.get("role") != "REFERENCE":
            continue
        key = str(source.get("source_id") or source.get("artifact_dir"))
        if key in seen:
            continue
        seen.add(key)
        try:
            reference = plot.load_artifact(source)
            reference["source"] = source
            references.append(reference)
        except Exception as exc:
            notes.append(
                f"{source.get('model')}: boundary reference could not be read: {type(exc).__name__}: {exc}"
            )
    return references, notes


def _matching_boundary(item, references, genes):
    """Match an own saved boundary first; otherwise require verified units."""
    source = _reference_metadata(item)
    source_id = source.get("source_id")
    scale = source.get("scale_group")
    exact, compatible = [], []
    for reference in references:
        meta = _reference_metadata(reference)
        if (
            meta.get("dataset") != source.get("dataset")
            or plot.source_panel(meta) != plot.source_panel(source)
            or plot.source_panel(source) is None
        ):
            continue
        if meta.get("scenario") not in (source.get("scenario"), "measured"):
            continue
        stage = _stage(reference)
        if stage.empty or not {"gene", "mean"}.issubset(stage):
            continue
        coverage = len(set(stage.gene.astype(str)) & set(genes))
        if not coverage:
            continue
        parents = meta.get("matching_parent_sources", [])
        if isinstance(parents, str):
            parents = [parents]
        parents = set(parents) | {meta.get("parent_source_id")}
        if source_id and source_id in parents:
            exact.append((coverage, reference, "exact saved parent source"))
        elif scale and "__unverified_" not in str(scale) and meta.get("scale_group") == scale:
            compatible.append((coverage, reference, "verified matching observed scale group"))
    candidates = exact or compatible
    if not candidates:
        return None, "no compatible saved observed boundary"
    candidates.sort(
        key=lambda candidate: (
            -candidate[0],
            str(_reference_metadata(candidate[1]).get("source_id", "")),
        )
    )
    return candidates[0][1], candidates[0][2]


def _drift_table(items, references=None):
    records = []
    if references is None:
        references = [a for a in items if _reference_metadata(a).get("role") == "REFERENCE"]
    for item in items:
        if item["source"].get("role") in ("GT", "REFERENCE"):
            continue
        stage = _stage(item)
        if stage.empty or not {"gene", "mean"}.issubset(stage.columns):
            continue
        wide = stage.pivot_table(index="time", columns="gene", values="mean").sort_index()
        if wide.empty:
            continue
        if item["source"].get("scenario") == "history":
            # First recursive history output is the latest saved generated age,
            # closest to the observed boundary, not the oldest/farthest age.
            wide = wide.sort_index(ascending=False)
        boundary, reference_match = _matching_boundary(item, references, wide.columns)
        refstage = _stage(boundary) if boundary is not None else pd.DataFrame()
        if not refstage.empty:
            refwide = refstage.pivot_table(index="time", columns="gene", values="mean").sort_index()
            reference_time = (
                float(refwide.index.min())
                if item["source"].get("scenario") == "history"
                else float(refwide.index.max())
            )
            common = sorted(set(wide.columns) & set(refwide.columns))
            wide = wide.reindex(columns=common)
            first = refwide.loc[reference_time, common].to_numpy(float)
            reference = "explicit saved observed boundary reference"
        else:
            first = wide.iloc[0].to_numpy(float)
            reference_time = float(wide.index[0])
            reference = "first saved prediction stage; no boundary reference available"
        previous = None
        cells = item.get("cells", pd.DataFrame())
        for t, row in wide.iterrows():
            current = row.to_numpy(float)
            common = np.isfinite(first) & np.isfinite(current)
            distance = float(np.linalg.norm(current[common] - first[common]))
            pearson = (
                float(np.corrcoef(current[common], first[common])[0, 1])
                if common.sum() > 1
                and np.std(current[common]) > 1e-12
                and np.std(first[common]) > 1e-12
                else np.nan
            )
            denom = max(float(np.linalg.norm(first[common])), 1e-12)
            adjacent = (
                np.nan
                if previous is None
                else float(
                    np.linalg.norm(
                        current[np.isfinite(current) & np.isfinite(previous)]
                        - previous[np.isfinite(current) & np.isfinite(previous)]
                    )
                )
            )
            sampled = (
                cells[np.isclose(pd.to_numeric(cells.time, errors="coerce"), t, rtol=0, atol=1e-6)]
                if not cells.empty and "time" in cells
                else pd.DataFrame()
            )
            refdist = pd.to_numeric(
                sampled.get("reference_distance", pd.Series(dtype=float)), errors="coerce"
            )
            records.append(
                {
                    "model": item["source"].get("model"),
                    "time": float(t),
                    "distance_to_reference_stage": distance,
                    "relative_distance_to_reference_stage": distance / denom,
                    "pseudobulk_pcc_to_reference_stage": pearson,
                    "adjacent_stage_distance": adjacent,
                    "observed_reference_pc_distance_median": (
                        float(refdist.median()) if len(refdist) else np.nan
                    ),
                    "n_common_genes": int(common.sum()),
                    "reference_time": reference_time,
                    "reference": reference,
                    "reference_source_id": (
                        _reference_metadata(boundary).get("source_id")
                        if boundary is not None
                        else item["source"].get("source_id")
                    ),
                    "reference_match": reference_match,
                    "scale_group": _reference_metadata(item).get("scale_group"),
                }
            )
            previous = current
    return pd.DataFrame(records)


def _interpolation_table(items):
    records = []
    for item in items:
        if item["source"].get("role") in ("GT", "REFERENCE"):
            continue
        stage = _stage(item)
        if stage.empty or not {"gene", "mean"}.issubset(stage):
            continue
        wide = stage.pivot_table(index="time", columns="gene", values="mean").sort_index()
        if len(wide) < 2:
            continue
        start, end = wide.iloc[0].to_numpy(float), wide.iloc[-1].to_numpy(float)
        direction = end - start
        valid = np.isfinite(direction)
        denom = float(np.dot(direction[valid], direction[valid]))
        for time, row in wide.iterrows():
            values = row.to_numpy(float)
            delta = values - start
            common = valid & np.isfinite(delta)
            progress = (
                float(np.dot(delta[common], direction[common]) / denom) if denom > 1e-12 else np.nan
            )
            deviation = (
                float(np.linalg.norm(delta[common] - progress * direction[common]))
                if np.isfinite(progress)
                else np.nan
            )
            records.append(
                {
                    "model": item["source"].get("model"),
                    "time": time,
                    "progress_between_first_last_saved_stages": progress,
                    "orthogonal_expression_deviation": deviation,
                    "global_mean_expression": float(np.nanmean(values)),
                    "pseudobulk_gene_variance": float(np.nanvar(values)),
                    "n_common_genes": int(common.sum()),
                    "reference": "first/last saved stages; generated versus observed anchor status exported separately",
                }
            )
    return pd.DataFrame(records)


def _temporal_descriptor_table(items, genes):
    """Exact saved-grid expression descriptors; never label crossing as onset."""
    records = []
    for item in items:
        if item["source"].get("role") in ("GT", "REFERENCE"):
            continue
        stage = _stage(item)
        if stage.empty or not {"gene", "mean", "time"}.issubset(stage):
            continue
        for gene, group in stage[stage.gene.astype(str).isin(genes)].groupby("gene", sort=True):
            group = group.sort_values("time")
            times = group.time.to_numpy(float)
            values = group["mean"].to_numpy(float)
            finite = np.isfinite(times) & np.isfinite(values)
            times = times[finite]
            values = values[finite]
            if not len(times):
                continue
            span = float(np.ptp(values))
            net = float(values[-1] - values[0])
            crossing = np.nan
            status = "insufficient_timepoints"
            if len(times) >= 3:
                if span <= 1e-12:
                    status = "constant"
                elif abs(net) <= 1e-12:
                    status = "no_net_direction"
                else:
                    sign = 1 if net > 0 else -1
                    half = (float(values.min()) + float(values.max())) / 2
                    status = (
                        "already_beyond_half_at_first_time"
                        if sign * (values[0] - half) >= 0
                        else "no_crossing"
                    )
                    if status == "no_crossing":
                        for i in range(1, len(times)):
                            if sign * (values[i - 1] - half) < 0 <= sign * (values[i] - half):
                                crossing = float(
                                    times[i - 1]
                                    + (half - values[i - 1])
                                    / (values[i] - values[i - 1])
                                    * (times[i] - times[i - 1])
                                )
                                status = "first_crossing"
                                break
            records.append(
                {
                    "model": item["source"].get("model"),
                    "gene": str(gene),
                    "n_saved_timepoints": len(times),
                    "first_saved_time": float(times[0]),
                    "last_saved_time": float(times[-1]),
                    "peak_time": float(times[np.argmax(values)]),
                    "net_change": net,
                    "expression_range": span,
                    "half_range_crossing_time": crossing,
                    "crossing_status": status,
                    "basis": "saved stage-gene means; half-range crossing is grid-dependent, not validated biological onset",
                    "has_unmeasured_truth": False,
                }
            )
    return pd.DataFrame(records)


def _rank_stability_table(items, genes):
    records = []
    for item in items:
        if item["source"].get("role") in ("GT", "REFERENCE"):
            continue
        stage = _stage(item)
        if stage.empty or not {"gene", "mean"}.issubset(stage):
            continue
        wide = (
            stage[stage.gene.astype(str).isin(genes)]
            .pivot_table(index="time", columns="gene", values="mean")
            .sort_index()
        )
        for i in range(1, len(wide)):
            left = wide.iloc[i - 1]
            right = wide.iloc[i]
            finite = np.isfinite(left) & np.isfinite(right)
            a = left[finite].rank(method="average").to_numpy()
            b = right[finite].rank(method="average").to_numpy()
            rho = (
                float(np.corrcoef(a, b)[0, 1])
                if len(a) >= 3 and np.std(a) > 0 and np.std(b) > 0
                else np.nan
            )
            records.append(
                {
                    "model": item["source"].get("model"),
                    "left_time": float(wide.index[i - 1]),
                    "right_time": float(wide.index[i]),
                    "n_common_selected_genes": int(finite.sum()),
                    "adjacent_rank_spearman": rho,
                    "basis": "rank stability among the declared selected genes; high stability can also indicate frozen expression",
                }
            )
    return pd.DataFrame(records)


def _rollout_coverage_table(items):
    records = []
    for item in items:
        source = _reference_metadata(item)
        if source.get("role") in ("GT", "REFERENCE") or source.get("scenario") not in (
            "history",
            "future",
        ):
            continue
        attrs = dict(source.get("source_metadata") or source.get("metadata") or {})
        completed = attrs.get("rollout_completed_steps", attrs.get("rollout_steps"))
        requested = attrs.get("rollout_steps_requested_effective", completed)
        # Undeclared horizons remain missing instead of being inferred from a
        # drawing sample whose timepoints may not cover every saved step.
        if completed is None or requested is None:
            continue
        completed, requested = int(completed), int(requested)
        stage = _stage(item)
        times = np.sort(stage.time.unique()) if not stage.empty else np.array([])
        history = source.get("scenario") == "history"
        records.append(
            {
                "model": source.get("model"),
                "scenario": source.get("scenario"),
                "completed_steps": completed,
                "requested_steps": requested,
                "fraction_horizon_retained": completed / requested if requested else np.nan,
                "full_requested_horizon_available": completed == requested,
                "last_retained_time": (
                    float(times[0] if history else times[-1]) if len(times) else np.nan
                ),
                "numerical_stability_status": attrs.get("numerical_stability_status", "undeclared"),
                "evaluation_view_schema": attrs.get("evaluation_view_schema", ""),
                "evaluation_stop_reason": attrs.get("evaluation_stop_reason", ""),
                "first_nonfinite_step": attrs.get("first_nonfinite_step", np.nan),
                "history_target_time_reached": (
                    attrs.get("history_target_time_reached", False) if history else ""
                ),
                "claim": "retained finite horizon coverage; numerical stability is not biological accuracy",
            }
        )
    return pd.DataFrame(records)


def _temporal_views(items, genes, output, options, title, mode):
    if mode == "rollout_coverage":
        table = _rollout_coverage_table(items)
    elif mode == "rank_stability":
        table = _rank_stability_table(items, genes)
    else:
        table = _temporal_descriptor_table(items, genes)
    if table.empty:
        return []
    table.to_csv(output / f"{mode}_summary.csv", index=False)
    if (
        mode == "rank_stability"
        and not np.isfinite(pd.to_numeric(table.adjacent_rank_spearman, errors="coerce")).any()
    ):
        return []

    def factory(item):
        frame = table[table.model == item["source"].get("model")]

        def draw(ax, legend=True):
            if frame.empty:
                ax.text(
                    0.5,
                    0.5,
                    "Observed context; descriptors apply to generated populations",
                    ha="center",
                    va="center",
                    wrap=True,
                    transform=ax.transAxes,
                    fontsize=7,
                )
                ax.set_axis_off()
                return
            if mode == "rank_stability":
                ax.plot(frame.right_time, frame.adjacent_rank_spearman, "o-", color="#397F86", ms=3)
                ax.set_ylim(-1.05, 1.05)
                ax.set(xlabel="Right saved time", ylabel="Adjacent selected-gene rank correlation")
            elif mode == "rollout_coverage":
                row = frame.iloc[0]
                ax.barh(
                    ["Requested", "Retained"],
                    [row.requested_steps, row.completed_steps],
                    color=["#969FA7", "#397F86"],
                )
                ax.set(
                    xlabel="Complete population steps",
                    title=f"{row.completed_steps}/{row.requested_steps} retained",
                )
                ax.text(
                    0.02,
                    0.02,
                    str(row.numerical_stability_status).replace("_", " "),
                    transform=ax.transAxes,
                    fontsize=6,
                    wrap=True,
                )
            else:
                ypos = np.arange(len(frame))
                ax.scatter(frame.peak_time, ypos, c="#397F86", s=18, label="Saved-grid peak")
                crossing = pd.to_numeric(frame.half_range_crossing_time, errors="coerce")
                keep = np.isfinite(crossing)
                ax.scatter(
                    crossing[keep],
                    ypos[keep],
                    c="#BA7146",
                    s=20,
                    marker="x",
                    label="Half-range crossing",
                )
                ax.set_yticks(ypos, frame.gene, fontsize=6)
                ax.set(xlabel="Saved reporting time", ylabel="Selected gene")
                if legend:
                    ax.legend(fontsize=6)

        return draw

    drawable = [item for item in items if item["source"].get("role") not in ("GT", "REFERENCE")]
    return _comparisons(
        drawable, factory, output / mode, options, title + " | " + mode.replace("_", " "), mode
    )


def _diagnostic_views(items, output, options, title, mode, references=None):
    table = (
        _interpolation_table(items)
        if mode == "interpolation_progress"
        else (
            _drift_table(items, references)
            if mode in ["drift", "horizon"]
            else _diversity_table(items)
        )
    )
    if table.empty:
        return []
    table.to_csv(output / f"{mode}_summary.csv", index=False)
    metrics = [
        c
        for c in table.columns
        if c not in ["model", "time", "n_genes", "n_common_genes", "reference", "reference_time"]
        and pd.api.types.is_numeric_dtype(table[c])
    ]
    if mode == "history_heatmap":
        metrics = ["relative_distance_to_reference_stage", "adjacent_stage_distance"]
    figures = []
    for metric in metrics:
        if mode == "history_heatmap":
            wide = table.pivot_table(index="model", columns="time", values=metric)
            fig, ax = plt.subplots(
                figsize=(max(6, wide.shape[1] * 0.22), max(3, wide.shape[0] * 0.35))
            )
            art = ax.imshow(wide.to_numpy(), aspect="auto", cmap="viridis")
            ax.set_yticks(
                range(len(wide)),
                [options.get("aliases", {}).get(str(m), str(m)) for m in wide.index],
                fontsize=7,
            )
            step = max(1, math.ceil(wide.shape[1] / 15))
            ax.set_xticks(
                range(0, wide.shape[1], step),
                [f"{v:g}" for v in wide.columns[::step]],
                rotation=45,
                fontsize=7,
            )
            fig.colorbar(art, ax=ax, label=metric.replace("_", " "))
            ax.set_xlabel("Saved history time")
            ax.set_title(title)
            figures.extend(_record(plot.save(fig, output / f"{mode}_{metric}", options), mode))
            plt.close(fig)
        else:

            def factory(item, m=metric):
                group = table[table.model == item["source"].get("model")].sort_values("time")

                def draw(ax, legend=True):
                    if group.empty:
                        ax.text(
                            0.5,
                            0.5,
                            "Observed context; diagnostic applies to generated populations",
                            ha="center",
                            va="center",
                            wrap=True,
                            fontsize=7,
                            transform=ax.transAxes,
                        )
                        ax.set_axis_off()
                        return
                    ax.plot(group.time, group[m], color="#397F86", lw=1.5, marker="o", ms=2.5)
                    ax.set_xlabel("Saved time")
                    ax.set_ylabel(m.replace("_", " "))
                    if m == "pseudobulk_pcc_to_reference_stage":
                        ax.set_ylim(-1.05, 1.05)

                return draw

            drawable = [item for item in items if item["source"].get("model") in set(table.model)]
            figures.extend(
                _comparisons(
                    drawable,
                    factory,
                    output / f"{mode}_{metric}",
                    options,
                    f'{title} | {metric.replace("_"," ")}',
                    mode,
                    metric,
                )
            )
    return figures


def _perturbation_table(cohort, options):
    frames, notes = [], []
    for source in plot.available_models(cohort, options):
        directory = Path(source.get("artifact_dir", ""))
        candidate = directory / "perturbation.parquet"
        if not candidate.is_file():
            summaries = source.get("summary_files", {})
            if isinstance(summaries, dict) and summaries.get("perturbation"):
                candidate = Path(summaries["perturbation"])
        if not candidate.is_file():
            notes.append(f"{source.get('model')}: no compact matched perturbation summary")
            continue
        frame = _read_table(candidate)
        if not {"intervention_gene", "response_gene", "time", "delta"}.issubset(frame.columns):
            notes.append(f"{candidate}: unsupported perturbation columns")
            continue
        frame["model"] = source.get("model", "unknown")
        frame["role"] = source.get("role", "FM")
        frame["panel"] = source.get("panel", "unknown")
        frame["artifact_dir"] = str(directory)
        frame["time"] = pd.to_numeric(frame.time, errors="coerce")
        frame["delta"] = pd.to_numeric(frame.delta, errors="coerce")
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), notes


def _perturbation_selection(table, options, output, cohort=None, gene_choices=None):
    if table.empty:
        return pd.DataFrame()
    eligible = None
    if cohort is not None:
        eligible = set()
        choices = (
            gene_choices
            if gene_choices is not None
            else select_genes(cohort, dict(options, max_entities=0))
        )
        if choices is not None and not choices.empty and "eligible" in choices:
            eligible = set(choices.loc[choices.eligible.astype(bool), "gene"].astype(str))
    grouped = table.groupby("intervention_gene", sort=False)
    rows = []
    for gene, group in grouped:
        response = group[~group.response_gene.astype(str).eq(str(gene))]
        score = (
            float(response.delta.abs().max())
            if not response.empty
            else float(group.delta.abs().max())
        )
        rows.append(
            {
                "gene": str(gene),
                "score": score,
                "effect": score,
                "p_value": np.nan,
                "q_value": np.nan,
                "reason": "Largest saved absolute downstream mean-expression response to a matched source-history edit; descriptive sensitivity, no perturbation truth/significance test",
                "n_models": group.model.nunique(),
                "eligible": eligible is None or str(gene) in eligible,
                "selected": eligible is None or str(gene) in eligible,
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["eligible", "score", "gene"], ascending=[False, False, True]
    )
    cap = int(options.get("max_entities", 0) or 0)
    if cap:
        indices = selection.index[selection.selected]
        selection.loc[indices[cap:], "selected"] = False
    selection.loc[
        ~selection.eligible, "reason"
    ] += "; excluded: absent from author/developmental evidence"
    selection.to_csv(output / "selected_entities.csv", index=False)
    return selection


def _response_selection(table, choices, interventions, options, output):
    """Audit evidence before ranking saved response effects, without p/q claims."""
    annotation = (
        choices.set_index("gene").to_dict("index")
        if choices is not None and not choices.empty
        else {}
    )
    exploratory = bool(options.get("allow_unannotated") or options.get("all_genes"))
    cap = int(options.get("max_response_genes", 20) or 0)
    all_timecourses = cap == 0 or cap > 20
    selected_targets = set(map(str, interventions))
    data = table.copy()
    data["_abs_delta"] = pd.to_numeric(data.delta, errors="coerce").abs()
    effects = (
        data.groupby(["intervention_gene", "response_gene"], sort=True, observed=True)
        .agg(
            effect=("_abs_delta", "max"), n_models=("model", "nunique"), n_times=("time", "nunique")
        )
        .reset_index()
    )
    rows = []
    chosen = {}
    for intervention, group in effects.groupby("intervention_gene", sort=True, observed=True):
        local = []
        for record in group.to_dict("records"):
            gene = str(record["response_gene"])
            evidence = annotation.get(gene, {})
            eligible = bool(evidence.get("eligible", False)) or exploratory
            label = str(evidence.get("reason", "unannotated exploratory feature")).split(";", 1)[0]
            if not eligible:
                label = "absent from supplied author/developmental evidence"
            local.append(
                {
                    **record,
                    "intervention_gene": str(intervention),
                    "gene": gene,
                    "annotation": label,
                    "evidence": evidence.get("evidence", ""),
                    "annotation_source": evidence.get("annotation_source", ""),
                    "evidence_location": evidence.get("evidence_location", ""),
                    "source_kind": evidence.get(
                        "source_kind", "exploratory" if exploratory else "unsupported"
                    ),
                    "eligible": eligible,
                    "intervention_selected": str(intervention) in selected_targets,
                    "is_direct_target": gene == str(intervention),
                    "selected": False,
                    "selected_for_timecourse": False,
                    "selected_for_network": False,
                    "p_value": np.nan,
                    "q_value": np.nan,
                    "selection_basis": "maximum saved absolute mean-expression delta across available models/ages; native units; descriptive sensitivity, no intervention truth/significance test",
                }
            )
        local.sort(
            key=lambda row: (
                not row["eligible"],
                -row["effect"] if np.isfinite(row["effect"]) else np.inf,
                row["gene"],
            )
        )
        eligible_rank = 0
        selected_genes = []
        for row in local:
            valid = row["eligible"] and np.isfinite(row["effect"])
            if valid:
                eligible_rank += 1
            row["response_rank"] = eligible_rank if valid else np.nan
            row["selected"] = (
                valid and row["intervention_selected"] and (cap == 0 or eligible_rank <= cap)
            )
            if row["selected"]:
                selected_genes.append(row["gene"])
                row["selected_for_timecourse"] = all_timecourses or eligible_rank <= 6
                row["selected_for_network"] = eligible_rank <= 20
            if not row["intervention_selected"]:
                why = "intervention not selected at its evidence/entity cap"
            elif not row["eligible"]:
                why = "excluded: no author/developmental evidence; --all-genes permits explicit exploration"
            elif not np.isfinite(row["effect"]):
                why = "excluded: no finite saved response effect"
            elif not row["selected"]:
                why = "outside max-response-genes cap"
            else:
                why = "effect-ranked after biological evidence screen; significance not established"
            row["reason"] = row["annotation"] + "; " + why
            rows.append(row)
        chosen[str(intervention)] = selected_genes
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "RESPONSE_GENE_SELECTION.csv", index=False)
    return audit, chosen


def _response_genes(table, intervention, options):
    choices = options.get("_response_gene_choices")
    if choices is not None:
        return list(choices.get(str(intervention), []))
    subset = table[table.intervention_gene.astype(str) == str(intervention)]
    order = (
        subset.groupby("response_gene")
        .delta.apply(lambda s: float(s.abs().max()))
        .sort_values(ascending=False)
    )
    cap = int(options.get("max_response_genes", 20) or 0)
    genes = order.index.astype(str).tolist()
    return genes[:cap] if cap else genes


def _perturbation_draw(table, model, intervention, mode, options):
    data = table[
        (table.model == model) & (table.intervention_gene.astype(str) == str(intervention))
    ].copy()
    response_genes = options.get(
        "_response_page_genes", _response_genes(table, intervention, options)
    )

    def draw(ax, legend=True):
        if data.empty:
            ax.text(
                0.5,
                0.5,
                "Intervention absent in native panel",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        if mode == "perturbation_heatmap":
            wide = (
                data[data.response_gene.astype(str).isin(response_genes)]
                .pivot_table(index="response_gene", columns="time", values="delta")
                .reindex([g for g in response_genes if g in set(data.response_gene.astype(str))])
            )
            if wide.empty:
                ax.text(
                    0.5,
                    0.5,
                    "Selected response genes outside native panel",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                return
            limit = max(
                float(
                    options.get(
                        "_response_limit", np.nanmax(np.abs(wide.to_numpy())) if wide.size else 0
                    )
                ),
                1e-8,
            )
            art = ax.imshow(
                wide.to_numpy(),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            ax.set_yticks(range(len(wide)), wide.index, fontsize=6)
            step = max(1, math.ceil(wide.shape[1] / 12))
            ax.set_xticks(
                range(0, wide.shape[1], step),
                [f"{v:g}" for v in wide.columns[::step]],
                rotation=45,
                fontsize=6,
            )
            if legend:
                plt.colorbar(art, ax=ax, fraction=0.05, pad=0.03, label="Edited − baseline mean")
            ax.set_ylabel("Saved response gene")
            ax.set_xlabel("Saved target time")
        elif mode == "perturbation_targets":
            direct = data[data.response_gene.astype(str) == str(intervention)].sort_values("time")
            if direct.empty:
                ax.text(
                    0.5,
                    0.5,
                    "Direct target outside response gene panel",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                return
            if {"baseline_mean", "edited_mean"}.issubset(direct.columns):
                ax.plot(
                    direct.time,
                    direct.baseline_mean,
                    color="#68717B",
                    lw=1.5,
                    marker="o",
                    ms=3,
                    label="Matched baseline",
                )
                ax.plot(
                    direct.time,
                    direct.edited_mean,
                    color="#A34C65",
                    lw=1.5,
                    marker="o",
                    ms=3,
                    label="Saved edit",
                )
                ax.set_ylabel("Stored expression mean")
            else:
                ax.plot(
                    direct.time,
                    direct.delta,
                    color="#A34C65",
                    marker="o",
                    ms=3,
                    label="Edited − baseline",
                )
                ax.set_ylabel("Mean-expression difference")
            ax.set_xlabel("Saved target time")
            if legend:
                ax.legend(fontsize=6)
        else:
            # Only plot covered response means, not artificial individual cell distributions.
            if not data.response_gene.astype(str).isin(response_genes).any():
                ax.text(
                    0.5,
                    0.5,
                    "Selected response genes outside native panel",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                return
            for gene in response_genes[: min(len(response_genes), 6)]:
                group = data[data.response_gene.astype(str) == gene].sort_values("time")
                if not group.empty:
                    ax.plot(group.time, group.delta, lw=1.2, marker="o", ms=2.5, label=gene)
            ax.axhline(0, color="#7F8991", lw=0.6)
            ax.set_xlabel("Saved target time")
            ax.set_ylabel("Edited − baseline mean")
            if legend:
                ax.legend(fontsize=6, ncol=2)

    return draw


def _configured_annotation(options, cohort, key):
    config = options.get("configuration", options.get("config", {}))
    datasets = options.get("datasets", config.get("datasets", {}))
    entry = datasets.get(cohort.get("dataset"), {}) if isinstance(datasets, dict) else {}
    path = options.get(key, entry.get(key))
    return Path(path) if path else None


def _provided_edges(options, cohort):
    path = _configured_annotation(options, cohort, "edges_path")
    if path is None or not path.is_file():
        return None
    edges = _read_table(path)
    columns = {str(column).lower(): column for column in edges}
    left = next(
        (
            columns[name]
            for name in ("source", "regulator", "tf", "source_gene", "regulator_gene")
            if name in columns
        ),
        None,
    )
    right = next(
        (columns[name] for name in ("target", "gene", "target_gene") if name in columns), None
    )
    return (
        edges.rename(columns={left: "source", right: "target"})
        if left is not None and right is not None
        else None
    )


def _network_draw(table, model, intervention, edges, options):
    subset = table[
        (table.model == model) & (table.intervention_gene.astype(str) == str(intervention))
    ]
    responses = (
        subset.groupby("response_gene").delta.apply(lambda x: float(x.loc[x.abs().idxmax()]))
        if not subset.empty
        else pd.Series(dtype=float)
    )
    genes = set(_response_genes(table, intervention, options)[:20]) | {str(intervention)}
    selected = edges[edges.source.astype(str).isin(genes) & edges.target.astype(str).isin(genes)]
    # Focus on declared intervention's real outgoing prior edges first; both endpoints
    # are selected gene responses. No edge is inferred from plotting correlations.
    nodes = sorted(set(selected.source.astype(str)) | set(selected.target.astype(str)))

    def draw(ax, legend=True):
        if not nodes:
            ax.text(
                0.5,
                0.5,
                "No supplied regulatory edges among covered response genes",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            return
        angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        pos = {g: (np.cos(a), np.sin(a)) for g, a in zip(nodes, angles)}
        for _, edge in selected.iterrows():
            start, end = pos[str(edge.source)], pos[str(edge.target)]
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#A8AFB5",
                    "alpha": 0.6,
                    "lw": 0.7,
                    "shrinkA": 10,
                    "shrinkB": 10,
                },
            )
        vals = np.array([responses.get(g, np.nan) for g in nodes])
        limit = max(float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 0, 1e-8)
        artist = ax.scatter(
            [pos[g][0] for g in nodes],
            [pos[g][1] for g in nodes],
            c=vals,
            cmap="RdBu_r",
            norm=Normalize(-limit, limit),
            s=[100 if g == str(intervention) else 55 for g in nodes],
            edgecolors="#424B53",
            linewidths=0.6,
        )
        for g in nodes:
            ax.text(pos[g][0] * 1.15, pos[g][1] * 1.15, g, ha="center", va="center", fontsize=6)
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.axis("off")
        if legend:
            plt.colorbar(
                artist, ax=ax, fraction=0.05, pad=0.03, label="Peak signed saved mean response"
            )

    return draw


def _perturbation_programs(table, cohort, options, output, title):
    records = _program_records(cohort, [], options)
    figures = []
    rows = []
    for program in records:
        name = str(program.get("program", program.get("name", "program")))
        genes = program.get("genes", [])
        if isinstance(genes, str):
            genes = [x for x in re.split("[,;| ]+", genes) if x]
        data = table[table.response_gene.astype(str).isin(genes)]
        if data.empty:
            continue
        agg = _weighted_program_rows(
            data, program, "response_gene", "delta", ["model", "intervention_gene", "time"]
        ).rename(columns={"weighted_score": "mean_response"})
        if agg.empty:
            continue
        agg["program"] = name
        rows.append(agg)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        # The caller has already selected biologically eligible intervention
        # targets. Program effects must not independently replace that choice.
        ranks = agg.intervention_gene.unique()
        for (model, intervention), group in agg[agg.intervention_gene.isin(ranks)].groupby(
            ["model", "intervention_gene"]
        ):
            ax.plot(
                group.time,
                group.mean_response,
                lw=1,
                alpha=0.7,
                label=f'{options.get("aliases",{}).get(model,model)} | {intervention}',
            )
        ax.axhline(0, color="#7F8991", lw=0.6)
        ax.set_xlabel("Saved target time")
        ax.set_ylabel("Weighted edited − baseline over covered genes")
        ax.set_title(f"{title} | {name}")
        if len(ranks) * agg.model.nunique() <= 12:
            ax.legend(fontsize=6, ncol=2)
        figures.extend(
            _record(
                plot.save(fig, output / f"program_response_{_safe(name)}", options),
                "perturbation_programs",
                name,
            )
        )
        plt.close(fig)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            output / "perturbation_program_response.csv", index=False
        )
    return figures


def _render_perturbation(question, cohort, output, options, progress):
    table, notes = _perturbation_table(cohort, options)
    if table.empty:
        return {
            "status": "unsupported",
            "figures": [],
            "tables": [],
            "notes": notes
            + [
                "No saved matched intervention/baseline mean summaries. Ordinary predictions cannot substitute."
            ],
        }
    _model_rows(cohort, options, output)
    gene_options = dict(
        options,
        max_entities=0,
        allow_unannotated=bool(options.get("allow_unannotated") or options.get("all_genes")),
    )
    gene_choices = select_genes(cohort, gene_options)
    selection = _perturbation_selection(table, options, output, cohort, gene_choices)
    interventions = selection.loc[selection.selected, "gene"].tolist()
    response_audit, response_choices = _response_selection(
        table, gene_choices, interventions, options, output
    )
    options = dict(options, _response_gene_choices=response_choices)
    if not interventions:
        notes.append(
            "No saved intervention target satisfies the declared author/developmental evidence filter; inspect selected_entities.csv or supply author_genes_path/gene_evidence_path."
        )
    title = question.get("title", "Saved source-history intervention sensitivity")
    views = [
        ALIASES.get(v, v)
        for v in question.get(
            "views", ["perturbation_heatmap", "perturbation_timecourse", "perturbation_targets"]
        )
    ]
    cap = int(options.get("max_views_per_entity", 0) or 0)
    if cap:
        views = views[:cap]
    sources = {s["model"]: s for s in plot.available_models(cohort, options)}
    figures = []
    edges = _provided_edges(options, cohort)
    # Selected response evidence remains complete across models/time, not only top plot genes.
    selected_table = table[table.intervention_gene.astype(str).isin(interventions)]
    selected_table.to_parquet(output / "selected_intervention_responses.parquet", index=False)
    for view in views:
        if view not in (
            "perturbation_programs",
            "perturbation_network",
            "perturbation_heatmap",
            "perturbation_timecourse",
            "perturbation_targets",
        ):
            notes.append(f"Unsupported perturbation view key: {view}")
            continue
        if view == "perturbation_programs":
            f = _perturbation_programs(selected_table, cohort, options, output, title)
            figures.extend(f)
            if not f:
                notes.append("No documented covered gene programs for perturbation program views.")
            _notify(progress, view, figures=len(figures))
            continue
        if view == "perturbation_network" and edges is None:
            notes.append(
                "Perturbation network unavailable: supply edges_path with genuine regulatory source/target edges."
            )
            continue
        for intervention in interventions:
            local_options = dict(options)
            sub = table[table.intervention_gene.astype(str) == str(intervention)]
            response_genes = _response_genes(table, intervention, options)
            if (
                view in ("perturbation_heatmap", "perturbation_timecourse", "perturbation_network")
                and not response_genes
            ):
                notes.append(
                    f"{intervention}: no finite response gene satisfies biological evidence and the response cap; no empty {view} figure exported."
                )
                continue
            if view == "perturbation_network":
                covered = set(response_genes[:20]) | {str(intervention)}
                matching = edges[
                    edges.source.astype(str).isin(covered) & edges.target.astype(str).isin(covered)
                ]
                if matching.empty:
                    notes.append(
                        f"{intervention}: no supplied regulatory edge connects covered selected response genes; network not fabricated."
                    )
                    continue
            plotted = sub[sub.response_gene.astype(str).isin(response_genes)]
            local_options["_response_limit"] = max(
                float(plotted.delta.abs().max()) if not plotted.empty else 0.0, 1e-8
            )
            response_cap = int(options.get("max_response_genes", 20) or 0)
            if view == "perturbation_heatmap":
                pages = [response_genes[i : i + 20] for i in range(0, len(response_genes), 20)]
            elif view == "perturbation_timecourse" and (response_cap == 0 or response_cap > 20):
                pages = [response_genes[i : i + 6] for i in range(0, len(response_genes), 6)]
            elif view == "perturbation_timecourse":
                pages = [response_genes[:6]]
            else:
                pages = [None]
            for page, genes in enumerate(pages, 1):
                page_options = dict(local_options)
                if genes is not None:
                    page_options["_response_page_genes"] = genes
                panels = []
                for model, source in sources.items():
                    draw = (
                        _network_draw(sub, model, intervention, edges, options)
                        if view == "perturbation_network"
                        else _perturbation_draw(sub, model, intervention, view, page_options)
                    )
                    panels.append((_label(source, options), source.get("role", "FM"), draw))
                base = output / f"{view}__{_safe(intervention)}"
                if len(pages) > 1:
                    base = output / f"{view}__{_safe(intervention)}__page_{page:03d}"
                caption = f"{title} | source-history edit: {intervention}" + (
                    f" | page {page}/{len(pages)}" if len(pages) > 1 else ""
                )
                paths = (
                    plot.compare_mosaics(panels, base, page_options, title=caption)
                    if hasattr(plot, "compare_mosaics")
                    else plot.mosaic(
                        [(label, draw) for label, role, draw in panels],
                        base,
                        page_options,
                        title=caption,
                    )
                )
                records = _record(paths, view, intervention)
                for record in records:
                    record.update(
                        page=page, n_pages=len(pages), genes=list(genes or response_genes[:20])
                    )
                figures.extend(records)
                _notify(progress, view, intervention, len(figures))
    for source in plot.available_models(cohort, options):
        warnings = source.get("warning", [])
        if isinstance(warnings, str):
            warnings = [warnings]
        notes.extend(f'{source.get("model")}: {w}' for w in warnings)
    notes.append(
        "Compact responses contain matched population means/variances only; all differences are descriptive model sensitivity without measured intervention truth."
    )
    notes.append(
        "Response genes are evidence screened then effect ranked; RESPONSE_GENE_SELECTION.csv records every saved candidate. Direct targets may appear in response heatmaps. Default response timecourses show the first six selected genes; response caps 0 or >20 paginate complete selected timecourses. Prior-network views use at most the first 20 selected response genes plus the intervention target."
    )
    return {
        "status": _view_status(figures, views, notes),
        "figures": figures,
        "tables": [str(p) for p in output.glob("*.csv")]
        + [str(p) for p in output.glob("*.parquet")],
        "notes": notes,
    }


ALIASES = {
    "temporal_geometry": "geometry",
    "future_geometry": "geometry",
    "history_geometry": "geometry",
    "reverse_geometry": "geometry",
    "temporal_trajectory": "trajectory",
    "centroid_trajectory": "trajectory",
    "temporal_storyboard": "storyboard",
    "expression_timecourse": "gene_timecourse",
    "history_expression": "gene_timecourse",
    "reverse_expression": "gene_timecourse",
    "history_timecourse": "gene_timecourse",
    "history_heatmap": "gene_heatmap",
    "heatmap": "gene_heatmap",
    "dotplot": "marker_dotplot",
    "gene_program_timecourse": "programs",
    "future_diversity": "diversity",
    "future_drift": "drift",
    "intervention_heatmap": "perturbation_heatmap",
    "intervention_timecourse": "perturbation_timecourse",
    "direct_target": "perturbation_targets",
}


def render_question(
    question: dict, cohort: dict, output: Path, options: dict, progress=None
) -> dict:
    """Render a supported biological question from prepared saved artifacts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plot.style()
    scenario = _scenario(cohort)
    if scenario == "perturbation" or str(question.get("family", "")).startswith("perturbation"):
        return _render_perturbation(question, cohort, output, options, progress)
    items, notes = _artifacts(cohort, options)
    if not items:
        return {
            "status": "unsupported",
            "figures": [],
            "tables": [],
            "notes": notes + ["No readable prepared scenario artifacts."],
        }
    boundary_references, boundary_notes = _boundary_references(cohort, options, items)
    notes.extend(boundary_notes)
    _model_rows(cohort, options, output, items, boundary_references)
    choices, genes = _gene_choices(cohort, items, options, output)
    policies = sorted(
        {str(item.get("meta", {}).get("reference_policy", "unspecified")) for item in items}
    )
    time_kinds = sorted(
        {str(item.get("meta", {}).get("time_kind", "unspecified")) for item in items}
    )
    notes.append(
        "Saved time-axis kinds: "
        + ", ".join(time_kinds)
        + ". Reference policies: "
        + "; ".join(policies)
        + "."
    )
    views = [
        ALIASES.get(v, v)
        for v in question.get("views", ["geometry", "trajectory", "gene_timecourse"])
    ]
    cap = int(options.get("max_views_per_entity", 0) or 0)
    if cap:
        views = views[:cap]
    title = question.get("title", "Saved biological scenario")
    figures = []
    if not genes:
        notes.append(
            "No selected genes satisfy the declared author/developmental evidence and current signal/cap policy; geometric biology remains drawable."
        )
    if genes and any(
        v in views
        for v in (
            "gene_timecourse",
            "gene_distribution",
            "gene_heatmap",
            "marker_dotplot",
            "temporal_descriptors",
            "rank_stability",
        )
    ):
        evidence = []
        for item in items:
            stage = _stage(item)
            if stage.empty or "gene" not in stage:
                continue
            frame = stage[stage.gene.astype(str).isin(genes)].copy()
            source = _reference_metadata(item)
            frame["model"] = source.get("model")
            frame["source_id"] = source.get("source_id")
            frame["role"] = source.get("role")
            frame["native_panel"] = source.get("panel")
            frame["scenario"] = source.get("scenario")
            frame["time_kind"] = source.get("time_kind", "undeclared")
            frame["basis"] = (
                "streamed stage-gene statistics; cell SD is dispersion, not independent biological replicate uncertainty"
            )
            evidence.append(frame)
        if evidence:
            pd.concat(evidence, ignore_index=True).to_csv(
                output / "selected_gene_timecourses.csv", index=False
            )
    all_times = [
        float(t)
        for item in items
        for t in _cells(item).get("time", pd.Series(dtype=float)).unique()
    ]
    if all_times:
        options = dict(options, _global_time_range=(min(all_times), max(all_times)))
    for view in views:
        if view in ["geometry", "trajectory", "density"]:
            if not any(not _cells(item).empty for item in items):
                notes.append(
                    f"{view}: no finite saved reference coordinates; no empty placeholder exported."
                )
                continue
            figures.extend(
                _comparisons(
                    items,
                    lambda a: _geometry(a, options, view),
                    output / view,
                    options,
                    title,
                    view,
                )
            )
            _notify(progress, view, figures=len(figures))
        elif view == "storyboard":
            figures.extend(_storyboard(items, output, options, title, progress, len(figures)))
            _notify(progress, view, figures=len(figures))
        elif view == "animation":
            figures.extend(_animation(items, output, options, title, progress, len(figures)))
            _notify(progress, view, figures=len(figures))
        elif view in ["gene_timecourse", "gene_distribution"]:
            for gene in genes:
                factory = (
                    (lambda a, g=gene: _gene_line(a, g))
                    if view == "gene_timecourse"
                    else (lambda a, g=gene: _gene_distribution(a, g))
                )
                figures.extend(
                    _comparisons(
                        items,
                        factory,
                        output / f"{view}__{_safe(gene)}",
                        options,
                        f"{title} | {gene}",
                        view,
                        gene,
                    )
                )
                _notify(progress, view, gene, len(figures))
        elif view in ["gene_heatmap", "marker_dotplot"]:
            if not genes:
                notes.append(
                    f"{view}: no biologically eligible selected genes; no empty placeholder figure exported."
                )
                continue
            means = [
                _stage(a).loc[_stage(a).gene.astype(str).isin(genes), "mean"].to_numpy(float)
                for a in items
                if not _stage(a).empty and "gene" in _stage(a)
            ]
            values = np.concatenate(means) if means else np.array([])
            values = values[np.isfinite(values)]
            limits = (float(values.min()), float(values.max())) if len(values) else None
            figures.extend(
                _comparisons(
                    items,
                    lambda a: _expression_heatmap(a, genes, view, limits),
                    output / view,
                    options,
                    title,
                    view,
                )
            )
            _notify(progress, view, figures=len(figures))
        elif view == "programs":
            f = _program_views(cohort, items, output, options, title)
            figures.extend(f)
            if not f:
                notes.append(
                    "No documented covered programs supplied; native gene biology views remain available."
                )
            _notify(progress, view, figures=len(figures))
        elif view in ["diversity", "drift", "horizon", "interpolation_progress"]:
            figures.extend(
                _diagnostic_views(items, output, options, title, view, boundary_references)
            )
            _notify(progress, view, figures=len(figures))
        elif view in ["temporal_descriptors", "rank_stability", "rollout_coverage"]:
            figures.extend(_temporal_views(items, genes, output, options, title, view))
            _notify(progress, view, figures=len(figures))
        else:
            notes.append(f"Unsupported scenario view key: {view}")
    # Export sampled centroid paths as biological data behind trajectory views.
    centers = []
    for item in items:
        cells = _cells(item)
        if not cells.empty:
            frame = cells.groupby("time")[["x", "y"]].mean().reset_index()
            frame["model"] = item["source"].get("model")
            frame["reference_policy"] = item.get("meta", {}).get(
                "reference_policy", item["source"].get("reference_policy", "unspecified")
            )
            frame["n_displayed_cells"] = cells.groupby("time").size().reindex(frame.time).to_numpy()
            centers.append(frame)
    if centers:
        pd.concat(centers, ignore_index=True).to_csv(
            output / "displayed_population_centroids.csv", index=False
        )
    notes.append(
        "Geometry is descriptive projection into one saved observed reference. Centroids/distributions use the declared display sample; stage-gene means/diversity use streamed summary rows."
    )
    if scenario in ["future", "history"]:
        notes.append(
            "No independent truth exists for unmeasured rollout stages. Drift uses an explicit saved observed boundary when available, otherwise the first saved prediction stage; the CSV identifies the reference. These are stress diagnostics, not accuracy scores."
        )
    return {
        "status": _view_status(figures, views, notes),
        "figures": figures,
        "tables": [str(p) for p in output.glob("*.csv")],
        "notes": notes,
    }
