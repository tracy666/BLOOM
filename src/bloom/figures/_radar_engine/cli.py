#!/usr/bin/env python3
"""Reference-style BLOOM radars from saved benchmark CSVs; no inference."""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import argparse
import html
import json
import sys
import time
from bloom.figures._radar_engine.bloom_benchmark.data import benchmark_input, write_csv, write_json
from bloom.figures._radar_engine.bloom_radar.data import prepare, make_jobs


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "audit"):
        s = sub.add_parser(name)
        s.add_argument("--input", required=True, help="Directory or ZIP of exported benchmark CSVs")
        s.add_argument("--config", default=str(Path(__file__).parent / "data/radar.json"))
        s.add_argument("--out-root", required=True)
        s.add_argument(
            "--tag", help="Optional fresh child folder name; existing folders are refused"
        )
        s.add_argument("--panel", choices=["all", "hvg", "author"], default="all")
        s.add_argument("--dataset", choices=["all", "big", "small", "sted"], default="all")
        s.add_argument(
            "--no-poor-filter",
            action="store_true",
            help="Keep all complete nonconstant candidate metrics in Version 5",
        )
        if name == "run":
            s.add_argument("--versions", default="all", help="all or comma-separated 1,2,3,4,5")
            s.add_argument("--formats", default="png,pdf,svg")
            s.add_argument(
                "--workers",
                default="auto",
                help="auto uses up to four independent figure processes",
            )
            s.add_argument("--scale", choices=["range", "rank"], default="range")
            s.add_argument(
                "--no-singles",
                action="store_true",
                help="Only combined/triptych views; omit extra individual datasets",
            )
            s.add_argument("--no-legend", action="store_true")
            s.add_argument("--no-scale-ticks", action="store_true")
    return p


def _draw(payload):
    job, out, config, formats = payload
    from bloom.figures._radar_engine.bloom_radar.plots import render_figure

    entry = render_figure(job, out, config, formats)
    if isinstance(entry["files"], dict):
        entry["files"] = list(entry["files"].values())
    return entry


def _workers(value, n_jobs):
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    if value == "auto":
        return min(4, cpus, max(1, n_jobs))
    result = int(value)
    if result < 1:
        raise ValueError("--workers must be auto or a positive integer")
    return min(result, max(1, n_jobs))


def _export_tables(out, data, config):
    write_csv(out / "RADAR_VALUES.csv", data["frame"])
    write_csv(out / "CORE_OVERALL_RANKINGS.csv", data["core"]["rankings"])
    write_csv(out / "CORE_NORMALIZATION_SENSITIVITY.csv", data["core"]["sensitivity"])
    write_csv(out / "CORE_ASPECT_WEIGHTS.csv", data["core"]["weights"])
    write_csv(out / "CORE_NORMALIZATION.csv", data["core"]["normalization"])
    write_csv(out / "CANDIDATE_NORMALIZATION.csv", data["broad"]["normalization"])
    write_csv(
        out / "AVERAGE_DISPLAY_RANGES.csv",
        [
            {
                "dataset": r["dataset"],
                "panel": r["panel"],
                "metric": "overall",
                "display_low": r["display_low"],
                "display_high": r["display_high"],
            }
            for r in data["frame"]
            if r["metric"] == "overall" and r["model_label"] == "BLOOM"
        ],
    )
    write_csv(out / "METRIC_SELECTION.csv", data["selection"])
    write_csv(out / "SOURCE_COMPARISON_COVERAGE.csv", data["broad_data"]["coverage"])
    write_csv(out / "SOURCE_COMPARISON_ISSUES.csv", data["broad_data"]["issues"])
    write_csv(out / "SCORING_ISSUES.csv", data["core"]["issues"] + data["broad"]["issues"])
    write_csv(out / "CONDITIONING_PROVENANCE.csv", data["core_data"]["conditions"])
    write_json(out / "CONFIG_USED.json", config)
    write_json(
        out / "METRIC_SELECTION.json",
        {
            "included_metrics": data["extended_metrics"],
            "rule": config["extended_selection"],
            "scope": data["selection_scope"],
            "screen_status": "Exploratory selection after inspecting benchmark results",
            "core_average_unchanged": True,
        },
    )


def _captions(out, data):
    text = """# Radar caption and interpretation notes

The panels compare BLOOM with all 22 saved TRANS models. Native HVG and author gene settings are kept separate. Label-box colors match the supplied PowerPoint screenshots: orange for Big, blue for Small, green for STED-EC. The overall/model paint order affects visual layering only; scores are never sorted separately and reassigned to models. Actual polygon crossings remain visible.

Range mode transforms each raw metric to a direction-aware 0–1 radius using its dataset/panel-specific declared bounds. Higher-is-better: (value − low)/(high − low); lower-is-better: (high − value)/(high − low). Values outside those bounds are clipped in the artwork and retained unchanged in the CSVs and JSON sidecars. Automatically expanded bounds are derived from all 23 models. Each spoke has its own raw numerical scale; outward is always better. Negative unbiased MMD estimates are not changed to zero in the source values.

The equal-aspect average always retains the four original endpoints: (gene PCC + RMSE + (MMD-U + sliced Wasserstein)/2)/3, after direction-aware empirical min-max normalization across all 23 models within each native dataset/panel. Display bounds never change that score. In Version 5 it is the same fixed core average, even if one core endpoint is excluded as an individual spoke. Broad related endpoints such as MSE/RMSE and PCC aggregation variants do not receive extra weights in this average. This is an exploratory custom summary, not a validated new metric; normalization sensitivity is provided separately.

Optional rank mode uses genuine average-tie ranks, radius=(23−rank)/22. The average spoke uses the rank of the fixed empirical-minmax core average. Rank-mode spokes show rank tick units, rather than raw values. Raw values remain in sidecars.

Version 5 uses one global metric exclusion set across all three datasets and both native panels. A severe-poor setting triggers exclusion only if BLOOM is both rank ≥19/23 and observed-range quality ≤0.25 in that same setting. This is a post-hoc favorable-metric screen requested for exploration; it should be disclosed when showing that version. Versions 1–4 retain the four fixed core endpoints regardless of performance. All candidate outcomes and exclusion reasons are supplied in METRIC_SELECTION.csv. Saved-positive fraction endpoints measure values >0 on the saved expression scale; they do not by themselves establish RNA detection fidelity.

Under the declared native evaluation protocol, BLOOM predicts all saved target cells for Small/STED, while TRANS predicts trajectory-covered targets. Big uses a shared saved evaluation population; both families use the reported 10% trajectory sampling in Big training. These protocol details and target counts are kept here and in numerical metadata, without coverage text inside the artwork.

BLOOM source metadata declares target-cell-type conditioning and target-cell-indexed priors. The evaluation is conditional. Numerical endpoint comparisons do not establish unconditional cell-type forecasting. No p-values or significance claims are inferred from the saved prediction runs.

Style reference: Figure 1(c), Chen et al., Towards a general-purpose foundation model for computational pathology, Nature Medicine (2024), https://www.nature.com/articles/s41591-024-02857-3 . These are newly drawn BLOOM benchmark charts, not the reference paper's numerical data.

## Saved evaluation target counts

| Dataset | Panel | BLOOM cells | TRANS cells |
|---|---|---:|---:|
"""
    seen = set()
    for record in data["core_data"]["coverage"]:
        key = record["dataset"], record["panel"]
        if key in seen:
            continue
        seen.add(key)
        counts = record["target_counts"]
        others = sorted({str(v) for model, v in counts.items() if model != "BLOOM"})
        text += f'| {key[0]} | {key[1]} | {counts.get("BLOOM", "")} | {", ".join(others)} |\n'
    selection = data.get("selection", [])
    if selection:
        rule = selection[0]
        text = text.replace("rank ≥19/23", f'rank ≥{rule["severe_rank_at_least"]:g}/23')
        text = text.replace("quality ≤0.25", f'quality ≤{rule["severe_quality_at_most"]:g}')
        if not rule["screen_enabled"]:
            text += "\nThe severe-performance filter was disabled for this run. Complete nonconstant candidate metrics remain eligible unless explicitly excluded globally by the user.\n"
    text = text.replace(
        "across all three datasets and both native panels",
        "across the requested dataset/native-panel settings",
    )
    scope = ", ".join(f"{row['dataset']}/{row['panel']}" for row in data["selection_scope"])
    text += f"\nAxis-selection scope for this run: {scope}.\n"
    (out / "CAPTIONS.md").write_text(text)


def _index(out, entries):
    sections = []
    for entry in entries:
        path = next((p for p in entry["files"] if p.endswith(".png")), entry["files"][0])
        rel = "figures/" + path
        title = f'{entry["panel"].upper()} · Version {entry["version"]} · {Path(path).stem}'
        preview = (
            f'<img loading="lazy" src="{html.escape(rel)}">'
            if path.endswith((".png", ".svg"))
            else "Open PDF"
        )
        downloads = " · ".join(
            f'<a href="figures/{html.escape(p)}">{Path(p).suffix[1:].upper()}</a>'
            for p in entry["files"]
        )
        downloads += f' · <a href="figures/{html.escape(entry["sidecar"])}">JSON values</a>'
        sections.append(
            f'<section><h2>{html.escape(title)}</h2><p>{downloads}</p><a href="{html.escape(rel)}">{preview}</a></section>'
        )
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>BLOOM radar gallery</title><style>body{font:16px system-ui,sans-serif;max-width:1700px;margin:35px auto;padding:0 24px;color:#28343d;background:#fafbfc}img{max-width:100%;background:white}section{border-top:1px solid #dbe0e5;margin-top:28px;padding-top:12px}a{color:#365b7c}h1{font-size:28px}</style></head><body><h1>BLOOM radar gallery</h1><p>Separate HVG and author settings. All 23 model profiles retain their actual score ordering. Version 5 uses a disclosed exploratory metric screen.</p><p><a href="RADAR_VALUES.csv">Plotted values</a> · <a href="METRIC_SELECTION.csv">Metric selection</a> · <a href="CORE_OVERALL_RANKINGS.csv">Core rankings</a> · <a href="CAPTIONS.md">Separate captions</a></p>"""
    (out / "INDEX.html").write_text(page + "".join(sections) + "</body></html>")


def main(argv=None):
    args = parser().parse_args(argv)
    config = json.loads(Path(args.config).expanduser().read_text())
    if args.no_poor_filter:
        config["extended_selection"]["exclude_severe"] = False
    if args.command == "run":
        versions = (
            [1, 2, 3, 4, 5]
            if args.versions == "all"
            else list(dict.fromkeys(int(x) for x in args.versions.split(",")))
        )
        if not versions or any(x not in range(1, 6) for x in versions):
            raise ValueError("--versions must be all or comma-separated 1–5")
        formats = tuple(dict.fromkeys(x.strip() for x in args.formats.split(",")))
        if not formats or any(x not in ("png", "pdf", "svg") for x in formats):
            raise ValueError("--formats must contain png,pdf,svg")
        if args.no_legend:
            config["legend"] = False
            config.setdefault("style", {})["legend"] = False
        if args.no_scale_ticks:
            config["show_scale_ticks"] = False
            config.setdefault("style", {})["show_scale_ticks"] = False
    # Never touch a previous output. Tags are simple child names, not paths.
    tag = args.tag or datetime.now(timezone.utc).strftime("radar_%Y%m%dT%H%M%S_%fZ")
    if Path(tag).name != tag or tag in {".", ".."}:
        raise ValueError("--tag must be a simple child folder name")
    out = Path(args.out_root).expanduser().resolve() / tag
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    print(f"Output: {out}", flush=True)
    write_json(out / "RUN_ARGUMENTS.json", vars(args))
    try:
        with benchmark_input(args.input, include_report=False) as root:
            data = prepare(root, config, dataset=args.dataset, panel=args.panel)
            _export_tables(out, data, config)
            _captions(out, data)
            entries = []
            if args.command == "run":
                jobs = make_jobs(
                    data,
                    config,
                    versions=versions,
                    panels=("hvg", "author") if args.panel == "all" else (args.panel,),
                    datasets=("big", "small", "sted") if args.dataset == "all" else (args.dataset,),
                    scale=args.scale,
                    include_singles=not args.no_singles,
                )
                write_json(out / "FIGURE_JOBS.json", jobs)
                workers = _workers(args.workers, len(jobs))
                print(
                    f'{len(jobs)} figures; {workers} CPU processes; {len(data["extended_metrics"])} Version-5 metrics plus optional fixed core average',
                    flush=True,
                )
                from tqdm import tqdm

                payloads = [(job, out / "figures", config, formats) for job in jobs]
                if workers == 1:
                    entries = [
                        _draw(payload)
                        for payload in tqdm(payloads, desc="Radar figures", unit="figure")
                    ]
                else:
                    with ProcessPoolExecutor(
                        max_workers=workers, mp_context=get_context("spawn")
                    ) as pool:
                        futures = {
                            pool.submit(_draw, payload): payload[0]["id"] for payload in payloads
                        }
                        for future in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc="Radar figures",
                            unit="figure",
                        ):
                            entries.append(future.result())
                entries.sort(key=lambda e: e.get("id", e["files"][0]))
                write_json(out / "FIGURE_MANIFEST.json", {"figures": entries})
                _index(out, entries)
            write_json(
                out / "RUN_STATUS.json",
                {
                    "status": "complete",
                    "elapsed_seconds": time.time() - start,
                    "source_global_sha256": data["core_data"]["input_sha256"],
                    "source_input": str(Path(args.input).expanduser().resolve()),
                    "figure_count": len(entries),
                    "extended_metric_count": len(data["extended_metrics"]),
                    "core_endpoints": [x["name"] for x in config["core_plan"]["metrics"]],
                    "model_count": 23,
                    "coverage_notes_inside_figures": False,
                    "core_average_definition_unchanged": True,
                    "selection_scope": data["selection_scope"],
                },
            )
        print(
            (
                f"Finished in {time.time()-start:.1f}s; open {out}/INDEX.html"
                if entries
                else f"Audit completed: {out}/METRIC_SELECTION.csv"
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        write_json(
            out / "RUN_STATUS.json",
            {
                "status": "failed",
                "elapsed_seconds": time.time() - start,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        print(f"FAILED: {type(exc).__name__}: {exc}; see {out}/RUN_STATUS.json", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
