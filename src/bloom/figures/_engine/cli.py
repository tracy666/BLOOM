#!/usr/bin/env python3
"""Internal source-preserving biology command implementation.

Use ``python -m bloom.figures`` for the paper-aligned public interface.
"""

from __future__ import annotations
import os

# Avoid 128 processes each starting a BLAS thread pool.
for _key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ[_key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse, importlib, json, sys
from pathlib import Path
from bloom.figures._engine.util import atomic_json, config_read, read_json, ALIASES, resources


def common(p):
    p.add_argument(
        "--config", default="configs/figures/config.json", help="Configuration created by configure"
    )


def scopes(p):
    p.add_argument(
        "--question", default="all", help="all, F2/F3/F4/F5, question ID, or comma-separated IDs"
    )
    p.add_argument("--dataset", default="all", help="all or comma-separated big,small,sted")
    p.add_argument(
        "--panel",
        choices=["all", "hvg", "200", "author"],
        default="all",
        help="Native assay: compare its own GT, v6, v7 and TRANS only; all runs both assays separately",
    )
    p.add_argument(
        "--saved-only",
        action="store_true",
        help="Explicitly render saved diagnostic tables without a prepared expression index; cell-level families are reported unavailable",
    )


def parallel(p):
    p.add_argument("--workers", default="auto", help="auto or positive process count")
    p.add_argument("--cpu-reserve", type=int, default=0)
    p.add_argument(
        "--worker-memory-gb",
        type=float,
        default=0.65,
        help="Minimum RAM allowance per rendering worker",
    )


def plotting(p):
    p.add_argument(
        "--max-entities",
        type=int,
        default=0,
        help="0 means all eligible genes/programs/states/interventions; positive numbers cap each entity family",
    )
    p.add_argument(
        "--max-views-per-entity",
        type=int,
        default=0,
        help="0 means every defined complementary view; positive numbers cap view families, not model panels",
    )
    p.add_argument(
        "--max-response-genes",
        type=int,
        default=20,
        help="Compact intervention response genes per target; 0=all eligible, with paginated views",
    )
    p.add_argument(
        "--max-display-genes",
        type=int,
        default=20,
        help="Reference-supported named genes for readable displays; 0=all eligible. Combines with max-entities using the smaller positive cap",
    )
    p.add_argument(
        "--max-display-states",
        type=int,
        default=12,
        help="GT-supported named states for readable displays; 0=all. Selection and omitted counts are audited",
    )
    p.add_argument("--profile", choices=["quick", "standard", "full"], default="standard")
    p.add_argument(
        "--formats", default=None, help="Comma separated png,pdf,svg; profile supplies default"
    )
    p.add_argument("--variants", default=None, help="legend,no_legend; profile supplies default")
    p.add_argument(
        "--layouts",
        default=None,
        help="all,gt_fm,gt_fm_trans,individual (profile supplies default)",
    )
    p.add_argument("--dpi", type=int, default=None)
    p.add_argument("--model-policy", choices=["all", "selected", "both"], default="all")
    p.add_argument(
        "--selected-models",
        default=None,
        help="Comma separated canonical TRANS names, or JSON policy file mapping big_author etc. FM and GT retained",
    )
    p.add_argument(
        "--scanpy",
        action="store_true",
        help="Also draw native Scanpy marker plots; install requirements-scanpy.txt first",
    )
    p.add_argument(
        "--scanpy-models",
        choices=["fm", "all"],
        default="fm",
        help="Native Scanpy model plots: observed + FM by default; all also includes available TRANS models",
    )
    p.add_argument("--background-alpha", type=float, default=0.8)
    p.add_argument("--background-color", default="#969FA7")
    p.add_argument(
        "--all-genes",
        action="store_true",
        help="Allow exploratory genes outside author/curated evidence; reasons remain exported",
    )
    p.add_argument(
        "--strict-significance",
        action="store_true",
        help="Use only annotated genes passing nominal reference-cell signal q-value screen",
    )
    p.add_argument("--signal-q", type=float, default=0.05)
    p.add_argument(
        "--max-stage-panels",
        type=int,
        default=12,
        help="Storyboard sampling only: 0=all stages, full-row statistics retained",
    )


def parser():
    p = argparse.ArgumentParser(
        description="Rebuild substantive BLOOM biology from saved expression and legacy tables"
    )
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser(
        "configure", help="Create a portable configuration, without changing any evaluation output"
    )
    c.add_argument("--v5-root", action="append", default=[])
    c.add_argument("--legacy-root", action="append", default=[])
    c.add_argument("--output-root", default=None)
    c.add_argument("--cache-root", default=None)
    c.add_argument("--from-config", default=None)
    c.add_argument("--out", default="configs/figures/config.json")
    c.add_argument(
        "--alias", action="append", default=[], help="canonical=display_name, repeatable"
    )
    c.add_argument(
        "--dataset-config",
        action="append",
        default=[],
        help="dataset=JSON annotation/time file, repeatable",
    )
    c.add_argument(
        "--selected",
        action="append",
        default=[],
        help="setting=TRANS1,TRANS2, e.g. small_hvg=TSMixer,Transformer",
    )
    c.add_argument(
        "--source-manifest",
        help="JSON list of explicit saved sources including path,dataset,model,role,panel,scenario",
    )
    c.add_argument(
        "--saved-diagnostic-manifest",
        help="JSON list of saved table/run folders including path,dataset,model,role,panel when native slot ancestry is absent",
    )
    c = sub.add_parser("check", help="Check dependencies and CPU/RAM availability")
    common(c)
    parallel(c)
    c = sub.add_parser("inventory", help="Discover and validate headers of saved H5/H5AD sources")
    common(c)
    c.add_argument("--out", default=None)
    c = sub.add_parser(
        "prepare", help="Read saved matrices once; cache geometry and full-row biology summaries"
    )
    common(c)
    parallel(c)
    c.add_argument("--embedding", choices=["pca", "umap"], default="pca")
    c.add_argument(
        "--sample-cells",
        type=int,
        default=20000,
        help="Stratified cell sample per source for display/distributions; full-row summaries are separate. 0=all display cells",
    )
    c.add_argument("--reference-cells", type=int, default=10000)
    c.add_argument(
        "--analysis-cap",
        type=int,
        default=0,
        help="0=all rows in streamed statistics; positive explicitly samples analysis rows",
    )
    c.add_argument("--chunk-size", type=int, default=65536)
    c.add_argument("--seed", type=int, default=1729)
    c = sub.add_parser("catalogue", help="Write the biological question catalogue")
    c.add_argument("--out", default="QUESTION_CATALOGUE.csv")
    c = sub.add_parser("plan", help="Question eligibility after prepare, without rendering")
    common(c)
    scopes(c)
    c.add_argument("--out", default=None)
    c = sub.add_parser(
        "run", help="Render each shared biological figure family once in a fresh trial"
    )
    common(c)
    scopes(c)
    parallel(c)
    plotting(c)
    c.add_argument("--run-name", default="biology_families")
    c.add_argument(
        "--resume-root",
        default=None,
        help="Explicit same-input trial resume; fresh output is default",
    )
    c.add_argument("--skip-legacy", action="store_true")
    c.add_argument(
        "--preflight",
        action="store_true",
        help="Alias for plan; does not create or claim rendered figures",
    )
    c = sub.add_parser(
        "legacy",
        help="Preserve every original legacy figure and redraw supported biological tables",
    )
    common(c)
    parallel(c)
    plotting(c)
    c.add_argument("--out", default=None)
    c = sub.add_parser(
        "metrics",
        help="Optional full-row supporting quantitative ranks, separate from biological questions",
    )
    common(c)
    parallel(c)
    plotting(c)
    c.add_argument("--dataset", default="all")
    c.add_argument("--panel", choices=["all", "hvg", "200", "author"], default="all")
    c.add_argument("--out", default=None)
    c = sub.add_parser(
        "verify", help="Check real rendered output files and biology/legacy coverage"
    )
    c.add_argument("--result-root", required=True)
    c = sub.add_parser("selftest", help="Run included integration checks")
    c.add_argument("--quick", action="store_true")
    return p


def normalize_options(args, config=None):
    o = vars(args).copy()
    config = config or {}
    profile = o.get("profile", "standard")
    defaults = {
        "quick": (["png"], ["legend"], ["all"], 120),
        "standard": (["png"], ["legend", "no_legend"], ["all", "gt_fm", "gt_fm_trans"], 150),
        "full": (
            ["png", "pdf", "svg"],
            ["legend", "no_legend"],
            ["all", "gt_fm", "gt_fm_trans", "individual"],
            240,
        ),
    }[profile]
    for key, value in zip(["formats", "variants", "layouts", "dpi"], defaults):
        if o.get(key) is None:
            o[key] = value
        elif key != "dpi" and isinstance(o[key], str):
            o[key] = o[key].split(",")
    selected = o.get("selected_models")
    if selected:
        if Path(selected).is_file():
            selected = read_json(selected)
        else:
            selected = selected.split(",")
    else:
        selected = config.get("selected_models", [])
    o["selected_models"] = selected
    o["aliases"] = config.get("aliases", ALIASES)
    o["include_legacy"] = not o.get("skip_legacy", False)
    o["allow_unannotated"] = o.get("all_genes", False)
    o["significant_only"] = o.get("strict_significance", False)
    o["selection_q"] = o.get("signal_q", 0.05)
    o["configuration"] = config
    if any(
        o.get(k, default) < 0
        for k, default in [
            ("max_entities", 0),
            ("max_views_per_entity", 0),
            ("max_response_genes", 20),
            ("max_display_genes", 20),
            ("max_display_states", 12),
        ]
    ):
        raise ValueError("Entity/view/display/response caps must be zero or positive")
    if not 0 <= o.get("background_alpha", 0.8) <= 1:
        raise ValueError("background-alpha must lie between 0 and 1")
    if o.get("worker_memory_gb", 0.65) <= 0:
        raise ValueError("worker-memory-gb must be positive")
    return o


def load_index(config):
    p = Path(config["cache_root"]) / "index.json"
    if not p.exists():
        raise ValueError(
            f"No prepared index: {p}. First run bloom.py inventory, then bloom.py prepare."
        )
    index = read_json(p)
    from bloom.figures._engine.prepare import VERSION

    if index.get("preparer_version") != VERSION:
        raise ValueError(
            "This prepared index predates the v2 native-assay reference pipeline. Use a fresh v2 cache and run bloom.py prepare before rendering."
        )
    # Preparation recovers author/program/prior paths from saved run provenance.
    for cohort in index.get("cohorts", []):
        settings = config.setdefault("datasets", {}).setdefault(cohort["dataset"], {})
        for key, value in cohort.get("configuration", {}).items():
            settings.setdefault(key, value)
    return index


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "configure":
        c = (
            config_read(args.from_config)
            if args.from_config
            else {
                "schema_version": 2,
                "v5_roots": [],
                "legacy_roots": [],
                "datasets": {},
                "aliases": ALIASES.copy(),
            }
        )
        # Apply the requested v2 publication names even when importing a v1
        # configuration. Explicit --alias values below can customize them.
        c.setdefault("aliases", {}).update(ALIASES)
        if args.v5_root:
            c["v5_roots"] = [str(Path(x).expanduser().resolve()) for x in args.v5_root]
        if args.legacy_root:
            c["legacy_roots"] = [str(Path(x).expanduser().resolve()) for x in args.legacy_root]
        if args.output_root or not c.get("output_root"):
            c["output_root"] = str(Path(args.output_root or "results").expanduser().resolve())
        if args.cache_root or not c.get("cache_root"):
            c["cache_root"] = str(
                Path(args.cache_root or str(Path(c["output_root"]) / "prepared"))
                .expanduser()
                .resolve()
            )
        for v in args.alias:
            key, sep, label = v.partition("=")
            if not sep or not key or not label:
                raise ValueError("--alias must be canonical=display_name")
            c["aliases"][key] = label
        for v in args.dataset_config:
            key, sep, path = v.partition("=")
            if not sep or key not in ("big", "small", "sted"):
                raise ValueError("--dataset-config must be big|small|sted=path.json")
            spec = read_json(path)
            base = Path(path).resolve().parent
            for k in [
                "obs_path",
                "programs_path",
                "edges_path",
                "author_genes_path",
                "gene_annotations_path",
                "gene_evidence_path",
                "author_graph_path",
                "graph_path",
                "graph_nodes_path",
                "graph_cell_map_path",
            ]:
                if spec.get(k):
                    spec[k] = str((base / Path(spec[k])).resolve())
            c.setdefault("datasets", {})[key] = spec
        for v in args.selected:
            key, sep, value = v.partition("=")
            if not sep:
                raise ValueError("--selected must be setting=TRANS1,TRANS2")
            c.setdefault("selected_models", {})[key] = value.split(",")
        if args.source_manifest:
            c["sources"] = read_json(args.source_manifest)
            base = Path(args.source_manifest).resolve().parent
            for s in c["sources"]:
                s["path"] = str((base / Path(s["path"])).resolve())
        if args.saved_diagnostic_manifest:
            c["saved_diagnostic_sources"] = read_json(args.saved_diagnostic_manifest)
            base = Path(args.saved_diagnostic_manifest).resolve().parent
            for s in c["saved_diagnostic_sources"]:
                s["path"] = str((base / Path(s["path"])).resolve())
        c.pop("config_path", None)
        atomic_json(args.out, c)
        print(f"Configuration: {Path(args.out).resolve()}")
        return 0
    if args.command == "selftest":
        import subprocess

        command = [sys.executable, "-m", "pytest", "-q", str(Path(__file__).parent / "tests")]
        if args.quick:
            command += ["-m", "not slow"]
        return subprocess.call(command, cwd=Path(__file__).parent)
    if args.command == "catalogue":
        from bloom.figures._engine.registry import QUESTIONS
        from bloom.figures._engine.run import csv_write

        csv_write(
            args.out,
            [
                dict(
                    q,
                    views=";".join(q["views"]),
                    scenario=";".join(q["scenario"]),
                    requires=";".join(q["requires"]),
                )
                for q in QUESTIONS
            ],
            ["id", "figure", "title", "handler", "scenario", "views", "requires", "note"],
        )
        print(f"{len(QUESTIONS)} biological question templates: {Path(args.out).resolve()}")
        return 0
    if args.command == "verify":
        from bloom.figures._engine.validation import verify

        report = verify(Path(args.result_root))
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 2
    config = config_read(args.config)
    if args.command == "check":
        failed = []
        for module in [
            "numpy",
            "scipy",
            "pandas",
            "matplotlib",
            "sklearn",
            "h5py",
            "pyarrow",
            "tqdm",
            "psutil",
            "PIL",
            "threadpoolctl",
        ]:
            try:
                m = importlib.import_module(module)
                print(f'{module}: {getattr(m,"__version__","OK")}')
            except Exception as e:
                failed.append(f"{module}: {e}")
        n, r = resources(args.workers, args.cpu_reserve, args.worker_memory_gb)
        print(f"CPU worker budget={n}: {r}")
        for problem in failed:
            print(problem, file=sys.stderr)
        return 2 if failed else 0
    options = normalize_options(args, config)
    if args.command == "inventory":
        from bloom.figures._engine.discovery import discover

        inv = discover(config)
        out = Path(args.out or str(Path(config["cache_root"]) / "INVENTORY.json"))
        atomic_json(out, inv)
        from collections import Counter

        print(
            f"Saved expression sources: {len(inv['sources'])}; by dataset/scenario: {dict(Counter((s['dataset'],s['scenario']) for s in inv['sources']))}; errors={len(inv['errors'])}"
        )
        for e in inv["errors"][:10]:
            print(f"INPUT ERROR: {e['path']}: {e['error']}")
        print(f"Inventory: {out}")
        return 0 if inv["sources"] else 2
    if args.command == "prepare":
        from bloom.figures._engine.prepare import prepare
        from tqdm import tqdm

        bars = {}

        def progress(e):
            phase = e.get("phase", "prepare")
            if phase not in bars:
                bars[phase] = tqdm(
                    total=e.get("total", 1), desc=phase, unit="source", dynamic_ncols=True
                )
            bar = bars[phase]
            bar.total = e.get("total", bar.total)
            bar.n = e.get("done", bar.n)
            bar.set_postfix_str(str(e.get("source_id", "")) + " " + str(e.get("status", "")))
            bar.refresh()

        try:
            index = prepare(config, options, progress)
        finally:
            for bar in bars.values():
                bar.close()
        print(
            f"Prepared {len(index.get('sources',[]))} artifacts in {len(index.get('cohorts',[]))} cohorts. Errors={len(index.get('errors',[]))}. Index: {Path(config['cache_root'])/'index.json'}"
        )
        return 0 if index.get("sources") else 2
    if args.command == "plan" or args.command == "run" and args.preflight:
        from bloom.figures._engine.run import make_plan, csv_write

        index = {"cohorts": [], "sources": []} if args.saved_only else load_index(config)
        plan = make_plan(index, config, args.question, args.dataset, args.panel)
        out = Path(
            getattr(args, "out", None) or str(Path(config["cache_root"]) / "QUESTION_PLAN.csv")
        )
        csv_write(out, plan, ["question", "figure", "title", "cohort", "eligible", "reason"])
        from bloom.figures._engine.families import make_family_plan
        from bloom.figures._engine.run import eligibility

        families = make_family_plan(plan, config, options, eligibility)
        family_out = out.with_name(out.stem + "_FAMILIES.csv")
        csv_write(
            family_out,
            [{**f, "questions": ";".join(f["questions"])} for f in families],
            [
                "family_id",
                "category",
                "family",
                "figure",
                "title",
                "cohort",
                "questions",
                "eligible",
                "reason",
            ],
        )
        byfigure = {f: sum(x["eligible"] for x in plan if x["figure"] == f) for f in range(2, 6)}
        print(
            f'Eligible question/cohort jobs by Figure: {byfigure}. Total eligible={sum(x["eligible"] for x in plan)}/{len(plan)}. Plan: {out}'
        )
        print(
            f'Shared drawable figure families={sum(f["eligible"] for f in families)}/{len(families)}. Family plan: {family_out}'
        )
        print("This checks actual prepared inputs. It does not create figures or test drawing.")
        return 0
    if args.command == "legacy":
        from bloom.figures._engine.archive import render_archive
        from bloom.figures._engine.run import trial_dir

        out = Path(args.out) if args.out else trial_dir(config, options) / "legacy"
        if out.exists() and any(out.iterdir()):
            raise ValueError(
                "--out must be a new/empty directory; legacy originals are never overwritten"
            )
        result = render_archive(config.get("legacy_roots", []), out, options)
        print(json.dumps(result, indent=2, default=str))
        print(f'Legacy gallery: {out/"INDEX.html"}')
        return (
            0
            if result.get("status") == "complete"
            and result.get("figures")
            and not result.get("errors")
            else 2
        )
    if args.command == "metrics":
        from bloom.figures._engine.benchmark import export_metrics
        from bloom.figures._engine.run import trial_dir

        options["run_name"] = "supporting_metrics"
        out = Path(args.out) if args.out else trial_dir(config, options)
        if args.out and out.exists() and any(out.iterdir()):
            raise ValueError("--out must be a fresh/empty directory")
        result = export_metrics(config, out, options)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("status") in ("complete", "saved_tables_only") else 2
    if args.command == "run":
        from bloom.figures._engine.run import run

        index = {"cohorts": [], "sources": []} if args.saved_only else load_index(config)
        root, results = run(config, index, options)
        errors = sum(r["status"] == "error" for r in results)
        coverage = read_json(root / "BIOLOGICAL_COVERAGE.json")
        legacy = coverage.get("legacy")
        if legacy and (
            legacy.get("errors") or not legacy.get("summary", {}).get("coverage_complete", False)
        ):
            errors += 1
        has_figures = any(r.get("figures") for r in results)
        return 2 if errors or not has_figures else 0
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        if len(sys.argv) > 1 and sys.argv[1] == "run":
            print(
                "Interrupted. Completed figures were kept; use --resume-root with the same options to continue.",
                file=sys.stderr,
            )
        else:
            print(
                "Interrupted. Completed files were kept. Re-run preparation to reuse its cache; use a fresh output directory for legacy or metrics.",
                file=sys.stderr,
            )
        sys.exit(130)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
