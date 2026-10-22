"""Parallel biological rendering with separate trial directories and live progress."""

from __future__ import annotations
import concurrent.futures as cf
import csv, json, os, time, uuid, traceback, html
from datetime import datetime, timezone
from pathlib import Path
from .registry import get_questions, QUESTIONS
from .util import (
    atomic_json,
    read_json,
    safe_name,
    resources,
    FM_MODELS,
    TRANS_MODELS,
    code_digest,
    process_pool,
    fingerprint,
)
from .families import make_family_plan, interpretation_markdown


def native_panel(source):
    """Exact native assay identity; unknown truth is never reused in both."""
    from .plot import source_panel

    return source_panel(source)


def split_cohorts(index, dataset="all", panel="all"):
    selected = []
    ds = set(dataset.split(",")) if dataset != "all" else None
    panels = (
        ["hvg", "author"]
        if panel == "all"
        else list(
            dict.fromkeys(
                "hvg" if p.lower() in ("200", "hvg", "hvg200") else p.lower()
                for p in panel.split(",")
            )
        )
    )
    if any(p not in ("hvg", "author") for p in panels):
        raise ValueError("Native panel must be hvg/200 or author, or all.")
    for cohort in index.get("cohorts", []):
        if ds is not None and cohort["dataset"] not in ds:
            continue
        for p in panels:
            sources = []
            for s in cohort["sources"]:
                if s.get("model") == "v6_200_old":
                    continue
                if native_panel(s) != p:
                    continue
                sources.append(s)
            if not any(s.get("role") not in ("GT", "REFERENCE") for s in sources):
                continue
            suffix = "__" + str(cohort["diagnostic_kind"]) if cohort.get("diagnostic_kind") else ""
            selected.append(
                {
                    **cohort,
                    "sources": sources,
                    "reference_sources": [
                        x for x in sources if x.get("role") in ("GT", "REFERENCE")
                    ],
                    "panel": p,
                    "key": f"{cohort['dataset']}__{cohort['scenario']}__{p}{suffix}",
                }
            )
    return selected


def eligibility(question, cohort, config):
    if cohort["scenario"] not in question["scenario"]:
        return False, "scenario does not apply"
    sources = [s for s in cohort["sources"] if s.get("role") not in ("GT", "REFERENCE")]
    if not sources:
        return False, "No generated source in this cohort"
    if question["handler"] == "saved_diagnostics":
        from . import saved_diagnostics

        check = getattr(saved_diagnostics, "eligibility", None)
        if check:
            return check(question, cohort, config)
        paths = [s.get("path") for s in sources if s.get("path")]
        if not any(Path(p).is_file() for p in paths):
            return False, "No readable saved diagnostic table in this native cohort"
        return (
            True,
            "Saved diagnostic tables are present; method-specific kinds and layout are checked during rendering",
        )
    caps = set().union(*(set(s.get("capabilities", [])) for s in sources))
    # Inspect actual artifact files too; older preparation metadata may be sparse.
    for s in sources:
        d = Path(s.get("artifact_dir", "__missing_artifact__"))
        if (d / "cells.parquet").exists():
            caps.add("cells")
        if (d / "expression.npy").exists():
            caps.add("expression")
        if (d / "stage_gene.parquet").exists():
            caps.add("stage")
        if (d / "perturbation.parquet").exists():
            caps.add("perturbation")
        if s.get("state_method", "unavailable") not in ("unavailable", "unknown", "none"):
            caps.add("state")
    spec = {
        **cohort.get("configuration", {}),
        **config.get("datasets", {}).get(cohort["dataset"], {}),
    }
    if spec.get("edges_path") and Path(spec["edges_path"]).is_file():
        caps.add("edges")
    graph = spec.get("author_graph_path") or spec.get("graph_path")
    if graph and Path(graph).is_file():
        caps.add("author_graph")
    missing = set(question["requires"]) - caps
    if missing:
        return False, "Missing actual capability: " + ", ".join(sorted(missing))
    return True, "Saved inputs support rendering; layout is verified only by actual rendering"


def make_plan(index, config, question="all", dataset="all", panel="all"):
    rows = []
    questions = get_questions(question)
    working_index = index
    diagnostic_errors = []
    if any(q["handler"] == "saved_diagnostics" for q in questions):
        from .saved_diagnostics import discover_saved_diagnostics_report

        diagnostic_report = discover_saved_diagnostics_report(config)
        diagnostics = diagnostic_report["cohorts"]
        diagnostic_errors = diagnostic_report["errors"]
        working_index = {**index, "cohorts": list(index.get("cohorts", [])) + diagnostics}
    available = split_cohorts(working_index, dataset, panel)
    for q in questions:
        cohorts = [c for c in available if c["scenario"] in q["scenario"]]
        if not cohorts:
            reason = (
                "No saved "
                + ",".join(q["scenario"])
                + " outputs are available in the requested native settings"
            )
            if q["handler"] == "saved_diagnostics" and diagnostic_errors:
                reason += "; diagnostic discovery: " + " | ".join(
                    e.get("path", "") + ": " + e.get("reason", "unknown error")
                    for e in diagnostic_errors[:3]
                )
            rows.append(
                {
                    "question": q["id"],
                    "title": q["title"],
                    "figure": q["figure"],
                    "cohort": "none",
                    "eligible": False,
                    "reason": reason,
                    "question_spec": q,
                    "cohort_spec": None,
                    "discovery_errors": (
                        diagnostic_errors if q["handler"] == "saved_diagnostics" else []
                    ),
                }
            )
        for c in cohorts:
            ok, why = eligibility(q, c, config)
            rows.append(
                {
                    "question": q["id"],
                    "title": q["title"],
                    "figure": q["figure"],
                    "cohort": c["key"],
                    "eligible": ok,
                    "reason": why,
                    "question_spec": q,
                    "cohort_spec": c,
                    "discovery_errors": (
                        diagnostic_errors if q["handler"] == "saved_diagnostics" else []
                    ),
                }
            )
    return rows


def csv_write(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def trial_dir(config, options):
    if options.get("resume_root"):
        root = Path(options["resume_root"]).resolve()
        if not (root / "RUN_CONFIG.json").is_file():
            raise ValueError(
                "--resume-root must point to an existing rebuild trial with RUN_CONFIG.json"
            )
        return root
    tag = safe_name(options.get("run_name") or "questions")
    e = options.get("max_entities", 0)
    v = options.get("max_views_per_entity", 0)
    label = f'{tag}_max-entities_{e or "all"}_max-views-per-entity_{v or "all"}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}'
    root = Path(config["output_root"]) / label
    root.mkdir(parents=True, exist_ok=False)
    return root


def _job(task):
    family, c, out, options = task
    q = family["question_spec"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(
        out / "FAMILY.json",
        {k: v for k, v in family.items() if k not in ("cohort_spec", "question_spec")},
    )
    atomic_json(out / "COHORT.json", c)
    guide = interpretation_markdown(family, c)
    (out / "INTERPRETATION.md").write_text(guide)
    (out / "GUIDE.md").write_text(guide)
    started = time.monotonic()

    def progress(e):
        atomic_json(
            out / "PROGRESS.json",
            {
                "question": family["family_id"],
                "family": family["family"],
                "cohort": c["key"],
                "elapsed_s": round(time.monotonic() - started, 1),
                **e,
            },
        )

    try:
        if q["handler"] == "measured":
            from .measured import render_question
        elif q["handler"] == "saved_diagnostics":
            from .saved_diagnostics import render_question
        else:
            from .scenarios import render_question
        result = render_question(q, c, out, options, progress)
        result.setdefault("status", "complete" if result.get("figures") else "unsupported")
        if not result.get("figures") and result["status"] == "complete":
            result["status"] = "empty"
        if c["scenario"] in ("measured", "reverse") and not any(
            s.get("role") == "GT" for s in c.get("sources", [])
        ):
            if result["status"] == "complete":
                result["status"] = "partial"
            result.setdefault("notes", []).append(
                "No same-native-panel measured GT is available. These figures are descriptive FM/TRANS comparisons; they do not establish measured reconstruction accuracy. An observed boundary is not matched GT for every target."
            )
    except Exception as e:
        result = {
            "status": "error",
            "figures": [],
            "tables": [],
            "notes": [f"{type(e).__name__}: {e}"],
            "traceback": traceback.format_exc(),
        }
    paper_figures = sorted(
        {family["figure"]}
        | {
            int(identifier[1])
            for identifier in family["questions"]
            if len(identifier) > 1 and identifier[1].isdigit()
        }
    )
    result.update(
        question=family["family_id"],
        family_id=family["family_id"],
        family=family["family"],
        category=family["category"],
        questions=family["questions"],
        question_ids=family["questions"],
        figure=q["figure"],
        paper_figures=paper_figures,
        title=q["title"],
        cohort=c["key"],
        dataset=c["dataset"],
        panel=c["panel"],
        scenario=c["scenario"],
        output_dir=str(out),
        elapsed_s=round(time.monotonic() - started, 2),
    )
    from .evidence import write_evidence

    try:
        result.setdefault("tables", []).extend(
            write_evidence(out, result, c, {**q, "question_ids": family["questions"]}, options)
        )
    except Exception as e:
        result["status"] = "error"
        result.setdefault("notes", []).append(
            "Evidence indexing failed: " + type(e).__name__ + ": " + str(e)
        )
        result["traceback"] = traceback.format_exc()
    atomic_json(out / "RESULT.json", result)
    if result.get("traceback"):
        (out / "ERROR.txt").write_text(result["traceback"])
    question_gallery(out, result)
    return result


def _relative(path, root):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def question_gallery(out, result):
    out = Path(out)
    figures = result.get("figures", [])
    paths = []
    for f in figures:
        p = Path(f.get("path", "") if isinstance(f, dict) else f)
        if p.suffix.lower() in (".png", ".gif") and p.is_file():
            try:
                relative = p.resolve().relative_to(out.resolve()).as_posix()
            except ValueError:
                continue
            paths.append((relative, f))
    cards = []
    for p, f in paths:
        label = (
            " / ".join(str(f.get(x, "")) for x in ["view", "entity", "model"] if f.get(x))
            if isinstance(f, dict)
            else p
        )
        cards.append(
            f'<article><a href="{html.escape(p,quote=True)}"><img loading="lazy" src="{html.escape(p,quote=True)}"></a><p>{html.escape(label)}</p></article>'
        )
    notes = "".join(f"<li>{html.escape(str(n))}</li>" for n in result.get("notes", []))
    (out / "INDEX.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(result.get("title","Biology"))}</title><style>body{{font:15px system-ui;margin:24px;color:#27343d}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:22px}}article{{border:1px solid #c6cfd3;border-radius:8px;padding:12px}}img{{width:100%;height:auto}}a{{color:#247d88}}</style><h1>{html.escape(result.get("title","Biology"))}</h1><p>{html.escape(result.get("cohort",""))}: {html.escape(result.get("status",""))}</p><ul>{notes}</ul><p><a href="INTERPRETATION.md">Biological interpretation</a> · <a href="FIGURE_EVIDENCE.csv">Figure evidence index</a> · <a href="RESULT.json">Result and source records</a> · <a href="selected_entities.csv">Entity selection</a></p><main>{"".join(cards)}</main>'
    )


def _summary(root, results, plan, archive=None, family_plan=None):
    rows = []
    for r in results:
        rows.append(
            {
                k: r.get(k)
                for k in [
                    "family_id",
                    "family",
                    "category",
                    "figure",
                    "title",
                    "cohort",
                    "status",
                    "elapsed_s",
                    "output_dir",
                ]
            }
            | {
                "questions": ";".join(r.get("questions", [])),
                "n_figures": len(r.get("figures", [])),
                "notes": " | ".join(str(x) for x in r.get("notes", [])),
            }
        )
    csv_write(
        root / "RUN_SUMMARY.csv",
        rows,
        [
            "family_id",
            "category",
            "family",
            "figure",
            "title",
            "cohort",
            "questions",
            "status",
            "n_figures",
            "elapsed_s",
            "notes",
        ],
    )
    csv_write(
        root / "QUESTION_PLAN.csv",
        plan,
        ["question", "figure", "title", "cohort", "eligible", "reason"],
    )
    diagnostic_errors = {
        json.dumps(e, sort_keys=True): e for row in plan for e in row.get("discovery_errors", [])
    }
    atomic_json(
        root / "SAVED_DIAGNOSTIC_DISCOVERY.json",
        {
            "errors": list(diagnostic_errors.values()),
            "native_cohorts": sorted(
                {
                    row["cohort"]
                    for row in plan
                    if row["question_spec"]["handler"] == "saved_diagnostics"
                    and row["cohort_spec"] is not None
                }
            ),
        },
    )
    counts = {f: sum(r["n_figures"] or 0 for r in rows if r["figure"] == f) for f in range(2, 6)}
    cards = []
    for row in rows:
        link = _relative(Path(row["output_dir"]) / "INDEX.html", root)
        cards.append(
            f'<tr><td>{html.escape(row["category"])}</td><td>Figure {row["figure"]}</td><td><a href="{html.escape(link,quote=True)}">{html.escape(row["family"])}</a></td><td>{html.escape(row["cohort"])}</td><td>{html.escape(row["questions"])}</td><td>{html.escape(row["status"])}</td><td>{row["n_figures"]}</td></tr>'
        )
    archive_link = (
        '<p><a href="legacy/INDEX.html">Preserved and redrawn legacy biology</a> · <a href="legacy/LEGACY_COVERAGE.csv">Every original figure: coverage ledger</a></p>'
        if archive
        else ""
    )
    (root / "INDEX.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>BLOOM shared biological figure gallery</title><style>body{font:15px system-ui;color:#27343d;margin:26px}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #c9d1d4;text-align:left;padding:10px}a{color:#247d88}input{padding:10px;width:60%}</style><h1>BLOOM shared biological figure gallery</h1><p>Each canonical figure family is rendered once per native assay cohort. HVG and author models and their ground truths are separate.</p><p>Unique newly rendered exports by suggested paper figure: '
        + html.escape(str(counts))
        + "</p>"
        + archive_link
        + '<p><a href="RUN_SUMMARY.csv">Family run summary</a> · <a href="FAMILY_PLAN.csv">Family eligibility</a> · <a href="QUESTION_PLAN.csv">Question eligibility</a> · <a href="QUESTION_FIGURE_MAP.md">Question ↔ shared figure map</a> · <a href="FIGURE_MANIFEST.json">Unique figure manifest</a> · <a href="MODEL_COVERAGE.csv">Model coverage</a></p><input placeholder="Filter family, question, cohort or status" oninput="for(const r of document.querySelectorAll(\'tbody tr\'))r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase())"><table><thead><tr><th>Category</th><th>Suggested paper figure</th><th>Family</th><th>Native cohort</th><th>Questions answered</th><th>Status</th><th>Unique exports</th></tr></thead><tbody>'
        + "".join(cards)
        + "</tbody></table>"
    )
    atomic_json(
        root / "BIOLOGICAL_COVERAGE.json",
        {
            "layout": "shared_families_v2",
            "new_exports_by_figure": {str(k): v for k, v in counts.items()},
            "completed_jobs": sum(r["status"] == "complete" for r in rows),
            "partial_jobs": sum(r["status"] == "partial" for r in rows),
            "error_jobs": sum(r["status"] == "error" for r in rows),
            "empty_jobs": sum(not r["n_figures"] for r in rows),
            "question_cohort_requests": len(plan),
            "shared_family_jobs": len(family_plan or []),
            "legacy": archive,
        },
    )
    _question_mapping(root, results, plan, family_plan or [])
    from .reporting import write_interpretation_report

    write_interpretation_report(root, results, family_plan or [], plan)
    homepage = root / "INDEX.html"
    homepage.write_text(
        homepage.read_text().replace(
            '<a href="RUN_SUMMARY.csv">Family run summary</a>',
            '<a href="BIOLOGICAL_INTERPRETATION_REPORT.md">Biological interpretation report</a> · <a href="RUN_SUMMARY.csv">Family run summary</a>',
        )
    )


def _question_mapping(root, results, plan, family_plan):
    """Publish link-only question pages and the unique figure manifest."""
    import hashlib

    result_by_id = {r["family_id"]: r for r in results}
    figure_records = []
    figure_ids = {}
    seen = set()
    for result in results:
        ids = []
        for figure in result.get("figures", []):
            f = figure if isinstance(figure, dict) else {"path": figure}
            path = _relative(f["path"], root)
            identity = hashlib.sha256(path.encode()).hexdigest()[:16]
            ids.append(identity)
            if path in seen:
                continue
            seen.add(path)
            figure_records.append(
                dict(
                    figure_id=identity,
                    path=path,
                    family_id=result["family_id"],
                    category=result["category"],
                    family=result["family"],
                    cohort=result["cohort"],
                    questions=result["questions"],
                    view=f.get("view", result["family"]),
                    entity=f.get("entity", ""),
                    model=f.get("model", "comparison"),
                    evidence=_relative(Path(result["output_dir"]) / "FIGURE_EVIDENCE.csv", root),
                    interpretation=_relative(
                        Path(result["output_dir"]) / "INTERPRETATION.md", root
                    ),
                )
            )
        figure_ids[result["family_id"]] = ids
    mapping = []
    csvrows = []
    for row in plan:
        references = []
        for family in family_plan:
            if row["question"] not in family["questions"] or row["cohort"] != family["cohort"]:
                continue
            result = result_by_id.get(family["family_id"], {})
            out = _family_output(root, family)
            ref = dict(
                family_id=family["family_id"],
                category=family["category"],
                family=family["family"],
                gallery=_relative(out / "INDEX.html", root),
                status=result.get("status", "pending"),
                eligible=family["eligible"],
                reason=family["reason"],
                figure_ids=figure_ids.get(family["family_id"], []),
            )
            references.append(ref)
            csvrows.append(
                dict(
                    question=row["question"],
                    title=row["title"],
                    cohort=row["cohort"],
                    original_figure=row["figure"],
                    suggested_figure=family["figure"],
                    **{
                        k: ref[k]
                        for k in (
                            "family_id",
                            "category",
                            "family",
                            "gallery",
                            "status",
                            "eligible",
                            "reason",
                        )
                    },
                    figure_ids=";".join(ref["figure_ids"]),
                )
            )
        complete = sum(ref["status"] == "complete" for ref in references)
        has_figures = any(ref["figure_ids"] for ref in references)
        status = (
            "complete"
            if references and complete == len(references)
            else (
                "partial"
                if has_figures
                else (
                    "pending"
                    if any(ref["status"] == "pending" for ref in references)
                    else "unsupported"
                )
            )
        )
        record = dict(
            question=row["question"],
            title=row["title"],
            cohort=row["cohort"],
            question_eligible=row["eligible"],
            eligibility_reason=row["reason"],
            status=status,
            families=references,
        )
        mapping.append(record)
        out = root / "questions" / row["question"] / row["cohort"]
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(out / "QUESTION_MAP.json", record)
        links = []
        for ref in references:
            href = os.path.relpath(root / ref["gallery"], out).replace(os.sep, "/")
            links.append(
                f'<li><a href="{html.escape(href,quote=True)}">{html.escape(ref["category"]+" / "+ref["family"])}</a> — {html.escape(ref["status"])} ({len(ref["figure_ids"])} unique exports)</li>'
            )
        (out / "INDEX.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>'
            + html.escape(row["question"])
            + "</title><style>body{font:16px system-ui;margin:28px;color:#27343d}li{margin:14px 0}a{color:#247d88}</style><h1>"
            + html.escape(row["question"] + " · " + row["title"])
            + "</h1><p>"
            + html.escape(row["cohort"] + " · " + status)
            + "</p><p>These links reuse canonical family outputs. Figures are generated once and are not copied into question folders.</p><p>"
            + html.escape(row["reason"])
            + "</p><ul>"
            + "".join(links)
            + '</ul><p><a href="QUESTION_MAP.json">Machine-readable mapping</a></p>'
        )
    atomic_json(
        root / "FIGURE_MANIFEST.json",
        dict(
            layout="shared_families_v2",
            n_unique_exports=len(figure_records),
            figures=figure_records,
        ),
    )
    atomic_json(
        root / "QUESTION_FIGURE_MAP.json", dict(layout="shared_families_v2", questions=mapping)
    )
    columns = [
        "question",
        "title",
        "cohort",
        "original_figure",
        "suggested_figure",
        "category",
        "family",
        "family_id",
        "gallery",
        "status",
        "eligible",
        "reason",
        "figure_ids",
    ]
    csv_write(root / "QUESTION_FIGURE_MAP.csv", csvrows, columns)
    markdown = [
        "# Biological questions and shared figure families",
        "",
        "Each canonical family/cohort is rendered once. Question pages contain links only. HVG and author assays stay separate. Paper figure assignments are narrative suggestions.",
        "",
        "| Question | Native cohort | Shared figure family | Status | Gallery |",
        "|---|---|---|---|---|",
    ]
    markdown.extend(
        "| "
        + r["question"]
        + " | "
        + r["cohort"]
        + " | "
        + r["category"]
        + " / "
        + r["family"]
        + " | "
        + r["status"]
        + " | [open]("
        + r["gallery"]
        + ") |"
        for r in csvrows
    )
    (root / "QUESTION_FIGURE_MAP.md").write_text("\n".join(markdown) + "\n")


def _family_output(root, family):
    # Handler distinguishes genuinely different measured/scenario methods with
    # the same view name. Scenario already disambiguates cohort directories.
    cohort = (
        family["cohort"]
        if family["cohort"] != "none"
        else "unavailable__" + family["handler"] + "__" + family["scenario"]
    )
    return root / "families" / family["category"] / family["family"] / safe_name(cohort)


def model_coverage(index, path):
    rows = []
    for d in sorted({c["dataset"] for c in index.get("cohorts", [])}):
        for scenario in [
            "measured",
            "reverse",
            "interpolation",
            "future",
            "history",
            "perturbation",
        ]:
            sources = [
                s
                for s in index.get("sources", [])
                if s.get("dataset") == d and s.get("scenario") == scenario
            ]
            for model in FM_MODELS:
                matches = [s for s in sources if s.get("model") == model]
                rows.append(
                    dict(
                        dataset=d,
                        scenario=scenario,
                        role="FM",
                        model=model,
                        panel="native",
                        available=bool(matches),
                        source=";".join(s.get("path", "") for s in matches),
                    )
                )
            if scenario == "measured":
                for panel in ["hvg", "author"]:
                    for model in TRANS_MODELS:
                        matches = [
                            s
                            for s in sources
                            if s.get("model") in (model, f"{model}_200", f"{model}_author")
                            and s.get("panel") == panel
                        ]
                        rows.append(
                            dict(
                                dataset=d,
                                scenario=scenario,
                                role="TRANS",
                                model=model,
                                panel=panel,
                                available=bool(matches),
                                source=";".join(s.get("path", "") for s in matches),
                            )
                        )
    csv_write(path, rows, ["dataset", "scenario", "role", "model", "panel", "available", "source"])
    return rows


def render_memory_estimate(tasks, options):
    """Conservative allowance for all native summaries in a comparison job."""
    sizes = {}
    largest = 0
    for _, cohort, _, _ in tasks:
        total = 0
        for source in {
            s["artifact_dir"]: s
            for s in cohort.get("sources", []) + cohort.get("reference_sources", [])
            if s.get("artifact_dir")
        }.values():
            directory = source["artifact_dir"]
            if directory not in sizes:
                rows = 0
                for name in ("stage_gene.parquet", "state_gene.parquet", "perturbation.parquet"):
                    p = Path(directory) / name
                    if p.is_file():
                        import pyarrow.parquet as pq

                        rows += pq.ParquetFile(p).metadata.num_rows
                # String/group columns expand considerably when read into pandas.
                sizes[directory] = (
                    rows * 256 + int(source.get("n_sample", 0)) * len(source.get("genes", [])) * 8
                )
            total += sizes[directory]
        largest = max(largest, total)
    return max(float(options.get("worker_memory_gb", 0.65)), 0.65 + largest / 2**30)


def run(config, index, options):
    from tqdm import tqdm

    root = trial_dir(config, options)
    plan = make_plan(
        index,
        config,
        options.get("question", "all"),
        options.get("dataset", "all"),
        options.get("panel", "all"),
    )
    family_plan = make_family_plan(plan, config, options, eligibility)
    family_digest = family_input_digest(family_plan, config)
    # Explicit resume must use the same configuration/analysis options.
    if options.get("resume_root"):
        previous = read_json(root / "RUN_CONFIG.json")
        compare = [
            "question",
            "dataset",
            "panel",
            "saved_only",
            "max_entities",
            "max_display_genes",
            "max_display_states",
            "max_views_per_entity",
            "max_response_genes",
            "profile",
            "formats",
            "variants",
            "model_policy",
            "selected_models",
            "scanpy",
            "scanpy_models",
            "max_stage_panels",
            "dpi",
            "sample_cells",
            "layouts",
            "background_alpha",
            "background_color",
            "allow_unannotated",
            "significant_only",
            "selection_q",
        ]
        changed = [k for k in compare if previous.get("options", {}).get(k) != options.get(k)]
        if previous.get("prepared_index_digest") != index_digest(index):
            changed.append("prepared source fingerprints")
        if previous.get("family_input_digest") != family_digest:
            changed.append("saved diagnostic/biological annotation files")
        if previous.get("code_digest") != code_digest():
            changed.append("renderer code")
        if previous.get("config") != config:
            changed.append("configuration/aliases/annotations")
        if changed:
            raise ValueError(
                "Resume inputs/options changed: "
                + ", ".join(changed)
                + ". Start a fresh trial instead."
            )
    atomic_json(
        root / "RUN_CONFIG.json",
        {
            "config": config,
            "options": options,
            "prepared_index_digest": index_digest(index),
            "family_input_digest": family_digest,
            "code_digest": code_digest(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    csv_write(
        root / "QUESTION_PLAN.csv",
        plan,
        ["question", "figure", "title", "cohort", "eligible", "reason"],
    )
    csv_write(
        root / "FAMILY_PLAN.csv",
        [{**p, "questions": ";".join(p["questions"])} for p in family_plan],
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
    model_coverage(index, root / "MODEL_COVERAGE.csv")
    diagnostics = []
    for cohort in index.get("cohorts", []):
        for source in cohort.get("sources", []):
            if native_panel(source) is None:
                diagnostics.append(
                    dict(
                        dataset=cohort.get("dataset"),
                        scenario=cohort.get("scenario"),
                        model=source.get("model"),
                        role=source.get("role"),
                        panel=source.get("panel", "unknown"),
                        path=source.get("path", source.get("artifact_dir", "")),
                        reason="Native HVG/author assay unresolved; excluded from both comparison settings",
                    )
                )
    csv_write(
        root / "COHORT_DIAGNOSTICS.csv",
        diagnostics,
        ["dataset", "scenario", "model", "role", "panel", "path", "reason"],
    )
    results = []
    tasks = []
    for row in family_plan:
        q = row["question_spec"]
        c = row["cohort_spec"]
        out = _family_output(root, row)
        if not row["eligible"]:
            r = {
                "question": row["family_id"],
                "family_id": row["family_id"],
                "family": row["family"],
                "category": row["category"],
                "questions": row["questions"],
                "title": q["title"],
                "figure": q["figure"],
                "cohort": row["cohort"],
                "status": "unsupported",
                "figures": [],
                "tables": [],
                "notes": [row["reason"]],
                "elapsed_s": 0,
                "output_dir": str(out),
            }
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out / "RESULT.json", r)
            atomic_json(
                out / "FAMILY.json",
                {k: v for k, v in row.items() if k not in ("cohort_spec", "question_spec")},
            )
            guide = interpretation_markdown(row, c)
            (out / "INTERPRETATION.md").write_text(guide)
            (out / "GUIDE.md").write_text(guide)
            question_gallery(out, r)
            results.append(r)
            continue
        result_file = out / "RESULT.json"
        if options.get("resume_root") and result_file.is_file():
            r = read_json(result_file)
            if (
                r["status"] == "complete"
                and r.get("figures")
                and all(
                    Path(f["path"] if isinstance(f, dict) else f).is_file()
                    for f in r.get("figures", [])
                )
                and all(Path(p).is_file() for p in r.get("tables", []))
            ):
                results.append(r)
                continue
        local = dict(
            options,
            dataset=c["dataset"],
            panel=c["panel"],
            reference_sources=c.get("reference_sources", []),
            dataset_config=config.get("datasets", {}).get(c["dataset"], {}),
            datasets=config.get("datasets", {}),
            aliases=config.get("aliases", {}),
        )
        # The cap was applied to requested views before unioning shared jobs.
        # Each atomic renderer must emit its full histogram/ECDF/etc. styles.
        local["max_views_per_entity"] = 0
        tasks.append((row, c, str(out), local))
    n, res = resources(
        options.get("workers", "auto"),
        options.get("cpu_reserve", 0),
        render_memory_estimate(tasks, options),
    )
    n = min(n, max(1, len(tasks)))
    print(f"New trial: {root}", flush=True)
    print(
        f'{len(tasks)} drawable shared family/cohort jobs for {len(plan)} question/cohort requests; {sum(not p["eligible"] for p in family_plan)} unsupported families. CPU workers={n}; {res}',
        flush=True,
    )
    _summary(root, results, plan, family_plan=family_plan)
    start = time.monotonic()
    last = 0
    last_summary = 0
    with tqdm(total=len(tasks), unit="family/cohort", dynamic_ncols=True, mininterval=1) as bar:
        with process_pool(n) as ex:
            pending = {ex.submit(_job, t): t for t in tasks}
            while pending:
                done, _ = cf.wait(pending, timeout=1, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    task = pending.pop(f)
                    try:
                        r = f.result()
                    except Exception as e:
                        fam, c, out, _ = task
                        r = {
                            "question": fam["family_id"],
                            "family_id": fam["family_id"],
                            "family": fam["family"],
                            "category": fam["category"],
                            "questions": fam["questions"],
                            "title": fam["title"],
                            "figure": fam["figure"],
                            "cohort": c["key"],
                            "status": "error",
                            "figures": [],
                            "notes": [str(e)],
                            "elapsed_s": 0,
                            "output_dir": str(out),
                        }
                        Path(out).mkdir(parents=True, exist_ok=True)
                        atomic_json(Path(out) / "RESULT.json", r)
                        question_gallery(out, r)
                    results.append(r)
                    bar.update(1)
                now = time.monotonic()
                if now - last >= 2:
                    active = []
                    exports = sum(len(r.get("figures", [])) for r in results)
                    for t in list(pending.values()):
                        f = Path(t[2]) / "PROGRESS.json"
                        if f.exists():
                            try:
                                e = read_json(f)
                                active.append(
                                    f'{e.get("question")}:{e.get("view",e.get("phase","loading"))} {e.get("entity","")}'
                                )
                                exports += int(e.get("figures", 0))
                            except (ValueError, OSError):
                                pass
                    bar.set_postfix_str(
                        f"active≤{min(n,len(pending))}; pending={len(pending)}; exports={exports}; elapsed={int(now-start)}s; "
                        + ("; ".join(active[:2]))
                    )
                    atomic_json(
                        root / "LIVE_PROGRESS.json",
                        dict(
                            completed=bar.n,
                            total=len(tasks),
                            exports_so_far=exports,
                            elapsed_s=round(now - start, 1),
                            active=active,
                        ),
                    )
                    last = now
                if done and now - last_summary >= 10:
                    _summary(root, results, plan, family_plan=family_plan)
                    last_summary = now
    archive_result = None
    if (
        options.get("include_legacy", True)
        and config.get("legacy_roots")
        and options.get("question", "all") == "all"
    ):
        from .archive import render_archive

        print("Preserving and redrawing legacy biological figures...", flush=True)
        archive_result = render_archive(config["legacy_roots"], root / "legacy", options)
    _summary(root, results, plan, archive_result, family_plan)
    atomic_json(
        root / "LIVE_PROGRESS.json",
        dict(
            status="finished",
            completed=len(tasks),
            total=len(tasks),
            exports_so_far=read_json(root / "FIGURE_MANIFEST.json")["n_unique_exports"],
            elapsed_s=round(time.monotonic() - start, 1),
            active=[],
            pending=0,
            complete_families=sum(r["status"] == "complete" for r in results),
            partial_families=sum(r["status"] == "partial" for r in results),
            error_families=sum(r["status"] == "error" for r in results),
        ),
    )
    print(f'Finished. Gallery: {root/"INDEX.html"}', flush=True)
    return root, results


def index_digest(index):
    import hashlib

    clean = {k: v for k, v in index.items() if k not in ["prepared_at", "created_at", "elapsed_s"]}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()


def family_input_digest(family_plan, config):
    """Resume identity for render-time evidence that bypasses prepare."""
    import hashlib

    paths = set()
    for spec in config.get("datasets", {}).values():
        for key, value in spec.items():
            if key.endswith("_path") and isinstance(value, str) and Path(value).is_file():
                paths.add(value)
    for family in family_plan:
        if family["handler"] != "saved_diagnostics" or family["cohort_spec"] is None:
            continue
        for source in family["cohort_spec"].get("sources", []):
            path = Path(source["path"])
            if path.is_file():
                paths.add(str(path))
            elif path.is_dir():
                paths.update(
                    str(p)
                    for p in path.iterdir()
                    if p.is_file() and p.suffix.lower() in (".csv", ".tsv", ".json")
                )
    records = [fingerprint(p) for p in sorted(paths)]
    return hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()
