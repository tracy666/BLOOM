"""Verify real output paths, independent of eligibility and success declarations."""

from __future__ import annotations
import csv, json
from pathlib import Path
from collections import Counter
from .util import atomic_json


def _path(value, directory):
    p = Path(value)
    if p.is_file():
        return p.resolve()
    if not p.is_absolute():
        candidate = directory / p
        if candidate.is_file():
            return candidate
    # Support moving an entire trial or historical relative-path ledgers.
    positions = [i for i, part in enumerate(p.parts) if part == directory.name]
    for i in reversed(positions):
        candidate = directory.joinpath(*p.parts[i + 1 :])
        if candidate.is_file():
            return candidate
    return p if p.is_absolute() else directory / p


def verify(root):
    root = Path(root).resolve()
    problems = []
    missing = []
    exports = []
    byfigure = Counter()
    questionfigures = Counter()
    views = Counter()
    statuses = Counter()
    canonical = root / "families" if (root / "families").exists() else root / "questions"
    seen = set()
    for p in sorted(canonical.rglob("RESULT.json")) if canonical.exists() else []:
        try:
            r = json.loads(p.read_text())
        except Exception as e:
            problems.append(f"{p}: {e}")
            continue
        statuses[r.get("status", "unknown")] += 1
        for f in r.get("figures", []):
            path = _path(f["path"] if isinstance(f, dict) else f, p.parent)
            if str(path) in seen:
                problems.append(f"Duplicate canonical figure declaration: {path}")
                continue
            seen.add(str(path))
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
                continue
            if path.suffix.lower() in (".png", ".gif"):
                try:
                    from PIL import Image

                    with Image.open(path) as im:
                        im.verify()
                except Exception as e:
                    problems.append(f"Unreadable image {path}: {e}")
            byfigure[str(r.get("figure", "?"))] += 1
            for paper in r.get("paper_figures", [r.get("figure", "?")]):
                questionfigures[str(paper)] += 1
            views[f.get("view", "unknown") if isinstance(f, dict) else "unknown"] += 1
            exports.append(str(path))
        if r.get("status") == "complete" and not r.get("figures"):
            problems.append(f"{p}: complete status with zero figures")
        if r.get("status") == "error":
            problems.append(
                f'{r.get("family",r.get("question"))}/{r.get("cohort")}: '
                + ";".join(r.get("notes", []))
            )
        cohort_file = p.parent / "COHORT.json"
        if cohort_file.is_file():
            cohort = json.loads(cohort_file.read_text())
            native = cohort.get("panel")
            from .plot import source_panel

            for source in cohort.get("sources", []) + cohort.get("reference_sources", []):
                if native in ("hvg", "author") and source_panel(source) != native:
                    problems.append(
                        f'{p.parent}: source {source.get("model")} is outside native {native} assay'
                    )
    legacy_summary = None
    legacy_paths = set()
    preserved = set()
    redrawn = set()
    directories = [root / "legacy", root]
    # Standalone CLI legacy creates trial/legacy; verifying trial works too.
    for directory in directories:
        report = next(
            (
                directory / n
                for n in [
                    "ARCHIVE_REPORT.json",
                    "LEGACY_COVERAGE_SUMMARY.json",
                    "COVERAGE_SUMMARY.json",
                ]
                if (directory / n).is_file()
            ),
            None,
        )
        if not report:
            continue
        legacy_summary = json.loads(report.read_text())
        ledger = directory / "LEGACY_COVERAGE.csv"
        if not ledger.is_file():
            ledger = directory / "LEGACY_COVERAGE_LEDGER.csv"
        if legacy_summary.get("preserved_originals", 0) > 0 and not ledger.is_file():
            problems.append("Reported preserved originals have no coverage ledger")
        for file, field, dest in [
            (ledger, "preserved_path", preserved),
            (directory / "BIOLOGICAL_TABLE_RERENDER_LEDGER.csv", "rerendered_paths", redrawn),
        ]:
            if not file.is_file():
                if (
                    field == "rerendered_paths"
                    and legacy_summary.get("rerendered_biological_figures", 0) > 0
                ):
                    problems.append("Reported biological re-renders have no table ledger")
                continue
            with file.open(newline="") as f:
                for row in csv.DictReader(f):
                    for value in row.get(field, "").split(";"):
                        if not value:
                            continue
                        actual = _path(value, directory)
                        dest.add(str(actual))
                        legacy_paths.add(str(actual))
                        if not actual.is_file() or actual.stat().st_size == 0:
                            missing.append(str(actual))
        if len(preserved) != legacy_summary.get("preserved_originals", len(preserved)):
            problems.append("Preserved-original ledger count differs from archive report")
        if len(redrawn) != legacy_summary.get("rerendered_biological_figures", len(redrawn)):
            problems.append("Biological re-render ledger count differs from archive report")
        if not legacy_summary.get("coverage_complete", False):
            problems.append("Legacy original coverage is incomplete")
        if legacy_summary.get("errors"):
            problems.append("Legacy biological table rendering reported errors")
        break
    if missing:
        problems.append(f"{len(set(missing))} declared figure paths are missing or empty")
    manifest_path = root / "FIGURE_MANIFEST.json"
    if canonical.name == "families":
        if not manifest_path.is_file():
            problems.append("Shared-family trial has no unique figure manifest")
        else:
            manifest = json.loads(manifest_path.read_text())
            records = manifest.get("figures", [])
            ids = [r.get("figure_id") for r in records]
            paths = [str(_path(r.get("path", ""), root)) for r in records]
            if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
                problems.append("Figure manifest contains duplicate IDs or paths")
            if set(paths) != set(exports):
                problems.append("Unique figure manifest differs from declared family exports")
            if manifest.get("n_unique_exports") != len(records):
                problems.append("Figure manifest export count is inconsistent")
            mapping_path = root / "QUESTION_FIGURE_MAP.json"
            if not mapping_path.is_file():
                problems.append("Shared-family trial has no question mapping")
            else:
                mapping = json.loads(mapping_path.read_text())
                for question in mapping.get("questions", []):
                    for family in question.get("families", []):
                        if set(family.get("figure_ids", [])) - set(ids):
                            problems.append(
                                f'Unknown mapped figure ID in {question.get("question")}'
                            )
        if list((root / "questions").rglob("*.png")):
            problems.append("Question link folders contain duplicate PNG exports")
    if not exports and not any(
        Path(p).is_file() and Path(p).stat().st_size > 0 for p in legacy_paths
    ):
        problems.append("No actual biological exports found")
    result = {
        "valid": not problems,
        "result_root": str(root),
        "unique_new_exports": len(exports),
        "new_exports_by_figure": dict(byfigure),
        "question_figure_coverage": dict(questionfigures),
        "coverage_note": "Primary suggested figure counts partition exports; question figure coverage may overlap because one family answers multiple questions.",
        "rendered_views": dict(views),
        "family_status_counts" if canonical.name == "families" else "question_status_counts": dict(
            statuses
        ),
        "legacy": legacy_summary,
        "legacy_declared_files_checked": len(legacy_paths),
        "problems": problems,
        "missing_paths": sorted(set(missing)),
    }
    atomic_json(root / "VERIFICATION.json", result)
    return result
