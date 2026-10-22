"""A biological narrative and actual coverage report for each generated trial."""

from __future__ import annotations
from collections import Counter
from pathlib import Path
import json


def _relative(path, root):
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_interpretation_report(root, results, family_plan, question_plan):
    root = Path(root)
    by_id = {result["family_id"]: result for result in results}
    categories = {}
    for family in family_plan:
        categories.setdefault(family["category"], []).append(family)
    paths = set()
    for result in results:
        for figure in result.get("figures", []):
            path = figure.get("path") if isinstance(figure, dict) else figure
            if path:
                paths.add(str(Path(path).resolve()))
    status_counts = Counter(result.get("status", "unknown") for result in results)
    config_path = root / "RUN_CONFIG.json"
    aliases = (
        json.loads(config_path.read_text()).get("config", {}).get("aliases", {})
        if config_path.is_file()
        else {}
    )
    lines = [
        "# Biological interpretation report",
        "",
        "This report organizes the actual shared figure families in this trial into a suggested paper narrative. The folder organization follows biological methods; the suggested Figures 2–5 remain metadata. Questions can share a family without copying or rerendering its figures.",
        "",
        f"Coverage at this report update: **{len(paths)} unique declared figure exports**, **{len(results)} completed or blocked family records** of **{len(family_plan)} planned shared family/cohort jobs**, for **{len(question_plan)} question/cohort requests**.",
        "Family statuses: "
        + ", ".join(f"{name}={count}" for name, count in sorted(status_counts.items()))
        + ".",
        "A complete family means its requested supported rendering finished; it does not mean the model has passed a biological test. Partial/unavailable families preserve their reasons and available evidence. A report generated during an active run is provisional.",
        "",
        "## Suggested paper logic",
        "",
        "| Suggested main figure | Biological role | Read first | Evidence standard |",
        "|---|---|---|---|",
        "| Figure 2 | Establish observed developmental reconstruction | Fixed atlas, density, gene localization, distributions, temporal expression and classifier reliability | Match native-panel GT at measured ages; inspect classifier reliability before generated recoverability. |",
        "| Figure 3 | Explain reconstructed state and program organization | Named-state occupancy/markers, state trajectories, documented programs and prior-edge associations | Preserve GT-supported biology with native coverage and explicit reference/scoring contracts. Associations do not establish causality. |",
        "| Figure 4 | Explore continuous and bidirectional temporal development | Interpolation, autoregressive future **and history**, temporal descriptors, diversity, drift and finite-horizon coverage | Evaluate plausibility, continuity, biological change and numerical support. Unmeasured query stages have no independent truth. |",
        "| Figure 5 | Examine model response to documented edits | Direct targets, response genes/programs, prior-network context and saved input sensitivity | Interpret matched edited-minus-baseline sensitivity in documented input/export units; no experimental perturbation truth is supplied. |",
        "",
        "Measured reverse prediction can support reconstruction fidelity because its exported targets have GT. Autoregressive history is a separate unobserved reconstruction task. Old `F5_Q04`–`F5_Q09` identifiers are retained for compatibility while their history families are suggested for Figure 4.",
        "",
        "## Comparison contracts",
        "",
        "HVG and author assays are separate settings. Each native mosaic uses the available GT, v6/v7 slots and TRANS trained on that setting’s gene assay. No overlap-based transfer of truth or cross-assay FM is permitted. Display aliases change names, not canonical model identities, source paths or assay contracts.",
        "Actual sources, missing slots and native panel contracts are recorded in [MODEL_COVERAGE.csv](MODEL_COVERAGE.csv), [COHORT_DIAGNOSTICS.csv](COHORT_DIAGNOSTICS.csv) and each family’s MODEL_ROWS.csv/COHORT.json. A missing source is not a poor-performing model. Saved table discovery errors are recorded in [SAVED_DIAGNOSTIC_DISCOVERY.json](SAVED_DIAGNOSTIC_DISCOVERY.json).",
        "Interpret spatial overlays only within the same verified reference axis. Expression values, program normalization and cell-state methods must be compatible before comparing magnitudes. Supplied/template labels, independent reference labels and classifier recoverability answer different questions.",
        "Gene/state display selection uses documented biological eligibility and observed support where available. Omitted genes/states, native coverage and meaningful source names are audited; anonymous numeric codes are not promoted to guessed anatomical names. Future/history hypotheses cannot use nonexistent query-stage GT for selection.",
        "",
        "## Available native comparison cohorts",
        "",
        "| Native cohort | GT source count | Available FM labels and canonical slots | TRANS slots | Observed boundaries |",
        "|---|---|---|---|---|",
    ]
    cohort_sources = {}
    for family in family_plan:
        if family.get("cohort_spec") is not None:
            cohort_sources.setdefault(family["cohort"], family["cohort_spec"]["sources"])
    for name, sources in sorted(cohort_sources.items()):
        fms = list(
            dict.fromkeys(s.get("model", "unknown") for s in sources if s.get("role") == "FM")
        )
        fm_labels = (
            "; ".join(str(aliases.get(model, model)) + " (`" + str(model) + "`)" for model in fms)
            or "none available"
        )
        lines.append(
            "| "
            + " | ".join(
                _escape(value)
                for value in [
                    name,
                    sum(s.get("role") == "GT" for s in sources),
                    fm_labels,
                    len({s.get("model") for s in sources if s.get("role") == "TRANS"}),
                    sum(s.get("role") == "REFERENCE" for s in sources),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "Source counts describe saved inputs, not quality or comparison sufficiency. A boundary is not query-stage GT. Model selection options can restrict the displayed mosaic; inspect actual panel labels and MODEL_ROWS.csv for each family.",
        "",
        "## Shared main categories and available results",
        "",
    ]
    for category, families in sorted(categories.items()):
        lines += ["### " + category.replace("_", " ").capitalize(), ""]
        purposes = list(dict.fromkeys(family["interpretation"]["purpose"] for family in families))
        lines.extend(purposes)
        lines += [
            "",
            "| Canonical family | Native cohort | Suggested figure | Status | Exports | Questions answered | Gallery and evidence |",
            "|---|---|---|---|---|---|---|",
        ]
        for family in families:
            result = by_id.get(family["family_id"], {})
            status = result.get("status", "pending")
            count = len(result.get("figures", []))
            directory = (
                Path(result["output_dir"])
                if result.get("output_dir")
                else root / "families" / category / family["family"] / family["cohort"]
            )
            gallery = _relative(directory / "INDEX.html", root)
            guide = _relative(directory / "GUIDE.md", root)
            evidence = _relative(directory / "FIGURE_EVIDENCE.csv", root)
            links = f"[gallery]({gallery}) · [guide]({guide})"
            if (directory / "FIGURE_EVIDENCE.csv").is_file():
                links += f" · [CSV evidence]({evidence})"
            qids = ", ".join(family["questions"])
            lines.append(
                "| "
                + " | ".join(
                    _escape(value)
                    for value in [
                        family["family"],
                        family["cohort"],
                        family["figure"],
                        status,
                        count,
                        qids,
                        links,
                    ]
                )
                + " |"
            )
        lines += ["", "**Favorable behavior and scientific limits**", ""]
        for view in dict.fromkeys(family["family"] for family in families):
            spec = next(family["interpretation"] for family in families if family["family"] == view)
            lines += [f"- **{view}**: {spec['favorable']} {spec['caveat']}"]
        notes = []
        for family in families:
            result = by_id.get(family["family_id"], {})
            if result.get("status") in ("partial", "unsupported", "empty", "error"):
                reason = "; ".join(map(str, result.get("notes", []))) or family["reason"]
                notes.append(f"- `{family['family']}` / `{family['cohort']}`: {reason}")
        if notes:
            lines += ["", "**Current limitations and missing evidence**", ""] + notes
        lines.append("")
    lines += [
        "## How to evaluate a candidate biological figure",
        "",
        "1. Read the native cohort, source count and ground-truth status before comparing model panels.",
        "2. Inspect the named gene/state/program selection and native coverage audit. Check whether the relevant GT pattern or documented biological expectation is actually present.",
        "3. Read the adjacent GUIDE.md for metric definitions and direction. Means, expressing-cell fractions, dispersion, occupancy and associations should agree in biologically coherent ways; no single number replaces the biological pattern.",
        "4. Consult FIGURE_EVIDENCE.csv and TABLE_CATALOGUE.csv for the actual numerical tables, row counts and columns behind each export. Missing support, incompatible scaling or template-conditioned labels limit the conclusion.",
        "5. For tasks without matched GT, separate plausible change from numerical stability. Lowest drift, smoothest curves, strongest perturbations or agreement among FMs do not by themselves establish the best model.",
        "",
        "This package does not automatically filter figures or assign model winners. The CSV evidence is prepared for a later filtering workflow and for manual scientific inspection.",
        "",
        "## Machine-readable navigation",
        "",
        "- [QUESTION_FIGURE_MAP.json](QUESTION_FIGURE_MAP.json), [CSV](QUESTION_FIGURE_MAP.csv) and [Markdown](QUESTION_FIGURE_MAP.md): many-to-many question/family mapping.",
        "- [FIGURE_MANIFEST.json](FIGURE_MANIFEST.json): unique figure paths and shared question references.",
        "- [FAMILY_PLAN.csv](FAMILY_PLAN.csv) and [QUESTION_PLAN.csv](QUESTION_PLAN.csv): atomic-family versus whole-question eligibility.",
        "- [RUN_SUMMARY.csv](RUN_SUMMARY.csv): actual shared family results.",
        "- [INDEX.html](INDEX.html): all-model comparison galleries grouped by main category.",
        "",
    ]
    (root / "BIOLOGICAL_INTERPRETATION_REPORT.md").write_text("\n".join(lines))
