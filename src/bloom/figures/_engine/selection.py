"""Reference-led biological feature selection and auditable partial marker sets.

Nominal ANOVA from streamed cell summaries is a descriptive cell-level screen.
It does not account for biological replicate structure, repeated target cells,
or correlated pseudotrajectories and must not be called replicate-level DE.
No gene is selected because a model beats another model.
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from .plot import load_artifact, native_sources, cohort_panel

# Transferred verbatim membership/evidence from the supplied earlier project.
# These are example marker sets, not an exhaustive embryo annotation.
BUNDLED_EVIDENCE = [
    {
        "gene": "Hmga2",
        "source_kind": "development_literature",
        "selection_reason": "Neuroepithelial marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Reln",
        "source_kind": "development_literature",
        "selection_reason": "Cajal-Retzius marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Trp73",
        "source_kind": "development_literature",
        "selection_reason": "Cajal-Retzius marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Sox2",
        "source_kind": "development_literature",
        "selection_reason": "Radial-glia marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Pax6",
        "source_kind": "development_literature",
        "selection_reason": "Radial-glia marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Hes1",
        "source_kind": "development_literature",
        "selection_reason": "Radial-glia marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Hes5",
        "source_kind": "development_literature",
        "selection_reason": "Radial-glia marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Eomes",
        "source_kind": "development_literature",
        "selection_reason": "Intermediate-progenitor marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Btg2",
        "source_kind": "development_literature",
        "selection_reason": "Intermediate-progenitor marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Neurog2",
        "source_kind": "development_literature",
        "selection_reason": "Intermediate-progenitor marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Gadd45g",
        "source_kind": "development_literature",
        "selection_reason": "Intermediate-progenitor marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Dcx",
        "source_kind": "development_literature",
        "selection_reason": "Immature-neuron marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Neurod1",
        "source_kind": "development_literature",
        "selection_reason": "Immature-neuron marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Neurod2",
        "source_kind": "development_literature",
        "selection_reason": "Immature-neuron marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Neurod6",
        "source_kind": "development_literature",
        "selection_reason": "Immature-neuron marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Tubb3",
        "source_kind": "development_literature",
        "selection_reason": "Immature-neuron marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Tbr1",
        "source_kind": "development_literature",
        "selection_reason": "Immature-neuron marker",
        "source_url": "https://www.nature.com/articles/s41586-025-09644-1",
        "source_path": "",
        "exact_location": "Fig. 2a–c",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Fezf2",
        "source_kind": "development_literature",
        "selection_reason": "Deep-layer-neuron marker",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Results: cell-type characterization",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Bcl11b",
        "source_kind": "development_literature",
        "selection_reason": "Deep-layer-neuron marker",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Results: cell-type characterization",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Satb2",
        "source_kind": "development_literature",
        "selection_reason": "Upper-layer-neuron marker",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Results: cell-type characterization",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Dlx1",
        "source_kind": "development_literature",
        "selection_reason": "Interneuron developmental regulator",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Introduction",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Dlx2",
        "source_kind": "development_literature",
        "selection_reason": "Interneuron developmental regulator",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Introduction",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Lhx6",
        "source_kind": "development_literature",
        "selection_reason": "Interneuron marker",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Results: cell-type characterization",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Gad2",
        "source_kind": "development_literature",
        "selection_reason": "Inhibitory-neuron marker",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Fig. 1d–e",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Olig2",
        "source_kind": "development_literature",
        "selection_reason": "Oligodendrocyte-lineage marker",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Results: cell-type characterization",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Aqp4",
        "source_kind": "development_literature",
        "selection_reason": "Astrocyte marker",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Results: cell-type characterization",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Mki67",
        "source_kind": "development_literature",
        "selection_reason": "Proliferation marker",
        "source_url": "https://www.nature.com/articles/s41467-018-08079-9",
        "source_path": "",
        "exact_location": "Fig. 1d–e",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "cortex development",
    },
    {
        "gene": "Efnb2",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic arterial marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/",
        "source_path": "",
        "exact_location": "Fig. 7C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Sox17",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic arterial marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/",
        "source_path": "",
        "exact_location": "Fig. 7C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Gja4",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic arterial marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/",
        "source_path": "",
        "exact_location": "Fig. 7C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Dll4",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic arterial marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/",
        "source_path": "",
        "exact_location": "Fig. 7C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Mecom",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic arterial marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/",
        "source_path": "",
        "exact_location": "Fig. 7C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Ephb4",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic venous marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/",
        "source_path": "",
        "exact_location": "Fig. 7C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Nr2f2",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic venous marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/",
        "source_path": "",
        "exact_location": "Fig. 7C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Nrp2",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic venous marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/",
        "source_path": "",
        "exact_location": "Fig. 7C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Flrt2",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic venous marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10906678/",
        "source_path": "",
        "exact_location": "Fig. 7C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Gja5",
        "source_kind": "development_literature",
        "selection_reason": "Embryonic arterial marker",
        "source_url": "https://elifesciences.org/articles/102440",
        "source_path": "",
        "exact_location": "Methods: E17.5 EC annotation",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Cxcr4",
        "source_kind": "development_literature",
        "selection_reason": "Arterial enhancer-associated gene",
        "source_url": "https://elifesciences.org/articles/102440",
        "source_path": "",
        "exact_location": "Fig. 1; Introduction",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Cxcl12",
        "source_kind": "development_literature",
        "selection_reason": "Arterial enhancer-associated gene",
        "source_url": "https://elifesciences.org/articles/102440",
        "source_path": "",
        "exact_location": "Fig. 1; Introduction",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Nrp1",
        "source_kind": "development_literature",
        "selection_reason": "Arterial enhancer-associated gene",
        "source_url": "https://elifesciences.org/articles/102440",
        "source_path": "",
        "exact_location": "Fig. 1; Introduction",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Unc5b",
        "source_kind": "development_literature",
        "selection_reason": "Arterial enhancer-associated gene",
        "source_url": "https://elifesciences.org/articles/102440",
        "source_path": "",
        "exact_location": "Fig. 1; Introduction",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Acvrl1",
        "source_kind": "development_literature",
        "selection_reason": "Arterial enhancer-associated gene",
        "source_url": "https://elifesciences.org/articles/102440",
        "source_path": "",
        "exact_location": "Fig. 1; Introduction",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Sox7",
        "source_kind": "development_literature",
        "selection_reason": "Vascular developmental regulator",
        "source_url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0143650",
        "source_path": "",
        "exact_location": "Fig. 1; Fig. 4",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Sox18",
        "source_kind": "development_literature",
        "selection_reason": "Vascular developmental regulator",
        "source_url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0143650",
        "source_path": "",
        "exact_location": "Fig. 1; Fig. 4",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Etv2",
        "source_kind": "development_literature",
        "selection_reason": "Endothelial developmental regulator",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4142794/",
        "source_path": "",
        "exact_location": "Abstract",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Pecam1",
        "source_kind": "development_literature",
        "selection_reason": "Endothelial differentiation marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4142794/",
        "source_path": "",
        "exact_location": "Fig. 3C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
    {
        "gene": "Cdh5",
        "source_kind": "development_literature",
        "selection_reason": "Endothelial differentiation marker",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4142794/",
        "source_path": "",
        "exact_location": "Fig. 3C",
        "species": "mouse",
        "dataset_id": "*",
        "biological_context": "endothelial development",
    },
]

COLUMNS = [
    "gene",
    "score",
    "p_value",
    "q_value",
    "effect",
    "effect_range",
    "basis",
    "reason",
    "evidence",
    "annotation_source",
    "evidence_location",
    "source_kind",
    "species",
    "reference_panel",
    "eligible",
    "significant",
    "selected",
    "display_rank",
    "display_cap",
]


def _setting(cohort, options, name, default=None):
    if name in options:
        return options[name]
    selection = options.get("gene_selection", {})
    if name in selection:
        return selection[name]
    if name in cohort:
        return cohort[name]
    dataset = options.get(
        "dataset_config", options.get("datasets", {}).get(cohort.get("dataset"), {})
    )
    if name in dataset:
        return dataset[name]
    config = options.get("configuration", cohort.get("configuration", {}))
    dataset = config.get("datasets", {}).get(cohort.get("dataset"), {})
    return dataset.get(name, config.get(name, default))


def _read(path):
    path = Path(path)
    if path.suffix == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            if not obj or isinstance(obj[0], str):
                return pd.DataFrame({"gene": obj})
            return pd.DataFrame(obj)
        if isinstance(obj, dict):
            rows = []
            for program, genes in obj.items():
                for gene in genes:
                    rows.append({"program": program, "gene": gene})
            return pd.DataFrame(rows)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    lines = [
        s.strip() for s in path.read_text().splitlines() if s.strip() and not s.startswith("#")
    ]
    return pd.DataFrame({"gene": lines})


def _gene_col(frame):
    return next(
        (c for c in ("gene", "symbol", "gene_symbol", "gene_id", "gene_name") if c in frame), None
    )


def _dataset_rows(frame, cohort):
    for column in ("dataset_id", "dataset"):
        if column in frame:
            values = frame[column].fillna("*").astype(str).str.lower()
            return frame[values.isin(("*", "", str(cohort.get("dataset", "")).lower()))]
    return frame


def _species(cohort, options):
    """Respect explicit species and never infer it from gene capitalization."""
    supplied = _setting(cohort, options, "species")
    if supplied:
        return str(supplied).strip().lower()
    declared = {
        str(s["species"]).strip().lower() for s in cohort.get("sources", []) if s.get("species")
    }
    if len(declared) == 1:
        return next(iter(declared))
    # These project dataset identities are documented mouse datasets. Big
    # retains author-list evidence, but needs source/config species provenance
    # before the mouse-only literature fallback is applied.
    dataset = str(cohort.get("dataset", "")).strip().lower()
    return "mouse" if dataset in ("small", "small_visual_cortex", "sted", "sted-ec") else "unknown"


def _display_cap(options):
    caps = [int(options.get("max_entities", 10)), int(options.get("max_display_genes", 20))]
    if any(cap < 0 for cap in caps):
        raise ValueError("Gene display limits must be nonnegative; zero means uncapped")
    positive = [cap for cap in caps if cap > 0]
    return min(positive) if positive else 0


def _text(value):
    return "" if value is None or pd.isna(value) else str(value)


def _bh(pvalues):
    values = np.asarray(pvalues, dtype=float)
    out = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return out
    order = valid[np.argsort(values[valid], kind="stable")]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    out[order] = np.minimum(1, np.minimum.accumulate(adjusted[::-1])[::-1])
    return out


def _summary_signal(frame, group_columns, min_n=2):
    """Compute one-way cell-summary ANOVA, standardized range, absolute range."""
    result = {}
    if frame.empty or not {"gene", "mean"}.issubset(frame.columns):
        return result
    grouping = [c for c in group_columns if c in frame]
    if not grouping:
        return result
    for gene, rows in frame.groupby("gene", sort=False, observed=True):
        # Avoid double counting when a source happens to carry duplicated rows.
        rows = rows.drop_duplicates(grouping)
        means = pd.to_numeric(rows["mean"], errors="coerce").to_numpy(float)
        counts = (
            pd.to_numeric(rows["n"], errors="coerce").to_numpy(float)
            if "n" in rows
            else np.ones(len(rows))
        )
        std = (
            pd.to_numeric(rows["std"], errors="coerce").to_numpy(float)
            if "std" in rows
            else np.full(len(rows), np.nan)
        )
        valid = np.isfinite(means) & np.isfinite(counts) & (counts >= min_n)
        means, counts, std = means[valid], counts[valid], std[valid]
        if not len(means):
            continue
        absolute = float(np.ptp(means))
        weight = np.maximum(counts - 1, 0)
        pooled_var = (
            float(np.nansum(weight * std**2) / weight.sum())
            if weight.sum() > 0 and np.all(np.isfinite(std))
            else np.nan
        )
        scale = np.sqrt(pooled_var) if np.isfinite(pooled_var) and pooled_var > 1e-12 else np.nan
        effect = absolute / scale if np.isfinite(scale) else absolute
        pvalue = np.nan
        if len(means) >= 2 and np.all(np.isfinite(std)):
            total = counts.sum()
            between = float(np.sum(counts * (means - np.average(means, weights=counts)) ** 2))
            within = float(np.sum(weight * std**2))
            df1, df2 = len(means) - 1, total - len(means)
            if df2 > 0:
                if within <= 1e-20:
                    pvalue = 0.0 if between > 1e-20 else 1.0
                else:
                    statistic = (between / df1) / (within / df2)
                    pvalue = float(f_distribution.sf(statistic, df1, df2))
        result[str(gene)] = {
            "p": pvalue,
            "effect": effect,
            "range": absolute,
            "groups": len(means),
            "n": float(counts.sum()),
        }
    return result


def _reference_sources(cohort, options):
    sources = list(cohort.get("sources", []))
    supplied = options.get("reference_sources", cohort.get("reference_sources", []))
    if supplied:
        sources = list(supplied) + sources
    sources = native_sources(cohort, options, sources=sources)
    # A hidden GT and the visible GT may name the same prepared artifact.
    unique = {}
    for source in sources:
        identity = str(source.get("artifact_dir", source.get("source_id", source.get("model", ""))))
        unique.setdefault(identity, source)
    sources = list(unique.values())
    gt = [s for s in sources if s.get("role") in ("GT", "REFERENCE")]
    if gt:
        return gt, "observed full-row summaries; cell-level descriptive screen"
    # FM-only scenarios remain drawable. This is explicitly not a GT screen.
    fm = [s for s in sources if s.get("role") == "FM"]
    fm.sort(key=lambda s: str(s.get("model", "")))
    if fm:
        return (
            fm[:1],
            "prediction-only descriptive screen; no observed reference summaries available",
        )
    return sources[:1], "source-only descriptive screen; no observed reference summaries available"


def select_genes(cohort, options=None):
    """Rank biologically annotated genes by observed stage/state signal.

    Significant genes are chosen first; if fewer than the requested cap exist,
    annotated genes with nonzero reference effects backfill and are explicitly
    labelled effect-ranked rather than statistically significant. Both entity
    and named-gene display limits must be zero for an exhaustive display. The
    audit always includes the complete native gene axis.
    """
    options = options or {}
    artifacts = [load_artifact(s) for s in native_sources(cohort, options)]
    universe = set(g for a in artifacts for g in a["genes"])
    author = set(
        g
        for a in artifacts
        if str(a["meta"].get("panel", "")).lower() == "author"
        for g in a["genes"]
    )
    species = _species(cohort, options)
    mouse = species in ("mouse", "mus musculus", "mus_musculus", "mm", "mmusculus")
    evidence = {r["gene"]: r for r in BUNDLED_EVIDENCE} if mouse else {}
    author_evidence = {}
    supplied_membership = (
        Path(__file__).resolve().parent / "data" / "author_membership_from_supplied_code.tsv"
    )
    if supplied_membership.is_file():
        frame = _dataset_rows(_read(supplied_membership), cohort)
        author_evidence = {str(r["gene"]): r.to_dict() for _, r in frame.iterrows()}
        author.update(author_evidence)
    evidence_path = _setting(cohort, options, "gene_evidence_path")
    if evidence_path and Path(evidence_path).is_file():
        frame = _dataset_rows(_read(evidence_path), cohort)
        col = _gene_col(frame)
        if col:
            evidence.update({str(r[col]): r.to_dict() for _, r in frame.iterrows()})
    author_path = _setting(cohort, options, "author_genes_path")
    if author_path and Path(author_path).is_file():
        frame = _dataset_rows(_read(author_path), cohort)
        col = _gene_col(frame)
        if col:
            author.update(frame[col].dropna().astype(str))
    programs_path = _setting(cohort, options, "programs_path")
    if programs_path and Path(programs_path).is_file():
        frame = _dataset_rows(_read(programs_path), cohort)
        col = _gene_col(frame)
        if col:
            for gene in frame[col].dropna().astype(str).unique():
                evidence.setdefault(
                    gene,
                    {
                        "gene": gene,
                        "selection_reason": "supplied biological program membership",
                        "source_url": str(programs_path),
                    },
                )
    reference_sources, basis = _reference_sources(cohort, options)
    observed_screen = any(s.get("role") in ("GT", "REFERENCE") for s in reference_sources)
    min_n = int(_setting(cohort, options, "min_group_cells", 2))
    signals = defaultdict(list)
    reference_temporal = False
    for source in reference_sources:
        artifact = load_artifact(source)
        if (
            source.get("role") in ("GT", "REFERENCE")
            and "time" in artifact["stage"]
            and artifact["stage"]["time"].nunique() >= 2
        ):
            reference_temporal = True
        for scope, table, group_cols in [
            ("time", artifact["stage"], ["time"]),
            ("state", artifact["state"], ["state", "time"]),
        ]:
            for gene, signal in _summary_signal(table, group_cols, min_n).items():
                signal["scope"] = scope
                signals[gene].append(signal)
    # A single saved boundary cannot screen developmental change. Display known
    # marker/author genes varying in the first canonical generated trajectory,
    # explicitly as a prediction-only exploratory ranking with no p/q claims.
    prediction_signals = {}
    scenario = str(options.get("scenario", cohort.get("scenario", "")))
    if scenario in ("future", "history", "interpolation") and not reference_temporal:
        generated = [a for a in artifacts if a["meta"].get("role") in ("FM", "legacy", "TRANS")]
        generated.sort(
            key=lambda a: (a["meta"].get("role") != "FM", str(a["meta"].get("model", "")))
        )
        for artifact in generated:
            for gene, signal in _summary_signal(artifact["stage"], ["time"], min_n).items():
                if signal["range"] > 0:
                    prediction_signals.setdefault(
                        gene,
                        {
                            **signal,
                            "scope": "generated time",
                            "model": artifact["meta"].get("model", "unknown"),
                        },
                    )
    rows = []
    allow_unannotated = bool(_setting(cohort, options, "allow_unannotated", False))
    for gene in sorted(universe):
        gene_signals = signals.get(gene, [])
        best = max(
            gene_signals,
            key=lambda s: (s["effect"], s["range"]),
            default={"effect": 0.0, "range": 0.0, "p": np.nan, "scope": "none"},
        )
        finite_p = [s["p"] for s in gene_signals if np.isfinite(s["p"])]
        # Correct the within-gene choice among overlapping stage/state screens.
        pvalue = min(1.0, min(finite_p) * len(gene_signals)) if finite_p else np.nan
        if not observed_screen:
            pvalue = np.nan
        gene_basis = basis + "; strongest " + best["scope"] + " signal"
        if gene in prediction_signals:
            best = prediction_signals[gene]
            pvalue = np.nan
            gene_basis = (
                "single observed boundary cannot screen temporal change; prediction-only descriptive temporal variation in "
                + str(best["model"])
                + "; no ground-truth significance or accuracy claim"
            )
        annotated = gene in author or gene in evidence
        eligible = annotated or allow_unannotated
        annotation_record = author_evidence.get(gene, evidence.get(gene, {}))
        annotation = (
            "member of supplied " + str(cohort.get("dataset", "")) + " author gene list"
            if gene in author_evidence
            else (
                "declared author-panel membership"
                if gene in author
                else str(
                    annotation_record.get("selection_reason", "unannotated exploratory feature")
                )
            )
        )
        reference = (
            _text(annotation_record.get("source_url"))
            or _text(annotation_record.get("source_path"))
            or str(author_path or "native author gene axis" if gene in author else "")
        )
        location = _text(annotation_record.get("exact_location"))
        rows.append(
            {
                "gene": gene,
                "score": float(best["effect"]),
                "p_value": pvalue,
                "effect": float(best["effect"]),
                "effect_range": float(best["range"]),
                "basis": gene_basis,
                "reason": annotation,
                "evidence": reference,
                "annotation_source": _text(annotation_record.get("transferred_from")) or reference,
                "evidence_location": location,
                "source_kind": _text(annotation_record.get("source_kind"))
                or ("native_author_axis" if gene in author else "exploratory"),
                "species": species,
                "reference_panel": cohort_panel(cohort, options),
                "eligible": bool(eligible),
                "selected": False,
            }
        )
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.DataFrame(rows)
    frame["q_value"] = _bh(frame.p_value)
    alpha = float(_setting(cohort, options, "selection_q", 0.05))
    min_effect = float(_setting(cohort, options, "selection_min_effect", 0.1))
    frame["significant"] = frame.q_value.le(alpha) & frame.effect.ge(min_effect) & frame.eligible
    frame.sort_values(
        ["eligible", "significant", "score", "gene"],
        ascending=[False, False, False, True],
        kind="stable",
        inplace=True,
    )
    cap = _display_cap(options)
    frame["display_cap"] = cap
    frame["display_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    candidates = frame[frame.eligible]
    if bool(_setting(cohort, options, "significant_only", False)):
        candidates = candidates[candidates.significant]
    # Flat markers remain audited but do not generate thousands of empty maps.
    # Explicit all-marker display is available when that absence is the question.
    elif not bool(_setting(cohort, options, "include_flat_markers", False)):
        candidates = candidates[candidates.effect_range.gt(0)]
    chosen = candidates if cap == 0 else candidates.head(max(cap, 0))
    frame.loc[chosen.index, "selected"] = True
    frame.loc[chosen.index, "display_rank"] = np.arange(1, len(chosen) + 1)
    for index in frame.index:
        if not frame.at[index, "eligible"]:
            suffix = "; excluded: absent from author list and bundled/supplied marker evidence"
        elif frame.at[index, "selected"] and frame.at[index, "significant"]:
            suffix = "; cell-summary nominal ANOVA/BH screen and nonzero reference effect"
        elif frame.at[index, "selected"]:
            suffix = "; effect-ranked descriptive display; significance not established"
        else:
            suffix = (
                "; omitted from named display: reference effect is zero"
                if frame.at[index, "effect_range"] <= 0
                and not bool(_setting(cohort, options, "include_flat_markers", False))
                else "; omitted from named display by reference-led gene cap "
                + (str(cap) if cap else "uncapped")
                + " or significance-only policy"
            )
        frame.at[index, "reason"] += suffix
    return frame[COLUMNS].reset_index(drop=True)


def select_programs(cohort, options=None):
    """Return covered marker/program dictionaries with original denominators.

    User-supplied definitions take precedence. Example literature marker panels
    are a labelled fallback, not independent cell identity calls or regulatory
    activity. At least two covered members are required by default.
    """
    options = options or {}
    artifacts = [load_artifact(s) for s in native_sources(cohort, options)]
    universe = set(g for a in artifacts for g in a["genes"])
    path = _setting(cohort, options, "programs_path")
    definitions = []
    if path and Path(path).is_file():
        frame = _dataset_rows(_read(path), cohort)
        gcol = _gene_col(frame)
        pcol = next(
            (c for c in ("program", "module", "entity", "marker_set", "cell_type") if c in frame),
            None,
        )
        if gcol and pcol:
            wcol = next(
                (
                    c
                    for c in ("weight", "prior_weight", "signed_weight", "importance", "score")
                    if c in frame
                ),
                None,
            )
            for name, rows in frame.groupby(pcol, sort=False, observed=True):
                members = list(dict.fromkeys(rows[gcol].dropna().astype(str)))
                warning = []
                if wcol is not None:
                    weights = dict.fromkeys(members, 0.0)
                    signed_weights = dict.fromkeys(members, 0.0)
                    numeric = pd.to_numeric(rows[wcol], errors="coerce").to_numpy(float)
                    missing = int((~np.isfinite(numeric)).sum())
                    if missing:
                        warning.append(
                            f"{missing} nonfinite supplied weights excluded from scoring; membership retained"
                        )
                    for gene, weight in zip(rows[gcol], numeric):
                        if pd.isna(gene) or not np.isfinite(weight):
                            continue
                        gene = str(gene)
                        weights[gene] += abs(float(weight))
                        signed_weights[gene] += float(weight)
                    scoring_basis = "normalized absolute supplied weights"
                    duplicate_policy = (
                        "sum absolute row weights by exact gene; unique-member coverage denominator"
                    )
                else:
                    weights = {gene: 1.0 for gene in members}
                    signed_weights = dict(weights)
                    scoring_basis = "uniform membership weights; no supplied weight column"
                    duplicate_policy = "unit weight per unique member"
                definitions.append(
                    {
                        "program": str(name),
                        "members": members,
                        "weights": weights,
                        "raw_signed_weights": signed_weights,
                        "weight_column": wcol,
                        "scoring_basis": scoring_basis,
                        "duplicate_weight_policy": duplicate_policy,
                        "input_rows": int(len(rows)),
                        "weight_warning": warning,
                        "basis": "supplied program membership",
                        "evidence": str(path),
                    }
                )
    species = _species(cohort, options)
    if not definitions and species in ("mouse", "mus musculus", "mus_musculus", "mm", "mmusculus"):
        groups = defaultdict(list)
        citations = defaultdict(set)
        for row in BUNDLED_EVIDENCE:
            groups[row["selection_reason"]].append(row["gene"])
            citations[row["selection_reason"]].add(row["source_url"])
        for name, members in groups.items():
            if len(members) >= 2:
                definitions.append(
                    {
                        "program": name,
                        "members": members,
                        "weights": {gene: 1.0 for gene in members},
                        "raw_signed_weights": {gene: 1.0 for gene in members},
                        "weight_column": None,
                        "scoring_basis": "uniform marker weights",
                        "duplicate_weight_policy": "unit weight per unique member",
                        "input_rows": len(members),
                        "weight_warning": [],
                        "basis": "bundled literature example marker panel; not a regulon",
                        "evidence": "; ".join(sorted(citations[name])),
                    }
                )
    min_members = int(_setting(cohort, options, "program_min_genes", 2))
    outputs = []
    for definition in definitions:
        members = definition["members"]
        covered = [g for g in members if g in universe]
        if len(covered) < min_members:
            continue
        per_model = {
            str(a["meta"].get("model", a["source"].get("model", "unknown"))): [
                g for g in members if g in a["genes"]
            ]
            for a in artifacts
        }
        weights = definition["weights"]
        full_weight_mass = float(sum(weights.values()))
        covered_weight_mass = float(sum(weights[g] for g in covered))
        model_masses = {
            model: float(sum(weights[g] for g in genes)) for model, genes in per_model.items()
        }
        normalized = {
            model: (
                {g: float(weights[g] / model_masses[model]) for g in genes}
                if model_masses[model] > 0
                else {}
            )
            for model, genes in per_model.items()
        }
        outputs.append(
            {
                **definition,
                "genes": covered,
                "covered_genes": covered,
                "total": len(members),
                "n_covered": len(covered),
                "coverage": len(covered) / len(members),
                "coverage_by_model": per_model,
                "full_weight_mass": full_weight_mass,
                "covered_weight_mass": covered_weight_mass,
                "weighted_coverage": (
                    covered_weight_mass / full_weight_mass if full_weight_mass > 0 else None
                ),
                "weight_mass_by_model": model_masses,
                "normalized_weights_by_model": normalized,
                "n_scoring_members": sum(weights[g] > 0 for g in covered),
                "model_mask": {
                    model: {g: g in genes for g in members} for model, genes in per_model.items()
                },
                "weight_formula": "sum_g abs(w_g)*expression_g / sum_g abs(w_g), normalized over covered native genes; duplicate row weights aggregated per gene",
                "reason": f"{definition['basis']}; {len(covered)}/{len(members)} native-panel members represented; {definition['scoring_basis']}; restricted-panel expression signature",
            }
        )
    outputs.sort(key=lambda r: (-r["n_covered"], r["program"]))
    cap = int(options.get("max_entities", 10))
    return outputs if cap == 0 else outputs[: max(cap, 0)]
