"""Canonical, shared figure families and their biological interpretation.

Questions are a many-to-many index of these render jobs.  The unit of work is
one canonical view in one native assay cohort, never one question.
"""

from __future__ import annotations

SCENARIO_ALIASES = {
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
    "horizon": "drift",
}

# Requirements apply to an atomic view.  A missing graph annotation must not
# suppress a drawable expression atlas merely because both answer one question.
FAMILY_SPECS = {}


def _add(category, views, requirements, purpose, favorable, caveat):
    for view in views.split(","):
        FAMILY_SPECS[view] = dict(
            category=category,
            requires=requirements.split(",") if requirements else [],
            purpose=purpose,
            favorable=favorable,
            caveat=caveat,
        )


_add(
    "developmental_atlas",
    "atlas_time,geometry,stage_storyboard,storyboard,animation",
    "cells",
    "Locate saved populations across developmental stages on fixed observed-reference coordinates.",
    "For measured ages, preserve reference-supported regions and age-specific population organization; for unobserved ages, inspect continuous movement and plausible reference proximity.",
    "Projection geometry is descriptive expression similarity. It does not establish lineage, fate or accuracy at unobserved stages.",
)
_add(
    "cell_state_atlas",
    "atlas_state",
    "cells,state",
    "Locate named cell-state regions in the observed-reference atlas.",
    "Preserve GT-supported state locations, coverage and stage-specific organization under the same state annotation method.",
    "Template-conditioned labels cannot independently prove autonomous cell-state generation; numeric labels need an explicit vocabulary.",
)
_add(
    "state_trajectory",
    "trajectory",
    "cells",
    "Compare sampled population paths and developmental centroids.",
    "At observed stages, paths and centroids follow the matching GT; away from observations, smooth paths with retained diversity are useful plausibility checks.",
    "Population centroid paths do not trace individual cells or establish lineages.",
)
_add(
    "state_trajectory",
    "state_trajectory,state_graph",
    "cells,state",
    "Compare named state-specific trajectories and expression-neighborhood connectivity.",
    "Retain observed state neighborhoods, developmental ordering and supported connectivity without losing minority states.",
    "Neighborhood edges are associations rather than causal transitions; coordinates must share a verified reference.",
)
_add(
    "population_density",
    "density,density_residual,density_contour",
    "cells",
    "Inspect atlas occupancy, overrepresentation and missing population regions.",
    "Measured reconstructions preserve matching GT support; residuals are small in biologically populated regions. Unobserved populations retain support without collapsing to one point.",
    "Density depends on the declared drawing sample, weights, smoothing and common reference; unobserved density has no independent accuracy target.",
)
_add(
    "population_density",
    "density_state",
    "cells,state",
    "Inspect density and coverage in GT-supported named cell-state regions.",
    "Preserve common and minority observed regions with compatible state labels and sampling weights.",
    "A supplied state label is not an independently recovered identity; compare only the same annotation method.",
)
_add(
    "gene_localization",
    "gene_feature",
    "expression,cells",
    "Show where reference-selected biological genes localize across the atlas.",
    "Retain GT-supported expression domains, heterogeneity and developmental localization on matching native gene axes.",
    "Expression colors use stored units and declared samples; a gene outside a model native panel cannot be evaluated.",
)
_add(
    "gene_histogram",
    "gene_distribution",
    "expression",
    "Compare gene-expression distributions with histogram, violin and ECDF variants where implemented.",
    "At measured stages, retain GT range, modes, tails and heterogeneity rather than matching the mean alone. For unobserved stages, avoid unsupported collapse or extreme distributional drift.",
    "Cell samples are descriptive draws, not independent biological replicates; model variance is not experimental uncertainty.",
)
_add(
    "gene_dynamics",
    "gene_timecourse",
    "stage",
    "Compare temporal gene means and available dispersion across stored developmental ages.",
    "At measured ages, preserve GT temporal peaks and trends. At unobserved ages, inspect plausible extension, continuity and compatibility with known gene evidence.",
    "A smooth or flat extrapolation is not automatically biologically correct; no unobserved truth means no accuracy claim.",
)
_add(
    "gene_dynamics",
    "gene_detection",
    "stage",
    "Compare expressing-cell fractions alongside mean expression.",
    "Preserve observed stage-specific fractions together with expression magnitude.",
    "Fraction > 0 describes saved expression units; it is not molecular RNA detection when inputs are scaled.",
)
_add(
    "gene_dynamics",
    "gene_phase",
    "stage",
    "Compare matched stage expression or expression/fraction relationships.",
    "Preserve GT-supported stage-specific expression relationships.",
    "A relationship across stages is descriptive and must use matched stage and scale definitions.",
)
_add(
    "gene_cascade",
    "cascade,gene_heatmap",
    "stage",
    "Gather selected developmental genes into a temporal cascade or heatmap.",
    "Measured data preserve reference gene activation ordering. Unobserved extensions maintain interpretable temporal structure without unsupported abrupt changes.",
    "Reference-frozen scaling is required for comparisons; an ordered heatmap is not a regulatory mechanism.",
)
_add(
    "state_markers",
    "marker_dotplot,marker_matrix,marker_violin",
    "expression,state",
    "Gather selected named genes against GT-supported named states using dotplot, matrix and violin styles.",
    "Preserve observed marker specificity, expression magnitude, positive fractions and within-state heterogeneity.",
    "Full-row summaries can use template strata, while sampled violins use declared inferred labels; check MODEL_ROWS.csv before treating this as autonomous state recovery.",
)
_add(
    "state_occupancy",
    "state_occupancy",
    "cells,state",
    "Compare developmental occupancy of selected GT-supported named states.",
    "Retain observed state proportions and temporal composition, including minority populations.",
    "Occupancy uses declared sample weights and state inference; template-conditioned labels cannot measure independent composition prediction.",
)
_add(
    "gene_programs",
    "program_heatmap,program_timecourse,program_feature,programs",
    "expression",
    "Compare documented author/marker program activity over time and atlas regions.",
    "At measured stages, retain observed coordinated activity and localization. At unobserved stages, inspect plausible program evolution and adequate native gene coverage.",
    "Check program membership, covered weight mass and scoring units; partially covered or mixed raw/z scores are not directly interchangeable.",
)
_add(
    "regulatory_associations",
    "coexpression",
    "expression",
    "Compare selected gene coexpression after within-age centering.",
    "Preserve observed associations among biologically eligible native genes.",
    "Correlation is associative and does not establish regulation or causality.",
)
_add(
    "regulatory_associations",
    "grn_activity,grn_support,grn_network,edge_rescue",
    "expression,edges",
    "Inspect supplied TF-target programs, expression-supported prior edges and reconstruction differences.",
    "Preserve matching observed within-age TF-target associations; use actual named baselines for error contrasts.",
    "Training priors are not independent validation. These are prior-edge association diagnostics, not causal GRN inference or unmeasured bio0/bio6 ablation results.",
)
_add(
    "author_topology",
    "author_graph",
    "cells,author_graph",
    "Display supplied author topology and occupancy where exact state-node mappings exist.",
    "Use documented node mappings and preserve GT-supported occupancy on the supplied topology.",
    "Unmapped topology is displayed alone; graph nodes must not be silently assigned to unrelated cell-state codes.",
)
_add(
    "selection_evidence",
    "candidate_evidence",
    "stage",
    "Show GT-only selection evidence and biological annotation for plotted genes.",
    "Prefer genes with documented membership/relevance and observed developmental signal, independently of model winners.",
    "Selection statistics are descriptive screening on existing cells, not biological replicate evidence or a model superiority filter.",
)
_add(
    "rollout_diagnostics",
    "diversity,drift,horizon,interpolation_progress",
    "stage",
    "Inspect diversity, observed-boundary drift, horizon behavior and interpolation continuity.",
    "Avoid pathological collapse, exploding magnitude and abrupt unsupported jumps while allowing biologically meaningful developmental change.",
    "The smallest drift is not automatically the best model: stationary predictions can fail to develop. These are stress/plausibility diagnostics without unobserved truth.",
)
_add(
    "perturbation_response",
    "perturbation_heatmap,perturbation_timecourse,perturbation_targets,perturbation_programs",
    "perturbation",
    "Inspect saved source-history interventions, direct targets and downstream gene/program responses.",
    "Responses are reproducible, stage-dependent where appropriate, gene-specific and compatible with documented interventions and prior biology.",
    "No measured perturbation truth is present. Edits may be performed in scaled input space; response direction and magnitude are model sensitivity, not validated knockout effects.",
)
_add(
    "perturbation_response",
    "perturbation_network",
    "perturbation,edges",
    "Relate saved intervention responses to supplied prior-network edges.",
    "Inspect response coherence with documented direct/indirect relations and native gene coverage.",
    "Supplied training priors are not independent perturbation validation; correlation and model sensitivity do not establish causal effects.",
)
_add(
    "regulatory_associations",
    "regulator_evidence",
    "stage",
    "Inspect supported regulator/target evidence from supplied priors and observed expression.",
    "Retain reference-supported regulatory associations with explicit native gene coverage and documented evidence.",
    "Evidence is associative; supplied training priors do not independently establish causal regulation.",
)
_add(
    "gene_dynamics",
    "temporal_descriptors,rank_stability",
    "stage",
    "Inspect saved temporal descriptors and stability of gene ordering across development.",
    "Preserve observed ordering/descriptors at measured ages; beyond measured ages, inspect biologically supported changes without abrupt unexplained rank instability.",
    "Rank stability and temporal smoothness are diagnostic properties, not proof of unobserved accuracy.",
)
_add(
    "rollout_diagnostics",
    "rollout_coverage",
    "stage",
    "Inspect which cells, ages and rollout horizons are present in saved outputs.",
    "Ensure claimed rollout stages have finite saved rows and disclose incomplete horizons or missing populations.",
    "Coverage is an output completeness diagnostic, not biological fidelity or unobserved prediction accuracy.",
)
_add(
    "saved_state_recoverability",
    "classifier_confusion,classifier_recoverability,classifier_composition",
    "",
    "Inspect saved independent classifier confusion, state recoverability and population composition evidence.",
    "Compare correctly named labels, class support and recovery under the same independently trained classifier and native assay.",
    "Classifier evidence depends on training domain, label mapping and held-out design. Saved tables cannot establish the performance of a classifier whose training contract is missing.",
)
_add(
    "saved_input_sensitivity",
    "input_sensitivity",
    "",
    "Inspect saved model sensitivity to documented biological input or prior channels.",
    "Find structured, documented sensitivity consistent with the controlled edit and supported biology.",
    "Input sensitivity is a model diagnostic, not a causal biological perturbation validation.",
)
_add(
    "differential_pathways",
    "deg_pathways",
    "",
    "Inspect actually saved differential-expression and pathway evidence for documented gene sets.",
    "Retain observed direction, native coverage and documented significance/enrichment contracts when comparing matched model outputs.",
    "Saved gene-set enrichment depends on the supplied background, mapping and comparison design. Missing pathway inputs cannot be replaced by invented enrichment.",
)
_add(
    "sampling_diagnostics",
    "sampling_convergence",
    "",
    "Inspect saved finite-sampling convergence of biological summaries.",
    "Increasing sample sizes stabilize the documented statistic without obscuring native gene and cell-state support.",
    "Sampling convergence assesses numerical stability, not biological truth or model superiority; inspect actual sample sizes and saved replicate design.",
)


def canonical_view(handler, view):
    return SCENARIO_ALIASES.get(view, view) if handler == "scenarios" else view


def family_spec(handler, view, scenario="measured"):
    canonical = canonical_view(handler, view)
    spec = dict(
        FAMILY_SPECS.get(
            canonical,
            dict(
                category="additional_biology",
                requires=[],
                purpose=canonical.replace("_", " "),
                favorable="Consult the method-specific evidence and source contracts.",
                caveat="Interpret only the available saved inputs; unsupported views are reported explicitly.",
            ),
        )
    )
    # Scenario marker dotplots are gene × age summaries, not cell-state plots.
    if handler == "scenarios" and canonical == "marker_dotplot":
        spec["requires"] = ["stage"]
        spec["category"] = "gene_dynamics"
        spec["purpose"] = "Show selected marker genes across saved ages using summary dotplots."
    if canonical == "gene_distribution" and handler == "scenarios":
        spec["requires"] = ["expression"]
    spec.update(view=canonical, handler=handler, scenario=scenario)
    if handler == "saved_diagnostics":
        spec["requires"] = []
    spec["suggested_figure"] = (
        5
        if scenario == "perturbation"
        else (
            4
            if scenario in ("interpolation", "future", "history")
            else (
                3
                if spec["category"]
                in (
                    "state_markers",
                    "state_occupancy",
                    "gene_programs",
                    "regulatory_associations",
                    "author_topology",
                    "cell_state_atlas",
                )
                else 2
            )
        )
    )
    spec["truth_status"] = (
        "measured GT comparison"
        if scenario in ("measured", "reverse")
        else "no independent query-stage/intervention GT"
    )
    if handler == "saved_diagnostics":
        spec["suggested_figure"] = (
            5
            if canonical == "input_sensitivity"
            else 2 if canonical in ("classifier_confusion", "sampling_convergence") else 3
        )
        spec["truth_status"] = (
            "saved diagnostic comparison; inspect held-out truth and source contracts"
        )
    return spec


def requested_views(question, options=None):
    """Apply the per-entity view limit once, before shared jobs are unioned."""
    views = list(
        dict.fromkeys(canonical_view(question["handler"], v) for v in question.get("views", []))
    )
    cap = int((options or {}).get("max_views_per_entity", 0) or 0)
    return views[:cap] if cap > 0 else views


def make_family_plan(question_plan, config, options, eligibility):
    """Build a deterministic union of canonical render jobs across questions."""
    grouped = {}
    for row in question_plan:
        q = row["question_spec"]
        cohort = row["cohort_spec"]
        scenario = cohort["scenario"] if cohort is not None else ",".join(q["scenario"])
        for view in requested_views(q, options):
            spec = family_spec(q["handler"], view, scenario)
            key = (q["handler"], scenario, view, row["cohort"])
            if key not in grouped:
                synthetic = dict(
                    id="family__" + q["handler"] + "__" + view,
                    figure=spec["suggested_figure"],
                    title=spec["purpose"],
                    handler=q["handler"],
                    scenario=[scenario],
                    views=[view],
                    family=view,
                    requires=spec["requires"],
                    note=spec["caveat"],
                )
                ok, reason = (
                    eligibility(synthetic, cohort, config)
                    if cohort is not None
                    else (False, row["reason"])
                )
                grouped[key] = dict(
                    family_id="__".join(key),
                    family=view,
                    category=spec["category"],
                    handler=q["handler"],
                    scenario=scenario,
                    figure=spec["suggested_figure"],
                    title=spec["purpose"],
                    cohort=row["cohort"],
                    eligible=ok,
                    reason=reason,
                    questions=[],
                    question_spec=synthetic,
                    cohort_spec=cohort,
                    interpretation=spec,
                )
            item = grouped[key]
            if q["id"] not in item["questions"]:
                item["questions"].append(q["id"])
    return list(grouped.values())


def interpretation_markdown(family, cohort=None):
    spec = family["interpretation"]
    cohort = cohort or {}
    scenario = family["scenario"]
    lines = [
        "# " + family["title"],
        "",
        f"Category: `{family['category']}`. Canonical view: `{family['family']}`.",
        f"Dataset: `{cohort.get('dataset','unavailable')}`. Native setting: `{cohort.get('panel','unavailable')}`. Scenario: `{scenario}`.",
        f"Suggested paper placement: Figure {family['figure']}; this is a narrative suggestion, not a compulsory folder assignment.",
        "",
        "## Biological question and reading guide",
        "",
        spec["purpose"],
        "",
        "## What favorable behavior means",
        "",
        spec["favorable"],
        "",
        "## Limits of the evidence",
        "",
        spec["caveat"],
        "",
        "Only compare models within this native HVG or author setting. GT, FM and TRANS use this setting’s gene axis; native coverage and observed reference identity are recorded in MODEL_ROWS.csv.",
        "Gene/state/program selection and response evidence are saved alongside the figures. Biological interpretation should use those tables together with the plotted patterns.",
        "",
        "## Questions answered by this shared output",
        "",
    ]
    lines += ["- `" + q + "`" for q in family["questions"]]
    meanings = []
    view = family["family"]
    category = family["category"]
    if category in (
        "gene_dynamics",
        "gene_cascade",
        "state_markers",
        "gene_histogram",
        "gene_localization",
    ):
        meanings += [
            "`mean` is the saved native expression average; `std` is cell-level dispersion, not experimental uncertainty. `pct` is the fraction of stored values > 0. Larger values are not universally better: measured fidelity means preserving the matching GT pattern and units."
        ]
    if view == "cascade":
        meanings += [
            "Cascade rows are ordered by the observed peak age. Colors use (stage mean − observed temporal mean) / observed temporal SD; residual colors use (model − observed stage mean) / observed temporal SD. Zero residual means stage-matched agreement, conditional on native coverage and the frozen reference."
        ]
    if view == "density_residual":
        meanings += [
            "Density residual = weighted generated probability − matching observed probability on fixed atlas bins. Positive values indicate local excess, negative values indicate local deficit. Near-zero residuals are favorable only where the observed reference and sampling contracts are compatible."
        ]
    if category == "regulatory_associations":
        meanings += [
            "Within-age association = correlation after subtracting each gene’s own age-specific mean. Association error = |model association − matched GT association|; lower is closer to GT. Named-baseline contrast = baseline error − model error; positive values indicate smaller error than that actual baseline. Prior-edge coverage and missing evidence remain explicit."
        ]
    if category == "gene_programs":
        meanings += [
            "Covered weight fraction = sum of covered absolute membership weights / full supplied absolute weight mass. Program activity is a covered-weight-normalized weighted score; consult the program coverage/scoring CSV for raw-expression versus observed-reference standardized units. High activity by itself is not a quality score."
        ]
    if view in ("drift", "horizon"):
        meanings += [
            "Expression distance and correlation use a documented matching observed boundary when available, otherwise a declared saved prediction-stage reference. `n_common_genes`, `reference_time`, `reference_source_id` and `scale_group` define each comparison. Distance magnitude is a change diagnostic, not a universal lower-is-better measure."
        ]
    if view == "temporal_descriptors":
        meanings += [
            "Peak age is the saved-grid argmax of gene mean expression. Net change = last saved mean − first saved mean. Range = maximum − minimum. Half-range crossing interpolates the first appropriate crossing of (minimum + maximum)/2; it is grid-dependent and is not validated biological onset. `n_saved_timepoints` and crossing status define support."
        ]
    if view == "rank_stability":
        meanings += [
            "Adjacent rank Spearman compares gene-mean ranks between consecutive saved ages on common selected genes. The table reports the number of common selected genes; undefined correlations remain missing. High stability can indicate either preserved ordering or pathologically frozen expression."
        ]
    if view == "rollout_coverage":
        meanings += [
            "Retained horizon fraction = completed saved finite steps / requested effective steps. This reports numerical output completeness. A complete horizon does not establish biologically accurate extrapolation."
        ]
    if scenario == "perturbation" or view == "input_sensitivity":
        meanings += [
            "Response delta = edited prediction − matched baseline prediction in saved native units. Direct targets, response-gene selection, available age support and covered program members are audited separately. Stronger or more numerous responses are not automatically better."
        ]
    if view.startswith("classifier_"):
        meanings += [
            "Inspect held-out observed classifier reliability before generated target-label recoverability. Confusion counts, class support and label mapping must use the same classifier contract. Agreement with requested/template labels is conditional classifier recoverability, not independently observed fate."
        ]
    if meanings:
        lines += ["", "## Numerical meaning and evidence support", ""] + meanings
    lines += [
        "",
        "The adjacent `FIGURE_EVIDENCE.csv` links every actual export to numerical tables. `TABLE_CATALOGUE.csv` lists each actual evidence table, its row count and column names. Use stage/cell/gene counts in those tables and `MODEL_ROWS.csv` to assess support; empty or missing comparisons cannot establish favorable performance.",
    ]
    if scenario in ("interpolation", "future", "history", "perturbation"):
        lines += [
            "",
            "## No-GT interpretation",
            "",
            "These saved generated query stages or intervention responses have no independent matched ground truth. A visually convincing pattern is a model hypothesis; it does not establish prediction accuracy or causal validity.",
            "For interpolation, saved measured anchors or query boundaries can support continuity checks but are not truth for every intermediate query. For history/future, boundary agreement is a stability check; meaningful development may require moving away from the boundary. For perturbation, use matched baseline/edit records and documented edit units.",
            "Do not rank a model simply by lowest drift, highest smoothness, strongest response, or agreement with another FM. Assess biological consistency, coverage, diversity and declared reference/scoring contracts together.",
        ]
    return "\n".join(lines) + "\n"
