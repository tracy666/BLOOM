"""Biological questions grounded in saved expression, geometry and intervention outputs."""

from __future__ import annotations
import re

QUESTIONS = []


def add(id, title, handler, scenario, views, requires=(), note=""):
    QUESTIONS.append(
        dict(
            id=id,
            figure=int(id[1]),
            title=title,
            handler=handler,
            scenario=scenario.split(","),
            views=views.split(","),
            family=views.split(",")[0],
            requires=list(requires),
            note=note,
        )
    )


add(
    "F2_Q01",
    "Where do generated developmental populations lie in a fixed observed atlas?",
    "measured",
    "measured",
    "atlas_time,stage_storyboard,trajectory",
    ["cells"],
)
add(
    "F2_Q02",
    "Do generated expression profiles occupy the same cell-state regions?",
    "measured",
    "measured",
    "atlas_state,state_trajectory,density_state",
    ["cells", "state"],
)
add(
    "F2_Q03",
    "Where are biologically relevant genes expressed across the atlas?",
    "measured",
    "measured",
    "gene_feature,gene_timecourse,gene_distribution",
    ["expression"],
)
add(
    "F2_Q04",
    "Which observed regions are overrepresented or missing in generated populations?",
    "measured",
    "measured",
    "density,density_residual,density_contour",
    ["cells"],
)
add(
    "F2_Q05",
    "Are developmental population paths and their stage centroids preserved?",
    "measured",
    "measured",
    "trajectory,stage_storyboard,state_trajectory",
    ["cells"],
)
add(
    "F2_Q06",
    "Are temporal changes in biologically relevant gene expression preserved?",
    "measured",
    "measured",
    "gene_timecourse,gene_phase,gene_detection",
    ["stage"],
)
add(
    "F2_Q07",
    "Are heterogeneous and multimodal gene-expression distributions reproduced?",
    "measured",
    "measured",
    "gene_distribution,gene_feature",
    ["expression"],
)
add(
    "F2_Q08",
    "Is the observed ordering of developmental gene activation reproduced?",
    "measured",
    "measured",
    "cascade,gene_timecourse",
    ["stage"],
)
add(
    "F2_Q09",
    "Are expressing-cell fractions reproduced alongside average expression?",
    "measured",
    "measured",
    "gene_detection,marker_dotplot",
    ["stage"],
)
add(
    "F2_Q10",
    "Which gene-state relationships agree with observed marker patterns?",
    "measured",
    "measured",
    "marker_dotplot,marker_matrix,marker_violin",
    ["expression", "state"],
)
add(
    "F2_Q11",
    "How do mean expression and detection together change along development?",
    "measured",
    "measured",
    "gene_phase,gene_timecourse,gene_detection",
    ["stage"],
)
add(
    "F2_Q12",
    "Which selected genes have both biological evidence and observed developmental signal?",
    "measured",
    "measured",
    "candidate_evidence,gene_feature,gene_timecourse",
    ["stage"],
)

add(
    "F3_Q01",
    "Are observed cell-state marker signatures retained?",
    "measured",
    "measured",
    "marker_dotplot,marker_matrix,marker_violin",
    ["expression", "state"],
)
add(
    "F3_Q02",
    "Where do state-specific expression signatures localize in generated populations?",
    "measured",
    "measured",
    "atlas_state,gene_feature,density_state",
    ["cells", "state"],
)
add(
    "F3_Q03",
    "Are cell-state proportions and their developmental changes preserved?",
    "measured",
    "measured",
    "state_occupancy,atlas_state,state_trajectory",
    ["cells", "state"],
)
add(
    "F3_Q04",
    "Are reference cell-state neighborhoods and their connectivity retained?",
    "measured",
    "measured",
    "state_graph,state_trajectory",
    ["cells", "state"],
    "Neighborhood connectivity is associative; it does not establish lineage.",
)
add(
    "F3_Q05",
    "Are author or marker gene-program dynamics preserved?",
    "measured",
    "measured",
    "program_timecourse,program_heatmap,program_feature",
    ["expression"],
)
add(
    "F3_Q06",
    "Are gene programs localized to the expected observed cell-state regions?",
    "measured",
    "measured",
    "program_feature,program_heatmap,atlas_state",
    ["expression", "state"],
)
add(
    "F3_Q07",
    "Is observed gene coexpression structure preserved?",
    "measured",
    "measured",
    "coexpression,gene_phase",
    ["expression"],
    "Coexpression does not establish regulation.",
)
add(
    "F3_Q08",
    "Do supplied prior TF-target modules retain coordinated expression?",
    "measured",
    "measured",
    "grn_activity,grn_support,grn_network",
    ["expression", "edges"],
    "Prior-edge expression support is not a newly inferred causal GRN.",
)
add(
    "F3_Q09",
    "Which supplied regulatory edges are supported by observed and generated expression?",
    "measured",
    "measured",
    "grn_support,grn_network",
    ["expression", "edges"],
)
add(
    "F3_Q10",
    "Is within-state gene heterogeneity retained?",
    "measured",
    "measured",
    "marker_violin,gene_distribution,marker_dotplot",
    ["expression", "state"],
)
add(
    "F3_Q11",
    "Are rare observed cell-state regions represented by generated cells?",
    "measured",
    "measured",
    "density_state,state_occupancy,atlas_state",
    ["cells", "state"],
)
add(
    "F3_Q12",
    "Does the spatial arrangement of state trajectories resemble the observed atlas?",
    "measured",
    "measured",
    "state_trajectory,state_graph,atlas_state",
    ["cells", "state"],
)
add(
    "F3_Q13",
    "Which programs change across developmental stages and cell states?",
    "measured",
    "measured",
    "program_heatmap,program_timecourse,program_feature",
    ["expression"],
)
add(
    "F3_Q14",
    "Do prediction differences concentrate in specific biological state neighborhoods?",
    "measured",
    "measured",
    "density_state,density_residual,state_occupancy",
    ["cells", "state"],
)

add(
    "F3_Q15",
    "How do observed and generated cell-state occupancies relate to supplied author topology?",
    "measured",
    "measured",
    "author_graph,state_graph",
    ["cells", "author_graph"],
    "Author node IDs require an explicit matching state mapping; otherwise topology is shown alone.",
)
add(
    "F3_Q16",
    "Which supplied prior-edge associations show the largest reconstruction differences?",
    "measured",
    "measured",
    "edge_rescue,grn_support,grn_network",
    ["expression", "edges"],
    "This compares current predictions with GT; it does not invent old ablation baselines.",
)

add(
    "F4_Q01",
    "Do saved interpolation populations form a bridge between observed anchors?",
    "scenarios",
    "interpolation",
    "storyboard,geometry,trajectory,interpolation_progress,animation",
    ["cells"],
)
add(
    "F4_Q02",
    "How does cell-state density evolve across intermediate developmental queries?",
    "scenarios",
    "interpolation",
    "density,storyboard",
    ["cells"],
)
add(
    "F4_Q03",
    "Do biologically relevant genes vary smoothly between saved anchors?",
    "scenarios",
    "interpolation",
    "gene_timecourse,gene_distribution,gene_heatmap",
    ["expression"],
)
add(
    "F4_Q04",
    "How do available gene programs evolve during interpolation?",
    "scenarios",
    "interpolation",
    "programs,gene_heatmap",
    ["expression"],
)
add(
    "F4_Q05",
    "How do intermediate marker patterns and expressing-cell fractions change?",
    "scenarios",
    "interpolation",
    "marker_dotplot,gene_timecourse",
    ["stage"],
)
add(
    "F4_Q06",
    "Does interpolation preserve population diversity rather than collapse it?",
    "scenarios",
    "interpolation",
    "diversity,density,horizon",
    ["cells"],
)
add(
    "F4_Q07",
    "Where do autoregressive future populations project relative to observed development?",
    "scenarios",
    "future",
    "geometry,storyboard,trajectory,animation",
    ["cells"],
    "No future truth: projection and plausibility, not accuracy.",
)
add(
    "F4_Q08",
    "How do biologically relevant genes evolve during future rollout?",
    "scenarios",
    "future",
    "gene_timecourse,gene_heatmap,gene_distribution",
    ["expression"],
)
add(
    "F4_Q09",
    "Which available gene programs drift during autoregressive future rollout?",
    "scenarios",
    "future",
    "programs,drift,horizon",
    ["expression"],
)
add(
    "F4_Q10",
    "Do future populations retain expression and geometric diversity?",
    "scenarios",
    "future",
    "diversity,density,horizon",
    ["cells"],
)
add(
    "F4_Q11",
    "How far do future populations move from observed reference support?",
    "scenarios",
    "future",
    "drift,geometry,trajectory",
    ["cells"],
)
add(
    "F4_Q12",
    "How do marker signatures change across future horizons?",
    "scenarios",
    "future",
    "marker_dotplot,gene_heatmap,gene_timecourse",
    ["stage"],
)
add(
    "F4_Q13",
    "Do FM future trajectories and gene responses agree qualitatively?",
    "scenarios",
    "future",
    "geometry,trajectory,gene_timecourse",
    ["cells"],
)
add(
    "F4_Q14",
    "How does developmental gene activation order extend beyond observations?",
    "scenarios",
    "future",
    "gene_heatmap,gene_timecourse",
    ["stage"],
)

add(
    "F5_Q01",
    "Does measured reverse prediction recover the observed developmental atlas?",
    "scenarios",
    "reverse",
    "reverse_geometry,trajectory,density",
    ["cells"],
)
add(
    "F5_Q02",
    "Are observed gene trajectories recovered in measured reverse prediction?",
    "scenarios",
    "reverse",
    "reverse_expression,gene_timecourse,gene_heatmap",
    ["expression"],
)
add(
    "F5_Q03",
    "Are reverse-predicted marker patterns and diversity preserved?",
    "scenarios",
    "reverse",
    "marker_dotplot,gene_distribution,diversity",
    ["expression"],
)
add(
    "F5_Q04",
    "Where do generated pre-observation history populations project?",
    "scenarios",
    "history",
    "history_geometry,storyboard,trajectory,animation",
    ["cells"],
    "No historical truth: reconstructed states are model hypotheses.",
)
add(
    "F5_Q05",
    "How do biologically relevant genes evolve into reconstructed history?",
    "scenarios",
    "history",
    "history_timecourse,history_heatmap,gene_distribution",
    ["expression"],
)
add(
    "F5_Q06",
    "Which gene programs drift as reconstructed history extends backward?",
    "scenarios",
    "history",
    "programs,drift,horizon",
    ["expression"],
)
add(
    "F5_Q07",
    "Does reconstructed history retain diverse cellular expression states?",
    "scenarios",
    "history",
    "diversity,density,history_geometry",
    ["cells"],
)
add(
    "F5_Q08",
    "How does reconstructed history move relative to the earliest observed state?",
    "scenarios",
    "history",
    "drift,trajectory,history_geometry",
    ["cells"],
)
add(
    "F5_Q09",
    "How does gene activation ordering differ among FM history reconstructions?",
    "scenarios",
    "history",
    "history_heatmap,history_timecourse",
    ["stage"],
)
add(
    "F5_Q10",
    "Which interventions produce the largest predicted downstream responses?",
    "scenarios",
    "perturbation",
    "perturbation_heatmap,perturbation_timecourse,perturbation_targets",
    ["perturbation"],
)
add(
    "F5_Q11",
    "Do edited source genes change their own predicted expression?",
    "scenarios",
    "perturbation",
    "perturbation_targets,perturbation_timecourse",
    ["perturbation"],
)
add(
    "F5_Q12",
    "Which predicted downstream genes respond to each available intervention?",
    "scenarios",
    "perturbation",
    "perturbation_heatmap,perturbation_timecourse",
    ["perturbation"],
)
add(
    "F5_Q13",
    "Which available gene programs show predicted intervention responses?",
    "scenarios",
    "perturbation",
    "perturbation_programs,perturbation_heatmap",
    ["perturbation"],
)
add(
    "F5_Q14",
    "Are predicted intervention responses stage dependent?",
    "scenarios",
    "perturbation",
    "perturbation_timecourse,perturbation_heatmap",
    ["perturbation"],
)
add(
    "F5_Q15",
    "Do intervention responses differ among FM models?",
    "scenarios",
    "perturbation",
    "perturbation_heatmap,perturbation_targets",
    ["perturbation"],
)
add(
    "F5_Q16",
    "How do predicted intervention responses relate to supplied prior-network edges?",
    "scenarios",
    "perturbation",
    "perturbation_network,perturbation_heatmap",
    ["perturbation", "edges"],
)

# Paper placement is a suggestion. Shared views below render once even when
# both the historical v1 IDs and the new interpretation questions select them.
add(
    "F3_Q17",
    "Which regulators have complementary observed temporal, state, prior-network and reconstruction evidence?",
    "measured",
    "measured",
    "regulator_evidence",
    ["stage"],
    "Descriptive evidence components remain separate; no causal regulator ranking or invented ablation baseline.",
)
add(
    "F4_Q15",
    "How do reconstructed pre-observation populations and gene programs extend development backward?",
    "scenarios",
    "history",
    "geometry,storyboard,trajectory,gene_timecourse,gene_heatmap,programs",
    ["cells"],
    "History was v1 F5_Q04–Q09; it also belongs in a bidirectional-development Figure 4 narrative.",
)
add(
    "F4_Q16",
    "Which history horizons are actually saved and numerically usable?",
    "scenarios",
    "history",
    "rollout_coverage,drift,horizon",
    ["stage"],
    "Finite retained prefixes are not extrapolated or padded.",
)
add(
    "F4_Q17",
    "How do saved-grid peak times, half-range crossings and temporal amplitude differ across history reconstructions?",
    "scenarios",
    "history",
    "temporal_descriptors,rank_stability",
    ["stage"],
    "Descriptive summaries over saved ages; no unobserved historical truth or validated biological onset.",
)
add(
    "F4_Q18",
    "Which future horizons remain usable and how do temporal gene patterns evolve?",
    "scenarios",
    "future",
    "rollout_coverage,temporal_descriptors,rank_stability",
    ["stage"],
)
add(
    "F4_Q19",
    "How do temporal gene descriptors and adjacent gene ranks change between interpolation anchors?",
    "scenarios",
    "interpolation",
    "temporal_descriptors,rank_stability",
    ["stage"],
    "Generated intermediate points have no independent truth.",
)
add(
    "F2_Q13",
    "Can the saved independent classifier reliably recognize held-out observed states?",
    "saved_diagnostics",
    "saved_diagnostics",
    "classifier_confusion",
    ["saved_diagnostics"],
    "Held-out truth reliability must be established before interpreting generated label recoverability.",
)
add(
    "F3_Q18",
    "Do generated target-label recoverability and state composition resemble reliable observed classifier diagnostics?",
    "saved_diagnostics",
    "saved_diagnostics",
    "classifier_recoverability,classifier_composition",
    ["saved_diagnostics"],
    "Template-label recoverability is not independently observed cell fate or accuracy.",
)
add(
    "F3_Q19",
    "What do the saved V5 temporal gene descriptors and rank stability reveal?",
    "saved_diagnostics",
    "saved_diagnostics",
    "temporal_descriptors,rank_stability",
    ["saved_diagnostics"],
)
add(
    "F5_Q17",
    "Which input-history edits produce saved downstream sensitivity?",
    "saved_diagnostics",
    "saved_diagnostics",
    "input_sensitivity",
    ["saved_diagnostics"],
    "Model sensitivity in saved input units is not an experimentally validated knockout.",
)
add(
    "F3_Q20",
    "Which saved differential-expression and pathway summaries support temporal interpretation?",
    "saved_diagnostics",
    "saved_diagnostics",
    "deg_pathways",
    ["saved_diagnostics"],
    "Conditional: requires actual exported DEG/pathway tables; cell-level tests are not biological-replicate validation.",
)
add(
    "F2_Q14",
    "How do saved sampling-convergence diagnostics change with population size?",
    "saved_diagnostics",
    "saved_diagnostics",
    "sampling_convergence",
    ["saved_diagnostics"],
    "Conditional: uses existing exported convergence tables; does not create new sampling experiments or repeated seeds.",
)


def canonical_question(value):
    value = str(value).upper()
    m = re.fullmatch(r"F([2-5])_Q0*(\d+)", value)
    return f"F{m[1]}_Q{int(m[2]):02d}" if m else value


def get_questions(selector="all"):
    if selector == "all":
        return QUESTIONS.copy()
    values = {canonical_question(v.strip()) for v in selector.split(",")}
    found = [q for q in QUESTIONS if q["id"] in values or f"F{q['figure']}" in values]
    unknown = values - {q["id"] for q in found} - {f"F{x}" for x in range(2, 6)}
    if unknown:
        raise ValueError(
            f"Unknown questions: {sorted(unknown)}. Run bloom.py catalogue to see supported question IDs."
        )
    return found
