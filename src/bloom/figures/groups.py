"""Map the final manuscript structure to the original analysis questions."""

GROUPS = {
    "measured": {
        "sections": "2.2--2.3",
        "description": "Measured-stage population structure and gene expression",
        "questions": tuple(f"F2_Q{i:02d}" for i in range(1, 15)),
    },
    "programmes": {
        "sections": "2.4--2.5",
        "description": "Cell-state programmes and regulatory associations",
        "questions": tuple(f"F3_Q{i:02d}" for i in range(1, 21)),
    },
    "temporal": {
        "sections": "2.6--2.7",
        "description": "Reverse reconstruction, interpolation, history and future rollouts",
        "questions": tuple(f"F4_Q{i:02d}" for i in range(1, 20))
        + tuple(f"F5_Q{i:02d}" for i in range(1, 10)),
    },
    "perturbation": {
        "sections": "2.8",
        "description": "Responses to single-gene input perturbations",
        "questions": tuple(f"F5_Q{i:02d}" for i in range(10, 18)),
    },
}

DATASET_ALIASES = {
    "prenatal": "big",
    "cortex": "small",
    "sted": "sted",
    "big": "big",
    "small": "small",
    "all": "all",
}


def question_selector(group):
    """Return the retained question IDs for one paper group or all groups."""
    keys = tuple(GROUPS) if group == "all" else (group,)
    return ",".join(dict.fromkeys(q for key in keys for q in GROUPS[key]["questions"]))


def native_datasets(value):
    """Translate public dataset names while retaining saved source identities."""
    try:
        return ",".join(DATASET_ALIASES[item.strip()] for item in value.split(","))
    except KeyError as error:
        raise ValueError(
            "Dataset must be prenatal, cortex, sted, all, or a comma-separated list"
        ) from error
