"""Truth-first, expression-only cell-state diagnostic for V4/V5 measured H5.

This is an evaluator diagnosis, NOT proof of forecasting accuracy. The default
same-stage split measures how separable real states are in the deployed panel.
Only real training/validation data select the classifier. Generated expression
never selects preprocessing, hyperparameters, taxonomy, or the winning model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


class DiagnosticError(RuntimeError):
    pass


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def evidence_audit(path, output):
    """Reproduce validation-only selection from an existing eval11 CSV."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"scheme", "model", "endpoint", "balanced_accuracy"}
    if not rows or not required.issubset(rows[0]):
        raise DiagnosticError(f"Evidence CSV requires {sorted(required)}")
    report = []
    for scheme in sorted({r["scheme"] for r in rows}):
        local = [r for r in rows if r["scheme"] == scheme]
        eligible = [
            r for r in local if r["endpoint"] == "validation_truth" and r["model"] != "stage_prior"
        ]
        if not eligible:
            continue
        best = max(eligible, key=lambda r: (float(r["balanced_accuracy"]), r["model"]))
        selected = {r["endpoint"]: r for r in local if r["model"] == best["model"]}
        endpoints = {
            k: {m: float(v[m]) for m in ("accuracy", "balanced_accuracy", "macro_f1") if v.get(m)}
            for k, v in selected.items()
        }
        truth = endpoints.get("heldout_truth", {}).get("balanced_accuracy")
        generated = endpoints.get("generated_prediction", {}).get("balanced_accuracy")
        report.append(
            {
                "scheme": scheme,
                "selected_model": best["model"],
                "selection": "real validation balanced accuracy only",
                "endpoints": endpoints,
                "generated_minus_real_balanced_accuracy": (
                    None if truth is None or generated is None else generated - truth
                ),
                "note": "A small generated-real gap with weak real classification is inconclusive biological validation.",
            }
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "existing_evidence_audit.json", report)
    print(json.dumps(report, indent=2))
    return report


def imports():
    global np, pd, h5py
    try:
        import numpy as np
        import pandas as pd
        import h5py
    except ImportError as exc:
        raise DiagnosticError(
            "Install numpy pandas h5py scipy scikit-learn matplotlib; parquet inputs also need pyarrow"
        ) from exc


def decode(values):
    return np.asarray(
        [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values], dtype=str
    )


def choose(handle, keys, required=True):
    key = next((k for k in keys if k in handle), None)
    if required and key is None:
        raise DiagnosticError(f"H5 lacks {keys}")
    return key


def read_rows(dataset, rows):
    """h5py fancy indexing requires increasing unique indices."""
    rows = np.asarray(rows, dtype=np.int64)
    unique, inverse = np.unique(rows, return_inverse=True)
    return np.asarray(dataset[unique], dtype=np.float32)[inverse]


def h5_metadata(path):
    with h5py.File(path, "r") as f:
        reporting = None
        protocol = f.attrs.get("prediction_protocol", "")
        protocol = protocol.decode("utf-8") if isinstance(protocol, bytes) else str(protocol)
        if protocol == "direct_cache_univariate_gene_forecast_v1":
            from . import read_direct_reporting

            reporting = read_direct_reporting(f)
        keys = {
            "truth": choose(f, ["truth_expression", "ground_truth", "truth"]),
            "generated": choose(f, ["prediction_expression", "prediction"]),
            "time": (
                "prediction_reporting_time"
                if reporting
                else choose(
                    f,
                    [
                        "prediction_actual_time",
                        "prediction_time",
                        "prediction_day_index",
                        "day_index",
                    ],
                )
            ),
            "type": choose(f, ["truth_type_id", "target_type_id"]),
            "names": choose(f, ["celltype_names", "cell_type_names"]),
            "genes": choose(f, ["gene_names", "genes", "var_names"]),
            "cell_index": choose(f, ["truth_cell_index", "target_cell_index"], required=False),
        }
        if reporting:
            keys["time_unit"] = reporting[0]["reporting_unit"]
            keys["time_axis_sha256"] = reporting[0]["reporting_axis_sha256"]
        shape = f[keys["truth"]].shape
        if len(shape) != 2 or f[keys["generated"]].shape != shape:
            raise DiagnosticError(
                "Measured truth/generated must be equal-shape cell-by-gene matrices"
            )
        type_id = np.asarray(f[keys["type"]][...]).reshape(-1)
        if not np.issubdtype(type_id.dtype, np.integer):
            raise DiagnosticError(
                "Truth type IDs must be integers; fractional IDs cannot be silently cast"
            )
        names = decode(f[keys["names"]][...])
        genes = decode(f[keys["genes"]][...])
        times = decode(np.asarray(f[keys["time"]][...]).reshape(-1))
        if len(type_id) != shape[0] or len(times) != shape[0] or len(genes) != shape[1]:
            raise DiagnosticError("Expression, label, time, or gene axes disagree")
        if len(set(genes)) != len(genes) or len(set(names)) != len(names):
            raise DiagnosticError(
                "Duplicate H5 genes or cell-type names: fix the export contract before fitting"
            )
        if np.any(type_id < 0) or np.any(type_id >= len(names)):
            raise DiagnosticError("Invalid or missing truth cell type IDs")
        indices = (
            None
            if keys["cell_index"] is None
            else np.asarray(f[keys["cell_index"]][...]).reshape(-1)
        )
        if indices is not None and (
            len(indices) != shape[0]
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
        ):
            raise DiagnosticError("Invalid truth cell indices")
        links_identical = f[keys["truth"]].id == f[keys["generated"]].id
    if links_identical:
        raise DiagnosticError("Truth and generated H5 datasets point to the same object")
    return keys, shape, names[type_id], times, genes, indices


def sample_unique_rows(labels, times, indices, max_rows, seed):
    """Deduplicate real identities BEFORE any split, then sample proportionally."""
    if indices is None:
        raise DiagnosticError(
            "H5 lacks truth_cell_index/target_cell_index; export real identity before classifier fitting (audit-only still works)"
        )
    mapping = {}
    for row, (identity, label, stage) in enumerate(
        zip(indices.tolist(), labels.tolist(), times.tolist())
    ):
        if identity in mapping:
            previous = mapping[identity]
            if labels[previous] != label or times[previous] != stage:
                raise DiagnosticError(f"Repeated truth cell {identity} has inconsistent label/time")
        else:
            mapping[identity] = row
    rows = np.asarray(list(mapping.values()), dtype=np.int64)
    if max_rows and len(rows) > max_rows:
        rng = np.random.default_rng(seed)
        rows = rng.choice(rows, size=max_rows, replace=False)
    return np.sort(rows)


def make_splits(labels, stages, seed=42, validation_times=(), test_times=(), groups=None):
    """Splits refer to already unique real cells; no labels used as features."""
    positions = np.arange(len(labels))
    if validation_times or test_times:
        if not validation_times or not test_times or set(validation_times) & set(test_times):
            raise DiagnosticError(
                "Time-disjoint diagnosis requires non-overlapping --validation-times and --test-times"
            )
        validation = positions[np.isin(stages, validation_times)]
        test = positions[np.isin(stages, test_times)]
        train = positions[~np.isin(stages, list(validation_times) + list(test_times))]
        for stage in list(validation_times) + list(test_times):
            if stage not in stages:
                raise DiagnosticError(f"Requested time {stage!r} absent; use exact H5 time values")
        if groups is not None:
            if (
                set(groups[train]) & set(groups[validation])
                or set(groups[train]) & set(groups[test])
                or set(groups[validation]) & set(groups[test])
            ):
                raise DiagnosticError(
                    "Biological group crosses selected time splits; choose disjoint specimens"
                )
        mode = "time-disjoint evaluator generalization (not a generator split)"
    elif groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        trainval, test = next(first.split(positions, groups=groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
        train0, val0 = next(second.split(trainval, groups=groups[trainval]))
        train, validation = trainval[train0], trainval[val0]
        mode = "specimen/group-disjoint evaluator generalization"
    else:
        rng = np.random.default_rng(seed)
        assignments = [[], [], []]
        for stage, label in sorted(set(zip(stages.tolist(), labels.tolist()))):
            local = positions[(stages == stage) & (labels == label)]
            rng.shuffle(local)
            # Rare strata remain in training; their heldout coverage is reported.
            n_test = max(1, int(len(local) * 0.2)) if len(local) >= 5 else 0
            n_val = n_test
            assignments[2].extend(local[:n_test])
            assignments[1].extend(local[n_test : n_test + n_val])
            assignments[0].extend(local[n_test + n_val :])
        train, validation, test = [np.asarray(x, dtype=int) for x in assignments]
        mode = "same-stage real-cell separability diagnostic; not heldout-stage forecast validation"
    if any(len(x) == 0 for x in (train, validation, test)):
        raise DiagnosticError("A split is empty: need enough unique real cells/groups/stages")
    if len(set(labels[train])) < 2:
        raise DiagnosticError("Training split needs at least two supported classes")
    return train, validation, test, mode


def matrix_summary(x):
    finite = np.isfinite(x)
    vals = x[finite]
    return {
        "rows": int(x.shape[0]),
        "genes": int(x.shape[1]),
        "nonfinite": int((~finite).sum()),
        "quantiles": np.quantile(vals, [0, 0.01, 0.5, 0.99, 1]).tolist() if len(vals) else [],
        "zero_fraction": float(np.mean(x == 0)),
        "negative_fraction": float(np.mean(x < 0)),
        "constant_genes": int(np.sum(np.nanstd(x, axis=0) < 1e-10)),
    }


def table(path):
    path = Path(path)
    return (
        pd.read_parquet(path)
        if path.suffix == ".parquet"
        else pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    )


def source_audit(args, rows, labels, genes, indices, truth):
    report, full = {}, None
    if args.obs:
        if indices is None:
            raise DiagnosticError("Source obs check requires real truth indices")
        obs = table(args.obs)
        if indices[rows].max() >= len(obs):
            raise DiagnosticError("H5 truth index outside source obs")
        if args.label_column not in obs:
            raise DiagnosticError(f"Obs has no label column {args.label_column!r}")
        obs_labels = obs.iloc[indices[rows]][args.label_column].astype(str).to_numpy()
        agreement = float(np.mean(obs_labels == labels[rows]))
        report["label_agreement"] = agreement
        if agreement < 1.0:
            raise DiagnosticError(
                f"H5/source obs label mismatch (agreement={agreement:.6f}); repair row contract first"
            )
    if args.source_x or args.source_var:
        if not args.source_x or not args.source_var or indices is None:
            raise DiagnosticError(
                "Source audit needs --source-x, --source-var and H5 truth indices"
            )
        from scipy import sparse

        path = Path(args.source_x)
        source = np.load(path, mmap_mode="r") if path.suffix == ".npy" else sparse.load_npz(path)
        var = table(args.source_var)
        if len(var) != source.shape[1]:
            raise DiagnosticError(
                "Source var must describe ALL source X columns in original order, not only selected genes"
            )
        if args.gene_column:
            if args.gene_column not in var:
                raise DiagnosticError(f"Source var lacks {args.gene_column}")
            all_genes = var[args.gene_column].astype(str).to_numpy()
        elif not isinstance(var.index, pd.RangeIndex):
            all_genes = var.index.astype(str).to_numpy()
        else:
            raise DiagnosticError("Use --gene-column for source var without a named gene index")
        if len(set(all_genes)) != len(all_genes):
            raise DiagnosticError(
                "Source gene identifiers not unique; choose unique Ensembl IDs or explicit unambiguous mapping"
            )
        lookup = {g: i for i, g in enumerate(all_genes)}
        absent = sorted(set(genes) - lookup.keys())
        if absent:
            raise DiagnosticError(f"H5 genes missing from source axis: {absent[:10]}")
        if indices[rows].max() >= source.shape[0]:
            raise DiagnosticError("H5 truth index outside source X")
        columns = np.asarray([lookup[g] for g in genes])
        selected = source[indices[rows]][:, columns]
        selected = selected.toarray() if sparse.issparse(selected) else np.asarray(selected)
        difference = np.abs(selected - truth)
        match = bool(np.allclose(selected, truth, rtol=args.source_rtol, atol=args.source_atol))
        report.update(
            source_panel_match=match,
            source_mean_absolute_difference=float(difference.mean()),
            source_max_absolute_difference=float(difference.max()),
            source_panel_column_order_sha256=hashlib.sha256(columns.tobytes()).hexdigest(),
        )
        if not match:
            raise DiagnosticError(
                f"H5 truth != source X in explicit gene order (mean abs={difference.mean():.6g}); check original row order, selected columns, and declared expression preprocessing. No scale conversion is guessed."
            )
        if args.full_panel_ceiling:
            memory_gib = len(rows) * source.shape[1] * 4 / 1024**3
            if memory_gib > args.max_dense_gib:
                raise DiagnosticError(
                    f"Full-panel array needs {memory_gib:.2f} GiB before fitting; lower --max-cells or raise --max-dense-gib"
                )
            full = source[indices[rows]]
            full = (
                full.toarray().astype(np.float32)
                if sparse.issparse(full)
                else np.asarray(full, dtype=np.float32)
            )
            if not np.isfinite(full).all():
                raise DiagnosticError("Non-finite full source features")
            report["full_panel_genes"] = int(full.shape[1])
    elif args.full_panel_ceiling:
        raise DiagnosticError("--full-panel-ceiling needs source X and full var")
    return report, full


def metrics(y, pred, trained_classes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    present = sorted(set(y))
    known = np.isin(y, trained_classes)
    return {
        "n_cells": len(y),
        "classes_present": len(present),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1_present_truth_classes": float(
            f1_score(y, pred, labels=present, average="macro", zero_division=0)
        ),
        "fraction_labels_seen_in_training": float(np.mean(known)),
        "uniform_random_over_training_classes_expected_accuracy": 1.0
        / len(trained_classes)
        * float(np.mean(known)),
    }


def fit_candidates(x, y, split, seed, n_jobs, max_pca, output, prefix):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold

    train, validation, test = split
    if not np.isfinite(x).all():
        raise DiagnosticError(
            "Non-finite expression must be fixed upstream; values were not silently replaced"
        )
    n_components = max(1, min(max_pca, x.shape[1], len(train) - 1))
    candidates = {
        "centroid_scaled": make_pipeline(VarianceThreshold(), StandardScaler(), NearestCentroid()),
        "logistic_C0.1": make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=seed),
        ),
        "logistic_C1": make_pipeline(
            VarianceThreshold(),
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed),
        ),
        "knn_pca": make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=seed),
            KNeighborsClassifier(
                n_neighbors=min(15, len(train)), weights="distance", n_jobs=n_jobs
            ),
        ),
    }
    records, fitted = [], {}
    for name, model in candidates.items():
        print(
            f"[classifier] {prefix}: {name} | train={len(train)} val={len(validation)}", flush=True
        )
        try:
            model.fit(x[train], y[train])
            score = metrics(y[validation], model.predict(x[validation]), np.unique(y[train]))
            records.append(dict(model=name, status="complete", **score))
            fitted[name] = model
        except Exception as exc:
            records.append(dict(model=name, status="failed", error=f"{type(exc).__name__}: {exc}"))
        pd.DataFrame(records).to_csv(output / f"{prefix}_validation_candidates.csv", index=False)
    good = [r for r in records if r["status"] == "complete"]
    if not good:
        raise DiagnosticError(f"All {prefix} classifier candidates failed; see candidate CSV")
    selected = max(good, key=lambda r: (r["balanced_accuracy"], r["model"]))["model"]
    return fitted[selected], selected, records


def plots(output, endpoint_rows, confusions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    frame = pd.DataFrame(endpoint_rows)
    labels = frame["endpoint"].str.replace("_", " ")
    axes[0].barh(labels, frame["balanced_accuracy"], color="#67a7d3")
    axes[0].set(xlim=(0, 1), xlabel="Balanced accuracy", title="Selected with real validation only")
    y, pred = confusions
    from sklearn.metrics import confusion_matrix

    order = sorted(set(y) | set(pred))
    matrix = confusion_matrix(y, pred, labels=order, normalize="true")
    axes[1].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axes[1].set(title="Real heldout confusion", xlabel="Predicted", ylabel="True")
    if len(order) <= 30:
        axes[1].set_xticks(range(len(order)), order, rotation=90, fontsize=6)
        axes[1].set_yticks(range(len(order)), order, fontsize=6)
    figure.tight_layout()
    figure.savefig(output / "classifier_diagnostic.png", dpi=240)
    plt.close(figure)


def run(args):
    imports()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    keys, shape, labels, stages, genes, identities = h5_metadata(args.prediction_h5)
    contract = {
        "prediction_h5": str(Path(args.prediction_h5).resolve()),
        "shape": list(shape),
        "keys": keys,
        "gene_order_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "has_truth_cell_index": identities is not None,
        "times": sorted(set(stages.tolist())),
        "truth_classes": sorted(set(labels.tolist())),
        "no_nonfinite_replacement": True,
        "generated_data_used_for_selection": False,
    }
    # Audit mode needs no classifier identity split and reads a bounded sample.
    if args.audit_only:
        rows = np.sort(
            np.random.default_rng(args.seed).choice(
                shape[0], size=min(args.max_cells or shape[0], shape[0]), replace=False
            )
        )
    else:
        rows = sample_unique_rows(labels, stages, identities, args.max_cells, args.seed)
        contract["n_unique_real_cells_in_file"] = len(set(identities.tolist()))
        contract["duplicates_removed_before_sampling"] = (
            shape[0] - contract["n_unique_real_cells_in_file"]
        )
    with h5py.File(args.prediction_h5, "r") as f:
        truth = read_rows(f[keys["truth"]], rows)
        generated = read_rows(f[keys["generated"]], rows)
    contract["sampled_rows"] = len(rows)
    contract["truth"] = matrix_summary(truth)
    contract["generated"] = matrix_summary(generated)
    contract["truth_generated_exact_equal_fraction"] = float(np.mean(truth == generated))
    contract["truth_generated_mean_absolute_difference"] = (
        float(np.mean(np.abs(truth - generated)))
        if np.isfinite(truth).all() and np.isfinite(generated).all()
        else None
    )
    save_json(output / "contract_audit.json", contract)
    if not np.isfinite(truth).all() or not np.isfinite(generated).all():
        raise DiagnosticError(
            "Non-finite matrix values; see contract_audit.json. No replacement was performed"
        )
    if np.array_equal(truth, generated):
        raise DiagnosticError(
            "Sampled truth and generated matrices are exactly identical; investigate accidental prediction reuse"
        )
    real_var = truth.var(axis=0)
    gen_var = generated.var(axis=0)
    pd.DataFrame(
        {
            "gene": genes,
            "real_mean": truth.mean(axis=0),
            "generated_mean": generated.mean(axis=0),
            "real_variance": real_var,
            "generated_variance": gen_var,
            "generated_over_real_variance": np.divide(
                gen_var, real_var, out=np.full_like(real_var, np.nan), where=real_var > 1e-12
            ),
            "real_positive_fraction": (truth > 0).mean(axis=0),
            "generated_positive_fraction": (generated > 0).mean(axis=0),
        }
    ).to_csv(output / "gene_expression_diagnostic.tsv", sep="\t", index=False)
    source, full = source_audit(args, rows, labels, genes, identities, truth)
    save_json(output / "source_alignment_audit.json", source)
    if args.audit_only:
        save_json(output / "status.json", {"status": "audit_complete", "classifier_fitted": False})
        return
    y, t = labels[rows], stages[rows]
    # Taxonomy is fixed before fitting; unmapped labels fail, never silently drop.
    if args.label_map:
        mapping = table(args.label_map)
        required = {"fine_label", "coarse_label"}
        if not required.issubset(mapping) or mapping.fine_label.duplicated().any():
            raise DiagnosticError("Label map requires unique fine_label/coarse_label columns")
        lookup = dict(zip(mapping.fine_label.astype(str), mapping.coarse_label.astype(str)))
        if set(labels) - set(lookup):
            raise DiagnosticError(
                "Label map misses observed labels; retain every class or document separate predeclared subgroup dataset"
            )
        y = np.asarray([lookup[v] for v in y])
        if (
            "include" in mapping
            and not mapping["include"].astype(str).str.lower().isin(["true", "1"]).all()
        ):
            raise DiagnosticError(
                "This diagnostic does not silently remove include=false classes; use a full predeclared taxonomy mapping"
            )
    groups = None
    if args.group_column:
        if not args.obs:
            raise DiagnosticError("--group-column needs --obs")
        obs = table(args.obs)
        if args.group_column not in obs:
            raise DiagnosticError("Group column missing from obs")
        groups = obs.iloc[identities[rows]][args.group_column]
        if groups.isna().any():
            raise DiagnosticError("Missing specimen/group identities")
        groups = groups.astype(str).to_numpy()
    train, validation, test, split_mode = make_splits(
        y, t, args.seed, args.validation_times or (), args.test_times or (), groups
    )
    split = (train, validation, test)
    assignments = np.full(len(rows), "", dtype=object)
    for name, locations in zip(("train", "validation", "test"), split):
        assignments[locations] = name
    registry = pd.DataFrame(
        {
            "h5_row": rows,
            "truth_cell_index": identities[rows],
            "time": t,
            "label": y,
            "split": assignments,
        }
    )
    if groups is not None:
        registry["group"] = groups
    registry.to_csv(output / "split_registry.tsv", sep="\t", index=False)
    registry.groupby(["split", "time", "label"]).size().rename("n").reset_index().to_csv(
        output / "class_stage_support.tsv", sep="\t", index=False
    )
    selected, selected_name, candidates = fit_candidates(
        truth, y, split, args.seed, args.n_jobs, args.pca_components, output, "panel"
    )
    trained = np.unique(y[train])
    endpoints = []
    all_predictions = {}
    for name, x, positions in (
        ("train_truth", truth, train),
        ("validation_truth", truth, validation),
        ("heldout_truth", truth, test),
        ("generated_target_label_recoverability", generated, test),
    ):
        pred = selected.predict(x[positions])
        all_predictions[name] = pred
        endpoints.append(
            dict(endpoint=name, model=selected_name, **metrics(y[positions], pred, trained))
        )
    majority = pd.Series(y[train]).value_counts().index[0]
    endpoints.append(
        dict(
            endpoint="majority_negative_control",
            model="train_majority",
            **metrics(y[test], np.repeat(majority, len(test)), trained),
        )
    )
    rng = np.random.default_rng(args.seed)
    shuffled = y[test].copy()
    rng.shuffle(shuffled)
    endpoints.append(
        dict(
            endpoint="label_shuffle_negative_control",
            model=selected_name,
            **metrics(shuffled, all_predictions["heldout_truth"], trained),
        )
    )
    if full is not None:
        full_model, full_name, _ = fit_candidates(
            full, y, split, args.seed, args.n_jobs, args.pca_components, output, "full_panel"
        )
        endpoints.append(
            dict(
                endpoint="full_panel_real_heldout_ceiling",
                model=full_name,
                **metrics(y[test], full_model.predict(full[test]), trained),
            )
        )
    pd.DataFrame(endpoints).to_csv(output / "endpoint_metrics.csv", index=False)
    pred_table = registry.iloc[test].copy()
    pred_table["real_expression_predicted_label"] = all_predictions["heldout_truth"]
    pred_table["generated_expression_predicted_label"] = all_predictions[
        "generated_target_label_recoverability"
    ]
    pred_table.to_csv(output / "heldout_predictions.tsv", sep="\t", index=False)
    stage_metrics = []
    for stage in sorted(set(t[test])):
        take = t[test] == stage
        for endpoint in ("heldout_truth", "generated_target_label_recoverability"):
            stage_metrics.append(
                dict(
                    time=stage,
                    endpoint=endpoint,
                    **metrics(y[test][take], all_predictions[endpoint][take], trained),
                )
            )
    pd.DataFrame(stage_metrics).to_csv(output / "stage_metrics.csv", index=False)
    # Population readout does not assume physical same-cell correspondence.
    composition_rows, composition_distance_rows = [], []
    for stage in sorted(set(t[test])):
        take = t[test] == stage
        label_order = sorted(set(y))
        real_labels = all_predictions["heldout_truth"][take]
        gen_labels = all_predictions["generated_target_label_recoverability"][take]
        a = np.asarray([np.mean(real_labels == label) for label in label_order])
        b = np.asarray([np.mean(gen_labels == label) for label in label_order])
        from scipy.spatial.distance import jensenshannon

        composition_distance_rows.append(
            {
                "time": stage,
                "n_cells": int(take.sum()),
                "total_variation": float(np.abs(a - b).sum() / 2),
                "jensen_shannon_distance_base2": float(jensenshannon(a, b, base=2)),
            }
        )
        for label, av, bv in zip(label_order, a, b):
            composition_rows.append(
                {
                    "time": stage,
                    "label": label,
                    "measured_annotation_fraction": float(np.mean(y[test][take] == label)),
                    "real_expression_classifier_fraction": float(av),
                    "generated_expression_classifier_fraction": float(bv),
                }
            )
    pd.DataFrame(composition_rows).to_csv(output / "state_composition.tsv", sep="\t", index=False)
    pd.DataFrame(composition_distance_rows).to_csv(
        output / "state_composition_distances.tsv", sep="\t", index=False
    )
    from sklearn.metrics import confusion_matrix

    label_order = sorted(set(y))
    for endpoint in ("heldout_truth", "generated_target_label_recoverability"):
        pd.DataFrame(
            confusion_matrix(y[test], all_predictions[endpoint], labels=label_order),
            index=label_order,
            columns=label_order,
        ).to_csv(output / f"confusion_{endpoint}.csv")
    real = next(r for r in endpoints if r["endpoint"] == "heldout_truth")
    generated_score = next(
        r for r in endpoints if r["endpoint"] == "generated_target_label_recoverability"
    )
    save_json(
        output / "diagnostic_summary.json",
        {
            "split_semantics": split_mode,
            "selected_classifier": selected_name,
            "selection_rule": "highest real validation balanced accuracy; no generated or real test tuning",
            "real_heldout_balanced_accuracy": real["balanced_accuracy"],
            "generated_target_label_recoverability": generated_score["balanced_accuracy"],
            "generated_minus_real": generated_score["balanced_accuracy"]
            - real["balanced_accuracy"],
            "interpretation": "Report real separability first. Close scores with weak real performance are inconclusive. Generated target-label recoverability is a conditional cell-state diagnostic, not physical cell identity matching or a population-only ground truth score.",
            "same_stage_is_not_forecasting_validation": True,
            "taxonomy": args.label_map or "unmerged exported truth labels",
        },
    )
    if not args.no_plots:
        plots(output, endpoints, (y[test], all_predictions["heldout_truth"]))
    save_json(
        output / "status.json",
        {"status": "complete", "classifier_fitted": True, "output": str(output.resolve())},
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction-h5", "--h5", dest="prediction_h5")
    p.add_argument(
        "--evidence-csv", help="Audit existing eval11 table; does not train a classifier"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--obs", help="Original source obs parquet/CSV/TSV for row-label verification")
    p.add_argument("--label-column", default="major_trajectory")
    p.add_argument(
        "--group-column", help="Specimen/donor column for group-disjoint evaluator splits"
    )
    p.add_argument(
        "--source-x", help="Original preprocessed expression .npz (scipy sparse) or .npy"
    )
    p.add_argument(
        "--source-var", help="All-column original source var, in original X column order"
    )
    p.add_argument("--gene-column", default="")
    p.add_argument("--source-atol", type=float, default=1e-5)
    p.add_argument("--source-rtol", type=float, default=1e-4)
    p.add_argument(
        "--full-panel-ceiling",
        action="store_true",
        help="Additional real-only source full-panel test; cannot classify missing generated genes",
    )
    p.add_argument(
        "--label-map",
        help="Predeclared full-coverage TSV fine_label,coarse_label; no posthoc class exclusion",
    )
    p.add_argument("--max-cells", type=int, default=30000)
    p.add_argument("--max-dense-gib", type=float, default=3.0)
    p.add_argument("--validation-times", nargs="+")
    p.add_argument("--test-times", nargs="+")
    p.add_argument("--pca-components", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.max_cells < 0 or args.pca_components < 1 or args.max_dense_gib <= 0:
            raise DiagnosticError("Sample and PCA/memory limits must be non-negative/positive")
        if args.evidence_csv:
            evidence_audit(args.evidence_csv, args.output)
        elif args.prediction_h5:
            run(args)
        else:
            raise DiagnosticError("Provide --prediction-h5 or --evidence-csv")
    except Exception as exc:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        save_json(
            Path(args.output) / "status.json",
            {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
        )
        print(f"Classifier diagnostic failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
