import ast
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

import numpy as np
import polars as pl
import pytest
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from notebooks.curate_ukbb_variants import select_trait_variants
from vep_comparisons.complementarity import eval_all_datasets, make_quadratic_model
from vep_comparisons.performance import (
    cached_bootstraps,
    run_bootstrap_estimates,
    sample_metric_deltas,
    sample_metrics,
)


@pytest.fixture(scope="module")
def analysis() -> Iterator[ModuleType]:
    """Load the remaining notebook-local trait curation helper."""
    notebook = Path(__file__).parents[1] / "notebooks" / "make_plots.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    module = ModuleType("_make_plots_analysis_test")
    sys.modules[module.__name__] = module
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith(("from collections import",)):
            exec(source, module.__dict__)
        elif "def restrict_ukbb_traits(" in source:
            tree = ast.parse(source)
            tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            exec(compile(tree, str(notebook), "exec"), module.__dict__)
    yield module
    del sys.modules[module.__name__]


def test_ukbb_selects_maximum_only_among_eligible_traits(tmp_path: Path) -> None:
    (tmp_path / "UKBB_94traits_release1.traits.filtered").write_text("trait\nallowed\n")
    variants = pl.DataFrame({
        "variant": ["a", "a", "a", "b", "c"],
        "trait": ["excluded", "allowed", "allowed", "excluded", "allowed"],
        "pip": [0.99, 0.02, 0.05, 0.8, 0.9],
        "allele1": ["A"] * 5,
        "allele2": ["C"] * 5,
        "LD_SV": [False, False, False, False, True],
        "LD_HWE": [False] * 5,
    })
    _, values = select_trait_variants.run(dataset_dir=tmp_path, pl=pl, variants=variants)
    result = values["pass_variants"]
    assert result["variant"].to_list() == ["a"]
    assert result["pip"].to_list() == [0.05]
    assert result["trait"].to_list() == ["allowed"]


def test_trait_restriction_rebuilds_gwas_labels_and_preserves_other_datasets(
    analysis: ModuleType,
) -> None:
    data = pl.DataFrame({
        "variant": ["eqtl", "eqtl", "multi", "ukbb_only", "eligible", "clinvar"],
        "gene": ["g1", "g2", "g3", "g4", "g5", "g6"],
        "ukbb_label": ["high_pip"] * 6,
        "ukbb_pip": [0.99] * 6,
        "multisusie_label": [None, None, "low_pip", None, None, None],
        "multisusie_pip": [None, None, 0.005, None, None, None],
        "eqtl_label": ["high_pip", "low_pip", None, None, None, None],
        "clinvar_label": [None, None, None, None, None, "Pathogenic"],
    })
    evidence = pl.DataFrame({
        "variant": ["eqtl", "multi", "ukbb_only", "eligible", "eligible", "clinvar"],
        "trait": ["excluded", "excluded", "excluded", "allowed", "excluded", "excluded"],
        "pip": [0.99, 0.99, 0.99, 0.6, 0.99, 0.99],
    })
    result = analysis.restrict_ukbb_traits(data, evidence, ["allowed"])
    assert result["variant"].to_list() == ["eqtl", "eqtl", "multi", "eligible", "clinvar"]
    assert result["ukbb_label"].to_list() == [None, None, None, "high_pip", None]
    assert result["ukbb_pip"].to_list() == [None, None, None, 0.6, None]
    assert result["gwas_label"].to_list() == [None, None, "low_pip", "high_pip", None]
    assert result["gwas_stringent_label"].to_list() == [None, None, "low_pip", None, None]
    assert result["eqtl_label"].to_list()[:2] == ["high_pip", "low_pip"]
    assert result["clinvar_label"].to_list()[-1] == "Pathogenic"


def test_quadratic_baseline_handles_nonlinearity_without_annotation_gain() -> None:
    # A constant annotation contains no information about this U-shaped outcome.
    # Each variant has two gene rows, and only one model needs the missing annotation.
    score = np.repeat(np.linspace(-2, 2, 120), 2)
    data = (
        pl.DataFrame({
            "variant": np.repeat([f"v{i}" for i in range(120)], 2),
            "score": score,
            "gnocchi_z": [None] * 2 + [0.0] * 238,
            "label_source": np.where(np.abs(score) > 1, "high_pip", "low_pip"),
        })
    )
    baseline = make_quadratic_model(n_jobs=1).set_params(
        logisticregressioncv__n_jobs=1, logisticregressioncv__cv=3,
        logisticregressioncv__Cs=[0.1, 1.0, 10.0],
    )
    combined = clone(baseline).set_params(
        logisticregressioncv__n_jobs=1, logisticregressioncv__cv=3,
        logisticregressioncv__Cs=[0.1, 1.0, 10.0],
    )
    models = {"score": (["score"], baseline), "combined": (["score", "gnocchi_z"], combined)}
    with parallel_backend("threading"):
        metrics, predictions, labels, groups = eval_all_datasets(
            data, {"test": ["label_source"]}, ["all"], models, n_folds=3,
        )
    key = ("test", "all")
    np.testing.assert_array_equal(groups[key], data["variant"].to_numpy()[2:])
    np.testing.assert_array_equal(labels[key], (np.abs(score) > 1)[2:])
    assert metrics["auroc"].min() > 0.98
    np.testing.assert_allclose(predictions[key]["score"], predictions[key]["combined"], atol=1e-5)


def test_cluster_bootstrap_matches_literal_whole_variant_resampling() -> None:
    # Nonadjacent gene rows for the same variant must always be drawn together.
    group_indices = np.array([0, 1, 0, 2, 1, 2, 2])
    y = np.array([0, 0, 1, 0, 1, 1, 0])
    ref = np.array([0.1, 0.4, 0.7, 0.2, 0.3, 0.8, 0.5])
    other = np.array([0.2, 0.1, 0.8, 0.5, 0.9, 0.4, 0.3])
    rng = np.random.RandomState(42)
    expected = {"auroc": [], "auprc": [], "auroc_delta": [], "auprc_delta": []}
    for _ in range(12):
        drawn_variants = rng.choice([0, 1, 2], size=3, replace=True)
        rows = np.concatenate([np.flatnonzero(group_indices == variant) for variant in drawn_variants])
        for metric, func in [("auroc", roc_auc_score), ("auprc", average_precision_score)]:
            value = func(y[rows], ref[rows])
            expected[metric].append(value)
            expected[f"{metric}_delta"].append(func(y[rows], other[rows]) - value)

    metrics = sample_metrics(y, ref, group_indices, job_idx=0, num_bootstraps=12)
    deltas = sample_metric_deltas(ref, other, y, group_indices, job_idx=0, num_bootstraps=12)
    for metric in ("auroc", "auprc"):
        np.testing.assert_allclose(metrics[metric], expected[metric])
        np.testing.assert_allclose(deltas[metric], expected[f"{metric}_delta"], atol=1e-15)


def test_bootstrap_runner_preserves_pairing_groups_and_biotypes() -> None:
    keys = [("test", "protein_coding"), ("test", "lncRNA")]
    y = np.array([0, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.8, 0.5, 0.6, 0.3, 0.7])
    metrics, differences = run_bootstrap_estimates(
        preds={key: {"reference": score, "same": score} for key in keys},
        labels={key: y for key in keys},
        groups={key: np.array(["a", "a", "b", "b", "c", "c"]) for key in keys},
        comparisons=[("reference", ("same",))], n_jobs=2, num_bootstraps_per_job=4,
    )
    assert metrics.height == 2 * 2 * 8
    assert differences.height == 2 * 8
    assert differences["biotype"].n_unique() == 2
    assert differences.select(pl.col("auroc", "auprc", "auroc_pct", "auprc_pct").abs().sum()).row(0) == (0, 0, 0, 0)


def test_bootstrap_cache_loads_and_regenerates_at_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "bootstrap.pkl"
    key = ("test", "all")
    y = np.array([0, 1, 0, 1, 0, 1])
    score = np.array([0.1, 0.8, 0.5, 0.6, 0.3, 0.7])
    predictions = {key: {"reference": score, "other": score}}
    labels = {key: y}
    groups = {key: np.repeat(["a", "b", "c"], 2)}
    comparisons = [("reference", ("other",))]
    metrics, differences = cached_bootstraps(
        path, predictions, labels, groups, comparisons, regenerate=False,
        n_jobs=2, num_bootstraps_per_job=4,
    )
    # Loading an existing cache requires no predictions or resampling.
    loaded = cached_bootstraps(path, {}, {}, {}, [], regenerate=False)
    assert metrics.equals(loaded[0])
    assert differences.equals(loaded[1])
    assert differences.height == 8

    predictions[key]["other"] = 1 - score
    regenerated = cached_bootstraps(
        path, predictions, labels, groups, comparisons, regenerate=True,
        n_jobs=2, num_bootstraps_per_job=4,
    )
    assert not differences.equals(regenerated[1])
    assert regenerated[1].equals(
        cached_bootstraps(path, {}, {}, {}, [], regenerate=False)[1]
    )
