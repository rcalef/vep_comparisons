"""Comparison semantics and reusable plotting, without loading the full notebook data."""
import ast
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from vep_comparisons.performance import (
    differences_from_metrics,
    sample_metric_deltas,
    sample_metrics,
)


@pytest.fixture(scope="module")
def complementarity() -> Iterator[ModuleType]:
    notebook = Path(__file__).parents[1] / "notebooks" / "make_plots.ipynb"
    module = ModuleType("_complementarity_plot_test")
    sys.modules[module.__name__] = module
    cells = json.loads(notebook.read_text())["cells"]
    in_complementarity = False
    for cell in cells:
        source = "".join(cell["source"])
        if source.startswith("# Complementarity"):
            in_complementarity = True
        if cell["cell_type"] != "code":
            continue
        if source.startswith((
            "from collections import", "# Helpful mappings", "from vep_comparisons",
            "model_specs =", "# Bind the notebook",
        )):
            exec(source, module.__dict__)
        elif in_complementarity:
            tree = ast.parse(source)
            tree.body = [
                node for node in tree.body
                if isinstance(node, ast.FunctionDef)
                or isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "metric_labels" for t in node.targets)
            ]
            exec(compile(tree, "<complementarity definitions>", "exec"), module.__dict__)
    yield module
    plt.close("all")
    del sys.modules[module.__name__]


def test_comparison_references_resolve_to_fitted_features(complementarity: ModuleType) -> None:
    models = ["Evo2", "ESM-C", "AlphaGenome AVI", "GPN-Star (V)"]
    for reference, alternatives in [
        (("model",), (("model", "phylop"), ("model", "gnocchi_z"))),
        (("gnocchi_z",), (("model",), ("model", "gnocchi_z"))),
        (("phylop",), (("model",), ("model", "phylop"))),
        (("model", "phylop"), (("model", "phylop", "gnocchi_z"),)),
    ]:
        comparisons = complementarity.make_comparisons(models, reference, alternatives)
        actual = {(ref, other) for ref, others in comparisons for other in others}
        expected = {
            (complementarity.prediction_name(model, reference),
             complementarity.prediction_name(model, features))
            for model in models for features in alternatives
        }
        assert actual == expected
        assert len(actual) == len(models) * len(alternatives)
    assert complementarity.make_comparisons(
        ["ESM-C"], reference=("model", "phylop"),
        alternatives=(("model", "gnocchi_z", "phylop"),),
    ) == [("esmc_phylop", ("esmc_gnocchi_phylop",))]


def test_annotation_deltas_use_annotation_denominator(complementarity: ModuleType) -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.repeat(np.arange(3), 2)
    annotation = np.array([0.3, 0.4, 0.2, 0.8, 0.6, 0.7])
    model = np.array([0.1, 0.9, 0.5, 0.6, 0.3, 0.8])
    baseline = sample_metrics(y, annotation, groups, 0, 12)
    alternative = sample_metrics(y, model, groups, 0, 12)
    differences = sample_metric_deltas(annotation, model, y, groups, 0, 12)
    metrics = pl.concat([
        pl.DataFrame(values).with_columns(
            model=pl.lit(name), dataset=pl.lit("test"), biotype=pl.lit("all"),
        )
        for name, values in [("Gnocchi", baseline), ("Evo2", alternative)]
    ])
    cached_differences = differences_from_metrics(
        metrics, [("Gnocchi", ("Evo2",))],
    )
    for metric in ("auroc", "auprc"):
        np.testing.assert_allclose(sorted(cached_differences[metric]), sorted(differences[metric]))
        np.testing.assert_allclose(sorted(cached_differences[f"{metric}_pct"]), sorted(differences[f"{metric}_pct"]))
        expected = np.array(alternative[metric]) - np.array(baseline[metric])
        np.testing.assert_allclose(differences[metric], expected)
        np.testing.assert_allclose(differences[f"{metric}_pct"], 100 * expected / baseline[metric])


def test_combined_cohorts_select_protein_by_alternative(complementarity: ModuleType) -> None:
    rows = {"ref": ["Gnocchi"] * 4, "compare": ["Evo2", "evo_gnocchi", "ESM-C", "esmc_gnocchi"]}
    missense = pl.DataFrame(rows).with_columns(cohort=pl.lit("missense"))
    genome_wide = pl.DataFrame(rows).with_columns(cohort=pl.lit("genome_wide"))
    actual = complementarity.combine_cohorts(missense, genome_wide)
    assert dict(actual.select("compare", "cohort").iter_rows()) == {
        "Evo2": "genome_wide", "evo_gnocchi": "genome_wide",
        "ESM-C": "missense", "esmc_gnocchi": "missense",
    }


@pytest.mark.parametrize("layout,models", [
    ("modality", ["RiNALMo"]),
    ("modality", ["RiNALMo", "Orthrus", "ESM-C"]),
    ("row", ["GPN-Star (V)", "AlphaGenome AVI"]),
])
@pytest.mark.parametrize("reference", [("model",), ("gnocchi_z",)])
def test_plot_supports_baselines_and_partial_modalities(
    complementarity: ModuleType, layout: str, models: list[str], reference: tuple[str, ...],
) -> None:
    alternatives = (("model", "gnocchi_z"),) if reference == ("model",) else (
        ("model",), ("model", "gnocchi_z"),
    )
    comparisons = complementarity.make_comparisons(models, reference, alternatives)
    data = pl.DataFrame([
        {"ref": ref, "compare": other, "dataset": "test", "biotype": biotype,
         "auprc_pct": value, "auprc": value / 100}
        for ref, others in comparisons for other in others
        for biotype, values in [("all", (-2.0, 2.0, 6.0)), ("lncRNA", (100.0,))]
        for value in values
    ])
    fig, axs = complementarity.plot_comparisons(
        data, "test", models=models, reference=reference,
        feature_order=("gnocchi_z", "model"), layout=layout,
        plot_col="auprc_pct", annotate_col="auprc", ymin=-5, sharey=False,
    )
    fig.canvas.draw()
    bar_axes = [ax for ax in axs.flat if ax.patches]
    assert len(bar_axes) == len(models)
    for ax in bar_axes:
        assert len(ax.patches) == len(alternatives)
        np.testing.assert_allclose([bar.get_height() for bar in ax.patches], 2.0)
        assert ax.get_ylim()[0] == -5
        assert [text.get_text() for text in ax.texts] == ["+0.020"] * len(alternatives)
    baseline_label = "base model" if reference == ("model",) else "human constraint"
    assert axs[0, 0].get_ylabel() == f"% AUPRC change\nover {baseline_label}"
    plt.close(fig)


@pytest.mark.parametrize("sharey", [False, True])
def test_y_limits_include_later_panels_and_ticks_are_capped(
    complementarity: ModuleType, sharey: bool,
) -> None:
    models = ["Evo2", "NTv3", "RiNALMo", "Orthrus", "ESM-C", "SaProt"]
    data = pl.DataFrame([
        {
            "ref": model, "compare": complementarity.prediction_name(model, ("model", "gnocchi_z")),
            "dataset": "test", "biotype": "all", "auroc_pct": value,
        }
        for i, model in enumerate(models)
        for value in (i * 10 + 1, i * 10 + 5, i * 10 + 9)
    ])
    fig, axs = complementarity.plot_comparisons(
        data, "test", models=models, sharey=sharey, ymin=-4,
    )
    fig.canvas.draw()
    bar_axes = [ax for ax in axs.flat if ax.patches]
    for ax in bar_axes:
        low, high = ax.get_ylim()
        assert low == -4
        assert max(bar.get_height() for bar in ax.patches) < high
        assert all(np.nanmax(line.get_ydata()) <= high for line in ax.lines)
        assert sum(low <= tick <= high for tick in ax.get_yticks()) <= 5
    if sharey:
        assert len({ax.get_ylim() for ax in bar_axes}) == 1
    plt.close(fig)


@pytest.mark.parametrize("sharey", [False, True])
def test_annotation_band_clears_bars_and_intervals_across_scales(
    complementarity: ModuleType, sharey: bool,
) -> None:
    models = ["Evo2", "NTv3", "RiNALMo", "Orthrus", "ESM-C", "SaProt"]
    data = pl.DataFrame([
        {
            "ref": model,
            "compare": complementarity.prediction_name(model, ("model", "gnocchi_z")),
            "dataset": "test", "biotype": "all",
            "auroc_pct": value * scale, "auroc": value * scale / 100,
        }
        for model, scale, values in zip(
            models, (0.001, 100, 0.01, 10, 0.1, 1),
            ((1, 2, 3), (-3, -2, -1), (-1, 0, 1), (0, 0, 0), (1, 2, 3), (-2, 0, 3)),
        )
        for value in values
    ])
    fig, axs = complementarity.plot_comparisons(
        data, "test", models=models, sharey=sharey, annotate_col="auroc",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bar_axes = [ax for ax in axs.flat if ax.patches]
    for ax in bar_axes:
        label_top = max(text.get_window_extent(renderer).y1 for text in ax.texts)
        label_bottom = min(text.get_window_extent(renderer).y0 for text in ax.texts)
        content_bottom = min(
            [bar.get_window_extent(renderer).y0 for bar in ax.patches]
            + [line.get_window_extent(renderer).y0 for line in ax.lines]
        )
        assert ax.bbox.y0 < label_bottom < label_top < content_bottom
        assert (content_bottom - ax.bbox.y0) / ax.bbox.height >= 0.15 - 1e-12
        low, high = ax.get_ylim()
        assert all(np.nanmax(line.get_ydata()) <= high for line in ax.lines)
        assert sum(low <= tick <= high for tick in ax.get_yticks()) <= 5
    if sharey:
        assert len({ax.get_ylim() for ax in bar_axes}) == 1
    plt.close(fig)
