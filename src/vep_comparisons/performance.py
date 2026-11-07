"""Variant-cluster bootstrap estimates of pair-level AUROC and AUPRC.

Rows represent variant–gene pairs. Bootstrap draws resample whole variants,
keeping every gene row and using the same draws for each prediction and comparison.
Intervals condition on the supplied out-of-fold predictions.
"""

from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from pickle import dump, load

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score


def variant_bootstrap_weights(
    group_indices: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Draw variants uniformly; every gene row inherits its variant's multiplicity.

    group_indices contains dense variant codes from np.unique(return_inverse=True).
    Integer metric weights exactly represent repeating all rows of each drawn variant.
    """
    n_groups = int(group_indices.max()) + 1
    draws = rng.randint(n_groups, size=n_groups)
    counts = np.bincount(draws, minlength=n_groups)
    return counts[group_indices]


def sample_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_indices: np.ndarray,
    job_idx: int,
    num_bootstraps: int = 100,
) -> dict[str, list[float]]:
    """Variant-cluster bootstrap estimates of pair-level AUROC and AUPRC."""
    rng = np.random.RandomState(seed=42 + job_idx)
    results = defaultdict(list)
    for _ in range(num_bootstraps):
        weights = variant_bootstrap_weights(group_indices, rng)
        results["auroc"].append(roc_auc_score(y_true, y_pred, sample_weight=weights))
        results["auprc"].append(
            average_precision_score(y_true, y_pred, sample_weight=weights)
        )
    return results


def sample_metric_deltas(
    y_pred_ref: np.ndarray,
    y_pred_oth: np.ndarray,
    y_true: np.ndarray,
    group_indices: np.ndarray,
    job_idx: int,
    num_bootstraps: int = 100,
) -> dict[str, list[float]]:
    """Paired variant-cluster bootstrap of pair-level performance differences."""
    rng = np.random.RandomState(seed=42 + job_idx)
    results = defaultdict(list)
    for _ in range(num_bootstraps):
        weights = variant_bootstrap_weights(group_indices, rng)
        auroc_ref = roc_auc_score(y_true, y_pred_ref, sample_weight=weights)
        auroc_oth = roc_auc_score(y_true, y_pred_oth, sample_weight=weights)
        results["auroc"].append(auroc_oth - auroc_ref)
        results["auroc_pct"].append(100 * (auroc_oth - auroc_ref) / auroc_ref)

        auprc_ref = average_precision_score(y_true, y_pred_ref, sample_weight=weights)
        auprc_oth = average_precision_score(y_true, y_pred_oth, sample_weight=weights)
        results["auprc"].append(auprc_oth - auprc_ref)
        results["auprc_pct"].append(100 * (auprc_oth - auprc_ref) / auprc_ref)
    return results


def nested_metrics_to_df(
    x: list[dict[str, list[float]]],
    index_items: dict[str, str],
) -> pl.DataFrame:
    flat = defaultdict(list)
    for d in x:
        for k, values in d.items():
            flat[k].extend(values)
    flat.update(index_items)
    return pl.DataFrame(flat)


def run_bootstrap_estimates(
    preds: dict[tuple[str, str], dict[str, np.ndarray]],
    labels: dict[tuple[str, str], np.ndarray],
    groups: dict[tuple[str, str], np.ndarray],
    comparisons: list[tuple[str, tuple[str, ...]]],
    n_jobs: int = 8,
    num_bootstraps_per_job: int = 100,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return paired draws; the total draws equal n_jobs * num_bootstraps_per_job.

    Prediction, label, and variant arrays share row order within each
    (dataset, biotype) key. Keep worker and draw counts fixed when combining caches.
    """
    seen = set()
    bootstrapped_metrics = []
    bootstrapped_diffs = []

    with Pool(processes=n_jobs) as p:
        for key, y_true in labels.items():
            dataset, biotype = key
            _, group_indices = np.unique(groups[key], return_inverse=True)
            index = {"dataset": dataset, "biotype": biotype}
            for ref_model, compare_models in comparisons:
                y_ref = preds[key][ref_model]
                for model in dict.fromkeys((ref_model, *compare_models)):
                    if (key, model) in seen:
                        continue
                    metric_func = partial(
                        sample_metrics,
                        y_true,
                        preds[key][model],
                        group_indices,
                        num_bootstraps=num_bootstraps_per_job,
                    )
                    bootstrapped_metrics.append(
                        nested_metrics_to_df(
                            x=p.map(metric_func, range(n_jobs)),
                            index_items={"model": model, **index},
                        )
                    )
                    seen.add((key, model))

                for compare_model in compare_models:
                    metric_func = partial(
                        sample_metric_deltas,
                        y_ref,
                        preds[key][compare_model],
                        y_true,
                        group_indices,
                        num_bootstraps=num_bootstraps_per_job,
                    )
                    bootstrapped_diffs.append(
                        nested_metrics_to_df(
                            x=p.map(metric_func, range(n_jobs)),
                            index_items={
                                "ref": ref_model,
                                "compare": compare_model,
                                **index,
                            },
                        )
                    )
    return pl.concat(bootstrapped_metrics), pl.concat(bootstrapped_diffs)


def differences_from_metrics(
    metrics: pl.DataFrame,
    comparisons: list[tuple[str, tuple[str, ...]]],
) -> pl.DataFrame:
    """Recombine cached paired draws without fitting or resampling.

    run_bootstrap_estimates uses the same seed and variant ordering for every
    prediction in a cohort. Within each dataset/biotype/model, row position is
    therefore the paired draw index, including across its existing cache files.
    Use metrics from one cohort and the same bootstrap configuration only.
    """
    keys = ["dataset", "biotype", "bootstrap"]
    indexed = metrics.with_columns(
        bootstrap=pl.int_range(pl.len()).over("dataset", "biotype", "model")
    )
    pairs = pl.DataFrame(
        [(ref, other) for ref, others in comparisons for other in others],
        schema=["ref", "compare"],
        orient="row",
    )
    return (
        pairs
        .join(indexed.rename({"model": "ref"}), on="ref")
        .join(
            indexed.rename({"model": "compare"}),
            on=["compare", *keys],
            suffix="_other",
            validate="m:1",
        )
        .select(
            *[
                expr
                for metric in ("auroc", "auprc")
                for expr in (
                    (pl.col(f"{metric}_other") - pl.col(metric)).alias(metric),
                    (
                        100
                        * (pl.col(f"{metric}_other") - pl.col(metric))
                        / pl.col(metric)
                    ).alias(f"{metric}_pct"),
                )
            ],
            "ref",
            "compare",
            "dataset",
            "biotype",
        )
    )


def cached_bootstraps(
    path: Path,
    preds: dict[tuple[str, str], dict[str, np.ndarray]],
    labels: dict[tuple[str, str], np.ndarray],
    groups: dict[tuple[str, str], np.ndarray],
    comparisons: list[tuple[str, tuple[str, ...]]],
    regenerate: bool,
    n_jobs: int = 8,
    num_bootstraps_per_job: int = 100,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reuse bootstrap results unless the corresponding predictions were regenerated."""
    if regenerate or not path.exists():
        result = run_bootstrap_estimates(
            preds, labels, groups, comparisons, n_jobs, num_bootstraps_per_job
        )
        with path.open("wb") as fh:
            dump(result, fh)
        return result
    with path.open("rb") as fh:
        return load(fh)
