"""Quadratic logistic regression and grouped evaluation for complementarity.

Model specifications map prediction names to (feature columns, estimator) pairs.
All specifications in an evaluation share complete-case rows and outer folds;
variants are held out together so their gene rows never cross outer folds.
"""

from collections import defaultdict
from typing import Literal

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

type ModelSpecs = dict[str, tuple[list[str], BaseEstimator]]


def make_quadratic_model(n_jobs: int = 8) -> Pipeline:
    """Build the notebook's scaled degree-two model with log-loss tuning."""
    return make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        LogisticRegressionCV(
            l1_ratios=(0.0,),
            scoring="neg_log_loss",
            use_legacy_attributes=False,
            n_jobs=n_jobs,
        ),
    )


def model_to_specs(score_name: str, model: BaseEstimator) -> ModelSpecs:
    """Combine one model score with phyloP, Gnocchi, or both."""
    shorthand = score_name.replace("_score", "")
    return {
        f"{shorthand}_gnocchi": ([score_name, "gnocchi_z"], model),
        f"{shorthand}_phylop": ([score_name, "phylop"], model),
        f"{shorthand}_gnocchi_phylop": ([score_name, "gnocchi_z", "phylop"], model),
    }


def make_model_specs(
    score_to_model: dict[str, str],
    model: BaseEstimator | None = None,
) -> ModelSpecs:
    """Build annotation-only, score-only, and combined specifications.

    score_to_model maps score columns to display/prediction names. Estimators
    are cloned by cross-validation, so specifications can share one estimator.
    """
    if model is None:
        model = make_quadratic_model()
    specs = {
        "PhyloP": (["phylop"], model),
        "Gnocchi": (["gnocchi_z"], model),
    }
    for score, name in score_to_model.items():
        specs[name] = ([score], model)
        specs.update(model_to_specs(score, model))
    return specs


def prepare_labeled_data(
    data: pl.DataFrame,
    feature_cols: list[str],
    label_columns: list[str],
) -> pl.DataFrame:
    """Collapse labels to single binary 0/1 labels."""
    return (
        data
        .filter(
            pl.all_horizontal(pl.col(feature_cols).is_not_null()),
            pl.any_horizontal(pl.col(label_columns).is_not_null()),
        )
        .with_columns(
            label=pl.any_horizontal(
                (pl.col(label_columns) == "high_pip").fill_null(False),
                (pl.col(label_columns).str.contains("athogenic")).fill_null(False),
            ).cast(pl.Int8),
        )
    )


def select_biotype_group(
    data: pl.DataFrame,
    biotype: Literal["all", "protein_coding", "lncRNA"],
) -> pl.DataFrame:
    return data if biotype == "all" else data.filter(pl.col("biotype") == biotype)


def cv_eval_models(
    full_data: pl.DataFrame,
    label_cols: list[str],
    models: ModelSpecs,
    cv: StratifiedGroupKFold,
    n_jobs: int = 2,
) -> tuple[pl.DataFrame, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Evaluate models on common rows with variant-grouped outer folds."""
    # All paired comparisons must use the same variant-gene rows and folds.
    all_features = list(
        dict.fromkeys(feat for features, _ in models.values() for feat in features)
    )
    labeled_data = prepare_labeled_data(full_data, all_features, label_cols)
    groups = labeled_data.get_column("variant").to_numpy()
    y = labeled_data.get_column("label").to_numpy()
    splits: list[tuple[np.ndarray, np.ndarray]] = list(cv.split(labeled_data, y, groups))

    predictions = {}
    metrics = []
    for model_name, (features, model) in models.items():
        prediction = cross_val_predict(
            model,
            labeled_data.select(features).to_numpy(),
            y,
            cv=splits,
            method="predict_proba",
            n_jobs=n_jobs,
        )[:, 1]
        predictions[model_name] = prediction
        metrics.append({
            "model": model_name,
            "auroc": roc_auc_score(y, prediction),
            "auprc": average_precision_score(y, prediction),
        })
    return pl.DataFrame(metrics), y, predictions, groups


def eval_all_datasets(
    full_data: pl.DataFrame,
    dataset_specs: dict[str, list[str]],
    want_biotypes: list[str],
    model_specs: ModelSpecs,
    n_folds: int = 5,
    n_jobs: int = 2,
) -> tuple[
    pl.DataFrame,
    dict[tuple[str, str], dict[str, np.ndarray]],
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
]:
    """Return metrics, OOF predictions, labels, and variants by dataset/biotype."""
    complementarity_cv = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=42,
    )

    oof_predictions = {}
    oof_outcomes = {}
    oof_groups = {}
    model_metrics = []
    for dataset, label_columns in dataset_specs.items():
        for biotype in want_biotypes:
            data = select_biotype_group(full_data, biotype)
            metrics, _y, _predictions, _groups = cv_eval_models(
                full_data=data,
                label_cols=label_columns,
                models=model_specs,
                cv=complementarity_cv,
                n_jobs=n_jobs,
            )
            key = (dataset, biotype)
            oof_outcomes[key] = _y
            oof_groups[key] = _groups
            oof_predictions[key] = _predictions
            model_metrics.append(
                metrics.with_columns(dataset=pl.lit(dataset), biotype=pl.lit(biotype))
            )
    model_metrics = pl.concat(model_metrics).sort(
        "dataset", "biotype", "auroc", descending=[False, False, True]
    )
    return model_metrics, oof_predictions, oof_outcomes, oof_groups


def prediction_name(
    model: str,
    features: tuple[str, ...],
    *,
    model_to_score: dict[str, str],
    model_specs: ModelSpecs,
) -> str:
    """Resolve a feature set to a fitted prediction; 'model' means this model's score."""
    wanted = {model_to_score[model] if f == "model" else f for f in features}
    return next(name for name, (cols, _) in model_specs.items() if set(cols) == wanted)


def make_comparisons(
    models: list[str],
    reference: tuple[str, ...] = ("model",),
    alternatives: tuple[tuple[str, ...], ...] = (
        ("model", "phylop"),
        ("model", "gnocchi_z"),
        ("model", "gnocchi_z", "phylop"),
    ),
    *,
    model_to_score: dict[str, str],
    model_specs: ModelSpecs,
) -> list[tuple[str, tuple[str, ...]]]:
    """Define paired differences as alternative minus reference on the same cohort."""
    comparisons = defaultdict(list)
    for model in models:
        ref = prediction_name(
            model, reference, model_to_score=model_to_score, model_specs=model_specs
        )
        comparisons[ref].extend(
            prediction_name(
                model, features, model_to_score=model_to_score, model_specs=model_specs
            )
            for features in alternatives
        )
    return [(ref, tuple(dict.fromkeys(others))) for ref, others in comparisons.items()]


def combine_cohorts(
    missense: pl.DataFrame,
    genome_wide: pl.DataFrame,
    model_col: str = "compare",
    *,
    protein_scores: list[str],
    model_specs: ModelSpecs,
) -> pl.DataFrame:
    """Use missense results for protein models and genome-wide results for the rest."""
    protein_specs = [
        name
        for name, (features, _) in model_specs.items()
        if set(protein_scores).intersection(features)
    ]
    return pl.concat([
        missense.filter(pl.col(model_col).is_in(protein_specs)),
        genome_wide.filter(~pl.col(model_col).is_in(protein_specs)),
    ])
