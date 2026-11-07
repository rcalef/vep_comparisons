"""Chromosome-held-out linear probes with nested regularization selection."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import warnings

from joblib import Parallel, delayed, parallel_config
import numpy as np
import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .constants import DEFAULT_MODELS, DIMENSIONS, PROTEIN_MODELS

CS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]


def make_pipeline(seed: int, max_iter: int, *, warm_start: bool = False) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(l1_ratio=0.0, solver='lbfgs', max_iter=max_iter,
                                          random_state=seed, warm_start=warm_start)),
    ])


def align_embeddings(rows: pl.DataFrame, embedding_file: Path) -> tuple[pl.DataFrame, np.ndarray]:
    with np.load(embedding_file, allow_pickle=False) as data:
        identifiers = pl.DataFrame({'site_id': data['site_id'], 'context_id': data['context_id']}).with_row_index('embedding_row')
        aligned = (
            rows
            .sort('site_id', 'context_id')
            .join(identifiers, on=['site_id', 'context_id'], how='left', validate='1:1', maintain_order='left')
        )
        assert aligned['embedding_row'].null_count() == 0
        vectors = data['embeddings'][aligned['embedding_row'].to_numpy()]
    assert np.isfinite(vectors).all()
    return aligned.drop('embedding_row'), vectors


def inner_splits(chromosomes: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(GroupKFold(n_splits=3).split(np.zeros(len(chromosomes)), groups=chromosomes))


def search_regularization_path(
    vectors: np.ndarray, labels: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]],
    *, seed: int, max_iter: int, cv_jobs: int, description: str,
) -> tuple[Pipeline, float, np.ndarray]:
    """Fit the complete C grid, reusing coefficients only within one inner fold."""
    def fit_path(inner_fold: int, train: np.ndarray, validation: np.ndarray) -> list[float]:
        pipeline = make_pipeline(seed, max_iter, warm_start=True)
        scores = []
        for c in CS:
            pipeline.set_params(classifier__C=c)
            pipeline.fit(vectors[train], labels[train])
            scores.append(roc_auc_score(labels[validation], pipeline.predict_proba(vectors[validation])[:, 1]))
            print(f'{description} inner_fold={inner_fold} C={c:g} AUROC={scores[-1]:.4f}', flush=True)
        return scores

    scores = np.array(Parallel(n_jobs=cv_jobs, prefer='threads')(
        delayed(fit_path)(inner_fold, train, validation)
        for inner_fold, (train, validation) in enumerate(splits)
    )).T
    c = CS[int(np.argmax(scores.mean(axis=1)))]
    # Refit the selected model from scratch on all outer training chromosomes.
    pipeline = make_pipeline(seed, max_iter).set_params(classifier__C=c)
    pipeline.fit(vectors, labels)
    return pipeline, c, scores


def evaluate_task(
    rows: pl.DataFrame, vectors: np.ndarray, model: str, *, seed: int = 20261005, max_iter: int = 3000,
    cv_jobs: int = 1,
    warm_start_search: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    labels = rows['label'].to_numpy()
    folds = rows['outer_fold'].to_numpy()
    chromosomes = rows['chromosome'].to_numpy()
    zero_shot = rows[f'{model}_score'].to_numpy()
    probabilities = np.full(len(rows), np.nan)
    chosen_cs = np.full(len(rows), np.nan)
    metrics, validation, assignments = [], [], []
    keys = {'model': model, 'cohort': rows['cohort'][0], 'task': rows['task'][0]}
    assert set(folds) == set(range(5))
    assert rows.group_by('chromosome').agg(pl.col('outer_fold').n_unique())['outer_fold'].max() == 1
    for fold in range(5):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        splits = inner_splits(chromosomes[train])
        for inner_fold, (_, held_out) in enumerate(splits):
            assignments.extend({**keys, 'outer_fold': fold, 'chromosome': chromosome, 'inner_fold': inner_fold}
                               for chromosome in sorted(set(chromosomes[train][held_out])))
        with warnings.catch_warnings(), parallel_config(backend='threading'):
            warnings.simplefilter('error', ConvergenceWarning)
            if warm_start_search:
                pipeline, c, inner_scores = search_regularization_path(
                    vectors[train], labels[train], splits, seed=seed, max_iter=max_iter,
                    cv_jobs=cv_jobs, description=f'{model} {keys["cohort"]} {keys["task"]} fold={fold}',
                )
            else:
                search = GridSearchCV(
                    make_pipeline(seed, max_iter), {'classifier__C': CS}, scoring='roc_auc',
                    cv=splits, refit=True, n_jobs=cv_jobs, error_score='raise',
                )
                search.fit(vectors[train], labels[train])
                pipeline = search.best_estimator_
                c = search.best_params_['classifier__C']
                inner_scores = np.array([search.cv_results_[f'split{i}_test_score'] for i in range(3)]).T
        probabilities[test] = pipeline.predict_proba(vectors[test])[:, 1]
        chosen_cs[test] = c
        for index, c_value in enumerate(CS):
            for inner_fold in range(3):
                validation.append({**keys, 'outer_fold': fold, 'C': c_value, 'inner_fold': inner_fold,
                                   'auroc': float(inner_scores[index, inner_fold])})
        probe_auroc = roc_auc_score(labels[test], probabilities[test])
        zero_auroc = roc_auc_score(labels[test], zero_shot[test])
        probe_ap = average_precision_score(labels[test], probabilities[test])
        zero_ap = average_precision_score(labels[test], zero_shot[test])
        metrics.append({**keys, 'outer_fold': fold, 'chosen_C': c, 'n_sites': len(test),
                        'n_low': int((labels[test] == 0).sum()), 'n_high': int((labels[test] == 1).sum()),
                        'probe_auroc': probe_auroc, 'zero_shot_auroc': zero_auroc,
                        'delta_auroc': probe_auroc - zero_auroc,
                        'probe_ap': probe_ap, 'zero_shot_ap': zero_ap, 'delta_ap': probe_ap - zero_ap})
        print(f'{model} {keys["cohort"]} {keys["task"]} fold={fold} C={c:g} '
              f'AUROC probe={probe_auroc:.4f} zero-shot={zero_auroc:.4f}', flush=True)
    assert np.isfinite(probabilities).all()
    predictions = (
        rows
        .select('site_id', 'context_id', 'chromosome', 'cohort', 'task', 'label', 'outer_fold')
        .with_columns(model=pl.lit(model), probability=pl.Series(probabilities),
                      chosen_C=pl.Series(chosen_cs), zero_shot_score=pl.Series(zero_shot))
    )
    return predictions, pl.DataFrame(metrics), pl.DataFrame(validation), pl.DataFrame(assignments)


def run_task(
    manifest: Path, embedding_dir: Path, output_dir: Path, model: str, cohort: str, task: str,
    seed: int, max_iter: int, cv_jobs: int = 1, warm_start_search: bool = False,
) -> Path:
    rows = pl.read_parquet(manifest).filter((pl.col('cohort') == cohort) & (pl.col('task') == task))
    rows, vectors = align_embeddings(rows, embedding_dir / f'{model}.npz')
    with threadpool_limits(limits=1):
        predictions, metrics, validation, assignments = evaluate_task(
            rows, vectors, model, seed=seed, max_iter=max_iter, cv_jobs=cv_jobs,
            warm_start_search=warm_start_search,
        )
    directory = output_dir / model / cohort / task
    directory.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(directory / 'predictions.parquet')
    metrics.write_csv(directory / 'fold_metrics.tsv', separator='\t')
    validation.write_csv(directory / 'inner_validation.tsv', separator='\t')
    assignments.write_csv(directory / 'inner_chromosome_folds.tsv', separator='\t')
    return directory


def summarize_metrics(metrics: pl.DataFrame) -> pl.DataFrame:
    return (
        metrics
        .group_by('model', 'cohort', 'task')
        .agg(n_folds=pl.len(), n_sites=pl.col('n_sites').sum(), n_low=pl.col('n_low').sum(), n_high=pl.col('n_high').sum(),
             **{name: pl.col(name).mean() for name in ('probe_auroc', 'zero_shot_auroc', 'delta_auroc', 'probe_ap', 'zero_shot_ap', 'delta_ap')},
             probe_auroc_fold_sd=pl.col('probe_auroc').std(), zero_shot_auroc_fold_sd=pl.col('zero_shot_auroc').std())
        .sort('cohort', 'task', 'model')
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--embedding-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=list(DIMENSIONS), default=list(DEFAULT_MODELS))
    parser.add_argument('--cohorts', nargs='+', choices=['dna_rna', 'shared_missense'], default=['dna_rna', 'shared_missense'])
    parser.add_argument('--tasks', nargs='+', choices=['phylop', 'gnocchi_z'], default=['phylop', 'gnocchi_z'])
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--cv-jobs', type=int, default=1,
                        help='concurrent inner-CV fits per task; each fit uses one BLAS thread')
    parser.add_argument('--warm-start-search', action='store_true',
                        help='reuse coefficients along the C grid within each inner fold')
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--max-iter', type=int, default=3000)
    args = parser.parse_args()
    directories = Parallel(n_jobs=args.jobs)(
        delayed(run_task)(args.manifest, args.embedding_dir, args.output_dir, model, cohort, task,
                          args.seed, args.max_iter, args.cv_jobs, args.warm_start_search)
        for model in args.models for cohort in args.cohorts for task in args.tasks
        if model not in PROTEIN_MODELS or cohort == 'shared_missense'
    )
    predictions = pl.concat([pl.read_parquet(directory / 'predictions.parquet') for directory in directories])
    predictions.write_parquet(args.output_dir / 'predictions.parquet')
    for name in ('fold_metrics', 'inner_validation', 'inner_chromosome_folds'):
        data = pl.concat([pl.read_csv(directory / f'{name}.tsv', separator='\t') for directory in directories])
        data.write_csv(args.output_dir / f'{name}.tsv', separator='\t')
        if name == 'fold_metrics':
            summarize_metrics(data).write_csv(args.output_dir / 'summary.tsv', separator='\t')
    settings = {'seed': args.seed, 'outer_folds': 5, 'inner_folds': 3, 'C_grid': ','.join(map(str, CS)),
                'solver': 'lbfgs', 'max_iter': args.max_iter, 'tol': 1e-4,
                'scikit_learn': version('scikit-learn'), 'numpy': version('numpy'), 'polars': version('polars')}
    if args.warm_start_search:
        settings['warm_start_search'] = True
    pl.DataFrame({'setting': list(settings), 'value': list(map(str, settings.values()))}).write_csv(args.output_dir / 'settings.tsv', separator='\t')


if __name__ == '__main__':
    main()
