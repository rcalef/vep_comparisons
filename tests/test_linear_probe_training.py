from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from vep_comparisons.linear_probes.collate import collate_model
from vep_comparisons.linear_probes.train import align_embeddings, evaluate_task, inner_splits, summarize_metrics
from vep_comparisons.linear_probes.merge_results import merge_results


def synthetic_task() -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 88
    labels = np.tile([0, 1, 0, 1], 22)
    rows = pl.DataFrame({
        'site_id': [f's{i:03d}' for i in range(n)], 'context_id': [f'c{i:03d}' for i in range(n)],
        'chromosome': [f'chr{i // 4 + 1}' for i in range(n)],
        'outer_fold': [(i // 4) % 5 for i in range(n)], 'label': labels,
        'cohort': ['dna_rna'] * n, 'task': ['phylop'] * n, 'orthrus_score': -labels.astype(float),
    })
    vectors = np.column_stack([np.arange(n), labels + rng.normal(0, 0.8, n), rng.normal(size=n)])
    return rows, vectors


def test_collation_reorders_vectors_and_checks_full_coverage(tmp_path: Path) -> None:
    rows, _ = synthetic_task()
    rows = rows.head(4)
    for shard, indices in enumerate(([2, 0], [3, 1])):
        vectors = np.repeat(np.array(indices, dtype=np.float32)[:, None], 512, axis=1)
        np.savez(tmp_path / f'shard-{shard:05d}-of-00002.npz',
                 site_id=np.array(rows[indices]['site_id'].to_list()), context_id=np.array(rows[indices]['context_id'].to_list()),
                 embeddings=vectors)
    output = tmp_path / 'collated.npz'
    summary = collate_model(pl.concat([rows, rows]), 'orthrus', tmp_path, output, 2)
    assert summary['sites'] == 4 and summary['dimension'] == 512
    with np.load(output, allow_pickle=False) as data:
        assert data['context_id'].tolist() == rows['context_id'].to_list()
        np.testing.assert_array_equal(data['embeddings'][:, 0], np.arange(4))
    aligned, vectors = align_embeddings(rows.reverse(), output)
    assert aligned['site_id'].to_list() == rows['site_id'].to_list()
    np.testing.assert_array_equal(vectors[:, 0], np.arange(4))


@pytest.mark.parametrize('failure', ['duplicate', 'wrong_context', 'nan', 'dimension'])
def test_collation_rejects_incorrect_embedding_sets(tmp_path: Path, failure: str) -> None:
    rows, _ = synthetic_task()
    rows = rows.head(4)
    ids = np.array(rows['site_id'].to_list())
    contexts = np.array(rows['context_id'].to_list())
    vectors = np.ones((4, 511 if failure == 'dimension' else 512), dtype=np.float32)
    if failure == 'duplicate':
        ids[1], contexts[1] = ids[0], contexts[0]
    if failure == 'wrong_context':
        contexts[0] = 'wrong'
    if failure == 'nan':
        vectors[0, 0] = np.nan
    np.savez(tmp_path / 'shard-00000-of-00001.npz', site_id=ids, context_id=contexts, embeddings=vectors)
    with pytest.raises(AssertionError):
        collate_model(rows, 'orthrus', tmp_path, tmp_path / 'output.npz', 1)


@pytest.mark.parametrize('model', ['orthrus', 'evo2'])
@pytest.mark.parametrize('warm_start_search', [False, True])
def test_nested_cv_fits_scalers_only_on_training_chromosomes(
    monkeypatch: pytest.MonkeyPatch, model: str, warm_start_search: bool,
) -> None:
    rows, vectors = synthetic_task()
    rows = rows.rename({'orthrus_score': f'{model}_score'})
    fits = []
    native_fit = StandardScaler.fit

    def record_fit(self: StandardScaler, x: np.ndarray, y: np.ndarray | None = None,
                   sample_weight: np.ndarray | None = None) -> StandardScaler:
        fits.append(frozenset(x[:, 0].astype(int)))
        return native_fit(self, x, y, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, 'fit', record_fit)
    predictions, metrics, validation, assignments = evaluate_task(rows, vectors, model, warm_start_search=warm_start_search)
    expected = []
    folds = rows['outer_fold'].to_numpy()
    chromosomes = rows['chromosome'].to_numpy()
    for fold in range(5):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        assert set(chromosomes[train]).isdisjoint(chromosomes[test])
        for inner_train, inner_test in inner_splits(chromosomes[train]):
            assert set(chromosomes[train][inner_train]).isdisjoint(chromosomes[train][inner_test])
            expected.extend([frozenset(train[inner_train])] * 7)
        expected.append(frozenset(train))
        assigned = assignments.filter(pl.col('outer_fold') == fold)['chromosome'].to_list()
        assert set(assigned) == set(chromosomes[train])
        assert len(assigned) == len(set(assigned))
        scored = predictions.filter(pl.col('outer_fold') == fold)
        metric = metrics.filter(pl.col('outer_fold') == fold).row(0, named=True)
        assert metric['probe_auroc'] == roc_auc_score(scored['label'], scored['probability'])
        assert metric['probe_ap'] == average_precision_score(scored['label'], scored['probability'])
        assert metric['zero_shot_auroc'] == 0.0  # retain the supplied sign, no fold-wise flipping
        assert scored['zero_shot_score'].to_list() == rows[test][f'{model}_score'].to_list()
    assert Counter(fits) == Counter(expected)
    assert predictions.height == predictions.unique('site_id').height == rows.height
    assert predictions['probability'].is_between(0, 1).all()
    assert metrics.height == 5 and validation.height == 5 * 7 * 3
    summary = summarize_metrics(metrics)
    assert summary['probe_auroc'][0] == pytest.approx(metrics['probe_auroc'].mean())
    assert summary['delta_auroc'][0] == pytest.approx(metrics['delta_auroc'].mean())
    assert summary['n_sites'][0] == rows.height


def test_merge_saved_evaluations_retains_predictions_and_rejects_duplicates(tmp_path: Path) -> None:
    rows, vectors = synthetic_task()
    sources, saved_predictions = [], []
    for model in ('orthrus', 'evo2'):
        source = tmp_path / model
        model_dir = source / model
        model_dir.mkdir(parents=True)
        model_rows = rows.rename({'orthrus_score': f'{model}_score'})
        predictions, metrics, validation, assignments = evaluate_task(model_rows, vectors, model)
        predictions.write_parquet(source / 'predictions.parquet')
        predictions.write_parquet(model_dir / 'predictions.parquet')
        for name, data in (('fold_metrics', metrics), ('inner_validation', validation),
                           ('inner_chromosome_folds', assignments), ('summary', summarize_metrics(metrics))):
            data.write_csv(source / f'{name}.tsv', separator='\t')
        pl.DataFrame({'setting': ['seed', 'max_iter'], 'value': ['20261005', '20000' if model == 'evo2' else '3000']}).write_csv(source / 'settings.tsv', separator='\t')
        sources.append(source)
        saved_predictions.append(predictions)
    output = tmp_path / 'combined'
    summary = merge_results(sources, output)
    assert summary.height == 2 and summary['n_folds'].to_list() == [5, 5]
    assert pl.read_parquet(output / 'predictions.parquet').equals(pl.concat(saved_predictions))
    settings = dict(pl.read_csv(output / 'settings.tsv', separator='\t').iter_rows())
    assert settings == {'seed': 20261005, 'max_iter_orthrus': 3000, 'max_iter_evo2': 20000}
    for source, predictions in zip(sources, saved_predictions, strict=True):
        assert pl.read_parquet(source / 'predictions.parquet').equals(predictions)
        assert pl.read_parquet(output / source.name / 'predictions.parquet').equals(predictions)
    with pytest.raises(AssertionError):
        merge_results([sources[0], sources[0]], tmp_path / 'duplicate')


@pytest.mark.parametrize('warm_start_search', [False, True])
def test_parallel_inner_fits_preserve_predictions_and_selected_regularization(warm_start_search: bool) -> None:
    rows, vectors = synthetic_task()
    serial = evaluate_task(rows, vectors, 'orthrus')
    parallel = evaluate_task(rows, vectors, 'orthrus', cv_jobs=2, warm_start_search=warm_start_search)
    assert all(left.equals(right) for left, right in zip(serial, parallel, strict=True))
