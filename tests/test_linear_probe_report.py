import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl

from vep_comparisons.linear_probes.constants import MODEL_LABELS, PROTEIN_MODELS
from vep_comparisons.linear_probes.report import comparison_table, plot_comparisons, save_figures
from vep_comparisons.linear_probes.train import summarize_metrics


def report_metrics() -> pl.DataFrame:
    return pl.DataFrame([
        {'model': model, 'cohort': cohort, 'task': task, 'outer_fold': fold,
         'n_sites': 10, 'n_low': 5, 'n_high': 5,
         'probe_auroc': 0.7 + fold / 100, 'zero_shot_auroc': 0.6,
         'delta_auroc': 0.1 + fold / 100,
         'probe_ap': 0.71 + fold / 100, 'zero_shot_ap': 0.61, 'delta_ap': 0.1 + fold / 100}
        for cohort in ('dna_rna', 'shared_missense') for task in ('phylop', 'gnocchi_z')
        for model in MODEL_LABELS for fold in range(5)
        if cohort == 'shared_missense' or model not in PROTEIN_MODELS
    ])


def test_report_separates_cohorts_and_preserves_metric_scale() -> None:
    folds = report_metrics()
    summary = summarize_metrics(folds)
    broad = comparison_table(summary, 'dna_rna')
    shared = comparison_table(summary, 'shared_missense')
    dna_models = len(MODEL_LABELS) - len(PROTEIN_MODELS)
    assert broad.height == 2 * dna_models and shared.height == 2 * len(MODEL_LABELS)
    assert set(broad['annotation']) == {'phyloP', 'Gnocchi'}
    assert broad['n_sites'].to_list() == [50] * (2 * dna_models)
    assert broad['probe_auroc'].to_list() == [0.72] * (2 * dna_models)
    assert set(shared['model']) == set(MODEL_LABELS.values())
    fig = plot_comparisons(folds)
    assert len(fig.axes) == 4
    assert [len(ax.get_yticklabels()) for ax in fig.axes] == [dna_models] * 2 + [len(MODEL_LABELS)] * 2
    assert all('N = 50' in ax.get_title() for ax in fig.axes)
    plt.close(fig)


def test_publication_figures_can_be_regenerated(tmp_path: Path) -> None:
    paths = save_figures(report_metrics(), tmp_path)
    assert len(paths) == 6
    assert all(path.stat().st_size > 1000 for path in paths)
