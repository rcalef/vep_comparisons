"""Regenerate comparison tables and figures from saved probe evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import polars as pl

from .constants import MODEL_LABELS

COHORT_LABELS = {'dna_rna': 'DNA/RNA: missense + lncRNA', 'shared_missense': 'Shared missense'}
TASK_LABELS = {'phylop': 'phyloP', 'gnocchi_z': 'Gnocchi'}


def comparison_table(summary: pl.DataFrame, cohort: str) -> pl.DataFrame:
    return (
        summary
        .filter(pl.col('cohort') == cohort)
        .with_columns(model=pl.col('model').replace(MODEL_LABELS), annotation=pl.col('task').replace(TASK_LABELS))
        .select('model', 'annotation', 'n_sites', 'n_low', 'n_high', 'probe_auroc', 'zero_shot_auroc',
                'delta_auroc', 'probe_auroc_fold_sd', 'zero_shot_auroc_fold_sd', 'probe_ap', 'zero_shot_ap', 'delta_ap')
        .sort('annotation', 'model')
    )


def plot_comparisons(folds: pl.DataFrame, metric: str = 'auroc', *, differences: bool = False) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for row, (cohort, cohort_label) in enumerate(COHORT_LABELS.items()):
        for column, (task, task_label) in enumerate(TASK_LABELS.items()):
            ax = axes[row, column]
            data = folds.filter((pl.col('cohort') == cohort) & (pl.col('task') == task))
            models = [model for model in MODEL_LABELS if model in data['model']]
            n = int(data.filter(pl.col('model') == models[0])['n_sites'].sum())
            for index, model in enumerate(models):
                values = data.filter(pl.col('model') == model).sort('outer_fold')
                if differences:
                    deltas = values[f'delta_{metric}'].to_numpy()
                    ax.scatter(deltas, index + np.linspace(-0.12, 0.12, len(deltas)), color='#1b7837', alpha=0.4, s=20)
                    ax.scatter(deltas.mean(), index, color='#1b7837', marker='D', s=45)
                else:
                    means = [values[f'{method}_{metric}'].mean() for method in ('zero_shot', 'probe')]
                    ax.plot(means, [index, index], color='0.7', linewidth=1)
                    for offset, method, label, color in ((-0.1, 'zero_shot', 'Zero-shot', '#666666'),
                                                       (0.1, 'probe', 'Linear probe', '#2166ac')):
                        points = values[f'{method}_{metric}'].to_numpy()
                        ax.scatter(points, index + offset + np.linspace(-0.04, 0.04, len(points)), color=color, alpha=0.35, s=18)
                        ax.scatter(points.mean(), index + offset, color=color, s=45, label=label if index == 0 else None)
            ax.set_yticks(range(len(models)), [MODEL_LABELS[model] for model in models])
            ax.invert_yaxis()
            ax.set_title(f'{cohort_label}\n{task_label}; N = {n:,}')
            ax.set_xlabel(('Probe − zero-shot ' if differences else '') + ('AUROC' if metric == 'auroc' else 'average precision'))
            if differences:
                ax.axvline(0, color='0.6', linestyle='--', linewidth=1)
            else:
                ax.set_xlim(0, 1)
                if metric == 'auroc':
                    ax.axvline(0.5, color='0.7', linestyle='--', linewidth=0.8)
                ax.legend(loc='lower left', fontsize=8)
            ax.grid(axis='x', alpha=0.2)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Paired improvements' if differences else 'Frozen-representation linear probes versus matched zero-shot scores')
    return fig


def save_figures(folds: pl.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, metric, differences in [('linear_probes_auroc', 'auroc', False),
                                      ('linear_probes_ap', 'ap', False),
                                      ('linear_probes_delta_auroc', 'auroc', True)]:
        fig = plot_comparisons(folds, metric, differences=differences)
        for suffix in ('pdf', 'png'):
            path = output_dir / f'{name}.{suffix}'
            fig.savefig(path, dpi=200)
            paths.append(path)
        plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--figure-dir', type=Path, required=True)
    args = parser.parse_args()
    summary = pl.read_csv(args.results_dir / 'summary.tsv', separator='\t')
    folds = pl.read_csv(args.results_dir / 'fold_metrics.tsv', separator='\t')
    predictions = pl.read_parquet(args.results_dir / 'predictions.parquet')
    assert predictions.height == predictions.unique(['model', 'cohort', 'task', 'site_id']).height == int(summary['n_sites'].sum())
    for cohort in COHORT_LABELS:
        comparison_table(summary, cohort).write_csv(args.results_dir / f'comparison_{cohort}.tsv', separator='\t')
    for path in save_figures(folds, args.figure_dir):
        print(path)


if __name__ == '__main__':
    main()
