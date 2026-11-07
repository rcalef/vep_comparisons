"""Combine saved evaluations on a fixed benchmark without refitting existing probes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import shutil

import polars as pl

from .train import summarize_metrics


def merge_results(sources: Sequence[Path], output_dir: Path) -> pl.DataFrame:
    source_settings = [pl.read_csv(source / "settings.tsv", separator="\t") for source in sources]
    optimization = ["max_iter", "warm_start_search"]
    settings = source_settings[0].filter(~pl.col("setting").is_in(optimization))
    # An increased iteration ceiling allows convergence at the same tolerance.
    assert all(settings.equals(data.filter(~pl.col("setting").is_in(optimization))) for data in source_settings)
    predictions = pl.concat([pl.read_parquet(source / "predictions.parquet") for source in sources])
    assert predictions.height == predictions.unique(["model", "cohort", "task", "site_id"]).height
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output_dir / "predictions.parquet")
    summary = None
    for name in ("fold_metrics", "inner_validation", "inner_chromosome_folds"):
        data = pl.concat([pl.read_csv(source / f"{name}.tsv", separator="\t") for source in sources])
        data.write_csv(output_dir / f"{name}.tsv", separator="\t")
        if name == "fold_metrics":
            assert data.height == data.unique(["model", "cohort", "task", "outer_fold"]).height
            summary = summarize_metrics(data)
            assert predictions.height == int(summary["n_sites"].sum())
            summary.write_csv(output_dir / "summary.tsv", separator="\t")
    for source, data in zip(sources, source_settings, strict=True):
        iterations = data.filter(pl.col("setting").is_in(optimization))
        for model in pl.read_csv(source / "summary.tsv", separator="\t")["model"].unique():
            shutil.copytree(source / model, output_dir / model, dirs_exist_ok=True)
            settings = pl.concat([settings, iterations.with_columns(setting=pl.col("setting") + f"_{model}")])
    settings.write_csv(output_dir / "settings.tsv", separator="\t")
    assert summary is not None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = merge_results(args.sources, args.output_dir)
    print(f"Merged {summary.height} evaluations and {summary['n_folds'].sum()} outer folds into {args.output_dir}")


if __name__ == "__main__":
    main()
