"""Extend a fixed probe manifest with Evo2 scores, retaining every membership and fold."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

KEYS = ["variant", "gene", "feature"]


def add_evo2_scores(manifest: pl.DataFrame, scores: pl.DataFrame) -> pl.DataFrame:
    assert (scores["model"] == "evo2-7b").all()
    assert (scores["window_length"] == 8192).all()
    selected = (
        scores
        .select(*KEYS, "score")
        .with_columns(evo2_score=-pl.col("score"))
        .drop("score")
    )
    extended = (
        manifest
        .drop("evo2_score", strict=False)
        .join(selected, on=KEYS, how="left", validate="m:1", maintain_order="left")
    )
    assert extended.height == manifest.height
    assert extended["evo2_score"].is_finite().all() and extended["evo2_score"].null_count() == 0
    return extended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True,
                        help="standard Evo2 score table(s), including extraction TSV shards")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = pl.concat([pl.read_csv(path, separator="\t") for path in args.scores])
    extended = add_evo2_scores(pl.read_parquet(args.manifest), scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extended.write_parquet(args.output)
    print(f"Saved {extended.height} unchanged memberships with Evo2 scores to {args.output}")


if __name__ == "__main__":
    main()
