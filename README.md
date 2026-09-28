# VEP comparisons

This repository supports the analyses our work focused on whether prevailing
language models capture deep evolutionary conservation and human mutational 
constraint, and what the consequences may be for human variant effect prediction
performance. It contains the workflows used to prepare variant datasets, score
variants with biological language models, add genomic annotations, and produce
the paper's analyses and figures.

## Overview

The analysis proceeds in four broad stages:

1. Prepare and curate the UK Biobank, MultiSuSiE, cis-eQTL, and ClinVar variant
   datasets with the notebooks in `notebooks/curate_*.py`.
2. Combine the curated datasets with `notebooks/collate_variants.py`.
3. Score variants with protein, DNA, and RNA language models through the
   `vep-comparisons` command-line interface. The scripts in `scripts/` record the
   cluster jobs used for the paper.
4. Add annotations and reproduce the comparisons and figures with
   `notebooks/add_annotations.py`, `notebooks/compare_annotations.py`, and
   `notebooks/make_plots.ipynb`.

Reference genomes, gene annotations, transcript/protein sequences, model
checkpoints, and the source datasets are external inputs and are not distributed
with this repository.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync --frozen
```

Protein and DNA model dependencies are available as optional environments:

```bash
uv sync --frozen --extra protein-models
uv sync --frozen --extra ntv3
```

The RNA models use separate environments because their dependency requirements
differ from the main project:

```bash
uv sync --project environments/orthrus --frozen
uv sync --project environments/rinalmo --frozen
```

## Usage

The unified CLI exposes commands for variant curation and protein, DNA, and RNA
scoring:

```bash
uv run vep-comparisons --help
uv run vep-comparisons curate --help
uv run vep-comparisons protein score --help
uv run vep-comparisons dna score --help
uv run vep-comparisons rna orthrus score --help
uv run vep-comparisons rna rinalmo score --help
```

Scoring jobs can be divided into shards for cluster execution and then combined
with the corresponding `collate` command. See `scripts/` for the invocations and
resource settings used in the paper. Paths in those scripts are specific to the
original compute environment and should be adapted to local data, checkpoints,
and scheduler configuration.

Run the test suite with:

```bash
uv run pytest
```
