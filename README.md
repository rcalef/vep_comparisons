# VEP comparisons

This repository contains dataset-preparation notebooks and a reusable command for
curating selected variants after annotation with Ensembl VEP.

## Variant curation command

Install the project environment with `uv sync`, then run:

```bash
curate-vep-variants \
  --vep-output variant_effect_output.txt.gz \
  --selected-variants filtered_variants.tsv.gz \
  --output-prefix final_variants
```

This writes two gzip-compressed, tab-separated files with headers:

- `final_variants.tsv.gz`: all variants passing curation.
- `final_variants.downsampled.tsv.gz`: all `high_pip` variants plus a sample of
  the unique `low_pip` variants in each retained biotype.

The low-PIP sampling fraction defaults to `0.1` and the random seed defaults to
`42`. They can be changed with `--low-pip-fraction` and `--seed`. Sampling is at
the variant level, so every gene annotation is retained when a multi-gene variant
is sampled. The command reports row and unique-variant counts for the input,
filtered, full, and downsampled stages.

### Input contract

The selected-variant input must be a headered TSV and contain one row per
`variant` with these columns:

```text
variant  chromosome  start  end  ref  alt  pip  label
```

Additional selected-variant metadata columns are passed through unchanged.
`label` is expected to contain `high_pip` or `low_pip`; chromosomes are expected
as autosomal `chrN` names; and coordinates use a zero-based `start` and one-based
`end`. The VEP input is the standard headered tabular output. Lines beginning
with `##` are ignored and `-` is read as null.

The VEP annotation fields appended to the selected metadata are `gene`,
`feature`, `consequence`, `cdna_position`, `cds_position`, `protein_position`,
`amino_acids`, `symbol`, and `biotype`.

### Fixed biological policy

The command applies the same policy to every dataset:

- Discard an annotation if any of its consequences is
  `downstream_gene_variant`, `upstream_gene_variant`, `intron_variant`, or
  `intergenic_variant`.
- Keep only variants represented by `protein_coding` or `lncRNA` annotations,
  and exclude a variant if it is represented by both biotypes.
- Retain annotations for multiple genes at a variant.
- Choose one transcript for each variant/gene pair, prioritizing MANE, canonical
  status, the lowest transcript support level (TSL), and finally lexical feature
  ID to break ties reproducibly.

The dataset notebooks prepare canonical selected-variant and VEP input files and
show their corresponding VEP and curation invocations:

- `notebooks/curate_ukbb_variants.py`
- `notebooks/curate_multisusie.py`
