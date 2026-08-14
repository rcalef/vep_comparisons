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
- `final_variants.downsampled.tsv.gz`: all positive variants plus a sample of
  the unique negative-label variants in each retained biotype.

The negative sampling fraction defaults to `0.1` and the random seed defaults to
`42`. They can be changed with `--neg-fraction` and `--seed`. The negative label
defaults to `low_pip`; use repeated `--neg-labels` arguments for datasets with
other negative classes. For example, ClinVar uses:

```bash
curate-vep-variants \
  --vep-output variant_effect_output.txt.gz \
  --selected-variants filtered_clinvar.tsv.gz \
  --neg-labels Benign \
  --neg-labels Likely_benign \
  --neg-fraction 0.25 \
  --output-prefix final_variants
```

Sampling is at the variant level, so every gene annotation is retained when a
multi-gene variant is sampled. Labels not listed with `--neg-labels` are all
preserved. The command reports row and unique-variant counts for the input,
filtered, full, and downsampled stages.

### Input contract

The selected-variant input must be a headered TSV and contain one row per
`variant` with these columns:

```text
variant  chromosome  start  end  ref  alt  pip  label
```

Additional selected-variant metadata columns are passed through unchanged.
By default, `label` is expected to contain `high_pip` or `low_pip`; other label
sets are supported by specifying their negative values with `--neg-labels`.
Chromosomes are expected as autosomal `chrN` names, and coordinates use a
zero-based `start` and one-based `end`. The VEP input is the standard headered
tabular output. Lines beginning with `##` are ignored and `-` is read as null.

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
- `notebooks/curate_clinvar.py`

## Protein variant scoring

`score-protein-variants` performs inference-only, masked-marginal scoring with
one local checkpoint on complete protein sequences. For example:

```bash
export MAGNETON_MODEL_DIR=/orcd/data/manoli/001/om/rcalef/model_weights

score-protein-variants \
  --variants final_variants.tsv.gz \
  --sequences gencode.v49.pc_translations.fa.gz \
  --model esmc-300m \
  --output esmc_300m_scores
```

The available models are `esmc-300m`, `esmc-600m`, `saprot-35m`, and
`saprot-650m`. SaProt additionally requires `--structure-tokens` pointing to a
complete ENST-keyed 3Di FASTA. Tokens must be lowercase Foldseek tokens (or `#`)
and exactly match translation lengths.

Eligible rows are protein-coding annotations whose comma-separated consequence
terms include `missense_variant`. All eligible rows are validated before model
loading. Invalid or incomplete inputs abort without publishing output. Proteins
above the configured full-sequence limit are retained with a null score; the
default limits are 2,046 residues for ESM-C and 1,024 for SaProt. Scores are
`log P(ALT) - log P(REF)`, so higher values mean greater model preference for
the alternate residue.

The command atomically writes `<output>.tsv.gz` with columns `variant`, `gene`,
`feature`, `protein_position`, `amino_acids`, `model`, and `score`. It is
single-process and non-resumable. `float32` is the default; use `--dtype` to
explicitly request `bfloat16` or `float16`.
