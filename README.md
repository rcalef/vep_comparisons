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
`amino_acids`, `symbol`, `biotype`, `gnomade_af`, `gnomadg_af`, `max_af`, and
`max_af_pops`.

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
one local checkpoint on complete protein sequences. Install its isolated model
stack with `uv sync --extra protein-models`. For example:

```bash
export MAGNETON_MODEL_DIR=/orcd/data/manoli/001/om/rcalef/model_weights

score-protein-variants \
  --variants final_variants.tsv.gz \
  --sequences gencode.v49.pc_translations.fa.gz \
  --model esmc-300m \
  --batch-size 1 \
  --output esmc_300m_scores
```

The available models are `esmc-300m`, `esmc-600m`, `saprot-35m`, and
`saprot-650m`. SaProt additionally requires `--structure-tokens` pointing to a
complete ENST-keyed 3Di FASTA. Tokens must be lowercase Foldseek tokens (or `#`)
and exactly match translation lengths. If structure coverage is intentionally
partial, `--ignore-missing-structure-tokens` omits candidates on absent
transcripts. Present records with a length mismatch still fail validation so
residue coordinates cannot be silently corrupted.

Length-mismatched structure records can be conservatively recovered from
existing AlphaFoldDB isoform models with `recover-saprot-structures`. The
preparation notebook writes the required enriched mismatch inventory. The
recovery command accepts only a unique monomer whose API sequence exactly
matches the full GENCODE translation, downloads the API-provided PDB URL, runs
Foldseek once over the accepted directory, validates both Foldseek sequences,
and merges without replacing an existing transcript:

```bash
recover-saprot-structures \
  --mismatches scoring/saprot_recovery/mismatch_candidates.tsv \
  --translations gencode.v50.pc_translations.fa.gz \
  --existing-tokens scoring/filtered_foldseek_toks.fa.bz2 \
  --output-dir scoring/saprot_recovery \
  --variants final_variants.tsv.gz \
  --foldseek ~/install/foldseek/bin/foldseek \
  --threads 8
```

The output directory contains the cached API responses and PDBs,
`foldseek_descriptors.tsv`, a per-transcript `recovery_manifest.tsv`, recovered
tokens, and a separately published merged token FASTA. Failed discovery,
downloads, ambiguous matches, partial models, and invalid descriptors remain
explicit unresolved manifest rows. Recovery deliberately uses no additional
pLDDT mask so its processing remains consistent with the existing token
artifact.

Eligible rows are protein-coding annotations whose comma-separated consequence
terms include `missense_variant`. All eligible rows are validated before model
loading. Invalid or incomplete inputs abort without publishing output. Proteins
that exceed model context are scored by default with full-capacity windows
tiled symmetrically from both termini at approximately 50% overlap. Only tiles
containing a requested variant are evaluated. Scores from overlapping tiles are
combined with normalized sigmoid edge weights; true protein termini are not
downweighted. This can require up to three contexts per variant. The default
window lengths are 2,046 residues for ESM-C and 1,024 for SaProt.

`--max-sequence-length` controls the per-window length and cannot exceed the
selected model's capacity. Use `--long-sequence-mode null` to restore the legacy
behavior in which proteins longer than that limit receive null scores. Windowing
cannot recover interactions between residues separated by more than one model
context. Scores are `log P(ALT) - log P(REF)`, so higher values mean greater
model preference for the alternate residue.

The command atomically writes `<output>.tsv.gz` with columns `variant`, `gene`,
`feature`, `protein_position`, `amino_acids`, `model`, and `score`. It is
single-process and non-resumable. `float32` is the default; use `--dtype` to
explicitly request `bfloat16` or `float16`. `--batch-size` controls the number
of masked protein positions evaluated per forward pass and defaults to `1`.

## NTv3 genomic SNV scoring

`score-dna-variants` implements offline, masked-marginal scoring with the pinned
`InstaDeepAI/NTv3_100M_pre` weights. It emits signed ALT-minus-REF logits for
both reference orientations, their mean, and a fixed-length gene-aware context
when the complete GENCODE gene fits with the requested variant margin. Rows
whose ALT allele is not A, C, G, or T are filtered and counted in the run
manifest.

The default is an 8,192-nt centered window. Omit `--genes` for centered-only
scoring; supplying a GTF additionally enables the gene-aware columns. Window
lengths can be changed with `--window-length` and must be divisible by 128.

Place the weights and gated custom-code snapshots under one directory and set
`NTV3_MODEL_ROOT`; the scorer validates every pinned file hash before importing
the model code. The checked-in manifest already pins and hashes the available
weights. Access to the gated code repository is still required once: check out
the manifest's code revision and replace its three `GATED_SNAPSHOT_REQUIRED`
values with SHA-256 hashes of the local files.

```bash
export NTV3_MODEL_ROOT=/path/to/model_weights
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run --extra ntv3 score-dna-variants \
  --variants collated_variants.tsv.gz \
  --reference hg38.fa.gz \
  --output-dir scoring/ntv3_100m_pre \
  --num-shards 32 --shard-index 0

uv run collate-dna-variant-scores \
  --variants collated_variants.tsv.gz \
  --input-dir scoring/ntv3_100m_pre \
  --num-shards 32 \
  --output scoring/ntv3_100m_pre.tsv.gz
```

The raw `score` is model preference for ALT over REF, not a calibrated
pathogenicity probability. The NTv3 license limits model use and derived
outputs to non-commercial use.

## Orthrus transcript RNA scoring

`score-orthrus-variants` scores mature GENCODE transcripts with the local
`antichronology/orthrus-mlm-6-track` checkpoint. It constructs the model's four
nucleotide channels plus CDS-codon-start and exon-end splice channels directly
from a transcript FASTA and GFF3. The genomic alleles are complemented on
minus-strand transcripts. At the requested one-based `cdna_position`, only the
four nucleotide channels are masked; the CDS and splice values are retained.

Orthrus's Mamba dependencies are incompatible with the main Python 3.13
environment. Its standalone uv project pins Python 3.10 and supplies the exact
runtime Torch as a build dependency for both CUDA extensions. With a CUDA
toolkit (`nvcc`) available, create the complete environment in one command:

```bash
uv sync --project environments/orthrus --frozen
export PYTHONPATH="$PWD/src"
```

The scorer is offline-only and validates the checkpoint Git revision and every
pinned file hash before model loading. It uses float32, defaults to CUDA and a
batch size of 32, and buckets causal transcript-prefix requests by length. It
does not truncate or window long transcripts.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PWD/src"
uv run --project environments/orthrus --frozen \
  python -m vep_comparisons.score_orthrus_variants_cli \
  --variants collated_variants.tsv.gz \
  --transcripts gencode.v50.transcripts.fa.gz \
  --gff3 gencode.v50.annotation.gff3.gz \
  --checkpoint /path/to/orthrus-mlm-6-track \
  --output-dir scoring/orthrus_mlm_6_track \
  --num-shards 32 --shard-index 0

uv run --project environments/orthrus --frozen \
  python -m vep_comparisons.collate_orthrus_scores_cli \
  --variants collated_variants.tsv.gz \
  --transcripts gencode.v50.transcripts.fa.gz \
  --gff3 gencode.v50.annotation.gff3.gz \
  --checkpoint /path/to/orthrus-mlm-6-track \
  --input-dir scoring/orthrus_mlm_6_track \
  --num-shards 32 \
  --output scoring/orthrus_mlm_6_track.tsv.gz
```

Only rows with an integer cDNA position and single A/C/G/T REF and ALT are
published. Metadata records excluded-row aggregates. Output includes genomic
and transcript-oriented alleles, REF and ALT log-probabilities, and
`score = log P(ALT) - log P(REF)`. The Slurm array helper is
`scripts/score_collated_variants_orthrus.sh`; collation remains a separate,
explicit step after the array completes.

## RiNALMo-giga transcript RNA scoring

`score-rinalmo-variants` applies masked-marginal scoring with only nucleotide
tokens from the official 650M RiNALMo `giga` checkpoint. GENCODE transcript
FASTA and GFF3 files remain authoritative for sequence, strand, exon geometry,
and cDNA-to-genomic validation; CDS features and annotation channels are not
used. Genomic alleles are complemented for minus-strand transcripts.

The standalone environment pins Python 3.11, CUDA 12.4 Torch 2.6.0,
FlashAttention 2.6.3, and the verified RiNALMo source revision. Build it with:

```bash
uv sync --project environments/rinalmo --frozen
export PYTHONPATH="$PWD/src"
```

The checked-in manifest verifies the source pin, exact 22-token alphabet,
giga architecture, and `rinalmo_giga_pretrained.pt` SHA-256 before loading.
`bfloat16` is the default; `float16` and `float32` are also accepted. Standard
attention is selected automatically for CPU and float32 inference.

```bash
uv run --project environments/rinalmo --frozen \
  python -m vep_comparisons.score_rinalmo_variants_cli \
  --variants collated_variants.tsv.gz \
  --transcripts gencode.v50.transcripts.fa.gz \
  --gff3 gencode.v50.annotation.gff3.gz \
  --weights /path/to/rinalmo_giga_pretrained.pt \
  --output-dir scoring/rinalmo_giga \
  --num-shards 32 --shard-index 0 --dtype bfloat16

uv run --project environments/rinalmo --frozen \
  python -m vep_comparisons.collate_rinalmo_scores_cli \
  --variants collated_variants.tsv.gz \
  --transcripts gencode.v50.transcripts.fa.gz \
  --gff3 gencode.v50.annotation.gff3.gz \
  --weights /path/to/rinalmo_giga_pretrained.pt \
  --input-dir scoring/rinalmo_giga --num-shards 32 \
  --dtype bfloat16 --output scoring/rinalmo_giga.tsv.gz
```

Inputs longer than the 1,024-token pretraining context are cropped around the
target using a fixed deterministic policy. CLS and EOS are included only when
the crop reaches the true transcript boundary. Every row records the half-open
transcript crop and total token count. The Slurm array helper is
`scripts/score_collated_variants_rinalmo.sh`; collation is a separate step.
