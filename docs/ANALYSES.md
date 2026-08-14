# Rebuilding the analyses

Every command below runs from the repository root with the virtual environment active. Outputs land under `results/`, which is not distributed — see the [Data availability](../README.md#data-availability) section for how to obtain the corpora first.

The stages are cumulative: Stage 1 produces the cleaned CSVs and spaCy DocBins that every later stage reads.

## Prerequisites

```bash
source .venv/bin/activate
python3 scripts/generate/download_data.py   # 2024-cohort corpora from OSF
# 2026-cohort corpora: see README § Data availability
```

## Stage 1 — cleaning, parsing, descriptives

```bash
python3 scripts/prepare/clean_corpus.py            # → <Model>.cleaned.csv, with quality + style flags
python3 -m aiidiolects.parse_corpus --n-process 4  # → <Model>.spacy DocBins
python3 scripts/analyse/descriptive_stats.py       # → results/phase2_mapping/stage1_descriptives/
python3 -m aiidiolects.visualize_stage1
```

Chapter §4 descriptors: output and sentence length, type–token ratio and vocabulary size, disclaimer and AI-self-reference rates.

The Phase 1 anchoring trio (Originals / Run 5 / Run 6) has its own view:

```bash
python3 scripts/figures/visualize_phase1.py  # → results/phase1_anchoring/stage1_descriptives/figures/
```

## Stage 2 — distances between corpora

```bash
python3 -m aiidiolects.build_distance_matrix  # → results/phase2_mapping/stage2_distance_matrix/
python3 scripts/figures/render_dendrogram.py --exclude run6_lmstudio
```

A single pairwise comparison on the ten fingerprint metrics, with diagnostic plots:

```bash
python3 -m aiidiolects.compare_fingerprints \
  --original   data/phase0_textualllmap/climate/Mistral-7b.csv \
  --replicated data/phase1_anchoring/run4_lmstudio/climate/Mistral-7b.csv \
  --output     results/phase1_anchoring/run4_lmstudio_vs_originals/climate/
```

Gap versus stochastic noise floor, for the anchoring trio:

```bash
python3 scripts/figures/render_phase1_trio.py  # → results/phase1_anchoring/stage2_distance_matrix/
```

Punctuation profile, discourse markers and hedges:

```bash
python3 -m aiidiolects.discourse_features  # → results/phase2_mapping/stage2_discourse_features/
python3 scripts/figures/render_discourse.py
```

Bootstrap confidence intervals over the distance matrix — **slow, roughly 6 hours single-machine**; use `scripts/run_on_lambda.sh` for the multi-process version:

```bash
python3 scripts/analyse/bootstrap_distances.py --n-reps 500
python3 scripts/figures/render_bootstrap_ci.py  # → .../stage2_distance_matrix/bootstrap/
```

## Human reference data

Prerequisite: `data/human_baseline/` built. The `.cleaned.csv` and `.spacy` siblings come from the Stage 1 scripts, whose input discovery already picks up `data/human_baseline/` — no extra flag needed there.

```bash
python3 scripts/prepare/download_human_baseline.py  # Webis-CMV-20 (Zenodo) + Reuters-50/C50 (UCI)
python3 scripts/prepare/build_human_baseline.py     # rebuilds the working subsets from FILTER_CONFIG

python3 scripts/analyse/within_author_baseline.py   # Reuters-50 within/cross-author Burrows' Δ
python3 scripts/figures/render_within_author.py --jpg

python3 scripts/analyse/descriptive_stats.py --include-human \
  --out-dir results/phase2_mapping/stage1_descriptives_with_human
python3 -m aiidiolects.visualize_stage1 --include-human
python3 -m aiidiolects.build_distance_matrix --include-human \
  --out-dir results/phase2_mapping/stage2_distance_matrix_with_human
python3 scripts/figures/render_dendrogram.py --include-human --exclude run6_lmstudio
python3 scripts/figures/render_human_distance.py --all --jpg
python3 -m aiidiolects.discourse_features --include-human \
  --out-dir results/phase2_mapping/stage2_discourse_features_with_human
python3 scripts/figures/render_discourse.py --include-human
```

These produce the numbers reported in [`../supplement/human_baseline.md`](../supplement/human_baseline.md).

## Stage 3 — syntax, within-corpus variation, keyness

Syntactic features load the existing DocBins, so no re-parse is needed:

```bash
python3 scripts/analyse/syntactic_features.py  # → results/phase2_mapping/stage3_syntactic/
python3 scripts/figures/render_syntactic.py --jpg
python3 scripts/figures/render_dendrogram.py --include-human \
  --in-dir  results/phase2_mapping/stage3_syntactic/matrices \
  --out-dir results/phase2_mapping/stage3_syntactic/figures \
  --exclude run6_lmstudio
```

Dependency length, mean dependency distance and mean hierarchical distance, tree depth, passive rate, subordination, noun-chunk length, POS n-gram JSD, and dependency-label JSD. Methodology follows Juzek, Krielke & Teich (UDW 2020).

```bash
python3 scripts/analyse/within_corpus_variation.py  # → .../stage3_within_corpus_variation/
python3 scripts/figures/render_within_corpus.py --jpg

python3 -m aiidiolects.keyness                      # → .../stage3_keyness/
python3 scripts/figures/render_keyness.py --jpg
```

Keyness is baseline-free: for each corpus, which words it over-uses relative to a reference pool of the others, by log-likelihood G² with Hardie's log-ratio effect size and a split-half consensus filter.

## Stage 4 — GPU stages

Perplexity scoring and the embedding map need a GPU (or a lot of RAM and patience). One command handles dependencies, data, and all three jobs on a fresh cloud instance:

```bash
bash scripts/run_on_lambda.sh
#   --ref-models gpt2-large          default; comma-separate for multiple yardsticks
#   --n-workers $(nproc)             default; for the multi-process bootstrap
#   --no-embedding-map | --no-bootstrap | --skip-data
```

Or individually:

```bash
python3 scripts/analyse/perplexity_scoring.py --ref-models gpt2-large
python3 scripts/figures/render_perplexity.py --jpg

python3 scripts/analyse/embedding_map.py  # intfloat/e5-large-v2 + UMAP
python3 scripts/figures/render_embedding_map.py --jpg
```

Full instance setup is in [`lambda_setup.md`](lambda_setup.md).

## JPG snapshots

Three render scripts emit PNG only (`visualize_stage1`, `render_dendrogram`, `render_discourse`). To produce JPG companions:

```bash
python3 -c "
from pathlib import Path
from PIL import Image
dirs = [
    'results/phase2_mapping/stage1_descriptives_with_human/figures',
    'results/phase2_mapping/stage2_distance_matrix_with_human/figures',
    'results/phase2_mapping/stage2_discourse_features_with_human/figures',
]
for d in dirs:
    for p in sorted(Path(d).glob('*.png')):
        Image.open(p).convert('RGB').save(
            p.with_suffix('.jpg'), quality=88, optimize=True, progressive=True)
"
```

## Analyses carried out in R

The stylometric PCA (chapter §5) and the contraction study (§6) were run in R. That code is a planned addition. 

> **AI-assistance** -- this document was written with substantial AI assistance (Claude Opus 5) and the final version was approved by the authors.
