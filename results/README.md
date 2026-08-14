# `results/` — analysis outputs

Nothing in this directory is tracked by git. Every file here is regenerated
from `data/` by the analysis scripts; [`docs/ANALYSES.md`](../docs/ANALYSES.md)
gives the command for each stage in order.

## What ends up here

| Path | Produced by | Chapter section |
|---|---|---|
| `phase2_mapping/stage1_descriptives/` | `scripts/analyse/descriptive_stats.py` → `python3 -m aiidiolects.visualize_stage1` | §4 — length, sentence length, TTR, vocabulary, openers |
| `phase2_mapping/stage2_distance_matrix/` | `python3 -m aiidiolects.build_distance_matrix` → `scripts/figures/render_dendrogram.py` | §4 — distance matrices, dendrograms, MDS |
| `phase2_mapping/stage2_distance_matrix/bootstrap/` | `scripts/analyse/bootstrap_distances.py` → `render_bootstrap_ci.py` | §4 — confidence intervals |
| `phase2_mapping/stage2_discourse_features/` | `python3 -m aiidiolects.discourse_features` → `render_discourse.py` | §4 — markers and hedges |
| `phase2_mapping/stage2_within_author_baseline/` | `scripts/analyse/within_author_baseline.py` → `render_within_author.py` | §4(iv) — human reference comparison |
| `phase2_mapping/stage3_keyness/` | `python3 -m aiidiolects.keyness` → `render_keyness.py` | §4 — log-likelihood keyness |
| `phase2_mapping/stage3_syntactic/` | `scripts/analyse/syntactic_features.py` → `render_syntactic.py` | §4 — dependency length, MDD/MHD, POS n-grams |
| `phase2_mapping/stage3_within_corpus_variation/` | `scripts/analyse/within_corpus_variation.py` → `render_within_corpus.py` | §4 — within-corpus heterogeneity |
| `phase2_mapping/stage4_perplexity/` | `scripts/analyse/perplexity_scoring.py` → `render_perplexity.py` | §4 — perplexity under a fixed yardstick LM |
| `phase2_mapping/stage4_embedding_map/` | `scripts/analyse/embedding_map.py` → `render_embedding_map.py` | §4 — sentence-embedding map |
| `phase1_anchoring/` | the same scripts with `--base`/`--out-dir` pointed at the Phase 1 data | [`docs/PHASE1_RESULTS.md`](../docs/PHASE1_RESULTS.md) |
| `chapter_figures/` | `scripts/figures/regen_figures.py` | the published §4 figures |

Stage 4 is GPU-oriented; `scripts/run_on_lambda.sh` drives it end to end on a
fresh cloud instance.

> **AI-assistance** -- this document was written with substantial AI assistance (Claude Opus 5) and the final version was approved by the authors.
