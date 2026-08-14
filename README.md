# ai-idiolects

Companion code for **Rudnicka & Juzek, "Beyond 'AI Language': The case for the idiolectal nature of LLM output"** ([arXiv:2608.06589](https://arxiv.org/abs/2608.06589)), a chapter in the De Gruyter *Digital Linguistics* series.

The chapter argues that LLM output is better understood as a set of distinct, model-specific linguistic signatures — idiolects — than as a single "AI language" super-variety. It compares two cohorts of six models each, generated on identical prompts: a **2024 cohort** taken from Improta, Veltri & Stella (2024), and a **2026 cohort** generated for this study.

This repository holds the Python generation and analysis pipeline behind that comparison.

## Layout

```
src/aiidiolects/     the library — modules the rest of the code imports
scripts/generate/    text generation and corpus fetching
scripts/prepare/     cleaning, parsing, human reference construction
scripts/analyse/     the metric and statistics stages
scripts/figures/     every plot in the chapter
docs/                methodology, model roster, rebuild guide
supplement/          the human reference data supplement
data/  results/       created at runtime; nothing here is tracked
```

The split is by *how code is used*: anything another module
imports lives in `src/aiidiolects/`, anything you only ever run lives in
`scripts/`. Library modules are runnable too, as `python3 -m aiidiolects.<name>`.

### The library — `src/aiidiolects/`

| Module | What it is |
|------|------------|
| `compare_fingerprints.py` | The metric library: length distributions with Cohen's *d* and KS, TTR, MATTR, hapax ratio, Yule's K, Jaccard top-100, bigram/trigram JSD, Burrows' Δ. |
| `build_distance_matrix.py` | Corpus discovery, the topic and quality-flag constants, and the cross-corpus distance matrices. |
| `visualize_stage1.py` | Shared plot conventions — model order, display labels, cohort colours — and the Stage 1 descriptive figures: output length, sentence length, type–token ratio, vocabulary size, opener style (chapter §4). |
| `parse_corpus.py` | spaCy tagging and dependency parsing (`en_core_web_lg`) into DocBins. |
| `keyness.py`, `discourse_features.py` | Log-likelihood keyness; discourse markers and hedges. |
| `paths.py` | `REPO_ROOT`, `DATA_DIR`, `RESULTS_DIR`. |

### The scripts — `scripts/`

| Script | What it is |
|------|------------|
| `generate/generate_texts.py` | Text generation against any OpenAI-compatible endpoint. Carries the four topic prompts verbatim from Improta et al. §2.1. |
| `generate/download_data.py` | Fetches the 2024-cohort corpora (TextualLLMap) from OSF. |
| `prepare/clean_corpus.py` | The regex/heuristic pre-parser: language and length filtering, truncation detection, repetition-loop and code-contamination flags, disclaimer/self-reference/refusal flags. This is the cleaning script referenced in §3 of the chapter. |
| `prepare/download_human_baseline.py`, `prepare/build_human_baseline.py` | The human reference sets (see below). |
| `analyse/descriptive_stats.py` | Per-text descriptive measures feeding the Stage 1 figures. |
| `analyse/bootstrap_distances.py` | Bootstrap confidence intervals over the distance matrix. |
| `analyse/syntactic_features.py`, `analyse/within_corpus_variation.py` | Dependency-length, MDD/MHD and POS-n-gram measures; within-corpus heterogeneity. |
| `analyse/within_author_baseline.py` | The within- versus between-author comparison behind chapter §4(iv). |
| `analyse/perplexity_scoring.py`, `analyse/embedding_map.py` | Perplexity under a fixed yardstick LM; sentence-embedding map with UMAP. GPU-oriented. |
| `figures/render_*.py`, `figures/visualize_phase1.py` | Dendrograms, MDS biplots, CI whiskers, keyness, syntax, discourse, perplexity and embedding plots. |
| `figures/regen_figures.py` | Produces the §4 chapter figures from the Stage 1 results, with the cohort-year tick labels used in the published figures. |
| `run_on_lambda.sh` | One-command driver for the GPU stages on a fresh cloud instance. |

Documentation: [Methodology](docs/METHODOLOGY.md), [model roster and generation parameters](docs/MODELS.md), [how to rebuild every analysis](docs/ANALYSES.md), [Phase 1 replication results](docs/PHASE1_RESULTS.md), [cloud GPU setup](docs/lambda_setup.md). Naming and attribution errors found after publication are recorded in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) rather than silently corrected.

The stylometric and contraction analyses behind chapter §5 and §6 were carried out in R and are a planned addition; see [`TODO.md`](TODO.md).

The pipeline is **inference-engine-agnostic**: no script touches a GPU directly. Generation goes through an OpenAI-compatible endpoint, so ollama, vLLM, llama.cpp, LM Studio and hosted APIs all work behind the same code path.

## Install

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # puts aiidiolects on the import path
python3 -m spacy download en_core_web_lg

# GPU stages (perplexity, embedding map) additionally need:
pip install -r requirements_stage4.txt
```

`pip install -e .` is what lets the scripts under `scripts/` import the library
under `src/`. Without it they will not run.

## Quickstart

```bash
# 1. Fetch the 2024-cohort corpora from OSF
python3 scripts/generate/download_data.py

# 2. Serve a model locally — example: ollama + Mistral-7B
ollama pull mistral:7b-instruct-v0.2-q8_0
ollama serve                      # listens on http://localhost:11434

# 3. Generate
python3 scripts/generate/generate_texts.py \
  --model mistral-7b \
  --model-id "mistral:7b-instruct-v0.2-q8_0" \
  --topic climate --n 1000 \
  --api-base http://localhost:11434/v1

# 4. Compare two corpora on the ten fingerprint metrics
python3 -m aiidiolects.compare_fingerprints \
  --original   data/phase0_textualllmap/climate/Mistral-7b.csv \
  --replicated data/phase1_anchoring/run1/climate/Mistral-7b.csv \
  --output     results/climate/
```

The four topics are `climate`, `global_warming`, `math_anxiety`, `misinfo_health`. The topic prompts are verbatim from Improta et al. §2.1; temperature 0.5 and the 1,000-texts-per-topic sample size are theirs too. The remaining sampling settings are *reconstructed* rather than inherited — their paper does not state them, so this study adopts the LM Studio defaults their run would have used. That reconstruction, and the two places where this study deliberately departs from their protocol, are set out in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md#generation-protocol). For the exact model identifiers, endpoints and parameters, see [`docs/MODELS.md`](docs/MODELS.md); to rebuild any analysis, see [`docs/ANALYSES.md`](docs/ANALYSES.md).

No API keys are stored here. `scripts/generate/generate_texts.py` takes a key via `--api-key` or the `OPENAI_API_KEY` environment variable; local servers usually need neither.

## Data provenance

The 2024 cohort comes from the **TextualLLMap deposit** at [`osf.io/uwv59`](https://osf.io/uwv59), retrieved 2026-03-24 by `scripts/generate/download_data.py` from four per-topic archives.

An important distinction the chapter does not draw: the deposit contains **more than the accompanying paper describes**. Improta, Veltri & Stella (2024) document five models and 28,000 English and Italian texts, and their §3 states that each topic folder holds **seven** per-model files. The deposit as retrieved holds **eleven** model–language streams per topic — six English, five Italian. Two of the English streams, **GPT-4o** and a **70B Llama**, appear nowhere in their paper, and no generation parameters are published for them anywhere.

This study uses the six English streams: **24,000 texts**, not the 28,000 the paper reports for its own (differently composed) roster.

`docs/MODELS.md` carries a table mapping each file to the `model` identifier recorded inside it, so that two inherited naming discrepancies stay visible rather than silently propagating. Both are described in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Data availability

**The corpora are not distributed in this repository.** They are being deposited separately, and this section will carry the DOI and download link once that deposit is live.

- **2024 cohort** — from the Improta, Veltri & Stella (2024) OSF deposit, fetched by `scripts/generate/download_data.py`.
- **2026 cohort** — generated for this study. Deposit pending; see [`TODO.md`](TODO.md).
- **Human reference sets** — not redistributable as text. `scripts/prepare/download_human_baseline.py` fetches Webis-CMV-20 (Zenodo) and Reuters-50/C50 (UCI) from source, and `scripts/prepare/build_human_baseline.py` rebuilds the exact working subsets from the filter configuration recorded in that script. No per-text identifiers or author manifests are shipped here; they are written locally at build time.

## Human reference data

Chapter §4, item (iv) reports a comparison against non-prompt-matched human reference data and points readers to this repository for the analysis itself. It is in [`supplement/human_baseline.md`](supplement/human_baseline.md), with the scripts that produce it listed above. The headline result: within-speaker variation in a given domain exceeds within-model variation, and between-speaker variation exceeds between-model variation.

The reference sets are deliberately supplementary rather than central — neither is prompt-matched to the generation task, which is stated in the chapter and detailed in the supplement's caveats.

## Citation

```bibtex
@article{rudnicka2026beyond,
  title   = {Beyond "AI Language": The case for the idiolectal nature of LLM output},
  author  = {Rudnicka, Karolina and Juzek, Thomas Stephan},
  year    = {2026},
  journal = {arXiv preprint arXiv:2608.06589},
  url     = {https://arxiv.org/abs/2608.06589}
}
```

Please also cite the source of the 2024-cohort corpora:

> Improta, R., Veltri, G., Stella, M. (2024). TextualLLMap: A dataset of 28,000 Large Language Models' writings designed to expose their biases on societal issues. OSF Preprint: `10.31234/osf.io/xwpe8`.

## License

MIT — see [LICENSE](LICENSE).

## Authors

Karolina Rudnicka (University of Gdańsk) and Thomas Stephan Juzek (Florida State University).

## AI-assistance disclosure

This project was developed with **substantial AI assistance**, disclosed prominently here so readers, reviewers, and collaborators can weigh the role of AI in its outputs.

- **Tool:** Anthropic [Claude Code](https://claude.com/claude-code) CLI, at "max" reasoning effort with extended thinking enabled.
- **Models:** Claude Opus 4.7 (`claude-opus-4-7`) for the pipeline, analyses, and documentation (April–June 2026); Claude Opus 5 (`claude-opus-5`) for the preparation of this repository (August 2026). Individual files carry a disclosure block naming the model used for that file.
- **Scope:** code drafting and review, methodology brainstorming, design discussion, statistical analysis, visualisation, and documentation.
- **Boundary:** final research direction, scientific judgements, validation against the data, and the prose of the published chapter remain the responsibility of the human authors.

AI-drafted material is reviewed by the authors before commit. Authorship of this work is human; no AI system is credited as an author or contributor.
