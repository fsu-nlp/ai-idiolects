# Known issues in this release

**This release reproduces [arXiv:2608.06589v1](https://arxiv.org/abs/2608.06589) as published.** The code and labels are the ones that produced the reported results. Where we have since found a description or labelling error, it is recorded here rather than silently corrected, so that this repository and the preprint agree. Corrections will land in a later release alongside the manuscript revision.

**No reported number is affected by anything below.** These are naming and attribution issues. Every corpus is real and was analysed as described.

---

## 1. The 2024-cohort Anthropic model is Claude 3 Haiku, not Claude 3.5 Haiku

Improta et al. identify their model as `haiku20240307` — that is `claude-3-haiku-20240307`, **Claude 3 Haiku** (March 2024). Claude 3.5 Haiku was released later in 2024 and cannot be in this data. Their deposited files label the model only as `Haiku` / `claude-haiku`, with no version.

The preprint, and the figure tick labels produced by `src/aiidiolects/visualize_stage1.py` and `scripts/figures/regen_figures.py`, say "Claude 3.5 Haiku". **This is wrong; read it as Claude 3 Haiku.**

Consequence for interpretation: the within-family diachronic pair is **Claude 3 → Claude Haiku 4.5**, a wider generational gap than the published label implies.

## 2. The 70B Llama's version is unresolved

The deposited file is named `Llama-3.1-70B.csv`, but the `model` column inside every row reads `Meta-Llama-3-70B-Q4` — the Hugging Face identifier for **Llama 3 70B** (April 2024), not Llama 3.1 70B (July 2024). It also records Q4 quantisation, which the source paper does not document for any model.

The discrepancy originates in the source deposit. This project inherited the filename, because every label in the pipeline derives from the filename rather than the column (`descriptive_stats.py`). We have not been able to determine which is correct; it is an open question with the depositors.

**Treat "Llama-3.1-70B" throughout this repository as "the 70B Llama stream in the TextualLLMap deposit", not as a verified version claim.**

## 3. Cohort year labels differ between two scripts

`src/aiidiolects/visualize_stage1.py` labels the earlier cohort `(2023)`; `scripts/figures/regen_figures.py` overrides those labels to `(2024)` and produced the figures used in the chapter.

The `(2024)` labels are the intended convention: **cohort labels denote the year the texts were generated, not the year each model was released.** Per-model release years span 2023–2026 and belong in prose. Figures rendered directly from `src/aiidiolects/visualize_stage1.py` will therefore disagree with the published figures. Use `scripts/figures/regen_figures.py` to reproduce them.

The internal era token in `src/aiidiolects/visualize_stage1.py` is still the string `"2023"`, and the render scripts key their legends and colours off it. This is a legacy internal identifier, not a claim about the data.

## 4. Attribution of the 2024 cohort to Improta et al.

See [README § Data provenance](README.md#data-provenance). In short: the TextualLLMap **paper** documents five models and 28,000 English + Italian texts, and its §3 states that each topic folder contains seven per-model files; the TextualLLMap **deposit** contains eleven model–language streams per topic, including GPT-4o and the 70B Llama, which the paper does not describe. This study uses the six English streams, totalling 24,000 texts. The preprint attributes all six to "Improta et al. (2024)" without drawing that distinction.

No generation parameters are published anywhere for the two undocumented streams.

## 5. Upstream duplicate texts in `misinfo_health`

Three deposited files — `Haiku.csv`, `Haiku(ITA).csv`, `GPT-3.5(ITA).csv` — contain duplicated texts carrying divergent EmoAtlas scores, so the raw row count exceeds the number of unique texts. `scripts/prepare/clean_corpus.py` deduplicates, keeping the first occurrence. If you use the upstream `zscores.*` / `fmnt.*` columns rather than the text, be aware that the retained row is an arbitrary one of two analysis passes.

## 6. `scripts/generate/download_data.py` keeps only `.csv` archive entries

Any non-CSV file in the OSF archives is skipped on extraction. The script also skips a topic directory that already contains CSVs, so re-running it will not repair a partial download — delete the directory first.

> **AI-assistance** -- this document was written with substantial AI assistance (Claude Opus 5) and the final version was approved by the authors.
