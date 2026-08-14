# Model roster and generation parameters

Documented here is everything needed to reproduce the generation side of the study: which models, served how, with which parameters.

## Cohort labels are generation years, not release years

The two cohorts are named for **when the texts were generated** (not: when each model was released). Per-model release dates span 2023–2026 and are reported in the chapter prose; figures label by cohort year, with the convention stated in the caption.

Of the six models in the 2026 cohort, only GPT-5.4 Mini was released in 2026. Mistral-Nemo dates from July 2024; Gemini 3 Flash, Claude Haiku 4.5, OLMo 3 and Qwen 3 from mid to late 2025. The Mistral-7B anchor uses December 2023 weights and appears in both cohorts — as a 2024-cohort member via Improta et al., and as a 2026 re-run that bridges the two.

## 2024 cohort

The six English streams of the TextualLLMap deposit (`osf.io/uwv59`), fetched by [`../scripts/generate/download_data.py`](../scripts/generate/download_data.py) on 2026-03-24. 1,000 texts per model per topic; 24,000 texts in total.

| File | `model` column in the data | Documented in Improta et al.? | Exact identifier from their §2 |
|---|---|---|---|
| `gpt-3.5.csv` | `gpt-3.5` | yes | `gpt-3.5-turbo-0125` |
| `Haiku.csv` | `Haiku` / `claude-haiku` | yes | `haiku20240307` → **Claude 3 Haiku** |
| `Llama-3-8B.csv` | `Llama-3-8B` | yes | `Llama-3-8B-Instruct-Q5_K_M` |
| `Mistral-7b.csv` | `Mistral-7b` | yes | `mistral-7b-instruct-v0.2.Q8_0` |
| `GPT-4o.csv` | `gpt-4o-2024-08-06` | **no** | — |
| `Llama-3.1-70B.csv` | `Meta-Llama-3-70B-Q4` | **no** | — |

Two things this table is meant to make visible, both discussed in [`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md):

- **The Haiku is Claude 3, not Claude 3.5.** The chapter and the figure labels say 3.5 and need to be updated. 
- **The 70B's filename and its own metadata disagree** (`Llama-3.1-70B` vs `Meta-Llama-3-70B-Q4`). Unresolved with the depositors. Pipeline labels derive from filenames, so "3.1" propagates throughout.

The deposit also contains five Italian streams (GPT-3.5, GPT-4o, Haiku, Llama-3.1-70B and LLaMAntino-2, each suffixed `(ITA)`), which this study does not use.

## 2026 cohort

Generated for this study with [`../scripts/generate/generate_texts.py`](../scripts/generate/generate_texts.py). All three closed-API providers expose OpenAI-compatible endpoints, so a single code path covers every model — only `--api-base`, `--api-key` and `--model-id` change.

### Local anchor

| Model | Serving | Notes |
|---|---|---|
| Mistral-7B-Instruct-v0.2 (Q8_0) | LM Studio 0.4.11, local | `--top-p 0.95`, `--no-system-role`. Engine continuity with Phase 1 is deliberate — see [`PHASE1_RESULTS.md`](PHASE1_RESULTS.md). |

### Closed APIs

| Model | Provider | `--model-id` | `--api-base` |
|---|---|---|---|
| GPT-5.4 Mini | OpenAI | `gpt-5.4-mini-2026-03-17` | `https://api.openai.com/v1` |
| Gemini 3 Flash | Google | `gemini-3-flash-preview` | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Claude Haiku 4.5 | Anthropic | `claude-haiku-4-5-20251001` | `https://api.anthropic.com/v1/` |

### Open weights

| Model | Lab | Serving |
|---|---|---|
| Qwen 3-14B-Instruct | Alibaba | vLLM on an A100 80 GB, FP16. Generated with `--no-thinking`. |
| OLMo 3-Instruct 7B | AI2 | vLLM on an A100 80 GB, FP16 |
| Mistral-Nemo-Instruct-2407 (12B) | Mistral AI + NVIDIA | vLLM on an A100 80 GB, FP16 |

All three open-weight models were served on the same instance at FP16 for consistency; at 16 GB of consumer VRAM, Qwen 14B would have required quantisation, which would have introduced a second source of variation on top of the model itself.

## Generation parameters

| Parameter | Value | Provenance |
|---|---|---|
| Topics | `climate`, `global_warming`, `math_anxiety`, `misinfo_health` | Improta et al. §2.1 |
| Prompts | verbatim; carried in `generate_texts.py` | Improta et al. §2.1 — checked string-for-string |
| Texts per model per topic | 1000 (Phase 1 anchor runs: 200) | Improta et al. — their sample size, matched |
| System prompt | "Below is an instruction…" | Improta et al. §2.1 — their *local-model* pre-prompt, applied here to every model |
| Temperature | 0.5 | Improta et al. §2.1 — applied here to every model, including the APIs they excepted |
| `top_p` | 0.95 | reconstructed — the LM Studio / llama.cpp default |
| top-k, repetition / frequency / presence penalties | unset | engine defaults, as in the source run |
| `max_tokens` | 2048 | this study |

Improta et al. report only model, quantisation and temperature. Because "all non-API models were employed locally through LM Studio … as .GGUF files" (their §2), the sampling parameters they left unstated took LM Studio's defaults — so this study reconstructs those: `top_p` is set to the LM Studio / llama.cpp default of 0.95, and top-k and the penalties are left unset so the serving engine's defaults apply.

This study also departs from their protocol in two places, deliberately: they used a different pre-prompt for ChatGPT ("You are a helpful assistant") and exempted it from the 0.5 temperature, whereas this study applies one system prompt and one temperature to every model so that differences between corpora are attributable to the models. See [`METHODOLOGY.md`](METHODOLOGY.md#generation-protocol) for the assumption the reconstruction rests on and the run that tested it.

Observed output length is short relative to the cap — a ~500-token median, with real cap hits concentrated in GPT-5.4 Mini (4–16% depending on topic) and below 0.5% for every other model. The truncation detector in `clean_corpus.py` flags a text when word count ≥ 0.80 × cap and there is no terminal punctuation.

## Selection constraints

Choices that shape what the roster can and cannot show, stated so they can be weighed:

- **Non-reasoning variants throughout.** Long chain-of-thought output makes text-level fingerprinting noisy and incomparable against non-thinking peers. A reasoning-idiolect finding would need a controlled sub-study rather than a mixed roster.
- **Open weights capped at roughly 15B.** Above that, family-internal variation is increasingly driven by multimodality and general capability rather than by language behaviour as such.
- **Cost-efficient tiers for the closed APIs.** Mini/Flash/Haiku are the tiers most high-volume production systems actually run, and Flash is the Gemini app default — so these are the consumer-facing idiolects rather than a lab's showcase model.
- **The Mistral-7B anchor carries the methodological link** to Improta et al.; other older small models were not re-run.

## Provider-specific handling in `generate_texts.py`

- `--no-system-role` merges the system prompt into the user turn. Required for LM Studio 0.4.11, which enforces the Mistral Jinja template strictly and offers no system slot.
- `--no-thinking` disables reasoning output for Qwen 3 served via vLLM.
- GPT-5 and o-series models are special-cased to use `max_completion_tokens` instead of `max_tokens`.
- `--delay` throttles request rate for hosted APIs.
- Generation resumes by row count, so an interrupted run continues rather than restarting.

> **AI-assistance** -- this document was written with substantial AI assistance (Claude Opus 5) and the final version was approved by the authors.
