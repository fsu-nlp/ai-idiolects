# Running the pipeline on a Lambda Cloud GPU

The pipeline itself is CPU-only Python; it only calls an external OpenAI-compatible endpoint. For open-weight models that don't fit on a desktop GPU, serve them from a Lambda Cloud GPU instance and point `scripts/generate/generate_texts.py` at it. The study's three open-weight corpora — Qwen 3-14B, OLMo 3-7B and Mistral-Nemo-12B — were generated this way, all at FP16 on one A100 so that quantisation would not vary between them.

## Prerequisites on Lambda

- NVIDIA GPU instance (A10 / A100 / H100 depending on model size).
- Lambda's AI env option has everything needed for this work. 

## Option A — ollama (fast to set up, good for 7B–70B)

```bash
# On the Lambda instance
git clone https://github.com/fsu-nlp/ai-idiolects.git
cd ai-idiolects

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # puts aiidiolects on the import path

# Install + start ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &   # listens on 127.0.0.1:11434

# Pull the target model (example: Mistral-7B Q8_0)
ollama pull mistral:7b-instruct-v0.2-q8_0

# Fetch originals and run replication
python3 scripts/generate/download_data.py
for topic in climate global_warming math_anxiety misinfo_health; do
  python3 scripts/generate/generate_texts.py \
    --model mistral-7b \
    --model-id "mistral:7b-instruct-v0.2-q8_0" \
    --topic $topic \
    --n 1000 \
    --api-base http://localhost:11434/v1
done
```

**Notes on ollama**: uses GGUF quantizations, defaults to `top_p=0.9`, chat-template handling evolved between 2024 and 2026 (see `PHASE1_RESULTS.md` for why this matters for fingerprint comparison). Use `--no-system-role` in `scripts/generate/generate_texts.py` when testing template sensitivity.

## Option B — vLLM (higher throughput, better for batch runs)

```bash
pip install vllm
python3 -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --port 8000 &

python3 scripts/generate/generate_texts.py \
  --model mistral-7b \
  --model-id "mistralai/Mistral-7B-Instruct-v0.2" \
  --topic climate --n 1000 \
  --api-base http://localhost:8000/v1
```

vLLM typically outperforms ollama on sustained batch workloads and supports the full HuggingFace model hub. For comparability, match the study's sampling settings — `temperature=0.5` and `--top-p 0.95` — rather than whichever defaults the engine ships; note that engine defaults differ (ollama uses `top_p=0.9`), which is why this study sets it explicitly. See [`METHODOLOGY.md`](METHODOLOGY.md#generation-protocol).

## Option C — remote APIs (OpenAI, together.ai, fireworks, etc.)

For models too large to self-host, or when using a managed endpoint:

```bash
export OPENAI_API_KEY=<your-key>          # or the provider's own key variable
python3 scripts/generate/generate_texts.py \
  --model gpt-5.4-mini \
  --model-id "gpt-5.4-mini-2026-03-17" \
  --topic climate --n 1000 \
  --api-base https://api.openai.com/v1
```

Google and Anthropic both expose OpenAI-compatible endpoints, so the same
command covers them with only `--model-id` and `--api-base` changing — that is
how the study's Gemini and Claude corpora were generated. No provider-specific
SDK is a dependency. The exact identifiers and endpoints are in
[`MODELS.md`](MODELS.md#closed-apis).


## After the run

Copy `data/phase2_mapping/<model>/` (or `data/phase1_anchoring/run*/` if you're running Phase 1 reproductions) back off the instance — or run `python3 -m aiidiolects.compare_fingerprints` on the instance and copy `results/` off. Spin the instance down — nothing about this pipeline needs to persist on the GPU box.

---

## Stage 4 — analysis jobs that need a GPU / a bigger box (turnkey)

The Stage-1/2/3 *analysis* code is CPU-only and runs on a 16 GB machine. A
small queue of analyses needs a Lambda instance — either a GPU (a reference
LM over the whole corpus; a transformer parse) or just >16 GB RAM (the
multi-process bootstrap). **One script does the whole thing:**

```bash
# Fresh Lambda Cloud GPU instance (A100 ideal; A10 fine), Ubuntu 22.04:
git clone https://github.com/fsu-nlp/ai-idiolects.git && cd ai-idiolects
bash scripts/run_on_lambda.sh         # ≈1–1.5 h on a GPU box
#   …then copy results/ back off the instance, and SPIN THE INSTANCE DOWN.
```

`scripts/run_on_lambda.sh` does, in order:

**(0) Dependencies** — `apt install p7zip-full`; a fresh `.venv`;
`pip install -r requirements.txt -r requirements_stage4.txt`;
`pip install -e .`; `spacy download en_core_web_lg`.

**(1) Data** — `scripts/generate/download_data.py` for the 2024 cohort;
`scripts/prepare/download_human_baseline.py` then `scripts/prepare/build_human_baseline.py` for
the human reference sets (the `FILTER_CONFIG` is locked, so this is
non-interactive); then `scripts/prepare/clean_corpus.py` and
`python3 -m aiidiolects.parse_corpus --n-process N`. Skip the whole step with
`--skip-data` if `data/` is already populated.

The 2026-cohort corpora are not distributed with this repository — see
[README § Data availability](../README.md#data-availability). If you hold them
as `phase2_<model>.7z` bundles, drop them in `data/phase2_mapping/archives/` and
the script unpacks them here; if that directory is absent the step is skipped
rather than failing.

**(2) Stage-4 analyses** — perplexity scoring
(`scripts/analyse/perplexity_scoring.py` → `scripts/figures/render_perplexity.py`);
the with-human bootstrap (`scripts/analyse/bootstrap_distances.py --include-human
--n-workers N`, memory-hungry on a big box; `scripts/figures/render_human_distance.py
--all` draws the 95 %-CI whiskers); and the sentence-embedding map
(`scripts/analyse/embedding_map.py` → `scripts/figures/render_embedding_map.py`)
unless `--no-embedding-map`.

**(3) Reporting** — prints the headline numbers each script echoes.

Flags: `--n-workers N` (default `nproc`), `--ref-models a,b` (default
`gpt2-large` — ungated; the Mistral/Llama *base* models are gated, so set
`HF_TOKEN` and accept the licence first), `--no-embedding-map`,
`--no-bootstrap`, `--skip-data`.

**Outputs** land in `results/phase2_mapping/stage4_perplexity/`,
`results/phase2_mapping/stage2_distance_matrix_with_human/bootstrap/`, and
`results/phase2_mapping/stage4_embedding_map/`. Copy them off the instance
before spinning it down; `results/` is not tracked by git.


> **AI-assistance** -- this document was written with substantial AI assistance (Claude Opus 4.7) and the final version was approved by the authors.
