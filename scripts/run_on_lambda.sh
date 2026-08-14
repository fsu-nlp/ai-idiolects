#!/usr/bin/env bash
# ===========================================================================
# Stage 4 — turnkey Lambda-GPU run for the AI-idiolects chapter.
#
# Fresh Lambda Cloud Ubuntu box:
#   git clone https://github.com/fsu-nlp/ai-idiolects.git && cd ai-idiolects
#   bash scripts/run_on_lambda.sh            # do it all, then print the commit recipe
#   bash scripts/run_on_lambda.sh --push     # ... and commit + push (if git is configured)
#
# What it does, in order:
#   0. deps   — apt p7zip-full; a fresh venv; pip -r requirements.txt -r requirements_stage4.txt;
#               pip install -e . (puts aiidiolects on the import path); spaCy en_core_web_lg
#   1. data   — download_data.py (OSF phase0), download_human_baseline.py + build_human_baseline.py
#               (phase 2b), extract+rename the phase2_*.7z archives, clean_corpus.py, parse_corpus.py
#               (skip the whole step with --skip-data if data/ is already populated)
#   2. analyses — perplexity_scoring.py → render_perplexity.py;  bootstrap_distances.py --include-human
#               (multi-process) → render_human_distance.py --all (now with 95%-CI whiskers);
#               embedding_map.py → render_embedding_map.py  (unless --no-embedding-map)
#   3. report — print the headline numbers (the scripts already printed them) + the
#               `git add … && git commit … && git push` recipe (or run it with --push)
#
# Then: copy results back / push, and **spin the Lambda instance down**.
# Flags:  --n-workers N | --ref-models a,b | --no-embedding-map | --no-bootstrap |
#         --skip-data | --push | -h
#
# AI-assistance disclosure: developed with substantial AI assistance from Anthropic
# Claude Opus 4.7 (claude-opus-4-7) via Claude Code at "max" reasoning effort.
# Reviewed by the human authors before commit. See the project README.
# ===========================================================================
set -euo pipefail

# --- locate the repo root (this script lives in scripts/, one level down) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- defaults / flags ---
NWORKERS="$(nproc 2>/dev/null || echo 4)"
REF_MODELS="gpt2-large"
DO_EMBEDDING=1
DO_BOOTSTRAP=1
SKIP_DATA=0
DO_PUSH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-workers) NWORKERS="$2"; shift 2 ;;
    --ref-models) REF_MODELS="$2"; shift 2 ;;
    --no-embedding-map) DO_EMBEDDING=0; shift ;;
    --no-bootstrap) DO_BOOTSTRAP=0; shift ;;
    --skip-data) SKIP_DATA=1; shift ;;
    --push) DO_PUSH=1; shift ;;
    -h|--help) grep -E '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==== %s  (%s)\033[0m\n' "$1" "$(date -u +%H:%M:%S)"; }
RES="results/phase2_mapping"

# --- 0. deps ----------------------------------------------------------------
say "0. dependencies"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq p7zip-full python3-venv
else
  command -v 7z >/dev/null 2>&1 || { echo "need 7z (p7zip) for the phase2 archives" >&2; exit 1; }
fi
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements.txt -r requirements_stage4.txt
python3 -m pip install -q -e .
python3 -m spacy download en_core_web_lg
python3 -c "import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# --- 1. data ----------------------------------------------------------------
if [[ "$SKIP_DATA" -eq 1 ]]; then
  say "1. data — SKIPPED (--skip-data); assuming data/ is populated and parsed"
else
  say "1. data — download + build + clean + parse"
  python3 scripts/generate/download_data.py
  python3 scripts/prepare/download_human_baseline.py
  python3 scripts/prepare/build_human_baseline.py
  # Phase-2a model corpora are not distributed in this repository (see the
  # "Data availability" section of README.md). If you have the 7z bundles,
  # drop them in data/phase2_mapping/archives/ and they are unpacked here:
  # phase2_<model>.7z -> data/phase2_mapping/<model>/
  if [[ -d data/phase2_mapping/archives ]]; then
    ( cd data/phase2_mapping
      shopt -s nullglob
      for f in archives/phase2_*.7z; do
        7z x -y "$f" >/dev/null
        d="$(basename "$f" .7z)"          # phase2_gemini-3-flash
        t="${d#phase2_}"                  # gemini-3-flash
        rm -rf "$t" && mv "$d" "$t"
        echo "  extracted $f -> $t/"
      done )
  else
    echo "  no data/phase2_mapping/archives/ — skipping phase-2a extraction"
  fi
  python3 scripts/prepare/clean_corpus.py
  python3 -m aiidiolects.parse_corpus --n-process "$NWORKERS"
fi

# --- 2. Stage-4 analyses ----------------------------------------------------
say "2a. perplexity scoring (E1)"
python3 scripts/analyse/perplexity_scoring.py --ref-models "$REF_MODELS"
python3 scripts/figures/render_perplexity.py --jpg

if [[ "$DO_BOOTSTRAP" -eq 1 ]]; then
  say "2b. with-human bootstrap CIs (E4, --n-workers $NWORKERS)"
  # reads stage2_distance_matrix_with_human/all_distances_long.csv +
  # the cleaned CSVs; writes .../bootstrap/bootstrap_summary.csv
  python3 scripts/analyse/bootstrap_distances.py --include-human --n-reps 500 --n-workers "$NWORKERS"
  python3 scripts/figures/render_human_distance.py --all --jpg   # now draws 95%-CI whiskers
else
  say "2b. with-human bootstrap — SKIPPED (--no-bootstrap)"
fi

if [[ "$DO_EMBEDDING" -eq 1 ]]; then
  say "2c. sentence-embedding semantic map (E3)"
  python3 scripts/analyse/embedding_map.py
  python3 scripts/figures/render_embedding_map.py --jpg
else
  say "2c. embedding map — SKIPPED (--no-embedding-map)"
fi

# --- 3. report + commit recipe ---------------------------------------------
say "3. done — results + commit recipe"
TO_ADD=("$RES/stage4_perplexity" "$RES/stage2_distance_matrix_with_human/figures")
[[ "$DO_BOOTSTRAP" -eq 1 ]] && TO_ADD+=("$RES/stage2_distance_matrix_with_human/bootstrap")
[[ "$DO_EMBEDDING" -eq 1 ]] && TO_ADD+=("$RES/stage4_embedding_map")
echo "Result dirs (small CSVs + figure JPGs):"
printf '  %s\n' "${TO_ADD[@]}"
echo
echo "Headline numbers were printed by each script above (perplexity ranking, bootstrap"
echo "CIs, internal-cosine ranking). Update docs/ANALYSES.md + the chapter with"
echo "them when the results come back."
echo
COMMIT_MSG="Stage 4 (Lambda): perplexity scoring + with-human bootstrap CIs"
[[ "$DO_EMBEDDING" -eq 1 ]] && COMMIT_MSG="$COMMIT_MSG + sentence-embedding map"
echo "To commit + push:"
echo "  git add ${TO_ADD[*]}"
echo "  git commit -m \"$COMMIT_MSG\""
echo "  git push"
echo
if [[ "$DO_PUSH" -eq 1 ]]; then
  if git config user.email >/dev/null 2>&1 && git config user.name >/dev/null 2>&1; then
    git add "${TO_ADD[@]}"
    git commit -m "$COMMIT_MSG"
    git push
    echo "  --> committed + pushed."
  else
    echo "  --push given but git identity not configured on this box. Run, then re-try:"
    echo "    git config user.email '<your-email>' && git config user.name '<your-name>'"
  fi
fi
echo
echo "Stage 4 complete. Copy results back if not pushed, then SPIN THE LAMBDA INSTANCE DOWN."
