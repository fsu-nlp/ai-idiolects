#!/usr/bin/env python3
"""Stage 4 (Lambda-GPU): perplexity / surprisal scoring.

Scores every cleaned text (all Phase 0/1/2a corpora + the Phase 2b human
baselines) under one or more *reference* causal-LM "yardsticks" — how
surprising is each model's prose to a fixed external LM? Per-corpus
perplexity distributions give a "how LM-optimised / low-surprisal is this
output" dimension the chapter doesn't otherwise have: the 2023↔2026 era
trend, and where the human corpora sit (expect: humans highest-perplexity,
the polished 2026 closed-API models lowest).

Default reference model: ``gpt2-large`` — the classic small baseline,
ungated on the HF Hub (downloads without a token), ~3 GB, fast. Pass
``--ref-models gpt2-large,mistralai/Mistral-7B-v0.1`` for more (the Mistral
/ Llama *base* models are gated — set ``HF_TOKEN`` and accept the licence
first). Runs on CUDA if available, CPU otherwise (~minutes/ref-model on an
A100, ~1–2 h on CPU — fine for a fire-and-forget Lambda run).

Long texts use a strided sliding-window perplexity (the standard HF recipe)
so a verbose 2026 model isn't unfairly penalised by first-N-tokens
truncation. The Stage-1a quality filter is applied (drop rows flagged
``repetition_loop`` / ``code_contam`` / ``non_english`` / ``truncation``)
so the per-corpus means aren't dragged down by degenerate outputs — pass
``--keep-flagged`` to score them too (their very low perplexity is itself a
finding, just not a clean corpus-mean input).

Output → ``results/phase2_mapping/stage4_perplexity/``:
``per_text_perplexity.csv`` (model_dir, topic, ref_model, n_tokens,
mean_nll, ppl — one row per text × ref-model; **not committed**, regenerable
like ``per_text.csv``) and ``summary.csv`` (per model_dir × scope ×
ref-model: n_texts + ppl/nll median/mean/std, committed). Render with
``render_perplexity.py``.

Usage::

    python3 scripts/analyse/perplexity_scoring.py
    python3 scripts/analyse/perplexity_scoring.py --ref-models gpt2-large --batch-not-used
    python3 scripts/analyse/perplexity_scoring.py --limit 20            # smoke test

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aiidiolects.build_distance_matrix import QUALITY_FLAGS_DROP, TOPICS, find_cleaned_csvs, model_dir_for
from aiidiolects.paths import DATA_DIR, RESULTS_DIR

SCOPES = ["pooled"] + TOPICS


# ---------------------------------------------------------------------------
# Per-text perplexity under one reference LM (strided sliding window).
# ---------------------------------------------------------------------------

def _text_nll(text: str, model, tokenizer, device, max_len: int, stride: int
              ) -> tuple[float, int]:
    """Return (mean negative log-likelihood per token, n_target_tokens) for
    ``text`` under ``model`` using a strided window — the canonical HF
    recipe. Empty / sub-tokenised texts return (nan, 0)."""
    import torch
    ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids
    n = ids.size(1)
    if n < 2:
        return float("nan"), 0
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0
    for begin in range(0, n, stride):
        end = min(begin + max_len, n)
        trg_len = end - prev_end          # tokens whose loss we count this window
        inp = ids[:, begin:end].to(device)
        tgt = inp.clone()
        tgt[:, :-trg_len] = -100          # mask out the "context" prefix
        with torch.no_grad():
            out = model(inp, labels=tgt)
        # out.loss is the *mean* NLL over the (trg_len) counted tokens.
        valid = trg_len  # number of target positions actually scored this window
        nll_sum += float(out.loss) * valid
        n_tokens += valid
        prev_end = end
        if end == n:
            break
    if n_tokens == 0:
        return float("nan"), 0
    return nll_sum / n_tokens, n_tokens


def _load_ref(name: str):
    """Load (model, tokenizer, device) for a reference causal LM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.to(device).eval()
    max_len = getattr(model.config, "n_positions", None) \
        or getattr(model.config, "max_position_embeddings", 1024)
    return model, tok, device, int(max_len)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage4_perplexity"),
    )
    ap.add_argument("--ref-models", default="gpt2-large",
                    help="comma-separated HF model ids (default: gpt2-large)")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="also score quality-flagged texts (default: drop them)")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, score only the first N texts per corpus (smoke test)")
    ap.add_argument("--stride-frac", type=float, default=0.5,
                    help="sliding-window stride as a fraction of the context length")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_models = [m.strip() for m in args.ref_models.split(",") if m.strip()]
    t0 = time.time()

    # Load all cleaned texts, quality-filtered, keyed by (model_dir, topic).
    print("Loading cleaned corpora …")
    corpora: dict[tuple[str, str], list[str]] = {}
    for csv in find_cleaned_csvs(Path(args.base), include_human=True):
        m = model_dir_for(csv)
        tp = csv.parent.name
        df = pd.read_csv(csv)
        if not args.keep_flagged:
            keep = pd.Series(True, index=df.index)
            for flag in QUALITY_FLAGS_DROP:
                if flag in df.columns:
                    keep &= ~df[flag].astype(bool)
            df = df.loc[keep]
        texts = df["text_clean"].fillna("").astype(str).tolist()
        texts = [t for t in texts if t.strip()]
        if args.limit:
            texts = texts[: args.limit]
        corpora[(m, tp)] = texts
    n_total = sum(len(v) for v in corpora.values())
    print(f"  {len(corpora)} (corpus, topic) cells, {n_total:,} texts; "
          f"reference LM(s): {ref_models}")

    rows: list[dict] = []
    for ref in ref_models:
        print(f"\n=== reference LM: {ref} ===")
        model, tok, device, max_len = _load_ref(ref)
        stride = max(1, int(max_len * args.stride_frac))
        print(f"  loaded on {device}; context={max_len}, stride={stride}")
        from tqdm import tqdm
        done = 0
        for (m, tp), texts in corpora.items():
            for txt in tqdm(texts, desc=f"{m}/{tp}", leave=False):
                nll, ntok = _text_nll(txt, model, tok, device, max_len, stride)
                if ntok == 0:
                    continue
                rows.append({"model_dir": m, "topic": tp, "ref_model": ref,
                             "n_tokens": ntok, "mean_nll": nll,
                             "ppl": float(math.exp(min(nll, 50.0)))})
                done += 1
            print(f"  {m:<22} {tp:<16} scored {len(texts):>5}  "
                  f"({done:,} total, {time.time() - t0:6.1f}s)")
        # free GPU memory before the next ref model
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    pt = pd.DataFrame(rows)
    pt_path = out_dir / "per_text_perplexity.csv"
    pt.to_csv(pt_path, index=False)
    print(f"\nWrote {pt_path}  ({len(pt):,} rows)")

    # Summary: per model_dir × scope × ref_model.
    summ_rows: list[dict] = []
    for ref in ref_models:
        rp = pt[pt["ref_model"] == ref]
        for scope in SCOPES:
            sub = rp if scope == "pooled" else rp[rp["topic"] == scope]
            for m, g in sub.groupby("model_dir"):
                summ_rows.append({
                    "model_dir": m, "scope": scope, "ref_model": ref,
                    "n_texts": int(len(g)),
                    "ppl_median": float(g["ppl"].median()),
                    "ppl_mean": float(g["ppl"].mean()),
                    "ppl_std": float(g["ppl"].std(ddof=1)) if len(g) > 1 else 0.0,
                    "nll_median": float(g["mean_nll"].median()),
                    "nll_mean": float(g["mean_nll"].mean()),
                    "n_tokens_median": float(g["n_tokens"].median()),
                })
    summ = pd.DataFrame(summ_rows)
    summ_path = out_dir / "summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"Wrote {summ_path}  ({len(summ):,} rows)")

    # Headline print: pooled median PPL per corpus, ranked (first ref model).
    if not summ.empty:
        ref0 = ref_models[0]
        ws = summ[(summ.scope == "pooled") & (summ.ref_model == ref0)].sort_values("ppl_median")
        print(f"\nMedian perplexity under {ref0} (pooled), ranked low→high:")
        for _, r in ws.iterrows():
            print(f"  {r.model_dir:<22} ppl={r['ppl_median']:7.2f}  (n={int(r['n_texts'])})")
    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
