#!/usr/bin/env python3
"""Stage 3: baseline-free per-model keyness — "top-N distinctive words".

The published LPR / LAS estimators (see the authors' prior work) need *prompt-
matched* human continuations and a split-halves design; our human baseline
(Webis-CMV-20) is topic-matched naturally-occurring opinion text, not
prompt-matched continuations, so that recipe doesn't transfer. The standard
register-free alternative — and what a corpus linguist (or AntConc) reaches
for — is **keyness**: for each corpus M, which words does M over-use relative
to a reference pool? Statistics: **log-likelihood G²** (Rayson & Garside /
Dunning — significance) and **log-ratio** (Hardie — effect size,
``log2[(c1+½)/N1 ÷ (c2+½)/N2]``). Rank by log-ratio among the
G²-significant words above a frequency floor (``c1 ≥ --min-count``).

The "establishing" — i.e. is M's top-10 robust or noise? — is done **via
M's own within-corpus variation**: recompute keyness on ``--n-resamples``
random half-subsamples of M's texts; a word's *selection rate* is the
fraction of resamples it lands in the top-N; the **consensus top-N** =
words with selection rate ≥ ``--consensus-rate`` (default 0.8). So
within-model variation is the stability filter for the signature list.

Reference pool (``ref_kind = "llm_rest"``, the default): the canonical
13-LLM roster (no run6) **minus** M, pooled across the scope's texts — i.e.
"what distinguishes M among the models". For the Phase-2b human targets
(webis / reuters), the reference is the full 13-LLM roster (they're not in
it) — "what distinguishes human prose from the LLM landscape". (A
``family_keyness`` mode — {Mistral corpora} vs {rest} etc. — is a noted
stretch goal; the default suppresses *family* signatures because the 2023
Mistral sits in the reference pool when M = Run 5.)

Three views, on the spaCy tokenisation (from the DocBins — lowercased,
no whitespace/punct):

* ``all``      — every token
* ``content``  — UPOS ∈ {NOUN, PROPN, VERB, ADJ, ADV}
* ``function`` — UPOS ∈ {DET, ADP, PRON, AUX, CCONJ, SCONJ, PART, NUM}

Scopes: pooled + the four Improta topics. Single-process; DocBins loaded
one corpus at a time (~1.5–2 GB peak — runs *after* within_corpus_variation.py,
no overlap).

Output → ``results/phase2_mapping/stage3_keyness/``:
``keyness_long.csv`` (target × scope × view × word: c1, c2, log_likelihood,
log_ratio, selection_rate, is_consensus) and ``consensus_topN.csv``
(the consensus signatures only, ordered). Render with ``render_keyness.py``.

Usage::

    python3 -m aiidiolects.keyness
    python3 -m aiidiolects.keyness --top-n 20 --n-resamples 50 --min-count 20

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import DocBin

from aiidiolects.build_distance_matrix import (
    QUALITY_FLAGS_DROP,
    TOPICS,
    find_cleaned_csvs,
    model_dir_for,
)
from aiidiolects.parse_corpus import cleaned_to_spacy_path
from aiidiolects.paths import DATA_DIR, RESULTS_DIR
from aiidiolects.visualize_stage1 import MODEL_ORDER as LLM_ROSTER_13   # 13 LLMs, no run6

# Keyness *targets*: the 13-LLM roster + the two Phase-2b human corpora.
# (run6_lmstudio is excluded as a target — redundant stochastic twin of run5.)
KEYNESS_TARGETS: list[str] = LLM_ROSTER_13 + ["webis_cmv_20", "reuters_50"]
SCOPES = ["pooled"] + TOPICS

CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
FUNCTION_POS = {"DET", "ADP", "PRON", "AUX", "CCONJ", "SCONJ", "PART", "NUM"}
VIEWS = ["all", "content", "function"]

# G² threshold ≈ p < 0.001 (1 df). Effect size (log-ratio) does the ranking.
G2_MIN = 10.83


# ---------------------------------------------------------------------------
# Load — per-text (lowercase, pos) token lists from the DocBins
# ---------------------------------------------------------------------------

def load_tagged(base: Path) -> dict[tuple[str, str], list[list[tuple[str, str]]]]:
    vocab = spacy.blank("en").vocab
    out: dict[tuple[str, str], list[list[tuple[str, str]]]] = {}
    for csv in find_cleaned_csvs(base, include_human=True):
        m = model_dir_for(csv)
        tp = csv.parent.name
        sp = cleaned_to_spacy_path(csv)
        if not sp.exists():
            print(f"  !! no DocBin for {csv} — skipping")
            continue
        df = pd.read_csv(csv)
        keep = pd.Series(True, index=df.index)
        for flag in QUALITY_FLAGS_DROP:
            if flag in df.columns:
                keep &= ~df[flag].astype(bool)
        keep = keep.to_numpy()
        docs = list(DocBin().from_disk(sp).get_docs(vocab))
        if len(docs) != len(df):
            print(f"  !! DocBin/CSV mismatch for {csv} — skipping")
            continue
        texts: list[list[tuple[str, str]]] = []
        for doc, k in zip(docs, keep):
            if not k:
                continue
            texts.append([(t.lower_, t.pos_) for t in doc
                          if not t.is_space and not t.is_punct and t.pos_])
        out[(m, tp)] = texts
        del docs
    return out


def _scope_texts(
    tagged: dict[tuple[str, str], list[list[tuple[str, str]]]],
    model_dir: str, scope: str,
) -> list[list[tuple[str, str]]]:
    if scope == "pooled":
        out: list[list[tuple[str, str]]] = []
        for tp in TOPICS + ["reuters_ccat"]:
            out.extend(tagged.get((model_dir, tp), []))
        return out
    return tagged.get((model_dir, scope), [])


def _view_counter(texts: list[list[tuple[str, str]]], view: str) -> Counter:
    c: Counter = Counter()
    if view == "all":
        for tx in texts:
            c.update(w for w, _ in tx)
    elif view == "content":
        for tx in texts:
            c.update(w for w, p in tx if p in CONTENT_POS)
    else:  # function
        for tx in texts:
            c.update(w for w, p in tx if p in FUNCTION_POS)
    return c


# ---------------------------------------------------------------------------
# Keyness statistics
# ---------------------------------------------------------------------------

def _ll(c1: int, c2: int, n1: int, n2: int) -> float:
    """Dunning log-likelihood G² for a word's counts in target vs reference."""
    if n1 == 0 or n2 == 0 or (c1 + c2) == 0:
        return 0.0
    e1 = n1 * (c1 + c2) / (n1 + n2)
    e2 = n2 * (c1 + c2) / (n1 + n2)
    s = 0.0
    if c1 > 0 and e1 > 0:
        s += c1 * math.log(c1 / e1)
    if c2 > 0 and e2 > 0:
        s += c2 * math.log(c2 / e2)
    return 2.0 * s


def _log_ratio(c1: int, c2: int, n1: int, n2: int) -> float:
    """Hardie log-ratio (effect size), +0.5 smoothing on both counts."""
    r1 = (c1 + 0.5) / n1 if n1 else 0.0
    r2 = (c2 + 0.5) / n2 if n2 else 0.0
    if r1 <= 0 or r2 <= 0:
        return 0.0
    return math.log2(r1 / r2)


def keyness_topn(
    target: Counter, ref: Counter, top_n: int, min_count: int,
) -> list[tuple[str, int, int, float, float]]:
    """Top-N (word, c1, c2, ll, log_ratio) by log-ratio among G²-significant
    words with c1 ≥ min_count."""
    n1, n2 = sum(target.values()), sum(ref.values())
    rows = []
    for w, c1 in target.items():
        if c1 < min_count:
            continue
        c2 = ref.get(w, 0)
        ll = _ll(c1, c2, n1, n2)
        if ll < G2_MIN:
            continue
        rows.append((w, c1, c2, ll, _log_ratio(c1, c2, n1, n2)))
    rows.sort(key=lambda r: r[4], reverse=True)
    return rows[:top_n]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage3_keyness"),
    )
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--n-resamples", type=int, default=50)
    ap.add_argument("--consensus-rate", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=20260511)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading tagged tokens from DocBins …")
    tagged = load_tagged(Path(args.base))
    present = sorted({m for m, _ in tagged})
    targets = [m for m in KEYNESS_TARGETS if m in present]
    ref_pool_13 = [m for m in LLM_ROSTER_13 if m in present]
    print(f"  targets ({len(targets)}): {targets}")
    print(f"  reference pool (13-LLM, minus target): {ref_pool_13}")

    rng = np.random.default_rng(args.seed)
    long_rows: list[dict] = []
    consensus_rows: list[dict] = []

    # Precompute, per (scope, view): each LLM-roster corpus's counter + the
    # full 13-roster sum (the reference for any target is that sum minus the
    # target's own counter, or the full sum for the human targets).
    for scope in SCOPES:
        for view in VIEWS:
            llm_counters = {
                m: _view_counter(_scope_texts(tagged, m, scope), view)
                for m in ref_pool_13
            }
            full_ref = Counter()
            for c in llm_counters.values():
                full_ref.update(c)

            for m in targets:
                m_texts = _scope_texts(tagged, m, scope)
                if len(m_texts) < 4:
                    continue
                m_counter = _view_counter(m_texts, view)
                if m in llm_counters:
                    ref = full_ref - llm_counters[m]   # 13-roster minus M
                else:
                    ref = full_ref                     # human target vs all 13
                if sum(ref.values()) == 0 or sum(m_counter.values()) == 0:
                    continue

                point = keyness_topn(m_counter, ref, args.top_n, args.min_count)
                point_words = {w for w, *_ in point}
                point_info = {w: (c1, c2, ll, lr) for w, c1, c2, ll, lr in point}

                # Stability via half-subsample resampling of M's texts.
                sel = Counter()
                n = len(m_texts)
                half = max(2, n // 2)
                for _ in range(args.n_resamples):
                    idx = rng.choice(n, size=half, replace=False)
                    sub = _view_counter([m_texts[i] for i in idx], view)
                    for w, *_ in keyness_topn(sub, ref, args.top_n, args.min_count):
                        sel[w] += 1
                rate = {w: sel[w] / args.n_resamples for w in sel}

                # Emit: every word that was top-N either at the point estimate
                # or in some resample.
                for w in point_words | set(sel):
                    c1, c2, ll, lr = point_info.get(
                        w, (m_counter.get(w, 0), ref.get(w, 0),
                            _ll(m_counter.get(w, 0), ref.get(w, 0),
                                sum(m_counter.values()), sum(ref.values())),
                            _log_ratio(m_counter.get(w, 0), ref.get(w, 0),
                                       sum(m_counter.values()), sum(ref.values()))))
                    r = rate.get(w, 0.0)
                    is_cons = (w in point_words) and (r >= args.consensus_rate)
                    long_rows.append({
                        "target": m, "scope": scope, "view": view,
                        "ref_kind": "llm_rest", "word": w,
                        "c1": c1, "c2": c2, "log_likelihood": ll,
                        "log_ratio": lr, "selection_rate": r,
                        "in_point_topn": w in point_words,
                        "is_consensus": is_cons,
                    })
                    if is_cons:
                        consensus_rows.append({
                            "target": m, "scope": scope, "view": view,
                            "word": w, "log_ratio": lr, "log_likelihood": ll,
                            "selection_rate": r, "c1": c1, "c2": c2,
                        })
        print(f"  scope {scope:<15} done ({time.time() - t0:5.1f}s)")

    long = pd.DataFrame(long_rows)
    long_path = out_dir / "keyness_long.csv"
    long.to_csv(long_path, index=False)
    print(f"\nWrote {long_path}  ({len(long):,} rows)")

    cons = pd.DataFrame(consensus_rows)
    if not cons.empty:
        cons = cons.sort_values(["scope", "view", "target", "log_ratio"],
                                ascending=[True, True, True, False])
    cons_path = out_dir / "consensus_topN.csv"
    cons.to_csv(cons_path, index=False)
    print(f"Wrote {cons_path}  ({len(cons):,} rows)")

    # Headline print: pooled / content consensus top-8 per target.
    if not cons.empty:
        cc = cons[(cons.scope == "pooled") & (cons.view == "content")]
        print("\nConsensus top content-word signatures (pooled), per target:")
        for m in targets:
            ws = cc[cc.target == m].head(8)["word"].tolist()
            print(f"  {m:<22} {', '.join(ws) if ws else '(none)'}")
    n_cons = cons.groupby(["scope", "view", "target"]).size() if not cons.empty else pd.Series(dtype=int)
    print(f"\nDone in {time.time() - t0:.1f}s. "
          f"Median consensus-list size: "
          f"{int(n_cons.median()) if len(n_cons) else 0}")


if __name__ == "__main__":
    main()
