#!/usr/bin/env python3
"""Stage 3: within-corpus / within-speaker variation.

For every corpus (the 14-corpus build roster + the two Phase 2b human
corpora — 16 in all), repeatedly split its texts into two disjoint random
halves and measure the distance *between the halves*. That distribution is
the corpus's **internal heterogeneity** — the noise floor against which the
between-corpus distances in the §4.1 matrix should be read: model A↔model B
of, say, 0.48 only means something if model A's own within-corpus distance
is ~0.05.

To keep the within-numbers on the §4.1 scale, the same **Burrows' Δ** and
**Cosine Δ** estimators as ``build_distance_matrix.py`` are used: a *global*
MFW (top-100 across all 16 corpora pooled) + a *global* per-text z-score
reference (mean / sd over every per-text MFW vector) — i.e. the per-text-
centroid formulation, not the per-pair one. Per split: z-scored half-centroid
vs z-scored half-centroid. (An n-gram-JSD within-corpus metric was scoped but
dropped — recomputing top-500 bigram distributions over hundreds of texts ×
hundreds of reps × 16 corpora × 5 scopes is the one expensive step, and
Burrows/Cosine already tell the within-vs-between story; an n-gram version is
a cheap follow-up if §4 wants it.)

Both within- and between-corpus distances here use *this script's* 16-corpus
global reference (so they're perfectly mutually comparable), which is a
slightly different reference set than the canonical 13-corpus or 15-corpus
``stage2_distance_matrix`` matrices — comparable in kind and scale, not
cell-for-cell. (Same caveat ``within_author_baseline.py`` already carries
about its own pairwise-MFW estimator.) For the "between" reference lines the
script emits, on this same reference: the Mistral 2023↔2026 (Run 5) engine
bridge, and every one of the 120 between-corpus full-corpus-centroid
distances (the render takes the median).

Splits are size-controlled: each split samples ``min(n, --sample-cap)``
texts (default cap 1000) and divides them into two halves, so the
within-corpus value isn't conflated with corpus size. Webis topics with
fewer texts than the cap use all of them (math (broad) n≈37 — a thin
estimate, flagged).

Plus **within-author** for the human corpora that carry an ``author_id``
column (Webis-CMV-20: a *thin* estimate — only ~25 r/CMV authors have ≥2
matched ``climate`` OPs, fewer elsewhere; Reuters-50's clean within-author
distribution is the dedicated ``within_author_baseline.py``).

Single-process by design (the 16 GB-RAM constraint — all corpora's token
lists are held at once, ~2 GB; multiprocessing the resampling defeats
fork-COW via refcount writes, as documented in ``bootstrap_distances.py``).

Output → ``results/phase2_mapping/stage3_within_corpus_variation/``:
``within_corpus_long.csv`` (one row per metric × scope × {within-split rep |
within-author author | between-pair}) and ``summary.csv`` (per-group
aggregates). Render with ``render_within_corpus.py``.

Usage::

    python3 scripts/analyse/within_corpus_variation.py
    python3 scripts/analyse/within_corpus_variation.py --n-reps 100 --sample-cap 1000

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from aiidiolects.build_distance_matrix import (
    MODEL_ORDER,
    QUALITY_FLAGS_DROP,
    TOPICS,
    burrows_delta_centroid,
    cosine_delta_centroid,
    find_cleaned_csvs,
    model_dir_for,
    per_text_relfreqs,
)
from aiidiolects.compare_fingerprints import tokenize
from aiidiolects.paths import DATA_DIR, RESULTS_DIR

ROSTER: list[str] = MODEL_ORDER + ["webis_cmv_20", "reuters_50"]
BRIDGE_PAIR = ("phase0_mistral-7b", "run5_lmstudio")
WITHIN_METRICS = ["burrows_delta", "cosine_delta"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_corpora(base: Path) -> dict[tuple[str, str], dict]:
    """{(model_dir, topic): {"tokens": list[list[str]], "authors": list[str]|None}}.
    Quality-filtered; raw text strings are dropped after tokenisation."""
    out: dict[tuple[str, str], dict] = {}
    for csv in find_cleaned_csvs(base, include_human=True):
        m = model_dir_for(csv)
        tp = csv.parent.name
        df = pd.read_csv(csv)
        keep = pd.Series(True, index=df.index)
        for flag in QUALITY_FLAGS_DROP:
            if flag in df.columns:
                keep &= ~df[flag].astype(bool)
        df = df.loc[keep]
        texts = df["text_clean"].fillna("").astype(str).tolist()
        tokens = [tokenize(t) for t in texts]
        authors = (
            df["author_id"].astype(str).tolist() if "author_id" in df.columns else None
        )
        out[(m, tp)] = {"tokens": tokens, "authors": authors}
    return out


def _concat_scope(
    corpora: dict[tuple[str, str], dict], model_dir: str, scope: str,
) -> dict:
    """Token lists (+ author ids) for one corpus in one scope."""
    if scope == "pooled":
        toks: list[list[str]] = []
        auths: list[str] = []
        have_auth = True
        for tp in TOPICS + ["reuters_ccat"]:
            d = corpora.get((model_dir, tp))
            if d is None:
                continue
            toks.extend(d["tokens"])
            if d["authors"] is None:
                have_auth = False
            else:
                auths.extend(d["authors"])
        return {"tokens": toks, "authors": auths if (have_auth and auths) else None}
    d = corpora.get((model_dir, scope))
    if d is None:
        return {"tokens": [], "authors": None}
    return {"tokens": d["tokens"], "authors": d["authors"]}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage3_within_corpus_variation"),
    )
    ap.add_argument("--n-reps", type=int, default=100,
                    help="random half-splits per (corpus × scope)")
    ap.add_argument("--sample-cap", type=int, default=1000,
                    help="cap on texts sampled per split (then divided into two "
                         "halves) — size-controls the within-corpus estimate")
    ap.add_argument("--n-features", type=int, default=100,
                    help="global MFW count for Burrows / Cosine Δ")
    ap.add_argument("--seed", type=int, default=20260511)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading + tokenising corpora …")
    corpora = load_corpora(Path(args.base))
    present_models = sorted({m for m, _ in corpora})
    in_roster = [m for m in ROSTER if m in present_models]
    skipped = [m for m in present_models if m not in ROSTER]
    print(f"  in-roster ({len(in_roster)}): {in_roster}")
    if skipped:
        print(f"  not in roster, skipped: {skipped}")
    scopes = ["pooled"] + TOPICS

    # --- global MFW + z-score reference over ALL corpora pooled-across-topics ---
    print("Building global MFW + z-score reference …")
    from collections import Counter
    gc: Counter = Counter()
    for d in corpora.values():
        for tl in d["tokens"]:
            gc.update(tl)
    features = [w for w, _ in gc.most_common(args.n_features)]

    # Per (model_dir, topic) per-text MFW relfreq vectors (raw, pre-z-score).
    raw_vecs: dict[tuple[str, str], np.ndarray] = {}
    for key, d in corpora.items():
        raw_vecs[key] = per_text_relfreqs(d["tokens"], features)
    all_rows = np.vstack([v for v in raw_vecs.values() if v.size])
    mu = all_rows.mean(axis=0)
    sigma = all_rows.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1e-12
    z_vecs: dict[tuple[str, str], np.ndarray] = {
        key: ((v - mu) / sigma) if v.size else np.zeros((0, len(features)))
        for key, v in raw_vecs.items()
    }
    del raw_vecs, all_rows, gc

    rng = np.random.default_rng(args.seed)
    long_rows: list[dict] = []

    # --- within-corpus split-half distributions ---
    for scope in scopes:
        for m in in_roster:
            zc = (
                np.vstack([z_vecs[(m, tp)] for tp in TOPICS + ["reuters_ccat"]
                           if (m, tp) in z_vecs and z_vecs[(m, tp)].size])
                if scope == "pooled"
                else z_vecs.get((m, scope), np.zeros((0, len(features))))
            )
            sc = _concat_scope(corpora, m, scope)
            n = len(sc["tokens"])
            if n < 4 or zc.shape[0] != n:
                continue
            k = min(n, args.sample_cap)
            for rep in range(args.n_reps):
                idx = rng.choice(n, size=k, replace=False)
                ha, hb = idx[: k // 2], idx[k // 2: 2 * (k // 2)]
                ca, cb = zc[ha].mean(axis=0), zc[hb].mean(axis=0)
                long_rows.append({"model_dir": m, "scope": scope,
                                  "kind": "within_split", "rep": rep,
                                  "burrows_delta": burrows_delta_centroid(ca, cb),
                                  "cosine_delta": cosine_delta_centroid(ca, cb)})
        print(f"  scope {scope:<15} done ({time.time() - t0:5.1f}s)")

    # --- within-author (corpora with an author_id column) ---
    for scope in scopes:
        for m in in_roster:
            sc = _concat_scope(corpora, m, scope)
            authors = sc["authors"]
            if authors is None:
                continue
            zc = (
                np.vstack([z_vecs[(m, tp)] for tp in TOPICS + ["reuters_ccat"]
                           if (m, tp) in z_vecs and z_vecs[(m, tp)].size])
                if scope == "pooled"
                else z_vecs.get((m, scope), np.zeros((0, len(features))))
            )
            if zc.shape[0] != len(authors):
                continue
            by_author: dict[str, list[int]] = {}
            for i, a in enumerate(authors):
                by_author.setdefault(a, []).append(i)
            multi = {a: ix for a, ix in by_author.items() if len(ix) >= 2}
            for a, ix in multi.items():
                ix = list(ix)
                rng.shuffle(ix)
                h = len(ix) // 2
                ha, hb = ix[:h], ix[h: 2 * h]
                ca, cb = zc[ha].mean(axis=0), zc[hb].mean(axis=0)
                long_rows.append({"model_dir": m, "scope": scope,
                                  "kind": "within_author", "rep": -1,
                                  "burrows_delta": burrows_delta_centroid(ca, cb),
                                  "cosine_delta": cosine_delta_centroid(ca, cb),
                                  "n_docs": len(ix)})
            if multi:
                print(f"  within-author: {m}/{scope}: {len(multi)} authors with ≥2 texts")

    # --- between-corpus reference distances (pooled, full-corpus centroids) ---
    pooled_cent: dict[str, np.ndarray] = {}
    for m in in_roster:
        zc = np.vstack([z_vecs[(m, tp)] for tp in TOPICS + ["reuters_ccat"]
                        if (m, tp) in z_vecs and z_vecs[(m, tp)].size])
        if zc.size:
            pooled_cent[m] = zc.mean(axis=0)
    for a, b in combinations([m for m in in_roster if m in pooled_cent], 2):
        is_bridge = {a, b} == set(BRIDGE_PAIR)
        long_rows.append({"model_dir": f"{a}|{b}", "model_a": a, "model_b": b,
                          "scope": "pooled",
                          "kind": "between_bridge" if is_bridge else "between_pair",
                          "rep": -1,
                          "burrows_delta": burrows_delta_centroid(pooled_cent[a], pooled_cent[b]),
                          "cosine_delta": cosine_delta_centroid(pooled_cent[a], pooled_cent[b])})

    long = pd.DataFrame(long_rows)
    long_path = out_dir / "within_corpus_long.csv"
    long.to_csv(long_path, index=False)
    print(f"\nWrote {long_path}  ({len(long):,} rows)")

    # --- summary ---
    summ_rows: list[dict] = []
    for (m, scope, kind), g in long.groupby(["model_dir", "scope", "kind"]):
        for metric in WITHIN_METRICS:
            vals = g[metric].dropna().to_numpy()
            if vals.size == 0:
                continue
            summ_rows.append({
                "model_dir": m, "scope": scope, "kind": kind, "metric": metric,
                "n": int(vals.size), "min": float(vals.min()),
                "p25": float(np.percentile(vals, 25)),
                "median": float(np.median(vals)),
                "p75": float(np.percentile(vals, 75)),
                "max": float(vals.max()), "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            })
    summ = pd.DataFrame(summ_rows)
    summ_path = out_dir / "summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"Wrote {summ_path}  ({len(summ):,} rows)")

    # Headline print: within-split medians (burrows) ranked, + reference lines.
    ws = summ[(summ.kind == "within_split") & (summ.scope == "pooled")
              & (summ.metric == "burrows_delta")].sort_values("median")
    print("\nWithin-corpus Burrows Δ (pooled, median), ranked:")
    for _, r in ws.iterrows():
        print(f"  {r.model_dir:<22} {r['median']:.4f}")
    bm = summ[(summ.kind == "between_pair") | (summ.kind == "between_bridge")]
    bp = bm[bm.metric == "burrows_delta"]
    if not bp.empty:
        bridge = bp[bp.kind == "between_bridge"]["median"]
        print(f"  --- engine bridge (Mistral 2023↔Run5): "
              f"{float(bridge.iloc[0]):.4f}" if len(bridge) else "")
        print(f"  --- median between-corpus pair:         "
              f"{float(bp['median'].median()):.4f}")
    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
