#!/usr/bin/env python3
"""Stage 2 — within-author calibration baseline (Reuters-50 / C50).

Reuses ``compare_fingerprints.burrows_delta`` (no metric reimplementation).
Three distance distributions over the 50 C50 authors × ~100 documents each:

* ``within_author`` — for each author, split their docs into two seeded random
  halves and take Burrows' Δ between the halves (``--n-splits`` splits per
  author). "How much does one human author vary across two samples of their
  own writing?"
* ``cross_author``  — Burrows' Δ between all (or ``--max-cross-pairs`` sampled)
  pairs of *different* authors, each using that author's full doc set. "How
  much do two different humans differ?"
* ``within_pool``   — ``--n-pool-splits`` random splits of the **whole** 5,000-
  doc pool into two halves. The closest structural analog to the LLM
  within-run noise floor (two large independent samples from one population),
  but the population here is "C50 journalists writing CCAT news", not a single
  author.

The headline calibration: ``within_author`` should sit well below
``cross_author`` (sanity — stylometry separates authors), and the **LLM
within-run noise floor** — Run 5 ↔ Run 6 ≈ 0.024 Burrows' Δ for Mistral-7B at
n=1000 (see ``bootstrap_distances.py`` / ``bootstrap_engine_bridge_long.csv``)
— can be read against ``within_pool`` (same-shape comparison) and
``within_author`` (note the much smaller per-author sample inflates Δ a bit).

This is a **pipeline calibration**, not a within-human variance estimate for
the Improta opinion-essay prompts — Reuters is topic-controlled corporate news
in a different register. See the §5
Discussion caveat in ``supplement/human_baseline.md``.

Input (first that exists):
    data/human_baseline/reuters_50/reuters_ccat/Reuters-50.cleaned.csv  (text_clean, quality-filtered)
    data/human_baseline/reuters_50/reuters_ccat/Reuters-50.csv          (raw text)

Output:
    results/phase2_mapping/stage2_within_author_baseline/reuters_within_author.csv  — every Δ measured
    results/phase2_mapping/stage2_within_author_baseline/summary.csv                — aggregates per kind + the LLM-floor reference
    + a short summary printed to stdout.

Usage::

    python3 scripts/analyse/within_author_baseline.py
    python3 scripts/analyse/within_author_baseline.py --n-splits 5 --max-cross-pairs 0   # all 1225 pairs
    python3 scripts/analyse/within_author_baseline.py --base data \
        --out-dir results/phase2_mapping/stage2_within_author_baseline

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import itertools
import random
import statistics
import sys
import time
from pathlib import Path

import pandas as pd

from aiidiolects.compare_fingerprints import burrows_delta, tokenize
from aiidiolects.paths import DATA_DIR, RESULTS_DIR

# Quality flags whose rows are dropped before tokenisation — mirrors
# build_distance_matrix.QUALITY_FLAGS_DROP so the human corpus is filtered
# the same way as the LLM corpora.
QUALITY_FLAGS_DROP = [
    "flag_repetition_loop", "flag_code_contam", "flag_non_english", "flag_truncation",
]

# Reference only (different implementation): the §4.1-figure-scale LLM
# within-run Burrows' Δ noise floor — Mistral-7B Run 5 ↔ Run 6, n=1000 each,
# computed by build_distance_matrix.burrows_delta_centroid over the 14-corpus
# *global* MFW + per-text z-scores. NOT comparable cell-for-cell with the
# numbers this script produces (it uses compare_fingerprints.burrows_delta:
# 100 MFW of the pair, 20 chunks per side). We also compute a same-scale LLM
# value below (load_llm_run_delta) so the calibration is apples-to-apples.
LLM_FLOOR_FIGURE_SCALE = 0.024
LLM_FLOOR_FIGURE_SCALE_NOTE = "Run5↔Run6 n=1000, build_distance_matrix global-MFW implementation (PHASE1_RESULTS.md)"

REUTERS_DIR = Path("reuters_50") / "reuters_ccat"
LLM_RUNS = ("run5_lmstudio", "run6_lmstudio")
LLM_TOPICS = ("climate", "global_warming", "math_anxiety", "misinfo_health")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_reuters_by_author(base: Path) -> dict[str, list[str]]:
    d = base / "human_baseline" / REUTERS_DIR
    cleaned, raw = d / "Reuters-50.cleaned.csv", d / "Reuters-50.csv"
    if cleaned.exists():
        df = pd.read_csv(cleaned)
        keep = pd.Series(True, index=df.index)
        for flag in QUALITY_FLAGS_DROP:
            if flag in df.columns:
                keep &= ~df[flag].astype(bool)
        df = df.loc[keep]
        text_col = "text_clean" if "text_clean" in df.columns else "text"
        src = cleaned
    elif raw.exists():
        df = pd.read_csv(raw)
        text_col = "text"
        src = raw
    else:
        sys.exit(f"ERROR: no Reuters-50 CSV under {d}/ — run "
                 f"`python3 build_human_baseline.py --only reuters` first "
                 f"(and optionally clean_corpus.py).")
    if "author_id" not in df.columns:
        sys.exit(f"ERROR: {src} has no 'author_id' column.")
    by_author: dict[str, list[str]] = {}
    for aid, grp in df.groupby("author_id"):
        texts = [t for t in grp[text_col].fillna("").astype(str).tolist() if t.strip()]
        if texts:
            by_author[str(aid)] = texts
    print(f"Loaded {src.name}: {len(by_author)} authors, "
          f"{sum(len(v) for v in by_author.values())} docs total "
          f"(median {statistics.median(len(v) for v in by_author.values()):.0f}/author)")
    return by_author


def _load_cleaned_texts(csv_path: Path) -> list[str]:
    """text_clean with quality-flagged rows dropped (mirrors build_distance_matrix.load_corpus_texts)."""
    df = pd.read_csv(csv_path)
    keep = pd.Series(True, index=df.index)
    for flag in QUALITY_FLAGS_DROP:
        if flag in df.columns:
            keep &= ~df[flag].astype(bool)
    col = "text_clean" if "text_clean" in df.columns else "text"
    return [t for t in df.loc[keep, col].fillna("").astype(str).tolist() if t.strip()]


def load_llm_run_delta(base: Path) -> dict | None:
    """Same-scale LLM within-run reference: Burrows' Δ (compare_fingerprints impl)
    between the pooled Run 5 and Run 6 Mistral-7B cleaned corpora. Returns None
    if the cleaned CSVs aren't present (run clean_corpus.py first)."""
    pooled: dict[str, list[str]] = {}
    for run in LLM_RUNS:
        rd = base / "phase1_anchoring" / run
        texts: list[str] = []
        for topic in LLM_TOPICS:
            c = rd / topic / "Mistral-7b.cleaned.csv"
            if c.exists():
                texts += _load_cleaned_texts(c)
        if not texts:
            return None
        pooled[run] = texts
    if len(pooled) != 2:
        return None
    a = [tokenize(t) for t in pooled[LLM_RUNS[0]]]
    b = [tokenize(t) for t in pooled[LLM_RUNS[1]]]
    return {
        "kind": "llm_within_run_same_metric",
        "n_docs_a": len(a), "n_docs_b": len(b),
        "burrows_delta": burrows_delta(a, b),
    }


# ---------------------------------------------------------------------------
# Distance distributions
# ---------------------------------------------------------------------------

def _toks(texts: list[str]) -> list[list[str]]:
    return [tokenize(t) for t in texts]


def within_author_deltas(tok_by_author: dict[str, list[list[str]]],
                         n_splits: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for ai, (aid, tls) in enumerate(sorted(tok_by_author.items())):
        if len(tls) < 4:
            continue
        for s in range(n_splits):
            rng = random.Random(seed + ai * 1000 + s)
            order = list(range(len(tls)))
            rng.shuffle(order)
            half = len(order) // 2
            a = [tls[i] for i in order[:half]]
            b = [tls[i] for i in order[half:2 * half]]
            rows.append({
                "kind": "within_author", "group_a": aid, "group_b": aid, "split": s,
                "burrows_delta": burrows_delta(a, b), "n_docs_a": len(a), "n_docs_b": len(b),
            })
    return rows


def cross_author_deltas(tok_by_author: dict[str, list[list[str]]],
                        max_pairs: int, seed: int) -> list[dict]:
    authors = sorted(tok_by_author)
    all_pairs = list(itertools.combinations(authors, 2))
    if max_pairs and len(all_pairs) > max_pairs:
        all_pairs = random.Random(seed).sample(all_pairs, max_pairs)
    rows: list[dict] = []
    for a_id, b_id in all_pairs:
        a, b = tok_by_author[a_id], tok_by_author[b_id]
        rows.append({
            "kind": "cross_author", "group_a": a_id, "group_b": b_id, "split": -1,
            "burrows_delta": burrows_delta(a, b), "n_docs_a": len(a), "n_docs_b": len(b),
        })
    return rows


def within_pool_deltas(tok_by_author: dict[str, list[list[str]]],
                       n_pool_splits: int, seed: int) -> list[dict]:
    pool = [tl for tls in tok_by_author.values() for tl in tls]
    rows: list[dict] = []
    for s in range(n_pool_splits):
        rng = random.Random(seed + 7_000_000 + s)
        order = list(range(len(pool)))
        rng.shuffle(order)
        half = len(order) // 2
        a = [pool[i] for i in order[:half]]
        b = [pool[i] for i in order[half:2 * half]]
        rows.append({
            "kind": "within_pool", "group_a": "ALL", "group_b": "ALL", "split": s,
            "burrows_delta": burrows_delta(a, b), "n_docs_a": len(a), "n_docs_b": len(b),
        })
    return rows


# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------

def _agg(vals: list[float]) -> dict:
    vals = sorted(vals)
    n = len(vals)
    q = statistics.quantiles(vals, n=4) if n >= 4 else [vals[0], statistics.median(vals), vals[-1]]
    return {
        "n": n, "min": round(vals[0], 5), "p25": round(q[0], 5),
        "median": round(statistics.median(vals), 5), "p75": round(q[2 if n >= 4 else -1], 5),
        "max": round(vals[-1], 5),
        "mean": round(statistics.fmean(vals), 5),
        "std": round(statistics.pstdev(vals), 5) if n > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument("--out-dir", default=str(RESULTS_DIR / "phase2_mapping" / "stage2_within_author_baseline"))
    ap.add_argument("--n-splits", type=int, default=3, help="random half-splits per author (within_author)")
    ap.add_argument("--max-cross-pairs", type=int, default=300,
                    help="cap on cross-author pairs (0 = all 1225)")
    ap.add_argument("--n-pool-splits", type=int, default=5, help="random whole-pool half-splits (within_pool)")
    ap.add_argument("--seed", type=int, default=20260511)
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_author = load_reuters_by_author(base)
    if len(by_author) < 2:
        sys.exit("ERROR: need ≥2 authors.")
    print("Tokenising...", flush=True)
    tok_by_author = {aid: _toks(texts) for aid, texts in by_author.items()}

    t0 = time.time()
    rows: list[dict] = []
    print(f"within_author ({args.n_splits} split(s) × {len(tok_by_author)} authors)...", flush=True)
    rows += within_author_deltas(tok_by_author, args.n_splits, args.seed)
    print(f"cross_author ({'all 1225' if not args.max_cross_pairs else f'≤{args.max_cross_pairs} sampled'} pairs)...", flush=True)
    rows += cross_author_deltas(tok_by_author, args.max_cross_pairs, args.seed)
    print(f"within_pool ({args.n_pool_splits} split(s) of the {sum(len(v) for v in tok_by_author.values())}-doc pool)...", flush=True)
    rows += within_pool_deltas(tok_by_author, args.n_pool_splits, args.seed)
    print(f"  {len(rows)} Δ measured in {time.time()-t0:.1f}s")

    # same-scale LLM within-run reference (Burrows' Δ via compare_fingerprints)
    llm_same = load_llm_run_delta(base)
    if llm_same is not None:
        rows.append({"group_a": LLM_RUNS[0], "group_b": LLM_RUNS[1], "split": -1, **llm_same})
        print(f"LLM within-run (same metric): Run5↔Run6 Burrows' Δ = {llm_same['burrows_delta']:.4f} "
              f"(n={llm_same['n_docs_a']}↔{llm_same['n_docs_b']})")
    else:
        print("LLM within-run (same metric): skipped — Run5/Run6 cleaned CSVs not found "
              "(run clean_corpus.py); the figure-scale reference 0.024 still shown below.")

    df = pd.DataFrame(rows)
    detail_path = out_dir / "reuters_within_author.csv"
    df.sort_values(["kind", "group_a", "group_b", "split"]).to_csv(
        detail_path, index=False, float_format="%.6f")
    print(f"Wrote {detail_path}  ({len(df)} rows)")

    # aggregates
    summary_rows = []
    for kind in ("within_author", "cross_author", "within_pool"):
        vals = df.loc[df["kind"] == kind, "burrows_delta"].tolist()
        if vals:
            summary_rows.append({"kind": kind, **_agg(vals)})
    if llm_same is not None:
        v = llm_same["burrows_delta"]
        summary_rows.append({"kind": "llm_within_run_same_metric", "n": 1,
                             "min": v, "p25": v, "median": v, "p75": v, "max": v, "mean": v, "std": 0.0})
    summary_rows.append({"kind": "llm_within_run_figure_scale_REF", "n": 1,
                         "min": LLM_FLOOR_FIGURE_SCALE, "p25": LLM_FLOOR_FIGURE_SCALE,
                         "median": LLM_FLOOR_FIGURE_SCALE, "p75": LLM_FLOOR_FIGURE_SCALE,
                         "max": LLM_FLOOR_FIGURE_SCALE, "mean": LLM_FLOOR_FIGURE_SCALE, "std": 0.0})
    sdf = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    sdf.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    print("\n=== Burrows' Δ — within/cross-author calibration (Reuters-50 / C50) ===")
    print("(within/cross/within_pool and llm_within_run_same_metric: compare_fingerprints.burrows_delta, "
          "100 MFW of the pair, 20 chunks/side. llm_within_run_figure_scale_REF=0.024 is the "
          "build_distance_matrix global-MFW implementation — a cross-reference, not on this scale.)")
    print(sdf.to_string(index=False))
    wa = next((r for r in summary_rows if r["kind"] == "within_author"), None)
    ca = next((r for r in summary_rows if r["kind"] == "cross_author"), None)
    if wa and ca:
        print(f"\nwithin-author median {wa['median']:.4f}  vs  cross-author median {ca['median']:.4f}  "
              f"→ ratio ≈ {ca['median']/wa['median']:.1f}× (stylometry separates authors as expected)")
    if llm_same is not None and wa:
        ratio = wa['median'] / llm_same['burrows_delta'] if llm_same['burrows_delta'] else float('nan')
        print(f"within-author median {wa['median']:.4f}  vs  LLM within-run (same metric) "
              f"{llm_same['burrows_delta']:.4f}  → within-human variation ≈ {ratio:.0f}× the LLM within-run floor")
    print(f"  [figure-scale cross-reference: {LLM_FLOOR_FIGURE_SCALE_NOTE}]")
    print("\nCaveat: Reuters-50 is corporate-news register, topic-controlled by construction — this is a "
          "pipeline calibration, NOT a within-human variance estimate for the Improta opinion-essay prompts. "
          "See data/human_baseline/README.md.")


if __name__ == "__main__":
    main()
