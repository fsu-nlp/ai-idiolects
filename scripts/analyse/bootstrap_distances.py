#!/usr/bin/env python3
"""Stage 2 D: bootstrap CIs over the model×model distance matrix.

For each (scope × metric × pair), resample 80% of each corpus's texts
without replacement, recompute the corpus statistics + the global
MFW + z-score delta state from the resamples, and compute the
pairwise distance. Repeat N times (default 1000) to obtain a
bootstrap distribution per metric per pair.

Outputs:

* ``bootstrap_summary.csv`` — one row per (scope × metric × ordered
  pair) with bootstrap mean, std, 2.5 / 50 / 97.5 percentiles, and
  the point-estimate distance from the full-data matrix
  (``all_distances_long.csv``). Used to attach 95% CIs to the §4.1
  figure and to flag pairs whose CI overlaps zero.
* ``bootstrap_engine_bridge_long.csv`` — full distribution
  (rep × metric × scope) for the engine-bridge pair only
  (phase0_mistral-7b ↔ run5_lmstudio). Small enough to commit; lets
  downstream plotting visualise the noise floor density.

Notes:

* MATTR is **excluded** from the bootstrap. Its sliding-window
  computation costs ~1-2 sec per corpus per rep and the bootstrap
  CI is essentially zero at this corpus size; the full-data MATTR
  remains in the matrix.
* Resampling unit is the text. The global MFW (top-100 across all
  14 corpora pooled) is recomputed per rep so that the CI on
  Burrows / Cosine reflects feature-selection variability as well
  as centroid variability.

Hardware: single-process by design on the 16 GB laptop. ~30 s/rep
for the pooled scope (~14 corpora × ~3,200 texts after 80 % resample),
~6 s/rep per topic. Total wall time at N=300 across all 5 scopes:
~4 hours (pooled 2.5 h + 4 topics × 30 min). Multi-process tempting
but unsafe at this RAM budget — even fork copy-on-write fails because
Python's per-object refcount writes duplicate the inherited bases
dict (~1.5 GB) per worker; 4 workers OOM-kill the box. See
``--n-workers`` help text.

Usage::

    python3 scripts/analyse/bootstrap_distances.py
    python3 scripts/analyse/bootstrap_distances.py --n-reps 200 \\
        --scopes pooled         # quick smoke test (~2 min)

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from aiidiolects.compare_fingerprints import (
    cohens_d,
    hapax_ratio,
    jaccard_top_n,
    jensen_shannon_divergence,
    ngram_distribution,
    sentence_lengths,
    tokenize,
    ttr,
    word_freq_vector,
    yules_k,
)
from aiidiolects.build_distance_matrix import (
    MODEL_ORDER,
    QUALITY_FLAGS_DROP,
    TOPICS,
    DeltaState,
    build_delta_state,
    burrows_delta_centroid,
    cosine_delta_centroid,
    find_cleaned_csvs,
    model_dir_for,
    per_text_relfreqs,
    zipf_exponent_fit,
)
from aiidiolects.paths import DATA_DIR, RESULTS_DIR


# Bootstrap excludes MATTR (slow + tiny CI at this scale). Order matches the
# bootstrap_summary.csv column ordering. See module docstring.
BOOTSTRAP_METRICS = [
    "text_length",
    "sentence_length",
    "ttr",
    "hapax",
    "yules_k",
    "jaccard_top100",
    "bigram_jsd",
    "trigram_jsd",
    "burrows_delta",
    "cosine_delta",
    "zipf_exponent",
]


# ---------------------------------------------------------------------------
# Per-corpus base data — pre-tokenised, indexable per text.
# ---------------------------------------------------------------------------

@dataclass
class CorpusBase:
    model_dir: str
    topic: str
    texts: list[str]                          = field(repr=False)
    token_lists_per_text: list[list[str]]     = field(repr=False)
    text_len_per_text: np.ndarray             = field(repr=False)
    sent_lens_per_text: list[np.ndarray]      = field(repr=False)


def build_corpus_base(model_dir: str, topic: str, texts: list[str]) -> CorpusBase:
    token_lists = [tokenize(t) for t in texts]
    text_len = np.array([len(tl) for tl in token_lists])
    # Per-text sentence lengths so resampling can splice them.
    sent_lens: list[np.ndarray] = []
    from aiidiolects.compare_fingerprints import sentencize
    for t in texts:
        lens = []
        for s in sentencize(t):
            lens.append(len(tokenize(s)))
        sent_lens.append(np.array(lens, dtype=int))
    return CorpusBase(
        model_dir=model_dir,
        topic=topic,
        texts=texts,
        token_lists_per_text=token_lists,
        text_len_per_text=text_len,
        sent_lens_per_text=sent_lens,
    )


# ---------------------------------------------------------------------------
# Per-corpus stats from a (possibly resampled) slice of the base.
# ---------------------------------------------------------------------------

@dataclass
class CorpusStatsLite:
    """Mirror of CorpusStats in build_distance_matrix but without MATTR."""
    model_dir: str
    topic: str
    n_texts: int
    n_tokens: int
    token_lists: list[list[str]]
    text_lens: np.ndarray
    sent_lens: np.ndarray
    ttr: float
    hapax: float
    yules_k: float
    top200_words: list[str]
    bigram_counter: Counter
    trigram_counter: Counter
    zipf_exponent: float


def stats_from_indices(
    base: CorpusBase, indices: np.ndarray
) -> CorpusStatsLite:
    sub_token_lists = [base.token_lists_per_text[i] for i in indices]
    sub_text_lens = base.text_len_per_text[indices]
    if base.sent_lens_per_text:
        per = [base.sent_lens_per_text[i] for i in indices if base.sent_lens_per_text[i].size]
        sub_sent_lens = np.concatenate(per) if per else np.array([], dtype=int)
    else:
        sub_sent_lens = np.array([], dtype=int)
    n_tokens = int(sub_text_lens.sum())
    top200, _ = word_freq_vector(sub_token_lists, top_n=100)
    return CorpusStatsLite(
        model_dir=base.model_dir,
        topic=base.topic,
        n_texts=len(sub_token_lists),
        n_tokens=n_tokens,
        token_lists=sub_token_lists,
        text_lens=sub_text_lens,
        sent_lens=sub_sent_lens,
        ttr=ttr(sub_token_lists),
        hapax=hapax_ratio(sub_token_lists),
        yules_k=yules_k(sub_token_lists),
        top200_words=top200,
        bigram_counter=ngram_distribution(sub_token_lists, n=2, top_k=500),
        trigram_counter=ngram_distribution(sub_token_lists, n=3, top_k=500),
        zipf_exponent=zipf_exponent_fit(sub_token_lists),
    )


# ---------------------------------------------------------------------------
# Build a delta state directly from CorpusStatsLite (mirrors
# build_distance_matrix.build_delta_state but typed for the lite stats).
# ---------------------------------------------------------------------------

def build_delta_state_lite(
    stats_by_model: dict[str, CorpusStatsLite],
    n_features: int = 100,
) -> DeltaState:
    """Per-text-vector global-reference delta state. Mirrors
    ``build_delta_state`` in build_distance_matrix.py exactly so the
    bootstrap statistic and the point estimate agree."""
    global_counter: Counter = Counter()
    for s in stats_by_model.values():
        for tl in s.token_lists:
            global_counter.update(tl)
    features = [w for w, _ in global_counter.most_common(n_features)]
    if not features:
        return DeltaState(features=[], centroids={})

    per_corpus_vecs: dict[str, np.ndarray] = {}
    for m, s in stats_by_model.items():
        per_corpus_vecs[m] = per_text_relfreqs(s.token_lists, features)

    all_rows = np.vstack([v for v in per_corpus_vecs.values() if v.size])
    if all_rows.shape[0] < 2:
        return DeltaState(features=features, centroids={})
    mu = all_rows.mean(axis=0)
    sigma = all_rows.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1e-12

    centroids = {
        m: ((v - mu) / sigma).mean(axis=0) if v.size else np.zeros(len(features))
        for m, v in per_corpus_vecs.items()
    }
    return DeltaState(features=features, centroids=centroids)


# ---------------------------------------------------------------------------
# Pair distances under the lite stats.
# ---------------------------------------------------------------------------

def pair_distances_lite(
    a: CorpusStatsLite,
    b: CorpusStatsLite,
    delta_state: DeltaState,
) -> dict[str, float]:
    ca = delta_state.centroids.get(a.model_dir)
    cb = delta_state.centroids.get(b.model_dir)
    if ca is None or cb is None:
        burrows = cosine = 0.0
    else:
        burrows = burrows_delta_centroid(ca, cb)
        cosine = cosine_delta_centroid(ca, cb)
    return {
        "text_length":     abs(cohens_d(a.text_lens, b.text_lens)),
        "sentence_length": abs(cohens_d(a.sent_lens, b.sent_lens)) if a.sent_lens.size and b.sent_lens.size else 0.0,
        "ttr":             abs(a.ttr - b.ttr),
        "hapax":           abs(a.hapax - b.hapax),
        "yules_k":         abs(a.yules_k - b.yules_k),
        "jaccard_top100":  1.0 - jaccard_top_n(a.top200_words, b.top200_words),
        "bigram_jsd":      jensen_shannon_divergence(a.bigram_counter, b.bigram_counter),
        "trigram_jsd":     jensen_shannon_divergence(a.trigram_counter, b.trigram_counter),
        "burrows_delta":   burrows,
        "cosine_delta":    cosine,
        "zipf_exponent":   abs(a.zipf_exponent - b.zipf_exponent),
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

# Roster including the Phase-2b human cross-author corpus (opt-in). Reuters
# never joins — 50 authors averaged into one centroid is meaningless for a
# cross-corpus matrix (its within-author distribution is within_author_baseline.py).
ROSTER_WITH_HUMAN = MODEL_ORDER + ["webis_cmv_20"]


def load_bases(
    base_path: Path, scope: str, include_human: bool = False,
) -> dict[str, CorpusBase]:
    """Build CorpusBase per model_dir in the roster for the given scope.
    For ``pooled``, concatenate texts across topics. With ``include_human``
    the Webis-CMV-20 corpus joins the roster."""
    csvs = find_cleaned_csvs(base_path, include_human=include_human)
    by_model: dict[str, list[str]] = {}
    for csv in csvs:
        md = model_dir_for(csv)
        topic = csv.parent.name
        if scope != "pooled" and topic != scope:
            continue
        df = pd.read_csv(csv)
        keep = pd.Series(True, index=df.index)
        for flag in QUALITY_FLAGS_DROP:
            if flag in df.columns:
                keep &= ~df[flag].astype(bool)
        sub_texts = (
            df.loc[keep, "text_clean"].fillna("").astype(str).tolist()
        )
        by_model.setdefault(md, []).extend(sub_texts)

    roster = ROSTER_WITH_HUMAN if include_human else MODEL_ORDER
    bases: dict[str, CorpusBase] = {}
    for m in roster:
        texts = by_model.get(m)
        if not texts:
            continue
        bases[m] = build_corpus_base(m, scope, texts)
    return bases


# Module-level state for the worker pool. The previous design passed
# ``bases`` (~1.5 GB for the pooled scope) as an arg to every Pool call;
# multiprocessing pickled it ~1000× per scope, which OOM-killed the
# session on a 16 GB laptop. We now stash the heavy data in module
# globals *before* the Pool is created and rely on fork()'s
# copy-on-write so children share the parent's pages without
# materialising a copy. Workers only receive small per-rep args
# (rep id + seed) over the pipe.
_BASES: dict[str, CorpusBase] = {}
_PAIRS: list[tuple[str, str]] = []
_SCOPE: str = ""
_SAMPLE_FRAC: float = 0.80
_WITH_REPLACEMENT: bool = True


def _set_global_state(
    bases: dict[str, CorpusBase],
    pairs: list[tuple[str, str]],
    scope: str,
    sample_frac: float,
    with_replacement: bool,
) -> None:
    global _BASES, _PAIRS, _SCOPE, _SAMPLE_FRAC, _WITH_REPLACEMENT
    _BASES = bases
    _PAIRS = pairs
    _SCOPE = scope
    _SAMPLE_FRAC = sample_frac
    _WITH_REPLACEMENT = with_replacement


BRIDGE_PAIR = ("phase0_mistral-7b", "run5_lmstudio")


def _one_rep(
    args: tuple[int, int],
) -> tuple[dict[tuple[str, str, str], float], list[dict]]:
    """Compute one bootstrap rep. Reads ``_BASES``, ``_PAIRS``,
    ``_SCOPE``, ``_SAMPLE_FRAC`` from module globals (inherited by
    fork). Designed for multiprocessing.Pool with chunksize=1."""
    rep, seed = args
    rng = np.random.default_rng(seed)
    present = [m for m in _BASES]

    sub_stats: dict[str, CorpusStatsLite] = {}
    for m in present:
        base = _BASES[m]
        n = len(base.texts)
        # Standard nonparametric bootstrap: same size as the original
        # corpus, sampled WITH replacement. The earlier 80% without-
        # replacement design biased Burrows / Cosine upward by ~70 %
        # because shrinking the corpus also shrunk each of the 20
        # Burrows chunks, which inflated per-chunk frequency noise and
        # therefore the z-score-centroid distance. Sampling with
        # replacement keeps total tokens at full-data scale, chunks
        # full-size, and the bootstrap mean centered on the point
        # estimate. ``--sample-frac`` is retained for diagnostics but
        # has no effect when ``--with-replacement`` is on (the default).
        if _WITH_REPLACEMENT:
            k = n
            idx = rng.choice(n, size=k, replace=True)
        else:
            k = max(1, int(round(_SAMPLE_FRAC * n)))
            idx = rng.choice(n, size=k, replace=False)
        sub_stats[m] = stats_from_indices(base, idx)

    delta_state = build_delta_state_lite(sub_stats)

    out: dict[tuple[str, str, str], float] = {}
    bridge_rows: list[dict] = []
    for mi, mj in _PAIRS:
        d = pair_distances_lite(sub_stats[mi], sub_stats[mj], delta_state)
        for k, v in d.items():
            out[(mi, mj, k)] = v
        if (mi, mj) == BRIDGE_PAIR:
            for k, v in d.items():
                bridge_rows.append({
                    "rep": rep, "scope": _SCOPE, "metric": k,
                    "model_a": mi, "model_b": mj, "distance": v,
                })
    return out, bridge_rows


def run_bootstrap_one_scope(
    bases: dict[str, CorpusBase],
    scope: str,
    n_reps: int,
    sample_frac: float,
    seed: int,
    n_workers: int,
    with_replacement: bool,
) -> tuple[dict[tuple[str, str, str], list[float]], list[dict]]:
    """Returns (per_pair_metric_distribution, engine_bridge_long_rows)."""
    present = [m for m in ROSTER_WITH_HUMAN if m in bases]
    pairs = [(present[i], present[j])
             for i in range(len(present))
             for j in range(i + 1, len(present))]

    # Stash heavy data in module globals so forked workers inherit it
    # via copy-on-write rather than receiving a pickled copy per call.
    _set_global_state(bases, pairs, scope, sample_frac, with_replacement)

    dist_per_pair_metric: dict[tuple[str, str, str], list[float]] = {
        (mi, mj, k): []
        for (mi, mj) in pairs
        for k in BOOTSTRAP_METRICS
    }
    bridge_rows_all: list[dict] = []

    # Per-rep RNG seed: deterministic from (scope, seed, rep) so reruns are
    # reproducible regardless of worker scheduling.
    base_seed = abs(hash((scope, seed))) % (2**31)
    rep_args = [(rep, base_seed + rep) for rep in range(n_reps)]

    t0 = time.time()
    done = 0
    if n_workers <= 1:
        for args in rep_args:
            out, bridge = _one_rep(args)
            for key, v in out.items():
                dist_per_pair_metric[key].append(v)
            bridge_rows_all.extend(bridge)
            done += 1
            if done % 25 == 0 or done == n_reps:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n_reps - done) / max(rate, 1e-6)
                print(
                    f"    {scope:>15}  rep {done:>4}/{n_reps}   "
                    f"{rate:.2f} rep/s  eta {eta:.0f}s",
                    flush=True,
                )
    else:
        # Force fork start method (Linux default; explicit for safety).
        ctx = get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for out, bridge in pool.imap_unordered(_one_rep, rep_args, chunksize=1):
                for key, v in out.items():
                    dist_per_pair_metric[key].append(v)
                bridge_rows_all.extend(bridge)
                done += 1
                if done % 25 == 0 or done == n_reps:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-6)
                    eta = (n_reps - done) / max(rate, 1e-6)
                    print(
                        f"    {scope:>15}  rep {done:>4}/{n_reps}   "
                        f"{rate:.2f} rep/s  eta {eta:.0f}s  "
                        f"({n_workers} workers)",
                        flush=True,
                    )

    return dist_per_pair_metric, bridge_rows_all


def summarise(
    dist_per_pair_metric: dict[tuple[str, str, str], list[float]],
    scope: str,
    point_long_df: pd.DataFrame,
) -> list[dict]:
    """Build summary rows: mean / std / quantiles + the point-estimate."""
    rows = []
    for (mi, mj, metric), values in dist_per_pair_metric.items():
        arr = np.asarray(values)
        # Look up the full-data point estimate.
        pe = point_long_df[
            (point_long_df["scope"] == scope)
            & (point_long_df["metric"] == metric)
            & (point_long_df["model_a"] == mi)
            & (point_long_df["model_b"] == mj)
        ]
        if pe.empty:
            pe = point_long_df[
                (point_long_df["scope"] == scope)
                & (point_long_df["metric"] == metric)
                & (point_long_df["model_a"] == mj)
                & (point_long_df["model_b"] == mi)
            ]
        point = float(pe["distance"].iloc[0]) if not pe.empty else float("nan")
        rows.append({
            "scope":      scope,
            "metric":     metric,
            "model_a":    mi,
            "model_b":    mj,
            "point":      point,
            "boot_mean":  float(arr.mean()),
            "boot_std":   float(arr.std(ddof=1)),
            "boot_2.5":   float(np.percentile(arr, 2.5)),
            "boot_50":    float(np.percentile(arr, 50)),
            "boot_97.5": float(np.percentile(arr, 97.5)),
            "n_reps":     int(len(arr)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument(
        "--include-human", action="store_true",
        help="add the Phase-2b Webis-CMV-20 corpus to the bootstrap roster "
             "(Reuters never joins). Defaults --out-dir / --point-long-csv to "
             "the stage2_distance_matrix_with_human/ paths. Build that matrix "
             "first with build_distance_matrix.py --include-human. Best run on "
             "a >16 GB box with --n-workers >1.",
    )
    ap.add_argument("--out-dir", default=None,
                    help="default: stage2_distance_matrix[/_with_human]/bootstrap")
    ap.add_argument("--point-long-csv", default=None,
                    help="default: stage2_distance_matrix[/_with_human]/all_distances_long.csv")
    ap.add_argument("--n-reps", type=int, default=1000)
    ap.add_argument("--sample-frac", type=float, default=0.80,
                    help="ignored when --with-replacement is on (the default)")
    ap.add_argument("--with-replacement", action="store_true", default=True,
                    help="standard nonparametric bootstrap (default ON)")
    ap.add_argument("--without-replacement", dest="with_replacement",
                    action="store_false",
                    help="diagnostic: 80%% subsample without replacement, "
                         "DO NOT USE for chapter CIs (biases distance metrics)")
    ap.add_argument("--seed", type=int, default=20260510)
    ap.add_argument(
        "--scopes",
        default="pooled,climate,global_warming,math_anxiety,misinfo_health",
    )
    ap.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help=(
            "parallel workers (default: 1 — single-process). "
            "DO NOT raise on a 16 GB laptop. Reason: even with fork "
            "(copy-on-write), Python's per-object refcount writes "
            "duplicate every page of the inherited ~1.5 GB bases dict, "
            "so each worker ends up with its own ~3 GB RSS and 4 "
            "workers OOM-kill the box. Multi-process is only safe with "
            "either (a) numpy-encoded token arrays (no per-element "
            "refcount) or (b) a host with >32 GB RAM."
        ),
    )
    args = ap.parse_args()
    print(f"Using n_workers={args.n_workers}", flush=True)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal", "MemAvailable", "SwapTotal")):
                    print(f"  {line.strip()}", flush=True)
    except OSError:
        pass

    base = Path(args.base)
    _mat = (
        str(RESULTS_DIR / "phase2_mapping" / "stage2_distance_matrix_with_human")
        if args.include_human
        else str(RESULTS_DIR / "phase2_mapping" / "stage2_distance_matrix")
    )
    out_dir = Path(args.out_dir or f"{_mat}/bootstrap")
    out_dir.mkdir(parents=True, exist_ok=True)
    point_long_csv = args.point_long_csv or f"{_mat}/all_distances_long.csv"

    point_long = pd.read_csv(point_long_csv)
    print(f"Loaded {len(point_long):,} point-estimate distances from "
          f"{point_long_csv}")

    summary_rows: list[dict] = []
    bridge_rows_all: list[dict] = []

    for scope in [s.strip() for s in args.scopes.split(",") if s.strip()]:
        print(f"\n=== bootstrap scope: {scope}  "
              f"(N={args.n_reps}, frac={args.sample_frac}) ===")
        bases = load_bases(base, scope, include_human=args.include_human)
        present = [m for m in ROSTER_WITH_HUMAN if m in bases]
        print(f"  {len(present)} corpora present: {present}")
        dist, bridge_rows = run_bootstrap_one_scope(
            bases, scope, args.n_reps, args.sample_frac, args.seed,
            args.n_workers, args.with_replacement,
        )
        summary_rows.extend(summarise(dist, scope, point_long))
        bridge_rows_all.extend(bridge_rows)

    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "bootstrap_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"\nWrote {summary_path}  ({len(summary):,} rows)")

    bridge_long = pd.DataFrame(bridge_rows_all)
    bridge_path = out_dir / "bootstrap_engine_bridge_long.csv"
    bridge_long.to_csv(bridge_path, index=False, float_format="%.6f")
    print(f"Wrote {bridge_path}  ({len(bridge_long):,} rows)")


if __name__ == "__main__":
    main()
