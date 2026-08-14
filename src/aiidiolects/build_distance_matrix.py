#!/usr/bin/env python3
"""Stage 2 A+B: model×model fingerprint distance matrix.

Computes pairwise distances over the 13-corpus AI-idiolects roster (6
Improta 2023 originals + Mistral-7B 2026 engine bridge + 6 Phase-2
production models) for 12 metrics:

* text length         — |Cohen's d| on per-text word counts
* sentence length     — |Cohen's d| on per-sentence word counts
* TTR                 — |corpus TTR_a - corpus TTR_b|
* MATTR (window=500)  — |MATTR_a - MATTR_b|
* hapax ratio         — |hapax_a - hapax_b|
* Yule's K            — |K_a - K_b|
* Jaccard top-100     — 1 - jaccard
* bigram JSD          — Jensen–Shannon divergence (top-500 bigrams, log2)
* trigram JSD         — Jensen–Shannon divergence (top-500 trigrams, log2)
* Burrows' Delta      — 100 MFW, 20 chunks per corpus
* Cosine Delta        — 100 MFW (Evert 2017), 20 chunks per corpus
* Zipf exponent       — |α_a - α_b| from log–log rank–frequency fit (top-1000)

Reuses the metric helpers from ``compare_fingerprints``; no metric
implementation is duplicated.

For each scope (pooled across topics + each topic individually) the
script writes:

* ``matrices/<metric>__<scope>.csv`` — symmetric 13×13 distance matrix
* ``all_distances_long.csv`` — long-format (metric, scope, model_a,
  model_b, distance) for downstream rendering / bootstrap.

Quality filter applied uniformly: rows with ``flag_repetition_loop``,
``flag_code_contam``, ``flag_non_english``, or ``flag_truncation`` are
dropped before tokenisation. This gives the cleanest cross-model
comparison; per-metric loosenings can be applied at the rendering step.

Usage::

    python3 -m aiidiolects.build_distance_matrix
    python3 -m aiidiolects.build_distance_matrix --base data \\
        --out-dir results/phase2_mapping/stage2_distance_matrix

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from aiidiolects.compare_fingerprints import (
    cohens_d,
    hapax_ratio,
    jaccard_top_n,
    jensen_shannon_divergence,
    mattr,
    ngram_distribution,
    sentence_lengths,
    text_lengths,
    tokenize,
    ttr,
    word_freq_vector,
    yules_k,
)
from aiidiolects.paths import DATA_DIR, RESULTS_DIR


# 13-corpus roster + era classification, mirrors visualize_stage1.py.
MODEL_ORDER = [
    # 2023 cohort (Improta originals)
    "phase0_gpt-3.5",
    "phase0_gpt-4o",
    "phase0_haiku",
    "phase0_llama-3-8b",
    "phase0_llama-3.1-70b",
    "phase0_mistral-7b",
    # 2023 → 2026 engine bridge (Run 5 = canonical anchor; Run 6 = stochastic
    # twin, kept in the matrix so the Phase 1 noise-floor analysis can use
    # the same global MFW + z-score reference as the §4.1 macro figure).
    "run5_lmstudio",
    "run6_lmstudio",
    # 2026 cohort (production roster)
    "gemini-3-flash",
    "gpt-5-4-mini",
    "haiku-4-5",
    "olmo3-7b",
    "nemo-12b",
    "qwen3-14b",
]
TOPICS = ["climate", "global_warming", "math_anxiety", "misinfo_health"]

QUALITY_FLAGS_DROP = [
    "flag_repetition_loop",
    "flag_code_contam",
    "flag_non_english",
    "flag_truncation",
]

# Metric key → (display name, unit hint). Order also defines column order
# in the long-format output.
METRICS = [
    ("text_length",     "|Cohen's d| on text length"),
    ("sentence_length", "|Cohen's d| on sentence length"),
    ("ttr",             "|TTR_a − TTR_b|"),
    ("mattr",           "|MATTR_a − MATTR_b| (w=500)"),
    ("hapax",           "|hapax_a − hapax_b|"),
    ("yules_k",         "|K_a − K_b|"),
    ("jaccard_top100",  "1 − Jaccard(top-100)"),
    ("bigram_jsd",      "JSD on top-500 bigrams"),
    ("trigram_jsd",     "JSD on top-500 trigrams"),
    ("burrows_delta",   "Burrows' Δ (100 MFW)"),
    ("cosine_delta",    "Cosine Δ (100 MFW; Evert 2017)"),
    ("zipf_exponent",   "|α_a − α_b| from log–log rank–freq fit"),
]


# ---------------------------------------------------------------------------
# Corpus discovery — mirrors clean_corpus / parse_corpus / descriptive_stats.
# ---------------------------------------------------------------------------

def model_dir_for(csv_path: Path) -> str:
    """Replicate the model_dir derivation in descriptive_stats.process_pair."""
    parent = csv_path.parent.parent.name
    if parent == "phase0_textualllmap":
        stem = csv_path.name[: -len(".cleaned.csv")]
        slug = stem.lower().replace(" ", "-")
        return f"phase0_{slug}"
    return parent


def find_cleaned_csvs(base: Path, include_human: bool = False) -> list[Path]:
    out: list[Path] = []
    phase2 = base / "phase2_mapping"
    if phase2.is_dir():
        for d in sorted(phase2.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("smoke_") or d.name == "archives":
                continue
            for csv in sorted(d.glob("*/*.cleaned.csv")):
                if "buggy-prompt" in csv.name:
                    continue
                out.append(csv)
    for run in ("run5_lmstudio", "run6_lmstudio"):
        anchor = base / "phase1_anchoring" / run
        if anchor.is_dir():
            for csv in sorted(anchor.glob("*/*.cleaned.csv")):
                if "buggy-prompt" in csv.name:
                    continue
                out.append(csv)
    originals = base / "phase0_textualllmap"
    if originals.is_dir():
        for csv in sorted(originals.glob("*/*.cleaned.csv")):
            if "(ITA)" in csv.name or csv.name.startswith("LLaMAntino"):
                continue
            out.append(csv)
    # Human baselines (opt-in via include_human). Webis-CMV-20 is a real
    # cross-author corpus and joins the distance roster (model_dir
    # ``webis_cmv_20``). Reuters-50/C50 is also discovered here so the shared
    # descriptive scripts (descriptive_stats / discourse_features) can pick it
    # up, but it is NOT in MODEL_ORDER — build_distance_matrix's roster filter
    # skips it (averaging 50 authors into one centroid is meaningless for a
    # cross-corpus matrix; the within-author analysis is within_author_baseline.py).
    if include_human:
        human = base / "human_baseline"
        if human.is_dir():
            for csv in sorted(human.glob("webis_cmv_20/*/Webis-CMV-20.cleaned.csv")):
                out.append(csv)
            for csv in sorted(human.glob("reuters_50/*/Reuters-50.cleaned.csv")):
                out.append(csv)
    return out


def load_corpus_texts(csv_path: Path) -> list[str]:
    """Read text_clean, drop quality-filtered rows, return plain text list."""
    df = pd.read_csv(csv_path)
    keep = pd.Series(True, index=df.index)
    for flag in QUALITY_FLAGS_DROP:
        if flag in df.columns:
            keep &= ~df[flag].astype(bool)
    return (
        df.loc[keep, "text_clean"]
        .fillna("")
        .astype(str)
        .tolist()
    )


# ---------------------------------------------------------------------------
# Per-corpus precompute (so each metric pair doesn't redo tokenisation).
# ---------------------------------------------------------------------------

@dataclass
class CorpusStats:
    model_dir: str
    topic: str            # "" for the pooled scope
    n_texts: int
    n_tokens: int
    token_lists: list[list[str]] = field(repr=False)
    text_lens: np.ndarray = field(repr=False)
    sent_lens: np.ndarray = field(repr=False)
    ttr: float = 0.0
    mattr: float = 0.0
    hapax: float = 0.0
    yules_k: float = 0.0
    top200_words: list[str] = field(default_factory=list, repr=False)
    bigram_counter: Counter = field(default_factory=Counter, repr=False)
    trigram_counter: Counter = field(default_factory=Counter, repr=False)
    zipf_exponent: float = 0.0


def zipf_exponent_fit(token_lists: list[list[str]], top_k: int = 1000) -> float:
    """Negative slope of log(rank) vs log(rel freq) on the top-k word ranks.

    Returns 0.0 for empty corpora; the magnitude is the Zipf-exponent
    estimate (typically 0.9–1.2 for natural English).
    """
    all_tokens = [t for tl in token_lists for t in tl]
    if not all_tokens:
        return 0.0
    freq = Counter(all_tokens)
    most_common = freq.most_common(top_k)
    if len(most_common) < 10:
        return 0.0
    ranks = np.arange(1, len(most_common) + 1)
    counts = np.array([c for _, c in most_common], dtype=float)
    rel = counts / counts.sum()
    log_rank = np.log(ranks)
    log_freq = np.log(rel)
    slope, _intercept, _r, _p, _se = stats.linregress(log_rank, log_freq)
    return float(-slope)   # report exponent as positive


def build_stats(
    model_dir: str, topic: str, texts: list[str]
) -> CorpusStats:
    token_lists = [tokenize(t) for t in texts]
    n_tokens = sum(len(tl) for tl in token_lists)
    top200_words, _ = word_freq_vector(token_lists, top_n=100)
    return CorpusStats(
        model_dir=model_dir,
        topic=topic,
        n_texts=len(texts),
        n_tokens=n_tokens,
        token_lists=token_lists,
        text_lens=text_lengths(token_lists),
        sent_lens=sentence_lengths(texts),
        ttr=ttr(token_lists),
        mattr=mattr(token_lists, window=500),
        hapax=hapax_ratio(token_lists),
        yules_k=yules_k(token_lists),
        top200_words=top200_words,
        bigram_counter=ngram_distribution(token_lists, n=2, top_k=500),
        trigram_counter=ngram_distribution(token_lists, n=3, top_k=500),
        zipf_exponent=zipf_exponent_fit(token_lists),
    )


# ---------------------------------------------------------------------------
# Burrows' Δ + Cosine Δ — built on a global MFW + per-text z-score backbone.
#
# Each text contributes one row to a (n_texts_total × n_features) relfreq
# matrix; we z-score per feature using the union of all 14 corpora's
# per-text vectors as the reference, then take each corpus's centroid as
# the mean of its z-scored per-text vectors. Pairwise Burrows Δ is the
# mean absolute difference between centroids; Cosine Δ (Evert 2017) is
# 1 − cosine_similarity between the same centroids.
#
# Why per-text rather than per-chunk: the previous design split each
# corpus's flat token stream into 20 consecutive chunks. That made the
# statistic *order-dependent* — shuffling the texts within a corpus
# changed Burrows Δ on the bridge pair from 0.143 to 0.241 because the
# chunk-grouping varied. With per-text vectors the statistic depends
# only on the multiset of texts (exchangeable), so the bootstrap is
# unbiased and the headline numbers don't move under arbitrary loading
# order. Per-text centroids are also the standard Burrows formulation
# for corpora with many short texts per author (Evert 2017 §3).
#
# Why global reference rather than per-pair: pooling only the two
# corpora being compared makes their z-score centroids mirror images
# by construction (their sum is zero on every feature), so Cosine Δ
# would always be 1 − (−1) = 2 and uninformative. Burrows under per-
# pair pooling still returns finite values but inflated against any
# global-reference baseline.
# ---------------------------------------------------------------------------

def per_text_relfreqs(
    token_lists: list[list[str]], features: list[str]
) -> np.ndarray:
    """Per-text MFW relative-frequency matrix, shape ``(n_texts,
    len(features))``. Empty / sub-tokenised texts contribute zero rows
    so the centroid math doesn't divide by zero."""
    if not token_lists or not features:
        return np.zeros((0, len(features)))
    feat_idx = {w: i for i, w in enumerate(features)}
    out = np.zeros((len(token_lists), len(features)))
    for r, tl in enumerate(token_lists):
        n = len(tl)
        if n == 0:
            continue
        for tok in tl:
            j = feat_idx.get(tok)
            if j is not None:
                out[r, j] += 1.0
        out[r, :] /= n
    return out


@dataclass
class DeltaState:
    """Shared state for the global-reference Burrows / Cosine Δ computation
    on one scope. Built once per scope; pair-wise distances become cheap."""
    features: list[str]
    centroids: dict[str, np.ndarray]    # model_dir -> centroid vector


def build_delta_state(
    stats_by_model: dict[str, "CorpusStats"],
    n_features: int = 100,
) -> DeltaState:
    # 1. Global MFW: top-N words across all corpora pooled.
    global_counter: Counter = Counter()
    for s in stats_by_model.values():
        for tl in s.token_lists:
            global_counter.update(tl)
    features = [w for w, _ in global_counter.most_common(n_features)]
    if not features:
        return DeltaState(features=[], centroids={})

    # 2. Per corpus: per-text relfreq vectors over the global features.
    per_corpus_vecs: dict[str, np.ndarray] = {}
    for m, s in stats_by_model.items():
        per_corpus_vecs[m] = per_text_relfreqs(s.token_lists, features)

    # 3. Global per-feature z-score stats from the union of all per-text
    # vectors (typically ~14 × 4000 = 56K rows for the pooled scope).
    all_rows = np.vstack([v for v in per_corpus_vecs.values() if v.size])
    if all_rows.shape[0] < 2:
        return DeltaState(features=features, centroids={})
    mu = all_rows.mean(axis=0)
    sigma = all_rows.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1e-12

    # 4. Per-corpus centroid: mean of z-scored per-text vectors.
    centroids = {
        m: ((v - mu) / sigma).mean(axis=0) if v.size else np.zeros(len(features))
        for m, v in per_corpus_vecs.items()
    }
    return DeltaState(features=features, centroids=centroids)


def burrows_delta_centroid(
    centroid_a: np.ndarray, centroid_b: np.ndarray
) -> float:
    return float(np.mean(np.abs(centroid_a - centroid_b)))


def cosine_delta_centroid(
    centroid_a: np.ndarray, centroid_b: np.ndarray
) -> float:
    na, nb = np.linalg.norm(centroid_a), np.linalg.norm(centroid_b)
    if na == 0 or nb == 0:
        return 0.0
    cos = float(np.dot(centroid_a, centroid_b) / (na * nb))
    return 1.0 - cos


# ---------------------------------------------------------------------------
# Pair distance — calls the metric helpers and returns one value per metric.
# ---------------------------------------------------------------------------

def pair_distances(
    a: CorpusStats,
    b: CorpusStats,
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
        "sentence_length": abs(cohens_d(a.sent_lens, b.sent_lens)),
        "ttr":             abs(a.ttr - b.ttr),
        "mattr":           abs(a.mattr - b.mattr),
        "hapax":           abs(a.hapax - b.hapax),
        "yules_k":         abs(a.yules_k - b.yules_k),
        "jaccard_top100":  1.0 - jaccard_top_n(a.top200_words, b.top200_words),
        "bigram_jsd":      jensen_shannon_divergence(
            a.bigram_counter, b.bigram_counter
        ),
        "trigram_jsd":     jensen_shannon_divergence(
            a.trigram_counter, b.trigram_counter
        ),
        "burrows_delta":   burrows,
        "cosine_delta":    cosine,
        "zipf_exponent":   abs(a.zipf_exponent - b.zipf_exponent),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_scope(
    corpora: dict[tuple[str, str], list[str]],
    model_order: list[str],
    scope: str,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Build distance matrices for one scope (a topic name or ``"pooled"``)."""
    print(f"\n=== scope: {scope} ===")
    stats_by_model: dict[str, CorpusStats] = {}

    if scope == "pooled":
        for m in model_order:
            texts: list[str] = []
            for tp in TOPICS:
                texts.extend(corpora.get((m, tp), []))
            if not texts:
                print(f"  WARN: empty pooled corpus for {m}")
                continue
            t0 = time.time()
            stats_by_model[m] = build_stats(m, "", texts)
            print(
                f"  precomputed {m:<24} n_texts={stats_by_model[m].n_texts:>4}"
                f"  tokens={stats_by_model[m].n_tokens:>8,}"
                f"  ({time.time()-t0:.1f}s)"
            )
    else:
        for m in model_order:
            texts = corpora.get((m, scope), [])
            if not texts:
                print(f"  WARN: empty corpus for {m}/{scope}")
                continue
            t0 = time.time()
            stats_by_model[m] = build_stats(m, scope, texts)
            print(
                f"  precomputed {m:<24} n_texts={stats_by_model[m].n_texts:>4}"
                f"  tokens={stats_by_model[m].n_tokens:>8,}"
                f"  ({time.time()-t0:.1f}s)"
            )

    present = [m for m in model_order if m in stats_by_model]
    matrices = {
        key: pd.DataFrame(
            np.zeros((len(present), len(present))),
            index=present,
            columns=present,
        )
        for key, _ in METRICS
    }
    long_rows: list[dict] = []

    # Global-reference Burrows / Cosine Δ state — built once per scope.
    t0 = time.time()
    delta_state = build_delta_state({m: stats_by_model[m] for m in present})
    print(
        f"  built delta-centroid state ({len(delta_state.features)} MFW, "
        f"{len(delta_state.centroids)} corpus centroids) in "
        f"{time.time() - t0:.1f}s"
    )

    n_pairs = len(present) * (len(present) - 1) // 2
    print(f"  computing {n_pairs} unique pairs across {len(METRICS)} metrics")
    t0 = time.time()
    done = 0
    for i, mi in enumerate(present):
        for j in range(i + 1, len(present)):
            mj = present[j]
            d = pair_distances(
                stats_by_model[mi], stats_by_model[mj], delta_state
            )
            for key, value in d.items():
                matrices[key].loc[mi, mj] = value
                matrices[key].loc[mj, mi] = value
                long_rows.append({
                    "metric": key,
                    "scope": scope,
                    "model_a": mi,
                    "model_b": mj,
                    "distance": value,
                })
            done += 1
            if done % 10 == 0 or done == n_pairs:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (n_pairs - done) / max(rate, 1e-6)
                print(
                    f"    pair {done:>3}/{n_pairs}   "
                    f"{rate:.2f} pair/s  eta {eta:.0f}s"
                )
    print(f"  scope {scope} done in {time.time()-t0:.1f}s")
    return matrices, long_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage2_distance_matrix"),
    )
    ap.add_argument(
        "--scopes",
        default="pooled,climate,global_warming,math_anxiety,misinfo_health",
        help="comma-separated list of scopes to build "
             "(default: pooled + all four topics)",
    )
    ap.add_argument(
        "--include-human", action="store_true",
        help="also include the Webis-CMV-20 human baseline as a 14th corpus "
             "(appended to MODEL_ORDER). Use with a separate --out-dir (e.g. "
             "..._with_human) so the canonical 13-corpus matrices are not "
             "overwritten. Reuters-50 is never included here — see "
             "within_author_baseline.py.",
    )
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    matrices_dir = out_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)

    model_order = MODEL_ORDER + (["webis_cmv_20"] if args.include_human else [])

    # 1. Discover and load every (model_dir, topic) corpus once.
    csvs = find_cleaned_csvs(base, include_human=args.include_human)
    print(f"Found {len(csvs)} cleaned CSV(s) under {base}"
          + ("  (incl. Webis-CMV-20 human baseline)" if args.include_human else ""))
    if args.include_human and out_dir.name == "stage2_distance_matrix":
        print("  NOTE: --include-human is writing into the canonical out-dir; "
              "the 13-corpus matrices there will be overwritten with 14-corpus ones. "
              "Pass --out-dir .../stage2_distance_matrix_with_human to keep both.")
    corpora: dict[tuple[str, str], list[str]] = {}
    for csv in csvs:
        md = model_dir_for(csv)
        topic = csv.parent.name
        corpora[(md, topic)] = load_corpus_texts(csv)

    present_models = sorted({md for md, _ in corpora})
    in_roster = [m for m in model_order if m in present_models]
    out_of_roster = [m for m in present_models if m not in model_order]
    print(f"In-roster corpora ({len(in_roster)}): {in_roster}")
    if out_of_roster:
        print(f"Skipped (not in roster): {out_of_roster}")

    # 2. Build matrices per scope.
    all_long: list[dict] = []
    for scope in [s.strip() for s in args.scopes.split(",") if s.strip()]:
        matrices, rows = build_scope(corpora, in_roster, scope)
        all_long.extend(rows)
        for key, df in matrices.items():
            out_path = matrices_dir / f"{key}__{scope}.csv"
            df.to_csv(out_path, float_format="%.6f")
            print(f"    wrote {out_path}")

    long_df = pd.DataFrame(all_long)
    long_path = out_dir / "all_distances_long.csv"
    long_df.to_csv(long_path, index=False, float_format="%.6f")
    print(f"\nWrote long-format distances: {long_path}  ({len(long_df):,} rows)")


if __name__ == "__main__":
    main()
