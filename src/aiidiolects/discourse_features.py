#!/usr/bin/env python3
"""Stage 2 C: punctuation profile + closed-list discourse markers / hedges.

Computes per-corpus rates per 1000 words for three families:

* **Punctuation** (raw character search): em-dash (U+2014), en-dash
  (U+2013), semicolon, colon, ellipsis (Unicode … or three ASCII dots),
  exclamation, question mark, comma, opening parenthesis. Targets the
  "GPT em-dash" cliché empirically — and the era frame (GPT-3.5 / 4o
  in 2023, GPT-5.4 Mini in 2026) makes the cliché directly testable.
* **Discourse markers** (case-insensitive whole-word regex): *however,
  furthermore, moreover, indeed, importantly, crucially, notably,
  ultimately*. Targets argumentative-essay register tendencies.
* **Hedges** (case-insensitive whole-word regex, with ``tend to`` as a
  bigram): *might, could, possibly, perhaps, arguably, suggest, tend
  to*. Targets confidence/qualification stance.

The same Stage 1a quality filter applies as in build_distance_matrix.py:
``flag_repetition_loop``, ``flag_code_contam``, ``flag_non_english``, and
``flag_truncation`` rows are dropped before counting.

Outputs (under
``results/phase2_mapping/stage2_discourse_features/``):

* ``rates_long.csv`` — one row per (model_dir, topic, feature) plus a
  ``pooled`` topic that aggregates over all four. Tracks count, total
  words, and rate per 1000 words.
* ``matrices/<category>__<scope>.csv`` — wide-format model × feature
  rate tables for each of the three categories × five scopes (4 topics
  + pooled).

Usage::

    python3 -m aiidiolects.discourse_features
    python3 -m aiidiolects.discourse_features --base data \\
        --out-dir results/phase2_mapping/stage2_discourse_features

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from aiidiolects.build_distance_matrix import (
    QUALITY_FLAGS_DROP,
    TOPICS,
    find_cleaned_csvs,
    model_dir_for,
)
from aiidiolects.paths import DATA_DIR, RESULTS_DIR


# Punctuation: raw substring search (no word boundaries).
PUNCTUATION_PATTERNS: dict[str, str] = {
    "em_dash":     r"—",        # — (U+2014)
    "en_dash":     r"–",        # – (U+2013)
    "semicolon":   r";",
    "colon":       r":",
    "ellipsis":    r"…|\.\.\.",  # … or three dots
    "exclamation": r"!",
    "question":    r"\?",
    "comma":       r",",
    "open_paren":  r"\(",
}

# Discourse markers — closed list, exact word match, case-insensitive.
DISCOURSE_WORDS: list[str] = [
    "however", "furthermore", "moreover", "indeed",
    "importantly", "crucially", "notably", "ultimately",
]

# Hedges — closed list. ``tend to`` is the only multi-word entry.
HEDGE_TOKENS: list[str] = [
    "might", "could", "possibly", "perhaps",
    "arguably", "suggest", "tend to",
]


def _build_word_regex(token: str) -> re.Pattern:
    """Word-boundary, case-insensitive regex. ``tend to`` becomes a
    flexible-whitespace bigram pattern."""
    parts = [re.escape(p) for p in token.split()]
    body = r"\s+".join(parts)
    return re.compile(rf"\b{body}\b", flags=re.IGNORECASE)


PUNCT_REGEXES = {
    name: re.compile(pat) for name, pat in PUNCTUATION_PATTERNS.items()
}
DISCOURSE_REGEXES = {w: _build_word_regex(w) for w in DISCOURSE_WORDS}
HEDGE_REGEXES = {h: _build_word_regex(h) for h in HEDGE_TOKENS}


CATEGORY_FEATURES: dict[str, list[str]] = {
    "punctuation":         list(PUNCTUATION_PATTERNS),
    "discourse_markers":   list(DISCOURSE_WORDS),
    "hedges":              list(HEDGE_TOKENS),
}
CATEGORY_REGEXES: dict[str, dict[str, re.Pattern]] = {
    "punctuation":       PUNCT_REGEXES,
    "discourse_markers": DISCOURSE_REGEXES,
    "hedges":            HEDGE_REGEXES,
}


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def count_features(text: str) -> dict[str, int]:
    """Return a flat dict of feature_name -> raw count for one text."""
    counts: dict[str, int] = {}
    for category, regexes in CATEGORY_REGEXES.items():
        for name, pattern in regexes.items():
            counts[name] = len(pattern.findall(text))
    return counts


def feature_to_category(feature: str) -> str:
    for cat, feats in CATEGORY_FEATURES.items():
        if feature in feats:
            return cat
    return "other"


def aggregate_corpus(
    df: pd.DataFrame, model_dir: str, topic: str
) -> list[dict]:
    """Return one row per feature for the given (model_dir, topic). Applies
    the Stage-2 quality filter before counting."""
    keep = pd.Series(True, index=df.index)
    for flag in QUALITY_FLAGS_DROP:
        if flag in df.columns:
            keep &= ~df[flag].astype(bool)
    sub = df.loc[keep].copy()
    sub["text_clean"] = sub["text_clean"].fillna("").astype(str)

    # Word-count denominator: whitespace tokens of the cleaned text.
    word_counts = sub["text_clean"].map(lambda t: len(t.split()))
    total_words = int(word_counts.sum())
    if total_words == 0:
        return []

    feature_counts: dict[str, int] = {f: 0 for cat in CATEGORY_FEATURES
                                       for f in CATEGORY_FEATURES[cat]}
    for text in sub["text_clean"]:
        for f, c in count_features(text).items():
            feature_counts[f] += c

    rows = []
    for feature, count in feature_counts.items():
        rows.append({
            "model_dir":      model_dir,
            "topic":          topic,
            "category":       feature_to_category(feature),
            "feature":        feature,
            "count":          count,
            "total_words":    total_words,
            "rate_per_1000":  1000.0 * count / total_words,
        })
    return rows


# ---------------------------------------------------------------------------
# Matrix pivoting (one wide-format CSV per category × scope).
# ---------------------------------------------------------------------------

def write_category_matrices(
    long_df: pd.DataFrame, out_dir: Path
) -> None:
    matrices_dir = out_dir / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    for category in CATEGORY_FEATURES:
        for scope in long_df["topic"].unique():
            sub = long_df[
                (long_df["category"] == category)
                & (long_df["topic"] == scope)
            ]
            if sub.empty:
                continue
            wide = sub.pivot(
                index="model_dir", columns="feature", values="rate_per_1000"
            ).reindex(columns=CATEGORY_FEATURES[category])
            out_path = matrices_dir / f"{category}__{scope}.csv"
            wide.to_csv(out_path, float_format="%.4f")
            print(f"    wrote {out_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument(
        "--out-dir",
        default=str(
            RESULTS_DIR / "phase2_mapping" / "stage2_discourse_features"
        ),
    )
    ap.add_argument(
        "--include-human", action="store_true",
        help="also include the human baselines (Webis-CMV-20 + Reuters-50) — "
             "requires clean_corpus.py to have produced their .cleaned.csv. Use "
             "with a separate --out-dir so the canonical 13-corpus output is kept.",
    )
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = find_cleaned_csvs(base, include_human=args.include_human)
    print(f"Found {len(csvs)} cleaned CSV(s) under {base}"
          + ("  (incl. human baselines)" if args.include_human else ""))

    rows: list[dict] = []
    # Per (model_dir, topic).
    for csv in csvs:
        md = model_dir_for(csv)
        topic = csv.parent.name
        df = pd.read_csv(csv)
        rows.extend(aggregate_corpus(df, md, topic))

    # Pooled across topics: combine cleaned dataframes per model_dir.
    by_model: dict[str, list[pd.DataFrame]] = {}
    for csv in csvs:
        md = model_dir_for(csv)
        by_model.setdefault(md, []).append(pd.read_csv(csv))
    for md, frames in by_model.items():
        merged = pd.concat(frames, ignore_index=True)
        rows.extend(aggregate_corpus(merged, md, "pooled"))

    long_df = pd.DataFrame(rows)
    long_path = out_dir / "rates_long.csv"
    long_df.to_csv(long_path, index=False, float_format="%.4f")
    print(f"\nWrote {long_path}  ({len(long_df):,} rows)")

    write_category_matrices(long_df, out_dir)


if __name__ == "__main__":
    main()
