#!/usr/bin/env python3
"""Stage 1c: parse-aware descriptive statistics.

Reads each ``<Model>.cleaned.csv`` and the matching ``<Model>.spacy`` DocBin
produced by Stages 1a and 1b. Computes per-text counts (tokens, alphabetic
words, sentences, lemma types, TTR, mean sentence length) and aggregates them
per (model_dir × topic), with vocabulary size accumulated only from
quality-clean texts (excluding repetition loops, code contamination,
non-English; truncated texts kept for length-independent aggregates).

Outputs::

    results/phase2_mapping/stage1_descriptives/per_text.csv
    results/phase2_mapping/stage1_descriptives/summary.csv

A short text summary is also printed to stdout for at-a-glance review.

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import Doc, DocBin

from aiidiolects.paths import DATA_DIR, RESULTS_DIR


def find_pairs(base: Path, include_human: bool = False) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
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
                stem = csv.name[: -len(".cleaned.csv")]
                sp = csv.with_name(stem + ".spacy")
                if sp.exists():
                    pairs.append((csv, sp))
    for run in ("run5_lmstudio", "run6_lmstudio"):
        anchor = base / "phase1_anchoring" / run
        if anchor.is_dir():
            for csv in sorted(anchor.glob("*/*.cleaned.csv")):
                if "buggy-prompt" in csv.name:
                    continue
                stem = csv.name[: -len(".cleaned.csv")]
                sp = csv.with_name(stem + ".spacy")
                if sp.exists():
                    pairs.append((csv, sp))
    # All 6 English originals from Improta et al.; ITA variants and
    # LLaMAntino-2 (Italian-tuned Llama) are out of scope for English-only.
    originals = base / "phase0_textualllmap"
    if originals.is_dir():
        for csv in sorted(originals.glob("*/*.cleaned.csv")):
            if "(ITA)" in csv.name or csv.name.startswith("LLaMAntino"):
                continue
            stem = csv.name[: -len(".cleaned.csv")]
            sp = csv.with_name(stem + ".spacy")
            if sp.exists():
                pairs.append((csv, sp))
    # Human-text baselines (Webis-CMV-20 + Reuters-50/C50), opt-in. Needs
    # clean_corpus.py + parse_corpus.py to have produced the (cleaned, spacy)
    # pair; otherwise the glob is a no-op.
    if include_human:
        human = base / "human_baseline"
        if human.is_dir():
            for csv in (sorted(human.glob("webis_cmv_20/*/Webis-CMV-20.cleaned.csv"))
                        + sorted(human.glob("reuters_50/*/Reuters-50.cleaned.csv"))):
                stem = csv.name[: -len(".cleaned.csv")]
                sp = csv.with_name(stem + ".spacy")
                if sp.exists():
                    pairs.append((csv, sp))
    return pairs


def per_text_stats(doc: Doc) -> dict[str, float | int]:
    n_tokens = len(doc)
    alpha = [t for t in doc if t.is_alpha]
    n_words = len(alpha)
    n_sents = sum(1 for _ in doc.sents)
    lemmas = {t.lemma_.lower() for t in alpha}
    n_types = len(lemmas)
    return {
        "n_chars": len(doc.text),
        "n_tokens": n_tokens,
        "n_words": n_words,
        "n_sents": n_sents,
        "n_types": n_types,
        "ttr": round(n_types / n_words, 4) if n_words else 0.0,
        "mean_sent_len": round(n_words / n_sents, 2) if n_sents else 0.0,
    }


def is_quality_clean(row: pd.Series) -> bool:
    return not (
        bool(row.get("flag_repetition_loop", False))
        or bool(row.get("flag_code_contam", False))
        or bool(row.get("flag_non_english", False))
    )


def process_pair(
    csv_path: Path, sp_path: Path, vocab
) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    dbin = DocBin().from_disk(sp_path)
    docs = list(dbin.get_docs(vocab))
    if len(docs) != len(df):
        raise RuntimeError(
            f"doc/row mismatch: {csv_path} has {len(df)} rows but "
            f"{sp_path} has {len(docs)} docs"
        )

    # Phase 0 has all 6 English originals under one parent dir; disambiguate
    # by filename so each model gets its own model_dir row in per_text/summary.
    parent = csv_path.parent.parent.name
    if parent == "phase0_textualllmap":
        stem = csv_path.name[: -len(".cleaned.csv")]
        slug = stem.lower().replace(" ", "-")
        model_dir = f"phase0_{slug}"
    else:
        model_dir = parent
    topic = csv_path.parent.name
    flag_cols = [c for c in df.columns if c.startswith("flag_")]

    rows = []
    pooled_lemmas: set[str] = set()
    pooled_alpha_tokens = 0

    for (_, row), doc in zip(df.iterrows(), docs):
        out = {
            "model_dir": model_dir,
            "topic": topic,
            "model": row["model"],
        }
        for c in flag_cols:
            out[c] = bool(row[c])
        out.update(per_text_stats(doc))
        rows.append(out)

        if is_quality_clean(row):
            for tok in doc:
                if tok.is_alpha:
                    pooled_lemmas.add(tok.lemma_.lower())
                    pooled_alpha_tokens += 1

    per_text_df = pd.DataFrame(rows)
    group_summary = {
        "model_dir": model_dir,
        "topic": topic,
        "vocab_size": len(pooled_lemmas),
        "corpus_tokens": pooled_alpha_tokens,
        "corpus_ttr": (
            round(len(pooled_lemmas) / pooled_alpha_tokens, 4)
            if pooled_alpha_tokens
            else 0.0
        ),
    }
    return per_text_df, group_summary


def build_summary(per_text: pd.DataFrame, group_summaries: list[dict]) -> pd.DataFrame:
    quality_clean = ~(
        per_text["flag_repetition_loop"]
        | per_text["flag_code_contam"]
        | per_text["flag_non_english"]
    )
    rows = []
    for gs in group_summaries:
        md, tp = gs["model_dir"], gs["topic"]
        grp = per_text[
            (per_text["model_dir"] == md) & (per_text["topic"] == tp)
        ]
        grp_clean = grp[quality_clean.loc[grp.index]]
        out = {
            "model_dir": md,
            "topic": tp,
            "n_total": len(grp),
            "n_quality_clean": len(grp_clean),
            "pct_truncated": round(grp["flag_truncation"].mean() * 100, 1),
            "vocab_size": gs["vocab_size"],
            "corpus_ttr": gs["corpus_ttr"],
        }
        for k in ("n_words", "n_sents", "n_types", "ttr", "mean_sent_len"):
            out[f"{k}_median"] = round(grp_clean[k].median(), 2)
            out[f"{k}_mean"] = round(grp_clean[k].mean(), 2)
            out[f"{k}_std"] = round(grp_clean[k].std(), 2)
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument(
        "--out-dir", default=str(RESULTS_DIR / "phase2_mapping" / "stage1_descriptives")
    )
    ap.add_argument("--model", default="en_core_web_lg")
    ap.add_argument(
        "--include-human", action="store_true",
        help="also include the human baselines (Webis-CMV-20 + Reuters-50) — "
             "requires clean_corpus.py + parse_corpus.py to have run on them. "
             "Use with a separate --out-dir to keep the canonical 13-corpus output.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nlp = spacy.load(args.model, disable=["ner"])
    pairs = find_pairs(Path(args.base), include_human=args.include_human)
    print(f"Found {len(pairs)} (cleaned, spacy) pairs"
          + ("  (incl. human baselines)" if args.include_human else ""))

    per_text_frames = []
    group_summaries = []
    for csv_path, sp_path in pairs:
        rel = csv_path.relative_to(Path(args.base))
        print(f"  {rel}")
        df, gs = process_pair(csv_path, sp_path, nlp.vocab)
        per_text_frames.append(df)
        group_summaries.append(gs)

    per_text = pd.concat(per_text_frames, ignore_index=True)
    per_text_path = out_dir / "per_text.csv"
    per_text.to_csv(per_text_path, index=False)
    print(f"Wrote {per_text_path} ({len(per_text)} rows)")

    summary = build_summary(per_text, group_summaries)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    print("\n=== Per-(model × topic) summary (clean-text aggregates) ===")
    cols = [
        "model_dir",
        "topic",
        "n_quality_clean",
        "pct_truncated",
        "n_words_median",
        "n_sents_median",
        "mean_sent_len_median",
        "ttr_median",
        "vocab_size",
        "corpus_ttr",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
