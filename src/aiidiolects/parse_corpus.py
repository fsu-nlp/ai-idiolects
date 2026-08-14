#!/usr/bin/env python3
"""Stage 1b: spaCy parse of the cleaned corpus.

Reads each ``<Model>.cleaned.csv`` produced by ``clean_corpus.py``, parses
the ``text_clean`` column with a spaCy English model, and writes the parsed
``Doc`` objects to ``<Model>.spacy`` (DocBin) alongside the cleaned CSV.

Stage 1c and Stage 3 load the DocBin instead of re-parsing.

Usage::

    python3 -m aiidiolects.parse_corpus                    # full run, en_core_web_lg
    python3 -m aiidiolects.parse_corpus --smoke            # one file
    python3 -m aiidiolects.parse_corpus --model en_core_web_trf  # if installed
    python3 -m aiidiolects.parse_corpus --n-process 4      # multi-process pipe

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin

from aiidiolects.paths import DATA_DIR


def find_cleaned_inputs(base: Path) -> list[Path]:
    inputs: list[Path] = []
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
                inputs.append(csv)
    for run in ("run5_lmstudio", "run6_lmstudio"):
        anchor = base / "phase1_anchoring" / run
        if anchor.is_dir():
            for csv in sorted(anchor.glob("*/*.cleaned.csv")):
                if "buggy-prompt" in csv.name:
                    continue
                inputs.append(csv)
    # All 6 English originals from Improta et al.; ITA variants and
    # LLaMAntino-2 (Italian-tuned Llama) are out of scope for English-only.
    originals = base / "phase0_textualllmap"
    if originals.is_dir():
        for csv in sorted(originals.glob("*/*.cleaned.csv")):
            if "(ITA)" in csv.name or csv.name.startswith("LLaMAntino"):
                continue
            inputs.append(csv)
    # Human-text baselines (Webis-CMV-20, Reuters-50/C50) — parse them too so
    # descriptive_stats.py can pick up the (cleaned, spacy) pairs when run with
    # --include-human. Cheap; the .spacy siblings are gitignored like the rest.
    human = base / "human_baseline"
    if human.is_dir():
        for csv in sorted(human.glob("webis_cmv_20/*/Webis-CMV-20.cleaned.csv")):
            inputs.append(csv)
        for csv in sorted(human.glob("reuters_50/*/Reuters-50.cleaned.csv")):
            inputs.append(csv)
    return inputs


def cleaned_to_spacy_path(cleaned_path: Path) -> Path:
    name = cleaned_path.name
    suffix = ".cleaned.csv"
    assert name.endswith(suffix), f"unexpected filename: {name}"
    return cleaned_path.with_name(name[: -len(suffix)] + ".spacy")


def parse_file(
    in_path: Path,
    out_path: Path,
    nlp: spacy.language.Language,
    batch_size: int,
    n_process: int,
) -> tuple[int, float]:
    df = pd.read_csv(in_path)
    texts = df["text_clean"].fillna("").astype(str).tolist()

    doc_bin = DocBin(store_user_data=False)
    t0 = time.time()
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        doc_bin.add(doc)
    elapsed = time.time() - t0

    doc_bin.to_disk(out_path)
    return len(texts), elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument("--model", default="en_core_web_lg")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only-human", action="store_true",
                    help="parse only the human_baseline/ cleaned corpora (skip the LLM .cleaned.csv already parsed)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--n-process",
        type=int,
        default=1,
        help="spaCy nlp.pipe n_process (use >1 for CPU multiprocessing)",
    )
    ap.add_argument(
        "--disable",
        default="ner",
        help="comma-separated pipe components to disable (default: ner)",
    )
    args = ap.parse_args()

    print(f"Loading spaCy model: {args.model}")
    disable = [c.strip() for c in args.disable.split(",") if c.strip()]
    nlp = spacy.load(args.model, disable=disable)
    print(f"  enabled pipes: {nlp.pipe_names}")
    print(f"  n_process: {args.n_process}, batch_size: {args.batch_size}")

    base = Path(args.base)
    inputs = find_cleaned_inputs(base)
    if args.only_human:
        inputs = [p for p in inputs if "human_baseline" in p.parts]
    if args.smoke:
        inputs = inputs[:1]
    print(f"Found {len(inputs)} cleaned input(s) under {base}")

    total_docs = 0
    total_time = 0.0
    for in_path in inputs:
        out_path = cleaned_to_spacy_path(in_path)
        rel = in_path.relative_to(base)
        n, dt = parse_file(
            in_path, out_path, nlp, args.batch_size, args.n_process
        )
        total_docs += n
        total_time += dt
        print(f"  {rel} -> {out_path.name}  ({n} docs, {dt:.1f}s, {n / dt:.1f} docs/s)")

    if total_docs:
        print(
            f"\nDone. Parsed {total_docs} docs in {total_time:.1f}s "
            f"({total_docs / total_time:.1f} docs/s avg)."
        )


if __name__ == "__main__":
    main()
