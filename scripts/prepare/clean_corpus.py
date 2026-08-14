#!/usr/bin/env python3
"""Stage 1a: clean and flag generated texts for the AI idiolects analysis.

Reads each phase2_<model>/<topic>/<Model>.csv and the Phase 1 anchor at
replicated_run5_lmstudio/<topic>/Mistral-7b.csv, adds flag columns plus a
``text_clean`` column, and writes ``<Model>.cleaned.csv`` next to the input.

Two flag families with different downstream treatment:

* **Stylistic annotations** (kept in corpus, become metrics in Stage 2):
  ``flag_ai_self_ref``, ``flag_disclaimer``, ``flag_refusal``,
  ``flag_hedge_heavy_opener``.

* **Quality exclusions** (filtered out of length/lexical metrics):
  ``flag_repetition_loop``, ``flag_code_contam``, ``flag_truncation``,
  ``flag_non_english``.

Usage::

    python3 scripts/prepare/clean_corpus.py                  # full run
    python3 scripts/prepare/clean_corpus.py --smoke          # one file only
    python3 scripts/prepare/clean_corpus.py --base path/...  # alternate data dir

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from aiidiolects.paths import DATA_DIR

try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    _HAVE_LANGDETECT = True
except ImportError:
    _HAVE_LANGDETECT = False


OPENER_LEN = 250

OPENER_PATTERNS: dict[str, list[str]] = {
    "ai_self_ref": [
        r"\bas an? (?:AI|artificial intelligence|A\.I\.|language model|LLM|machine|chatbot|computational|digital being|digital entity)\b",
        r"\bspeaking as an? (?:AI|language model|artificial)\b",
        r"\bI(?:'m| am) an? (?:AI|artificial intelligence|language model|chatbot|LLM)\b",
        r"\bfrom (?:the )?perspective of an? (?:AI|artificial intelligence|language model|chatbot)\b",
        r"\bbeing an? (?:AI|language model|artificial intelligence)\b",
    ],
    "disclaimer": [
        r"\bI (?:don'?t|do not) (?:have|hold|possess|experience|form) (?:any )?(?:personal )?(?:opinions?|beliefs?|feelings?|emotions?|views?|preferences?)\b",
        r"\bI (?:cannot|can'?t) (?:have|hold|form|possess) (?:personal )?(?:opinions?|beliefs?|feelings?|emotions?)\b",
        r"\bmy (?:so-called |alleged |\")?(?:beliefs?|opinions?|inner beliefs?|values?|views?)(?:\")? (?:are|aren'?t|do not|don'?t)\b",
        r"\bI (?:should|must|need to|want to) (?:clarify|note|preface|emphasize|acknowledge|disclose|first)\b",
        r"\bI appreciate (?:the |your )?(?:invitation|opportunity|prompt|chance|question)\b",
        r"\b(?:before|first,?) (?:I |let me )(?:address|note|clarify|preface|state)\b",
    ],
    "refusal": [
        r"\bI (?:cannot|can'?t|won'?t|will not|am unable|am not able) (?:help|assist|provide|generate|produce|engage|comply|do|share|express)\b",
        r"\bI(?:'m| am) (?:not able|unable) to\b",
        r"\bI(?:'m| am) (?:sorry|afraid),? (?:but )?I\b",
        r"\bI must (?:decline|refuse)\b",
    ],
}

HEDGES = frozenset({
    "might", "may", "could", "would", "should",
    "possibly", "perhaps", "arguably", "likely",
    "seems", "seem", "seemingly", "appears", "appear",
    "somewhat", "relatively", "potentially",
    "tends", "tend", "suggests", "suggest",
})

CODE_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
HTML_TAG_RE = re.compile(r"<(?:[a-zA-Z]+|/[a-zA-Z]+)\b[^<>]{0,200}>")
TERMINAL_PUNCT = (
    ".", "!", "?", "…",
    ".\"", "!\"", "?\"",
    ".'", "!'", "?'",
    ".)", "!)", "?)",
    ".]", "!]", "?]",
)

# Project constant: average English tokens per word (cross-tokenizer estimate
# from a separate project). Used to translate the generation request's
# max_tokens cap into an expected word-count ceiling.
ENGLISH_TOKEN_RATIO = 1.1047
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TRUNCATION_PCT = 0.80


def detect_openers(text: str) -> list[str]:
    head = text[:OPENER_LEN]
    matched: list[str] = []
    for label, patterns in OPENER_PATTERNS.items():
        for p in patterns:
            if re.search(p, head, flags=re.IGNORECASE):
                matched.append(label)
                break
    return matched


def hedge_count_opener(text: str) -> int:
    head_tokens = re.findall(r"[A-Za-z']+", text[:OPENER_LEN].lower())
    return sum(1 for t in head_tokens if t in HEDGES)


def detect_repetition_loop(
    text: str, n: int = 3, min_count: int = 8, min_ratio: float = 0.015
) -> tuple[bool, int]:
    """Flag a text as a repetition loop only when its top n-gram count is both
    absolutely high (>= ``min_count``) and a meaningful share of all n-grams
    (>= ``min_ratio``). This avoids flagging long, healthy texts whose common
    formulaic openers occur a handful of times naturally.
    """
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    if len(tokens) < n + 1:
        return False, 0
    grams = list(zip(*[tokens[i:] for i in range(n)]))
    if not grams:
        return False, 0
    _, count = Counter(grams).most_common(1)[0]
    threshold = max(min_count, int(min_ratio * len(grams)))
    return count >= threshold, count


def detect_code_contamination(text: str) -> dict[str, object]:
    fenced = CODE_FENCE_RE.findall(text)
    fence_chars = sum(len(f) for f in fenced)
    has_html = bool(HTML_TAG_RE.search(text))
    return {
        "has_code_fence": bool(fenced),
        "code_fence_n": len(fenced),
        "code_fence_ratio": fence_chars / max(len(text), 1),
        "has_html_tag": has_html,
    }


def strip_code_fences(text: str) -> str:
    return CODE_FENCE_RE.sub("[CODE_BLOCK]", text)


def detect_truncation(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    token_ratio: float = ENGLISH_TOKEN_RATIO,
    threshold_pct: float = DEFAULT_TRUNCATION_PCT,
) -> bool:
    """Flag a text as truncated only when both:

    1. it does not end with terminal punctuation, AND
    2. its word count is close enough to the max-tokens-derived ceiling that
       hitting the cap is plausible.

    The expected ceiling is ``max_tokens / token_ratio`` words (with
    ``token_ratio = 1.1047`` for English). A text well below
    ``threshold_pct`` of that ceiling is treated as a stylistic non-period
    ending (e.g., Gemini's em-dashes and colon-introduced lists), not a
    generation cut.
    """
    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped.endswith(TERMINAL_PUNCT):
        return False
    word_count = len(re.findall(r"[A-Za-z']+", stripped))
    max_words = max_tokens / token_ratio
    return word_count >= max_words * threshold_pct


def detect_non_english(text: str) -> bool | None:
    if not _HAVE_LANGDETECT:
        return None
    sample = text[:1000] if len(text) > 1000 else text
    if not sample.strip():
        return False
    try:
        return detect(sample) != "en"
    except Exception:
        return False


def process_file(
    in_path: Path,
    out_path: Path,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    token_ratio: float = ENGLISH_TOKEN_RATIO,
    truncation_pct: float = DEFAULT_TRUNCATION_PCT,
) -> pd.DataFrame | None:
    df = pd.read_csv(in_path)
    if "text" not in df.columns:
        print(f"  WARN: no 'text' column in {in_path}", file=sys.stderr)
        return None

    # Drop exact-duplicate rows by ``text``. The published Improta corpus
    # has one affected file (phase0_textualllmap/misinfo_health/Haiku.csv:
    # 1852 rows / 1000 unique). Deduplicating here keeps n balanced across
    # the 13-corpus matrix and is a no-op on every other file.
    n_before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    n_after = len(df)
    if n_after != n_before:
        print(
            f"  dropped {n_before - n_after} duplicate-text rows "
            f"({n_before} -> {n_after})",
            file=sys.stderr,
        )

    rows = []
    for _, row in df.iterrows():
        text = "" if pd.isna(row["text"]) else str(row["text"])

        openers = detect_openers(text)
        hedge_n = hedge_count_opener(text)
        rep_flag, rep_count = detect_repetition_loop(text)
        code = detect_code_contamination(text)
        truncated = detect_truncation(
            text,
            max_tokens=max_tokens,
            token_ratio=token_ratio,
            threshold_pct=truncation_pct,
        )
        non_en = detect_non_english(text)

        out = dict(row)
        out["text_clean"] = (
            strip_code_fences(text) if code["has_code_fence"] else text
        )

        out["flag_ai_self_ref"] = "ai_self_ref" in openers
        out["flag_disclaimer"] = "disclaimer" in openers
        out["flag_refusal"] = "refusal" in openers
        out["flag_hedge_heavy_opener"] = hedge_n >= 3
        out["opener_hedge_count"] = hedge_n

        out["flag_repetition_loop"] = rep_flag
        out["repetition_top_count"] = rep_count
        out["flag_code_contam"] = bool(code["has_code_fence"]) or bool(
            code["has_html_tag"]
        )
        out["code_fence_n"] = code["code_fence_n"]
        out["code_fence_ratio"] = round(code["code_fence_ratio"], 4)
        out["flag_truncation"] = truncated
        out["flag_non_english"] = bool(non_en) if non_en is not None else False

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    return out_df


def find_inputs(base: Path) -> list[Path]:
    inputs: list[Path] = []
    phase2 = base / "phase2_mapping"
    if phase2.is_dir():
        for d in sorted(phase2.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("smoke_") or d.name == "archives":
                continue
            for csv in sorted(d.glob("*/*.csv")):
                if csv.name.endswith(".cleaned.csv") or "buggy-prompt" in csv.name:
                    continue
                inputs.append(csv)
    # Phase 1 anchoring: Run 5 (canonical anchor) + Run 6 (stochastic baseline).
    for run in ("run5_lmstudio", "run6_lmstudio"):
        anchor = base / "phase1_anchoring" / run
        if anchor.is_dir():
            for csv in sorted(anchor.glob("*/*.csv")):
                if csv.name.endswith(".cleaned.csv") or "buggy-prompt" in csv.name:
                    continue
                inputs.append(csv)
    # Phase 0 originals (Improta et al.): all 6 English models per topic
    # (GPT-3.5, GPT-4o, Haiku, Llama-3-8B, Llama-3.1-70B, Mistral-7b). The 5
    # Italian variants (filenames containing "(ITA)" or starting "LLaMAntino")
    # are out of scope for this English-only chapter.
    originals = base / "phase0_textualllmap"
    if originals.is_dir():
        for csv in sorted(originals.glob("*/*.csv")):
            if csv.name.endswith(".cleaned.csv") or "buggy-prompt" in csv.name:
                continue
            if "(ITA)" in csv.name or csv.name.startswith("LLaMAntino"):
                continue
            inputs.append(csv)
    # Human-text baselines (Webis-CMV-20 cross-author, Reuters-50/C50 within-author).
    # Built by build_human_baseline.py; cleaned with the same flag pipeline as the
    # LLM corpora — the LLM-artifact opener flags just come out ~all-False on human
    # text, which is itself a useful contrast. (The raw/, survey/, manifest/ dirs
    # contain no Webis-CMV-20.csv / Reuters-50.csv, so the globs skip them.)
    human = base / "human_baseline"
    if human.is_dir():
        for csv in sorted(human.glob("webis_cmv_20/*/Webis-CMV-20.csv")):
            inputs.append(csv)
        for csv in sorted(human.glob("reuters_50/*/Reuters-50.csv")):
            inputs.append(csv)
    return inputs


FLAG_COLS = [
    "flag_ai_self_ref",
    "flag_disclaimer",
    "flag_refusal",
    "flag_hedge_heavy_opener",
    "flag_repetition_loop",
    "flag_code_contam",
    "flag_truncation",
    "flag_non_english",
]


def summarize(out_df: pd.DataFrame, label: str) -> str:
    n = len(out_df)
    if n == 0:
        return f"{label}: empty"
    parts = [f"n={n}"]
    for c in FLAG_COLS:
        if c in out_df.columns:
            rate = out_df[c].sum() / n * 100
            parts.append(f"{c.replace('flag_','')}={rate:.1f}%")
    return f"{label}: " + "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR), help="path to data dir")
    ap.add_argument(
        "--smoke", action="store_true", help="process only the first input file"
    )
    ap.add_argument(
        "--only-human", action="store_true",
        help="process only the human_baseline/ corpora (Webis-CMV-20 + Reuters-50) — "
             "skip re-cleaning the already-cleaned LLM corpora",
    )
    ap.add_argument(
        "--no-langdetect",
        action="store_true",
        help="skip language detection even if langdetect is installed",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="generation request max_tokens cap (used for truncation detection)",
    )
    ap.add_argument(
        "--token-ratio",
        type=float,
        default=ENGLISH_TOKEN_RATIO,
        help="average English tokens per word (default 1.1047)",
    )
    ap.add_argument(
        "--truncation-pct",
        type=float,
        default=DEFAULT_TRUNCATION_PCT,
        help="flag truncation only when word count >= (max_tokens / token_ratio) * truncation_pct",
    )
    args = ap.parse_args()

    global _HAVE_LANGDETECT
    if args.no_langdetect:
        _HAVE_LANGDETECT = False

    base = Path(args.base)
    inputs = find_inputs(base)
    if args.only_human:
        inputs = [p for p in inputs if "human_baseline" in p.parts]
    if args.smoke:
        inputs = inputs[:1]

    print(f"Found {len(inputs)} input file(s) under {base}")
    if not _HAVE_LANGDETECT:
        print("(langdetect unavailable or disabled — non-English check skipped)")
    max_words = args.max_tokens / args.token_ratio
    print(
        f"Truncation threshold: word count >= "
        f"{max_words * args.truncation_pct:.0f} "
        f"(= {args.max_tokens} max_tokens / {args.token_ratio} ratio "
        f"× {args.truncation_pct} pct) AND no terminal punctuation"
    )

    summaries: list[str] = []
    for in_path in inputs:
        out_path = in_path.with_name(in_path.stem + ".cleaned.csv")
        rel = in_path.relative_to(base)
        print(f"  {rel} -> {out_path.name}")
        out_df = process_file(
            in_path,
            out_path,
            max_tokens=args.max_tokens,
            token_ratio=args.token_ratio,
            truncation_pct=args.truncation_pct,
        )
        if out_df is not None:
            label = "/".join(in_path.parts[-3:-1]) + "/" + in_path.name
            summaries.append(summarize(out_df, label))

    print("\n=== Flag rate summary ===")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
