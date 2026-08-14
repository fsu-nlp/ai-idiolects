#!/usr/bin/env python3
"""Render Phase 1 anchoring trio (originals / Run 5 / Run 6) descriptive figures.

Mirrors ``visualize_stage1.py`` but for the Phase 1 anchoring corpus only:

- ``phase0_mistral-7b/<topic>/Mistral-7b.csv`` — Improta et al. originals
- ``phase1_anchoring/run5_lmstudio/<topic>/Mistral-7b.csv`` — canonical anchor
- ``phase1_anchoring/run6_lmstudio/<topic>/Mistral-7b.csv`` — stochastic baseline

Reads the same shared ``per_text.csv`` produced by ``descriptive_stats.py`` and
filters to the three trio entries via its own MODEL_ORDER. Imports the figure
helpers from ``visualize_stage1`` so figures are visually consistent with the
Phase 2 mapping suite.

Outputs::

    results/phase1_anchoring/stage1_descriptives/figures/
        fig1_length.png
        fig2_lexical_diversity.png
        fig3_opener_style.png
        fig4_topic_verbosity.png

The Phase 2 ``fig1b`` / ``fig4b`` truncation-included companions and
``fig5_truncation`` are intentionally omitted: zero texts in the trio
hit the 2048-token / 1483-word cap, so the truncation-included views
are byte-identical to the non-truncated ones and the truncation
heatmap is uniformly zero. The verbosity heatmap is rendered with a
``vmin``/``vmax`` matching the broader Phase 2 corpus span (350–1350
words median) so the trio reads as visually overlapping rather than
maximally separated, which would be misleading: the within-trio gap
is within stochastic noise.

Usage::

    python3 scripts/figures/visualize_phase1.py
    python3 scripts/figures/visualize_phase1.py --in-csv path/to/per_text.csv \\
                                          --out-dir path/to/figures

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aiidiolects.paths import RESULTS_DIR
from aiidiolects import visualize_stage1 as v1

# Override the trio's order/labels for Phase 1 anchoring figures.
PHASE1_MODEL_ORDER = [
    "phase0_mistral-7b",
    "run5_lmstudio",
    "run6_lmstudio",
]

PHASE1_MODEL_LABELS = {
    "phase0_mistral-7b": "Originals\n(Improta et al.)",
    "run5_lmstudio":       "Run 5\n(LM Studio 2026)",
    "run6_lmstudio":       "Run 6\n(stochastic baseline)",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    default_in = str(
        RESULTS_DIR / "phase2_mapping" / "stage1_descriptives" / "per_text.csv"
    )
    default_out = str(
        RESULTS_DIR / "phase1_anchoring" / "stage1_descriptives" / "figures"
    )
    ap.add_argument("--in-csv", default=default_in)
    ap.add_argument("--out-dir", default=default_out)
    args = ap.parse_args()

    # Swap MODEL_ORDER and MODEL_LABELS in the imported module so its figure
    # functions filter to the trio. This keeps visualize_stage1.py's figure
    # implementations as the single source of truth for plot styling.
    v1.MODEL_ORDER = PHASE1_MODEL_ORDER
    v1.MODEL_LABELS = PHASE1_MODEL_LABELS

    in_path = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    present = sorted(df["model_dir"].unique())
    print(f"Loaded {len(df):,} rows from {in_path}")
    print(f"Models present in CSV: {present}")
    trio_present = [m for m in PHASE1_MODEL_ORDER if m in present]
    print(f"Phase 1 trio rows visible: {trio_present}")
    missing = [m for m in PHASE1_MODEL_ORDER if m not in present]
    if missing:
        print(f"  WARNING: trio members missing from per_text.csv: {missing}")
        print(f"  Re-run clean_corpus.py + parse_corpus.py + descriptive_stats.py.")

    # summary.csv lives next to per_text.csv (descriptive_stats.py output),
    # not next to the Phase 1 figures dir.
    summary_path = in_path.with_name("summary.csv")

    # Wider vmin/vmax for fig4 keeps the trio visually close: their
    # actual range (~350–450) is small relative to the Phase 2 corpus
    # (~350–1350); the auto-fit colour scale would otherwise inflate
    # within-noise differences (e.g. 429 vs 447 reads as green vs yellow).
    figures = [
        ("fig1_length.png", v1.fig_length),
        (
            "fig2_lexical_diversity.png",
            lambda df, p: v1.fig_lexical_diversity(df, p, summary_path=summary_path),
        ),
        ("fig3_opener_style.png", v1.fig_opener_style),
        (
            "fig4_topic_verbosity.png",
            lambda df, p: v1.fig_topic_verbosity(
                df, p,
                vmin=250, vmax=550,
                subtitle=(
                    "(quality-clean texts; colour scale 250–550 words "
                    "— within-trio differences are within stochastic noise)"
                ),
            ),
        ),
    ]
    for name, fn in figures:
        out_path = out_dir / name
        print(f"  rendering {name}")
        fn(df, out_path)
        print(f"    wrote {out_path}")


if __name__ == "__main__":
    main()
