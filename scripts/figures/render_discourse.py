#!/usr/bin/env python3
"""Stage 2 C: render the punctuation / discourse-marker / hedge heatmaps.

Reads ``rates_long.csv`` (or the per-category × per-scope wide-format
matrices) written by ``discourse_features.py`` and produces three
era-coloured heatmaps per scope:

* punctuation (em-dash, en-dash, semicolon, colon, ellipsis, exclamation,
  question, comma, opening parens) — tests the "GPT em-dash" cliché.
* discourse markers (however, furthermore, ...).
* hedges (might, could, ..., tend to).

Row order + era colouring + cohort separators match
``visualize_stage1.py`` (2023 grey / bridge amber / 2026 blue, dashed
``axhline`` between cohorts). Rates are shown per 1000 words.

Output::

    results/phase2_mapping/stage2_discourse_features/figures/
        discourse__pooled.png         (3-panel headline)
        discourse__<topic>.png        (per-topic supplements)

Usage::

    python3 scripts/figures/render_discourse.py
    python3 scripts/figures/render_discourse.py --in-dir <matrices> \\
        --out-dir <figures>

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aiidiolects.discourse_features import CATEGORY_FEATURES
from aiidiolects.paths import RESULTS_DIR
from aiidiolects.visualize_stage1 import (
    ERA_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    MODEL_ORDER_WITH_HUMAN,
    era_of,
    era_separator_indices,
)

# Active row order. main() swaps this to MODEL_ORDER_WITH_HUMAN under
# --include-human; reorder() reads it.
_ACTIVE_ORDER: list[str] = MODEL_ORDER

# Scopes that are redundant once their single row is already in `pooled`.
SKIP_SCOPES = {"reuters_ccat"}


CATEGORY_TITLES: dict[str, str] = {
    "punctuation":       "Punctuation rate per 1000 words",
    "discourse_markers": "Discourse-marker rate per 1000 words",
    "hedges":            "Hedge rate per 1000 words",
}

FEATURE_LABELS: dict[str, str] = {
    # Punctuation: render the actual character so the cell label is
    # immediately recognisable.
    "em_dash":     "—  em-dash",
    "en_dash":     "–  en-dash",
    "semicolon":   ";  semicolon",
    "colon":       ":  colon",
    "ellipsis":    "…  ellipsis",
    "exclamation": "!  exclamation",
    "question":    "?  question",
    "comma":       ",  comma",
    "open_paren":  "(  paren",
    # Discourse markers + hedges: word as-is.
}

SCOPE_TITLES = {
    "pooled":         "pooled across topics",
    "climate":        "climate",
    "global_warming": "global warming",
    "math_anxiety":   "math anxiety",
    "misinfo_health": "health misinfo",
}


def reorder(matrix: pd.DataFrame, category: str) -> pd.DataFrame:
    rows = [m for m in _ACTIVE_ORDER if m in matrix.index]
    cols = [f for f in CATEGORY_FEATURES[category] if f in matrix.columns]
    return matrix.loc[rows, cols]


def heatmap_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    cmap: str = "magma_r",
    annotate_fmt: str = "{:.1f}",
) -> plt.cm.ScalarMappable:
    if matrix.empty:
        ax.set_visible(False)
        return None

    vmin = float(matrix.values[~np.isnan(matrix.values)].min())
    vmax = float(matrix.values[~np.isnan(matrix.values)].max())
    if vmax == vmin:
        vmax = vmin + 1e-9
    im = ax.imshow(
        matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
    )
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        [FEATURE_LABELS.get(f, f) for f in matrix.columns],
        rotation=35, ha="right", fontsize=8,
    )
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(
        [MODEL_LABELS.get(m, m).replace("\n", " ") for m in matrix.index],
        fontsize=8,
    )
    for tick, m in zip(ax.get_yticklabels(), matrix.index):
        tick.set_color(ERA_COLORS[era_of(m)])
    for i in era_separator_indices(list(matrix.index)):
        ax.axhline(i + 0.5, color="#999999", linewidth=0.5, linestyle="--")

    midpoint = vmin + 0.5 * (vmax - vmin)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if pd.isna(v):
                continue
            color = "white" if v > midpoint else "black"
            ax.text(j, i, annotate_fmt.format(v),
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title, fontsize=10, loc="left", pad=4)
    return im


def render_scope(
    matrices: dict[str, pd.DataFrame],
    scope: str,
    out_path: Path,
) -> None:
    """Three-panel heatmap (punctuation / discourse / hedges) for one scope."""
    cats = ["punctuation", "discourse_markers", "hedges"]
    widths = [
        max(1, len(CATEGORY_FEATURES[c]))
        for c in cats
    ]
    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 6.5),
        gridspec_kw={"width_ratios": widths},
    )

    for ax, cat in zip(axes, cats):
        m = reorder(matrices.get(cat, pd.DataFrame()), cat)
        im = heatmap_panel(ax, m, CATEGORY_TITLES[cat])
        if im is not None:
            fig.colorbar(im, ax=ax, shrink=0.8, label="per 1,000 words")

    fig.suptitle(
        f"Stage 2C: discourse features ({SCOPE_TITLES.get(scope, scope)})",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir", default=None,
        help=("matrices dir; default "
              "results/phase2_mapping/stage2_discourse_features/matrices "
              "(or *_with_human under --include-human)"),
    )
    ap.add_argument(
        "--out-dir", default=None,
        help=("figures dir; default "
              "results/phase2_mapping/stage2_discourse_features/figures "
              "(or *_with_human under --include-human)"),
    )
    ap.add_argument(
        "--include-human", action="store_true",
        help=("add the Webis-CMV-20 + Reuters-50 human rows (Webis in every "
              "scope, Reuters in the pooled heatmap) and read from / write to "
              "the stage2_discourse_features_with_human directories. The "
              "standalone reuters_ccat scope is skipped (redundant with the "
              "pooled Reuters row)."),
    )
    args = ap.parse_args()

    global _ACTIVE_ORDER
    if args.include_human:
        _ACTIVE_ORDER = MODEL_ORDER_WITH_HUMAN
        base = str(RESULTS_DIR / "phase2_mapping" / "stage2_discourse_features_with_human")
    else:
        base = str(RESULTS_DIR / "phase2_mapping" / "stage2_discourse_features")
    in_dir = Path(args.in_dir or f"{base}/matrices")
    out_dir = Path(args.out_dir or f"{base}/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(in_dir.glob("*__*.csv"))
    if not csvs:
        raise SystemExit(f"No matrix CSVs found in {in_dir}")

    by_scope: dict[str, dict[str, pd.DataFrame]] = {}
    for path in csvs:
        category, scope = path.stem.split("__", 1)
        if scope in SKIP_SCOPES:
            continue
        by_scope.setdefault(scope, {})[category] = pd.read_csv(
            path, index_col=0
        )

    for scope, matrices in sorted(by_scope.items(),
                                   key=lambda kv: kv[0] != "pooled"):
        out_path = out_dir / f"discourse__{scope}.png"
        render_scope(matrices, scope, out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
