#!/usr/bin/env python3
"""Stage 4 render — sentence-embedding semantic map.

Reads ``embeddings_2d.csv`` / ``summary.csv`` (from ``embedding_map.py``)
and draws a 2-panel UMAP scatter: left = points coloured by *era* (with the
per-corpus centroids overlaid as labelled markers), right = points coloured
by *topic*. A supplementary view: does the corpus collection separate by
era / topic / model in semantic space?

Plus a small ``fig_internal_redundancy.png`` — per-corpus mean pairwise
cosine of its text embeddings (high ⇒ semantically homogeneous / templated),
era-coloured, ranked.

Usage::

    python3 scripts/figures/render_embedding_map.py
    python3 scripts/figures/render_embedding_map.py --in-dir <stage4_embedding_map dir> --jpg

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

from aiidiolects.paths import RESULTS_DIR
from aiidiolects.visualize_stage1 import (
    ERA_COLORS, MODEL_LABELS, MODEL_ORDER_WITH_HUMAN,
    TOPIC_COLORS, TOPIC_LABELS, era_of,
)

# topic palette: the 4 Improta topics + a 5th slot for Reuters' pseudo-topic
TOPIC_PAL = dict(TOPIC_COLORS)
TOPIC_PAL["reuters_ccat"] = "#7f7f7f"
TOPIC_LAB = dict(TOPIC_LABELS)
TOPIC_LAB["reuters_ccat"] = "Reuters CCAT"


def _short(m: str) -> str:
    return MODEL_LABELS.get(m, m).replace("\n", " ")


def _save(fig: plt.Figure, png: Path, write_jpg: bool) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")
    if write_jpg:
        try:
            from PIL import Image
        except ImportError:
            return
        with Image.open(png) as im:
            im.convert("RGB").save(png.with_suffix(".jpg"), quality=88,
                                   optimize=True, progressive=True)
        print(f"  wrote {png.with_suffix('.jpg')}")


def fig_map(mds: pd.DataFrame, summ: pd.DataFrame, out_path: Path, write_jpg: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    # Panel 1: coloured by era. (Per-corpus pooled centroids are not drawn here:
    # because topic dominates UMAP geometry, each corpus's pooled centroid lands
    # at the average of its four topic-driven sub-clusters — i.e. somewhere in
    # the middle of topic space, not where any of its texts actually are — so
    # the labels pile up illegibly. Era is encoded in point colour instead.)
    ax = axes[0]
    for era_key, col in [("2023", ERA_COLORS["2023"]), ("bridge", ERA_COLORS["bridge"]),
                         ("2026", ERA_COLORS["2026"]), ("human", ERA_COLORS["human"])]:
        sel = mds["model_dir"].map(era_of) == era_key
        ax.scatter(mds.loc[sel, "x"], mds.loc[sel, "y"], s=3, alpha=0.18,
                   color=col, linewidths=0)
    ax.set_title("coloured by era", fontsize=10, loc="left")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", linestyle="", color=ERA_COLORS["2023"], label="2023 cohort"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=ERA_COLORS["bridge"], label="2026 engine bridge"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=ERA_COLORS["2026"], label="2026 cohort"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=ERA_COLORS["human"], label="human (Phase 2b)"),
    ], loc="best", fontsize=8, frameon=False)

    # Panel 2: coloured by topic.
    ax = axes[1]
    for tp in ["climate", "global_warming", "math_anxiety", "misinfo_health", "reuters_ccat"]:
        sel = mds["topic"] == tp
        if not sel.any():
            continue
        ax.scatter(mds.loc[sel, "x"], mds.loc[sel, "y"], s=3, alpha=0.12,
                   color=TOPIC_PAL.get(tp, "#333"), linewidths=0,
                   label=TOPIC_LAB.get(tp, tp))
    ax.set_title("coloured by topic", fontsize=10, loc="left")
    leg = ax.legend(loc="best", fontsize=8, frameon=False)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
        h.set_sizes([30])

    for ax in axes:
        ax.set_xlabel("UMAP dim 1", fontsize=9)
        ax.set_ylabel("UMAP dim 2", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Stage 4: sentence-embedding semantic map (e5-large-v2 → UMAP, cosine)\n"
        "supplementary — semantic geometry vs the lexical/stylometric §4.1 distances",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_path, write_jpg)


def fig_redundancy(summ: pd.DataFrame, out_path: Path, write_jpg: bool) -> None:
    g = summ[summ["scope"] == "pooled"].dropna(subset=["mean_internal_cosine"])
    present = set(g["model_dir"])
    models = [m for m in MODEL_ORDER_WITH_HUMAN if m in present]
    models = sorted(models, key=lambda m: g.set_index("model_dir").loc[m, "mean_internal_cosine"])
    vals = [float(g.set_index("model_dir").loc[m, "mean_internal_cosine"]) for m in models]
    cols = [ERA_COLORS[era_of(m)] for m in models]
    fig, ax = plt.subplots(figsize=(9, 6.0))
    ys = np.arange(len(models))
    ax.barh(ys, vals, color=cols, edgecolor="white", linewidth=0.5, height=0.66)
    for y, v in zip(ys, vals):
        ax.text(v, y, f" {v:.3f}", va="center", ha="left", fontsize=7, color="#333")
    ax.set_yticks(ys)
    ax.set_yticklabels([_short(m) for m in models], fontsize=8)
    for tick, m in zip(ax.get_yticklabels(), models):
        tick.set_color(ERA_COLORS[era_of(m)])
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlabel("mean pairwise cosine of the corpus's text embeddings "
                  "(high ⇒ semantically homogeneous / templated)", fontsize=9)
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("Stage 4: semantic internal redundancy per corpus (pooled, ranked low→high)",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_path, write_jpg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage4_embedding_map"),
    )
    ap.add_argument("--out-dir", default=None, help="default: <in-dir>/figures")
    ap.add_argument("--jpg", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    mds = pd.read_csv(in_dir / "embeddings_2d.csv")
    summ = pd.read_csv(in_dir / "summary.csv")
    print(f"Loaded {len(mds):,} points; corpora: {sorted(mds['model_dir'].unique())}")

    fig_map(mds, summ, out_dir / "embedding_map.png", args.jpg)
    fig_redundancy(summ, out_dir / "fig_internal_redundancy.png", args.jpg)


if __name__ == "__main__":
    main()
