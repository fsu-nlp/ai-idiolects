#!/usr/bin/env python3
"""Stage 2 A.cont: render dendrograms + MDS biplots from a distance matrix.

Reads the per-metric × per-scope CSVs written by ``build_distance_matrix.py``
and produces era-coloured visualisations:

* ``dendrogram__<metric>__<scope>.png`` — agglomerative clustering with
  the average-linkage method (linkage-method-agnostic w.r.t. the input
  metric, unlike Ward which assumes Euclidean distances).
* ``mds_biplot__<metric>__<scope>.png`` — 2-D metric MDS embedding of
  the same distance matrix, points coloured by era.
* ``headline.png`` — side-by-side panel of the headline figure
  (Burrows' Δ, pooled across topics): dendrogram + MDS biplot.

Era colouring matches ``visualize_stage1.py``: 2023 cohort grey, Mistral
2026 engine bridge amber, 2026 production roster blue.

Usage::

    python3 scripts/figures/render_dendrogram.py
    python3 scripts/figures/render_dendrogram.py --metric burrows_delta --scope pooled
    python3 scripts/figures/render_dendrogram.py --in-dir <matrices_dir> --out-dir <figs>

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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

# Re-use the era classification + display labels that the Stage 1 figures
# already use, so the dendrogram visual grammar matches.
from aiidiolects.paths import RESULTS_DIR
from aiidiolects.visualize_stage1 import (
    ERA_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    MODEL_ORDER_WITH_HUMAN,
    era_of,
)

# Active leaf order. main() swaps this to MODEL_ORDER_WITH_HUMAN under
# --include-human; reorder_for_display() and the era legend read it.
_ACTIVE_ORDER: list[str] = MODEL_ORDER

# Headline figure defaults — change here, not at the call site.
HEADLINE_METRIC = "burrows_delta"
HEADLINE_SCOPE = "pooled"

# Linkage method. "average" is the safest choice for arbitrary distance
# matrices (Ward assumes Euclidean inputs, which is wrong for JSD/Jaccard).
LINKAGE_METHOD = "average"

# Per-metric pretty titles for figure suptitles.
METRIC_TITLES: dict[str, str] = {
    "text_length":     "Text-length distance (|Cohen's d|)",
    "sentence_length": "Sentence-length distance (|Cohen's d|)",
    "ttr":             "TTR distance",
    "mattr":           "MATTR distance (window 500)",
    "hapax":           "Hapax-ratio distance",
    "yules_k":         "Yule's K distance",
    "jaccard_top100":  "1 − Jaccard (top-100 words)",
    "bigram_jsd":      "Bigram JSD",
    "trigram_jsd":     "Trigram JSD",
    "burrows_delta":   "Burrows' Δ (100 MFW)",
    "cosine_delta":    "Cosine Δ (100 MFW; Evert 2017)",
    "zipf_exponent":   "Zipf-exponent distance",
}

SCOPE_TITLES: dict[str, str] = {
    "pooled":         "pooled across topics",
    "climate":        "climate",
    "global_warming": "global warming",
    "math_anxiety":   "math anxiety",
    "misinfo_health": "health misinfo",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_label(model_dir: str) -> str:
    """One-line label suitable for dendrogram leaves and biplot labels."""
    return MODEL_LABELS.get(model_dir, model_dir).replace("\n", " ")


def reorder_for_display(matrix: pd.DataFrame) -> pd.DataFrame:
    """Reindex a square distance matrix to follow the active leaf order
    (era-grouped; human baselines last when --include-human)."""
    present = [m for m in _ACTIVE_ORDER if m in matrix.index]
    return matrix.reindex(index=present, columns=present)


def linkage_from_matrix(matrix: pd.DataFrame, method: str = LINKAGE_METHOD):
    """scipy.cluster.hierarchy.linkage from a square distance DataFrame."""
    arr = matrix.to_numpy()
    arr = 0.5 * (arr + arr.T)              # symmetrise (rounding tolerance)
    np.fill_diagonal(arr, 0.0)
    condensed = squareform(arr, checks=False)
    return linkage(condensed, method=method)


def colour_dendrogram_labels(ax: plt.Axes, leaf_models: list[str]) -> None:
    """Recolour x-tick labels (vertical-orientation dendrogram) by era."""
    for tick, m in zip(ax.get_xticklabels(), leaf_models):
        tick.set_color(ERA_COLORS[era_of(m)])


def era_legend_handles() -> list[plt.Line2D]:
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=ERA_COLORS["2023"], label="2023 cohort (Improta)"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=ERA_COLORS["bridge"], label="2026 engine bridge"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=ERA_COLORS["2026"], label="2026 cohort (Phase 2)"),
    ]
    if "webis_cmv_20" in _ACTIVE_ORDER:
        handles.append(
            plt.Line2D([0], [0], marker="o", linestyle="",
                       color=ERA_COLORS["human"],
                       label="human (r/CMV, naturally-occurring)")
        )
    return handles


# ---------------------------------------------------------------------------
# Single-metric figures
# ---------------------------------------------------------------------------

def render_dendrogram_panel(
    ax: plt.Axes, matrix: pd.DataFrame, title: str
) -> list[str]:
    """Draw the dendrogram on ``ax`` and return leaf labels in plotted order."""
    Z = linkage_from_matrix(matrix)
    labels = [short_label(m) for m in matrix.index]

    ddata = dendrogram(
        Z,
        ax=ax,
        labels=labels,
        leaf_rotation=45,
        leaf_font_size=8,
        color_threshold=0,           # uniform link colour; era is row-coded
        above_threshold_color="#666666",
    )
    leaf_idx = ddata["leaves"]
    leaf_models = [matrix.index[i] for i in leaf_idx]
    colour_dendrogram_labels(ax, leaf_models)
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    ax.set_ylabel("distance", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return leaf_models


def render_biplot_panel(
    ax: plt.Axes, matrix: pd.DataFrame, title: str, seed: int = 0
) -> None:
    """2-D metric MDS embedding of the distance matrix, era-coloured points."""
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=8,
        normalized_stress="auto",
    )
    arr = matrix.to_numpy()
    arr = 0.5 * (arr + arr.T)
    np.fill_diagonal(arr, 0.0)
    coords = mds.fit_transform(arr)

    for (x, y), m in zip(coords, matrix.index):
        c = ERA_COLORS[era_of(m)]
        ax.scatter(x, y, s=72, c=c, edgecolors="white",
                   linewidths=0.6, zorder=3)
        ax.annotate(
            short_label(m),
            (x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            color=c,
            zorder=4,
        )
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    ax.set_xlabel("MDS dim 1", fontsize=9)
    ax.set_ylabel("MDS dim 2", fontsize=9)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_metric_pair(
    matrix: pd.DataFrame,
    metric: str,
    scope: str,
    out_path: Path,
) -> None:
    """Side-by-side dendrogram + MDS biplot for one (metric, scope)."""
    matrix = reorder_for_display(matrix)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2),
                             gridspec_kw={"width_ratios": [1, 1]})
    metric_title = METRIC_TITLES.get(metric, metric)
    scope_title = SCOPE_TITLES.get(scope, scope)
    render_dendrogram_panel(
        axes[0],
        matrix,
        f"Dendrogram — {metric_title} ({scope_title})",
    )
    render_biplot_panel(
        axes[1],
        matrix,
        f"Metric MDS — {metric_title} ({scope_title})",
    )
    fig.legend(
        handles=era_legend_handles(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"Stage 2: model×model linguistic distance — "
        f"{metric_title}, {scope_title}",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    try:
        from PIL import Image
        with Image.open(out_path) as im:
            im.convert("RGB").save(out_path.with_suffix(".jpg"), quality=88,
                                   optimize=True, progressive=True)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        default=None,
        help=("matrices dir; default "
              "results/phase2_mapping/stage2_distance_matrix/matrices "
              "(or the *_with_human path under --include-human)"),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help=("figures dir; default "
              "results/phase2_mapping/stage2_distance_matrix/figures "
              "(or *_with_human under --include-human)"),
    )
    ap.add_argument(
        "--metric",
        default=None,
        help="render only one metric (default: all metrics found)",
    )
    ap.add_argument(
        "--scope",
        default=None,
        help="render only one scope (default: all scopes found)",
    )
    ap.add_argument(
        "--exclude",
        default="",
        help=(
            "comma-separated model_dirs to drop from the matrix before "
            "rendering (e.g. 'run6_lmstudio' to keep §4.1 figures at 13 "
            "leaves while the build matrix has 14 — or 14 leaves, 13 LLM + "
            "Webis, with --include-human --exclude run6_lmstudio)."
        ),
    )
    ap.add_argument(
        "--include-human", action="store_true",
        help=("place the Webis-CMV-20 human baseline as an extra leaf "
              "(green, 'human' cohort) and read from / write to the "
              "stage2_distance_matrix_with_human directories. Reuters-50 is "
              "not in the matrix, so it never appears here."),
    )
    args = ap.parse_args()
    excluded = {m.strip() for m in args.exclude.split(",") if m.strip()}

    global _ACTIVE_ORDER
    if args.include_human:
        _ACTIVE_ORDER = MODEL_ORDER_WITH_HUMAN
        base = str(RESULTS_DIR / "phase2_mapping" / "stage2_distance_matrix_with_human")
    else:
        base = str(RESULTS_DIR / "phase2_mapping" / "stage2_distance_matrix")
    in_dir = Path(args.in_dir or f"{base}/matrices")
    out_dir = Path(args.out_dir or f"{base}/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(in_dir.glob("*__*.csv"))
    if not csvs:
        raise SystemExit(f"No matrix CSVs found in {in_dir}")

    rendered = 0
    headline_matrix: pd.DataFrame | None = None
    for path in csvs:
        stem = path.stem
        if "__" not in stem:
            continue
        metric, scope = stem.split("__", 1)
        if args.metric and metric != args.metric:
            continue
        if args.scope and scope != args.scope:
            continue

        matrix = pd.read_csv(path, index_col=0)
        if excluded:
            keep = [m for m in matrix.index if m not in excluded]
            matrix = matrix.loc[keep, keep]
        out_path = out_dir / f"dendrogram_mds__{metric}__{scope}.png"
        render_metric_pair(matrix, metric, scope, out_path)
        print(f"  wrote {out_path}")
        rendered += 1

        if metric == HEADLINE_METRIC and scope == HEADLINE_SCOPE:
            headline_matrix = matrix.copy()

    # Headline figure (also saved as the canonical "headline.png").
    if headline_matrix is not None:
        out_path = out_dir / "headline.png"
        render_metric_pair(
            headline_matrix, HEADLINE_METRIC, HEADLINE_SCOPE, out_path
        )
        print(f"  wrote {out_path}  (headline)")

    print(f"\nRendered {rendered} (metric × scope) figure pair(s).")


if __name__ == "__main__":
    main()
