#!/usr/bin/env python3
"""Render the Reuters-50 within-author calibration figure.

Reads the CSVs written by ``within_author_baseline.py``:

* ``summary.csv`` — per-``kind`` aggregates (n, min, p25, median, p75,
  max, mean, std) for ``within_author`` (each author's docs split into two
  seeded random halves), ``cross_author`` (different-author pairs),
  ``within_pool`` (random half-splits of all 5,000 docs), plus two
  single-value LLM reference rows: ``llm_within_run_same_metric`` (Run 5 ↔
  Run 6 Mistral-7B, pooled, the *same* per-pair Burrows estimator used for
  the human distributions) and ``llm_within_run_figure_scale_REF`` (the
  §4.1-figure-scale global-MFW estimate of the same pair — labelled, not
  cell-for-cell comparable with the rest).
* ``reuters_within_author.csv`` — the individual Burrows' Δ measurements
  (one row per author-split / author-pair / pool-split).

Output (PNG; ``--jpg`` also writes a JPG snapshot):

* ``within_author_distributions.png`` — a horizontal strip + box plot of
  the three human Burrows' Δ distributions with the LLM within-run value
  marked, annotated with medians and the within-human ÷ within-LLM ratio.

This is a *pipeline calibration*: Reuters-50/C50 is topic-controlled
corporate news, so the within-author spread here measures the stylometry
pipeline's author-resolution on that register — not within-human variance
on the Improta opinion-essay prompts (no open English corpus has the
many-texts-per-author × four-topic structure for that; Talk2AI's Italian
conversational data is the cited proof-of-concept for the design).

Usage::

    python3 scripts/figures/render_within_author.py
    python3 scripts/figures/render_within_author.py --in-dir <dir> --out-dir <dir>

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


# Display order (bottom→top after invert): the "human floor", the
# "human ceiling", the shuffled-pool control.
KIND_ORDER = ["within_author", "within_pool", "cross_author"]
KIND_LABELS = {
    "within_author": "within-author\n(one author, two halves)",
    "within_pool":   "within-pool\n(random 2 500 / 2 500 split)",
    "cross_author":  "cross-author\n(different authors)",
}
KIND_COLORS = {
    "within_author": "#2e7d32",   # green — the human within-author floor
    "within_pool":   "#757575",   # grey — shuffled-pool control
    "cross_author":  "#c62828",   # red — the human cross-author ceiling
}

LLM_POINT_COLOR = "#d97706"        # amber — matches the engine-bridge cue
REF_POINT_COLOR = "#b08000"        # dim amber — the figure-scale reference


def _save_jpg(png_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("  (Pillow not installed — skipping JPG snapshot)")
        return
    jpg_path = png_path.with_suffix(".jpg")
    with Image.open(png_path) as im:
        im.convert("RGB").save(
            jpg_path, quality=88, optimize=True, progressive=True
        )
    print(f"  wrote {jpg_path}")


def render(detail: pd.DataFrame, summary: pd.DataFrame,
           out_path: Path, write_jpg: bool = False) -> None:
    summ = summary.set_index("kind")

    llm_within = (
        float(summ.loc["llm_within_run_same_metric", "median"])
        if "llm_within_run_same_metric" in summ.index else None
    )
    llm_ref = (
        float(summ.loc["llm_within_run_figure_scale_REF", "median"])
        if "llm_within_run_figure_scale_REF" in summ.index else None
    )

    kinds = [k for k in KIND_ORDER if k in detail["kind"].unique()]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    rng = np.random.default_rng(0)

    for i, kind in enumerate(kinds):
        vals = detail.loc[detail["kind"] == kind, "burrows_delta"].dropna().values
        col = KIND_COLORS.get(kind, "#333333")
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(vals, np.full(len(vals), i) + jitter, s=14, alpha=0.35,
                   color=col, linewidths=0, zorder=2)
        bp = ax.boxplot(
            vals, positions=[i], vert=False, widths=0.42, showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="white", edgecolor=col, linewidth=1.4),
            medianprops=dict(color=col, linewidth=2.4),
            whiskerprops=dict(color=col, linewidth=1.2),
            capprops=dict(color=col, linewidth=1.2),
            zorder=3,
        )
        med = float(np.median(vals))
        ax.text(med, i + 0.30, f"median {med:.3f}  (n={len(vals)})",
                ha="center", va="bottom", fontsize=8, color=col)

    # LLM within-run reference line(s) — drawn behind the strips, labelled
    # vertically alongside the line so they never collide with the x-axis.
    mid_y = (len(kinds) - 1) / 2.0
    if llm_within is not None:
        ax.axvline(llm_within, color=LLM_POINT_COLOR, linewidth=2.0,
                   linestyle="-", zorder=1, alpha=0.9)
        ax.text(llm_within, mid_y,
                f"LLM within-run (Run 5 ↔ Run 6, same estimator) = {llm_within:.3f}  ",
                rotation=90, ha="right", va="center", fontsize=8.5,
                color=LLM_POINT_COLOR)
    if llm_ref is not None and abs((llm_ref or 0) - (llm_within or 0)) > 1e-4:
        ax.axvline(llm_ref, color=REF_POINT_COLOR, linewidth=1.2,
                   linestyle=":", zorder=1, alpha=0.8)
        ax.text(llm_ref, mid_y, f"§4.1-figure-scale estimate {llm_ref:.3f}  ",
                rotation=90, ha="right", va="center", fontsize=7.5,
                color=REF_POINT_COLOR, style="italic")

    # Headline ratio annotation.
    if llm_within and "within_author" in detail["kind"].unique():
        wa_med = float(np.median(
            detail.loc[detail["kind"] == "within_author", "burrows_delta"]
        ))
        ratio = wa_med / llm_within
        ax.annotate(
            f"within-human variation ≈ {ratio:.0f}× the LLM within-run floor",
            xy=(0.5, 0.98), xycoords="axes fraction",
            ha="center", va="top", fontsize=9.5, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e1", ec="#d97706",
                      lw=0.8),
        )

    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels([KIND_LABELS.get(k, k) for k in kinds], fontsize=9)
    for tick, k in zip(ax.get_yticklabels(), kinds):
        tick.set_color(KIND_COLORS.get(k, "#333"))
    ax.invert_yaxis()
    ax.set_xlabel("Burrows' Δ (100 MFW of the pair, 20 chunks/side; "
                  "compare_fingerprints.burrows_delta)", fontsize=9)
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Reuters-50 / C50 within-author calibration\n"
        "human writing varies far more within one author than two LLM runs do "
        "— a pipeline-calibration figure (corporate-news register, not the "
        "Improta opinion-essay prompts)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    if write_jpg:
        _save_jpg(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage2_within_author_baseline"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(
            RESULTS_DIR / "phase2_mapping"
            / "stage2_within_author_baseline" / "figures"
        ),
    )
    ap.add_argument("--jpg", action="store_true",
                    help="also write a JPG snapshot next to the PNG")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    detail = pd.read_csv(in_dir / "reuters_within_author.csv")
    summary = pd.read_csv(in_dir / "summary.csv")
    print(f"Loaded {len(detail):,} measurements; "
          f"kinds: {sorted(detail['kind'].unique())}")

    render(detail, summary, Path(args.out_dir) / "within_author_distributions.png",
           write_jpg=args.jpg)


if __name__ == "__main__":
    main()
