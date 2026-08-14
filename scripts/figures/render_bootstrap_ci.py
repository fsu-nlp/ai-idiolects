#!/usr/bin/env python3
"""Stage 2 D: render bootstrap-CI forest plots.

Reads ``bootstrap_summary.csv`` (point + 95 % CI per scope × metric ×
ordered pair) and produces two figures:

* ``engine_bridge_ci__pooled.png`` — small-multiples forest plot
  showing the Mistral-7B 2023 ↔ Mistral-7B 2026 (Run 5) engine
  bridge's point estimate and 95 % CI under each of the 11
  bootstrapped Stage-2 metrics (MATTR is not bootstrapped — see
  ``bootstrap_distances.py`` header).
* ``ranked_pairs_burrows__pooled.png`` — forest plot of all 91 unique
  pairs ranked by Burrows' Δ point estimate, points + 95 % CI bars,
  era-coloured pair labels (2023 ↔ 2023 grey, 2023 ↔ bridge amber,
  2026 ↔ 2026 blue, cross-era purple). The engine-bridge pair is
  visually flagged as the chapter's noise scale-bar.

Both figures are saved as PNG; ``--jpg`` also writes JPG snapshots.

Usage::

    python3 scripts/figures/render_bootstrap_ci.py
    python3 scripts/figures/render_bootstrap_ci.py --in-csv path/to/summary.csv

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
    ERA_2023,
    ERA_2026,
    ERA_BRIDGE,
    ERA_COLORS,
    MODEL_LABELS,
    era_of,
)


METRIC_ORDER = [
    "burrows_delta",
    "cosine_delta",
    "trigram_jsd",
    "bigram_jsd",
    "jaccard_top100",
    "yules_k",
    "sentence_length",
    "text_length",
    "ttr",
    "hapax",
    "zipf_exponent",
]
METRIC_LABELS_SHORT: dict[str, str] = {
    "burrows_delta":   "Burrows' Δ",
    "cosine_delta":    "Cosine Δ",
    "trigram_jsd":     "trigram JSD",
    "bigram_jsd":      "bigram JSD",
    "jaccard_top100":  "1 − Jaccard\n(top-100)",
    "yules_k":         "Yule's K Δ",
    "sentence_length": "sent length\n|d|",
    "text_length":     "text length\n|d|",
    "ttr":             "TTR Δ",
    "hapax":           "hapax Δ",
    "zipf_exponent":   "Zipf-α Δ",
}

BRIDGE_PAIR = ("phase0_mistral-7b", "run5_lmstudio")


def short_pair_label(a: str, b: str) -> str:
    la = MODEL_LABELS.get(a, a).replace("\n", " ")
    lb = MODEL_LABELS.get(b, b).replace("\n", " ")
    return f"{la}  ↔  {lb}"


def pair_era_color(a: str, b: str) -> str:
    """Colour code:
       grey  — both 2023
       blue  — both 2026
       amber — bridge involvement (Run 5 against anything)
       purple — cross-era (2023 vs 2026 non-bridge)"""
    eras = {era_of(a), era_of(b)}
    if "bridge" in eras:
        return ERA_COLORS["bridge"]
    if eras == {"2023"}:
        return ERA_COLORS["2023"]
    if eras == {"2026"}:
        return ERA_COLORS["2026"]
    return "#6f42c1"


# ---------------------------------------------------------------------------
# Figure 1 — engine bridge across metrics
# ---------------------------------------------------------------------------

def render_engine_bridge_ci(
    summary: pd.DataFrame, out_path: Path, scope: str = "pooled"
) -> None:
    sub = summary[
        (summary["scope"] == scope)
        & (summary["model_a"] == BRIDGE_PAIR[0])
        & (summary["model_b"] == BRIDGE_PAIR[1])
    ].set_index("metric")

    metrics_present = [m for m in METRIC_ORDER if m in sub.index]
    n = len(metrics_present)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(13, 2.4 * nrows), squeeze=False
    )
    for ax_idx, metric in enumerate(metrics_present):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]
        row = sub.loc[metric]
        point = row["point"]
        lo = row["boot_2.5"]
        hi = row["boot_97.5"]
        mean = row["boot_mean"]

        ax.barh([0], [hi - lo], left=lo, color="#fee0b6",
                edgecolor="#d97706", height=0.45)
        ax.scatter([point], [0], color="#000", zorder=4, s=40,
                   marker="o", label="point estimate")
        ax.scatter([mean], [0], color="#d97706", zorder=3, s=20,
                   marker="|", label="bootstrap mean")
        ax.text(
            hi, 0.45,
            f" point {point:.4f}\n 95 % CI [{lo:.4f}, {hi:.4f}]",
            va="bottom", ha="right", fontsize=7, color="#333",
        )
        ax.set_yticks([])
        ax.set_xlim(0, max(hi * 1.18, point * 1.18))
        ax.set_title(METRIC_LABELS_SHORT.get(metric, metric),
                     fontsize=10, loc="left", pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)

    for ax_idx in range(n, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Stage 2 D: Mistral-7B 2023 ↔ Mistral-7B 2026 (engine bridge)\n"
        f"point estimate + 95 % bootstrap CI ({scope}, N={int(sub['n_reps'].iloc[0])} reps, "
        "with replacement, n=n_full per corpus)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — ranked pairs forest plot (Burrows pooled)
# ---------------------------------------------------------------------------

def render_ranked_pairs(
    summary: pd.DataFrame, out_path: Path,
    metric: str = "burrows_delta", scope: str = "pooled",
    top_n: int | None = None,
) -> None:
    sub = summary[(summary["scope"] == scope) & (summary["metric"] == metric)].copy()
    sub = sub.sort_values("point").reset_index(drop=True)
    if top_n:
        sub = sub.head(top_n)
    n = len(sub)

    fig, ax = plt.subplots(figsize=(11, max(6.5, 0.18 * n)))
    ys = np.arange(n)
    for i, row in sub.iterrows():
        col = pair_era_color(row["model_a"], row["model_b"])
        bridge_pair = {row["model_a"], row["model_b"]} == set(BRIDGE_PAIR)
        ax.plot(
            [row["boot_2.5"], row["boot_97.5"]],
            [i, i],
            color=col, linewidth=1.6 if bridge_pair else 0.8,
            alpha=0.95 if bridge_pair else 0.7,
        )
        ax.scatter(
            [row["point"]], [i],
            s=42 if bridge_pair else 18,
            color=col, edgecolor="white",
            linewidth=0.8 if bridge_pair else 0.3,
            zorder=4,
        )

    labels = [
        short_pair_label(r["model_a"], r["model_b"])
        for _, r in sub.iterrows()
    ]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7)
    for tick, (_, r) in zip(ax.get_yticklabels(), sub.iterrows()):
        tick.set_color(pair_era_color(r["model_a"], r["model_b"]))
    ax.invert_yaxis()
    ax.set_xlabel(
        f"{METRIC_LABELS_SHORT.get(metric, metric)}  "
        f"(point + 95 % CI, N={int(sub['n_reps'].iloc[0])} reps)",
        fontsize=9,
    )

    legend = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=ERA_COLORS["2023"], label="2023 ↔ 2023"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color="#6f42c1", label="2023 ↔ 2026 (cross-era)"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=ERA_COLORS["2026"], label="2026 ↔ 2026"),
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=ERA_COLORS["bridge"], label="bridge (engine drift on identical weights)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")

    fig.suptitle(
        f"Stage 2 D: all {n} unique pairs ranked by "
        f"{METRIC_LABELS_SHORT.get(metric, metric)} ({scope})\n"
        "engine-bridge pair (Mistral 2023 ↔ Mistral 2026 Run 5) is the "
        "thick amber row — chapter's noise scale-bar",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-csv",
        default=str(
            RESULTS_DIR / "phase2_mapping" / "stage2_distance_matrix"
            / "bootstrap" / "bootstrap_summary.csv"
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=str(
            RESULTS_DIR / "phase2_mapping" / "stage2_distance_matrix"
            / "bootstrap" / "figures"
        ),
    )
    args = ap.parse_args()

    summary = pd.read_csv(args.in_csv)
    print(f"Loaded {len(summary):,} rows from {args.in_csv}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge_path = out_dir / "engine_bridge_ci__pooled.png"
    render_engine_bridge_ci(summary, bridge_path, scope="pooled")
    print(f"  wrote {bridge_path}")

    ranked_path = out_dir / "ranked_pairs_burrows__pooled.png"
    render_ranked_pairs(
        summary, ranked_path, metric="burrows_delta", scope="pooled"
    )
    print(f"  wrote {ranked_path}")


if __name__ == "__main__":
    main()
