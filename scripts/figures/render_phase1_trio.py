#!/usr/bin/env python3
"""Phase 1 anchoring trio — beyond-descriptives view.

Filters the Stage 2 long-format distance CSV to the three-corpus
anchoring trio (Improta 2023 originals, our 2026 Run 5 anchor, our
2026 Run 6 stochastic baseline) and renders the chapter's Phase 1
question directly:

    Is the Originals ↔ Run 5 systematic gap distinguishable from
    the Run 5 ↔ Run 6 stochastic noise floor under each Stage 2
    metric?

For each metric we have three pairwise distances:

* **Originals ↔ Run 5** — systematic gap (engine + time drift on
  identical weights). The chapter's Phase 1 headline.
* **Originals ↔ Run 6** — independent draw of the same systematic
  gap; ought to be ≈ Originals ↔ Run 5 within stochastic noise.
* **Run 5 ↔ Run 6** — stochastic noise floor: same engine, same
  weights, same prompt, same sampling, different random seed.

If gap > noise (visibly), the systematic Originals-vs-2026
difference is real. If gap ≈ noise, the replication is
indistinguishable from re-running the same setup.

All distances use the same global MFW + global z-score reference as
the Phase 2 §4.1 macro matrix (built across all 14 corpora pooled);
absolute values are therefore directly comparable to the chapter's
macro figure.

Outputs::

    results/phase1_anchoring/stage2_distance_matrix/
      trio_distances.csv          (long-format: metric × scope × pair × d)
      gap_vs_noise_table.csv      (one row per metric × scope: gap, noise,
                                   ratio, distinguishable_yes_no)
      figures/
        gap_vs_noise__pooled.jpg  (12 metrics, 3 bars each — headline)
        gap_vs_noise__<topic>.jpg (per-topic supplements)

Usage::

    python3 scripts/figures/render_phase1_trio.py
    python3 scripts/figures/render_phase1_trio.py \\
        --in-csv path/to/all_distances_long.csv \\
        --out-dir path/to/output

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

# Match build_distance_matrix.py / visualize_stage1.py.
TRIO = ["phase0_mistral-7b", "run5_lmstudio", "run6_lmstudio"]
ORIGINALS, RUN5, RUN6 = TRIO

# Pair labels for readability.
PAIR_LABELS = {
    ("phase0_mistral-7b", "run5_lmstudio"): "Originals ↔ Run 5  (systematic gap)",
    ("phase0_mistral-7b", "run6_lmstudio"): "Originals ↔ Run 6  (systematic gap, independent draw)",
    ("run5_lmstudio",     "run6_lmstudio"): "Run 5 ↔ Run 6      (stochastic noise floor)",
}
# Visual encoding: noise floor in a different colour so the comparison reads.
PAIR_COLORS = {
    "Originals ↔ Run 5  (systematic gap)":                       "#666666",
    "Originals ↔ Run 6  (systematic gap, independent draw)":     "#999999",
    "Run 5 ↔ Run 6      (stochastic noise floor)":               "#d97706",
}

METRIC_ORDER = [
    "text_length", "sentence_length",
    "ttr", "mattr", "hapax", "yules_k",
    "jaccard_top100",
    "bigram_jsd", "trigram_jsd",
    "burrows_delta", "cosine_delta",
    "zipf_exponent",
]
METRIC_LABELS = {
    "text_length":     "text length\n(|Cohen's d|)",
    "sentence_length": "sentence length\n(|Cohen's d|)",
    "ttr":             "TTR Δ",
    "mattr":           "MATTR Δ\n(w=500)",
    "hapax":           "hapax Δ",
    "yules_k":         "Yule's K Δ",
    "jaccard_top100":  "1 − Jaccard\n(top-100)",
    "bigram_jsd":      "bigram JSD",
    "trigram_jsd":     "trigram JSD",
    "burrows_delta":   "Burrows' Δ\n(100 MFW)",
    "cosine_delta":    "Cosine Δ\n(100 MFW)",
    "zipf_exponent":   "Zipf-α Δ",
}

SCOPE_TITLES = {
    "pooled":         "pooled across topics",
    "climate":        "climate",
    "global_warming": "global warming",
    "math_anxiety":   "math anxiety",
    "misinfo_health": "health misinfo",
}


def trio_pair_distances(
    long_df: pd.DataFrame, scope: str
) -> pd.DataFrame:
    """Return one row per (metric × pair_label) for the given scope."""
    sub = long_df[(long_df["scope"] == scope)
                  & long_df["model_a"].isin(TRIO)
                  & long_df["model_b"].isin(TRIO)].copy()

    rows = []
    for _, r in sub.iterrows():
        a, b = sorted([r["model_a"], r["model_b"]])
        label = PAIR_LABELS.get((a, b))
        if label is None:
            continue
        rows.append({
            "metric":   r["metric"],
            "pair":     label,
            "distance": float(r["distance"]),
        })
    df = pd.DataFrame(rows)
    return df


def gap_vs_noise(long_df: pd.DataFrame) -> pd.DataFrame:
    """One summary row per (metric × scope) — gap (Originals↔Run5),
    noise (Run5↔Run6), and the ratio."""
    rows: list[dict] = []
    for scope in long_df["scope"].unique():
        scope_df = long_df[long_df["scope"] == scope]
        for metric in METRIC_ORDER:
            mdf = scope_df[scope_df["metric"] == metric]

            def find(a: str, b: str) -> float:
                a, b = sorted([a, b])
                row = mdf[
                    (mdf["model_a"] == a) & (mdf["model_b"] == b)
                ]
                if row.empty:
                    row = mdf[
                        (mdf["model_a"] == b) & (mdf["model_b"] == a)
                    ]
                return float(row["distance"].iloc[0]) if not row.empty else float("nan")

            gap1 = find(ORIGINALS, RUN5)
            gap2 = find(ORIGINALS, RUN6)
            noise = find(RUN5, RUN6)
            ratio = float("nan") if not noise else (max(gap1, gap2) / noise)
            rows.append({
                "metric":         metric,
                "scope":          scope,
                "gap_orig_run5":  gap1,
                "gap_orig_run6":  gap2,
                "noise_run5_run6": noise,
                "max_gap_over_noise": ratio,
            })
    return pd.DataFrame(rows)


def render_gap_vs_noise(
    pair_df: pd.DataFrame, scope: str, out_path: Path
) -> None:
    """One panel per metric, three bars per panel — gap vs noise floor."""
    metrics = [m for m in METRIC_ORDER if m in pair_df["metric"].unique()]
    n_metrics = len(metrics)
    ncols = 4
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(13, 2.6 * nrows), squeeze=False
    )

    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        sub = pair_df[pair_df["metric"] == metric]
        # Always plot in fixed order so colours map consistently.
        order = list(PAIR_LABELS.values())
        ys = [sub[sub["pair"] == p]["distance"].iloc[0]
              if not sub[sub["pair"] == p].empty else 0.0
              for p in order]
        colors = [PAIR_COLORS[p] for p in order]
        bars = ax.barh(range(len(order)), ys, color=colors, edgecolor="white")
        for bar, y in zip(bars, ys):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f"  {y:.3f}",
                va="center", fontsize=8, color="#333",
            )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([
            "Orig ↔ Run 5", "Orig ↔ Run 6", "Run 5 ↔ Run 6",
        ], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(METRIC_LABELS.get(metric, metric),
                     fontsize=9, loc="left", pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
        # Slight headroom for the value labels.
        if max(ys) > 0:
            ax.set_xlim(0, max(ys) * 1.18)

    # Hide unused axes.
    for idx in range(n_metrics, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PAIR_COLORS[p]) for p in PAIR_LABELS.values()
    ]
    fig.legend(
        handles=legend_handles,
        labels=list(PAIR_LABELS.values()),
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Phase 1 anchoring trio — gap vs stochastic noise floor "
        f"({SCOPE_TITLES.get(scope, scope)})\n"
        f"Originals (Improta) / Run 5 (LM Studio 2026) / Run 6 "
        f"(stochastic baseline). Same global reference as the §4.1 macro matrix.",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-csv",
        default=str(
            RESULTS_DIR / "phase2_mapping" / "stage2_distance_matrix"
            / "all_distances_long.csv"
        ),
    )
    ap.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "phase1_anchoring" / "stage2_distance_matrix"),
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.read_csv(args.in_csv)
    print(f"Loaded {len(long_df):,} rows from {args.in_csv}")

    trio_long = long_df[
        long_df["model_a"].isin(TRIO) & long_df["model_b"].isin(TRIO)
    ].copy()
    trio_long.to_csv(out_dir / "trio_distances.csv", index=False, float_format="%.6f")
    print(f"  wrote {out_dir / 'trio_distances.csv'}  "
          f"({len(trio_long):,} rows)")

    summary = gap_vs_noise(long_df)
    summary.to_csv(out_dir / "gap_vs_noise_table.csv", index=False, float_format="%.6f")
    print(f"  wrote {out_dir / 'gap_vs_noise_table.csv'}  "
          f"({len(summary):,} rows)")

    # Quick stdout digest of the pooled scope.
    pooled = summary[summary["scope"] == "pooled"].set_index("metric")
    print("\nPooled-scope summary (gap = max(orig↔run5, orig↔run6); ratio = gap / noise):")
    print(pooled[["gap_orig_run5", "gap_orig_run6",
                  "noise_run5_run6", "max_gap_over_noise"]].round(4))

    for scope in ["pooled"] + sorted(
        s for s in long_df["scope"].unique() if s != "pooled"
    ):
        pair_df = trio_pair_distances(long_df, scope)
        if pair_df.empty:
            continue
        out_path = fig_dir / f"gap_vs_noise__{scope}.png"
        render_gap_vs_noise(pair_df, scope, out_path)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
