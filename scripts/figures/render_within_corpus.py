#!/usr/bin/env python3
"""Stage 3 render — within-corpus variation ("within vs between").

Reads ``within_corpus_long.csv`` / ``summary.csv`` (from
``within_corpus_variation.py``) and draws, per corpus, the distribution of
half-split distances (the corpus's internal heterogeneity), ranked, era-
coloured, with two reference lines on the *same* scale: the Mistral 2023↔2026
(Run 5) engine bridge (amber — two independent generations of the same
weights) and the median between-corpus pair distance (grey dashed). The
message: model M's own within-corpus distance is X; its distance to model N
is Y; Y/X is how confidently the §4.1 metric separates them — and the human
corpora (Phase 2b) sit at the heterogeneous end.

Figures (PNG; ``--jpg`` also writes JPGs):

* ``within_corpus_<metric>__pooled.png`` — ranked strip + box, pooled scope.
* ``within_corpus_<metric>__by_topic.png`` — 1×4 small-multiples per topic.

Usage::

    python3 scripts/figures/render_within_corpus.py
    python3 scripts/figures/render_within_corpus.py --metric cosine_delta --jpg

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

from aiidiolects.visualize_stage1 import ERA_COLORS, MODEL_LABELS, era_of
from aiidiolects.paths import RESULTS_DIR

METRIC_TITLES = {
    "burrows_delta": "Burrows' Δ (100 MFW, per-text centroids)",
    "cosine_delta":  "Cosine Δ (100 MFW, per-text centroids)",
}
SCOPE_TITLES = {
    "pooled": "pooled across topics", "climate": "climate (broad)",
    "global_warming": "global warming", "math_anxiety": "math (broad) — thin",
    "misinfo_health": "health misinfo",
}
TOPIC_SCOPES = ["climate", "global_warming", "math_anxiety", "misinfo_health"]


def _short(m: str) -> str:
    return MODEL_LABELS.get(m, m).replace("\n", " ")


def _ref_lines(long: pd.DataFrame, metric: str) -> tuple[float | None, float | None]:
    """(engine-bridge value, median between-corpus pair value) — pooled scope."""
    bp = long[(long["scope"] == "pooled")
              & long["kind"].isin(["between_pair", "between_bridge"])]
    if bp.empty:
        return None, None
    bridge_v = bp.loc[bp["kind"] == "between_bridge", metric]
    bridge = float(bridge_v.iloc[0]) if len(bridge_v) else None
    med = float(bp[metric].median())
    return bridge, med


def _panel(ax: plt.Axes, long: pd.DataFrame, metric: str, scope: str,
           bridge: float | None, med: float | None,
           show_labels: bool = True) -> None:
    sub = long[(long["kind"] == "within_split") & (long["scope"] == scope)]
    # rank corpora by median within-distance
    med_by_m = sub.groupby("model_dir")[metric].median().sort_values()
    models = list(med_by_m.index)
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        vals = sub.loc[sub["model_dir"] == m, metric].dropna().to_numpy()
        col = ERA_COLORS[era_of(m)]
        jitter = rng.normal(0, 0.07, size=len(vals))
        ax.scatter(vals, np.full(len(vals), i) + jitter, s=10, alpha=0.35,
                   color=col, linewidths=0, zorder=2)
        ax.boxplot(vals, positions=[i], vert=False, widths=0.5, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor=col, linewidth=1.3),
                   medianprops=dict(color=col, linewidth=2.2),
                   whiskerprops=dict(color=col, linewidth=1.1),
                   capprops=dict(color=col, linewidth=1.1), zorder=3)
    # x-scale: zoom to the within-corpus boxes + the engine-bridge line. The
    # median between-corpus pair is ~10× larger — drawn on-scale if it fits,
    # otherwise annotated off the right edge so the within differences stay legible.
    within_max = float(sub[metric].max()) if not sub.empty else 0.1
    xmax = max(within_max * 1.25, (bridge or 0) * 1.15, 0.05)
    if bridge is not None:
        ax.axvline(bridge, color=ERA_COLORS["bridge"], linewidth=1.8,
                   linestyle="-", zorder=1, alpha=0.9)
    if med is not None:
        if med <= xmax:
            ax.axvline(med, color="#555", linewidth=1.2, linestyle="--",
                       zorder=1, alpha=0.85)
        else:
            ax.annotate(f"median between-corpus pair = {med:.2f}  →",
                        xy=(xmax, len(models) - 1), xytext=(xmax * 0.99, len(models) - 1),
                        ha="right", va="center", fontsize=7.5, color="#555",
                        style="italic")
    ax.set_yticks(range(len(models)))
    if show_labels:
        ax.set_yticklabels([_short(m) for m in models], fontsize=8)
        for tick, m in zip(ax.get_yticklabels(), models):
            tick.set_color(ERA_COLORS[era_of(m)])
    else:
        ax.set_yticklabels([])
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlabel(f"within-corpus half-split distance — "
                  f"{METRIC_TITLES.get(metric, metric)}", fontsize=8.5)
    ax.set_xlim(0, xmax)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(SCOPE_TITLES.get(scope, scope), fontsize=10, loc="left", pad=4)


def _legend(have_bridge: bool, have_med: bool) -> list:
    h = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["2023"],
                   label="2023 cohort"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["bridge"],
                   label="2026 engine bridge (Mistral)"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["2026"],
                   label="2026 cohort"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["human"],
                   label="human (Phase 2b)"),
    ]
    if have_bridge:
        h.append(plt.Line2D([0], [0], color=ERA_COLORS["bridge"], linewidth=1.8,
                            label="Mistral engine-bridge distance"))
    if have_med:
        h.append(plt.Line2D([0], [0], color="#555", linewidth=1.2, linestyle="--",
                            label="median between-corpus pair"))
    return h


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage3_within_corpus_variation"),
    )
    ap.add_argument("--out-dir", default=None, help="default: <in-dir>/figures")
    ap.add_argument("--metric", default="burrows_delta",
                    help="burrows_delta | cosine_delta")
    ap.add_argument("--jpg", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    long = pd.read_csv(in_dir / "within_corpus_long.csv")
    print(f"Loaded {len(long):,} rows; "
          f"corpora: {sorted(long.loc[long.kind=='within_split','model_dir'].unique())}")
    bridge, med = _ref_lines(long, args.metric)

    # pooled
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _panel(ax, long, args.metric, "pooled", bridge, med)
    fig.legend(handles=_legend(bridge is not None, med is not None),
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Stage 3: within-corpus variation — each corpus' internal heterogeneity "
        "(half-split distance)\nvs the engine-bridge noise floor and the typical "
        "between-corpus distance — human corpora (Phase 2b) at the heterogeneous end",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    _save(fig, out_dir / f"within_corpus_{args.metric}__pooled.png", args.jpg)

    # by-topic
    fig, axes = plt.subplots(1, len(TOPIC_SCOPES),
                             figsize=(5.2 * len(TOPIC_SCOPES), 5.8), squeeze=False)
    for ax, sc in zip(axes[0], TOPIC_SCOPES):
        _panel(ax, long, args.metric, sc, bridge, med,
               show_labels=(sc == TOPIC_SCOPES[0]))
    fig.legend(handles=_legend(bridge is not None, med is not None),
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Stage 3: within-corpus {METRIC_TITLES.get(args.metric, args.metric)} "
        "by topic (reference lines = pooled engine-bridge solid amber, median "
        "between-corpus pair dashed grey)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, out_dir / f"within_corpus_{args.metric}__by_topic.png", args.jpg)


if __name__ == "__main__":
    main()
