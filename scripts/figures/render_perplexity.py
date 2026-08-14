#!/usr/bin/env python3
"""Stage 4 render — perplexity / surprisal figures.

Reads ``per_text_perplexity.csv`` / ``summary.csv`` (from
``perplexity_scoring.py``) and draws per-corpus perplexity-distribution
boxplots, era-coloured, ranked by median (pooled + per-topic), one figure
per reference LM.

**Read with care:** perplexity is measured *under the reference LM(s)* — a
yardstick, not an absolute "fluency" or "quality" score. ``gpt2-large`` is a
2019 LM, so *low* perplexity ≈ "close to 2019 web-text style", *high* ≈
"drifted from a 2019 LM's expectations" (which is where the polished 2026
models land — they're high, not low). For a "distance from a *current* LM's
expectations" read, score under a recent base model too (``--ref-models``;
the Mistral/Llama base models are gated — needs ``HF_TOKEN``).

Usage::

    python3 scripts/figures/render_perplexity.py
    python3 scripts/figures/render_perplexity.py --in-dir <stage4_perplexity dir> --jpg

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

from aiidiolects.visualize_stage1 import ERA_COLORS, MODEL_LABELS, MODEL_ORDER_WITH_HUMAN, era_of
from aiidiolects.paths import RESULTS_DIR

SCOPE_TITLES = {"pooled": "pooled across topics", "climate": "climate (broad)",
                "global_warming": "global warming", "math_anxiety": "math (broad)",
                "misinfo_health": "health misinfo", "reuters_ccat": "Reuters CCAT"}
TOPIC_SCOPES = ["climate", "global_warming", "math_anxiety", "misinfo_health"]


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


def _panel(ax: plt.Axes, pt: pd.DataFrame, ref: str, scope: str,
           ranked: bool = True, show_labels: bool = True) -> list[str]:
    sub = pt[pt["ref_model"] == ref]
    if scope != "pooled":
        sub = sub[sub["topic"] == scope]
    present = set(sub["model_dir"])
    models = [m for m in MODEL_ORDER_WITH_HUMAN if m in present]
    if ranked:
        med = sub.groupby("model_dir")["ppl"].median()
        models = sorted(models, key=lambda m: med.get(m, float("inf")))
    rng = np.random.default_rng(0)
    for i, m in enumerate(models):
        v = sub.loc[sub["model_dir"] == m, "ppl"].dropna().to_numpy()
        col = ERA_COLORS[era_of(m)]
        jitter = rng.normal(0, 0.07, size=len(v))
        ax.scatter(v, np.full(len(v), i) + jitter, s=6, alpha=0.18,
                   color=col, linewidths=0, zorder=2)
        ax.boxplot(v, positions=[i], vert=False, widths=0.55, showfliers=False,
                   patch_artist=True,
                   boxprops=dict(facecolor="white", edgecolor=col, linewidth=1.3),
                   medianprops=dict(color=col, linewidth=2.2),
                   whiskerprops=dict(color=col, linewidth=1.1),
                   capprops=dict(color=col, linewidth=1.1), zorder=3)
    ax.set_yticks(range(len(models)))
    if show_labels:
        ax.set_yticklabels([_short(m) for m in models], fontsize=8)
        for tick, m in zip(ax.get_yticklabels(), models):
            tick.set_color(ERA_COLORS[era_of(m)])
    else:
        ax.set_yticklabels([])
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlabel(f"perplexity under {ref}", fontsize=8.5)
    ax.set_xscale("log")
    ax.grid(True, axis="x", alpha=0.25, linestyle=":", which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(SCOPE_TITLES.get(scope, scope), fontsize=10, loc="left", pad=4)
    return models


def _legend() -> list:
    return [
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["2023"],
                   label="2023 cohort"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["bridge"],
                   label="2026 engine bridge (Mistral)"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["2026"],
                   label="2026 cohort"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["human"],
                   label="human (Phase 2b)"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage4_perplexity"),
    )
    ap.add_argument("--out-dir", default=None, help="default: <in-dir>/figures")
    ap.add_argument("--jpg", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    pt = pd.read_csv(in_dir / "per_text_perplexity.csv")
    refs = sorted(pt["ref_model"].unique())
    print(f"Loaded {len(pt):,} per-text rows; reference LM(s): {refs}")

    for ref in refs:
        tag = ref.replace("/", "_")
        # pooled
        fig, ax = plt.subplots(figsize=(10, 6.4))
        _panel(ax, pt, ref, "pooled")
        fig.legend(handles=_legend(), loc="lower center", ncol=4, frameon=False,
                   fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f"Stage 4: per-corpus perplexity under {ref} (pooled, ranked low→high)\n"
            "yardstick metric — low ≈ close to the reference LM's distribution; "
            "for gpt2-large (a 2019 LM) the polished 2026 models sit high, not low",
            fontsize=11, y=1.0,
        )
        fig.tight_layout(rect=[0, 0.05, 1, 0.94])
        _save(fig, out_dir / f"perplexity__{tag}__pooled.png", args.jpg)
        # by-topic
        fig, axes = plt.subplots(1, len(TOPIC_SCOPES),
                                 figsize=(5.2 * len(TOPIC_SCOPES), 6.2), squeeze=False)
        for ax, sc in zip(axes[0], TOPIC_SCOPES):
            _panel(ax, pt, ref, sc, show_labels=(sc == TOPIC_SCOPES[0]))
        fig.legend(handles=_legend(), loc="lower center", ncol=4, frameon=False,
                   fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(f"Stage 4: per-corpus perplexity under {ref}, by topic "
                     "(ranked low→high within each panel)", fontsize=11, y=1.0)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        _save(fig, out_dir / f"perplexity__{tag}__by_topic.png", args.jpg)


if __name__ == "__main__":
    main()
