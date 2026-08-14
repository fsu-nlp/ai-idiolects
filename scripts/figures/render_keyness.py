#!/usr/bin/env python3
"""Stage 3 render — per-model keyness ("top-N distinctive words").

Reads ``consensus_topN.csv`` (from ``keyness.py``) and draws:

* ``keyness_consensus__<view>__<scope>.png`` — a grid of small panels, one
  per target corpus, each a horizontal-bar list of that corpus's *consensus*
  signature words (those that survive the within-corpus split-half stability
  filter at selection-rate ≥ 0.8), ranked by log-ratio, era-coloured. One
  figure per view (all / content / function).
* ``keyness_signature_overlap__content__<scope>.png`` — a heatmap of the
  content-word signatures that are shared by ≥2 corpora (rows = corpora,
  era-grouped; cols = the shared words; cell = log-ratio): which signature
  words are common across cohorts, and whether a 2023-model signature
  persists into its 2026 successor.

Usage::

    python3 scripts/figures/render_keyness.py
    python3 scripts/figures/render_keyness.py --scope climate --jpg

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
from aiidiolects.keyness import KEYNESS_TARGETS, VIEWS
from aiidiolects.paths import RESULTS_DIR

VIEW_TITLES = {"all": "all words", "content": "content words (NOUN/PROPN/VERB/ADJ/ADV)",
               "function": "function words (DET/ADP/PRON/AUX/CONJ/PART/NUM)"}
SCOPE_TITLES = {"pooled": "pooled across topics", "climate": "climate (broad)",
                "global_warming": "global warming", "math_anxiety": "math (broad)",
                "misinfo_health": "health misinfo"}


def _short(m: str) -> str:
    return MODEL_LABELS.get(m, m).replace("\n", " ")


def _save(fig: plt.Figure, png: Path, write_jpg: bool) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
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


def fig_consensus_grid(cons: pd.DataFrame, view: str, scope: str,
                       out_path: Path, write_jpg: bool, per_panel: int = 12) -> None:
    sub = cons[(cons["view"] == view) & (cons["scope"] == scope)]
    targets = [m for m in KEYNESS_TARGETS if m in sub["target"].unique()]
    if not targets:
        print(f"  (no consensus rows for view={view} scope={scope})")
        return
    ncols = 4
    nrows = int(np.ceil(len(targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows),
                             squeeze=False)
    for idx, m in enumerate(targets):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        g = sub[sub["target"] == m].sort_values("log_ratio", ascending=False).head(per_panel)
        g = g.iloc[::-1]  # so the top word is at the top after barh
        col = ERA_COLORS[era_of(m)]
        ys = np.arange(len(g))
        # selection rate → bar alpha (0.5–1.0)
        alphas = 0.5 + 0.5 * g["selection_rate"].clip(0, 1).to_numpy()
        for y, (_, row) in zip(ys, g.iterrows()):
            ax.barh([y], [row["log_ratio"]], color=col,
                    alpha=0.5 + 0.5 * min(1.0, row["selection_rate"]),
                    edgecolor="white", linewidth=0.4, height=0.7)
        ax.set_yticks(ys)
        ax.set_yticklabels(g["word"].tolist(), fontsize=7)
        ax.set_xlim(left=0)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(_short(m), fontsize=9, color=col, loc="left", pad=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.2, linestyle=":")
        if len(g) == 0:
            ax.text(0.5, 0.5, "(no consensus words)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="#999")
    for idx in range(len(targets), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(
        f"Stage 3: per-corpus consensus keyness — {VIEW_TITLES.get(view, view)}, "
        f"{SCOPE_TITLES.get(scope, scope)}\n"
        "words that stay top-20 (by log-ratio vs the rest of the 13-LLM roster) "
        "in ≥80% of within-corpus half-subsamples; bar length = log-ratio, "
        "opacity = selection rate; human corpora vs the LLM landscape",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_path, write_jpg)


def fig_overlap_heatmap(cons: pd.DataFrame, scope: str, out_path: Path,
                        write_jpg: bool, view: str = "content") -> None:
    sub = cons[(cons["view"] == view) & (cons["scope"] == scope)]
    targets = [m for m in KEYNESS_TARGETS if m in sub["target"].unique()]
    if not targets:
        return
    # words that are a consensus signature for ≥2 targets
    word_targets: dict[str, list[str]] = {}
    for _, row in sub.iterrows():
        word_targets.setdefault(row["word"], []).append(row["target"])
    shared = {w: ms for w, ms in word_targets.items() if len(ms) >= 2}
    if not shared:
        print(f"  (no signature words shared by ≥2 corpora for {view}/{scope})")
        return
    # order columns by (count desc, then by era of the first target that uses it)
    era_rank = {"2023": 0, "bridge": 1, "2026": 2, "human": 3, "other": 4}
    cols = sorted(shared, key=lambda w: (-len(shared[w]),
                  min(era_rank[era_of(m)] for m in shared[w]), w))
    M = pd.DataFrame(np.nan, index=targets, columns=cols)
    for _, row in sub.iterrows():
        if row["word"] in shared:
            M.loc[row["target"], row["word"]] = row["log_ratio"]

    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(cols)), 0.5 * len(targets) + 2))
    vmax = float(np.nanmax(M.values))
    im = ax.imshow(M.values, cmap="magma_r", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels([_short(m) for m in targets], fontsize=8)
    for tick, m in zip(ax.get_yticklabels(), targets):
        tick.set_color(ERA_COLORS[era_of(m)])
    for i in range(len(targets)):
        for j in range(len(cols)):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if v > vmax * 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log-ratio (vs rest of 13-LLM roster)")
    fig.suptitle(
        f"Stage 3: keyness signatures shared by ≥2 corpora — {VIEW_TITLES.get(view, view)}, "
        f"{SCOPE_TITLES.get(scope, scope)}\n"
        "blank = not a consensus signature for that corpus; "
        "look for 2023→2026 same-family persistence and cohort-wide clichés",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_path, write_jpg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage3_keyness"),
    )
    ap.add_argument("--out-dir", default=None, help="default: <in-dir>/figures")
    ap.add_argument("--scope", default="pooled")
    ap.add_argument("--jpg", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    cons = pd.read_csv(in_dir / "consensus_topN.csv")
    print(f"Loaded {len(cons):,} consensus rows; "
          f"scopes={sorted(cons['scope'].unique())}, views={sorted(cons['view'].unique())}")

    for view in VIEWS:
        fig_consensus_grid(cons, view, args.scope,
                           out_dir / f"keyness_consensus__{view}__{args.scope}.png",
                           args.jpg)
    fig_overlap_heatmap(cons, args.scope,
                        out_dir / f"keyness_signature_overlap__content__{args.scope}.png",
                        args.jpg, view="content")


if __name__ == "__main__":
    main()
