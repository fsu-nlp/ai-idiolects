#!/usr/bin/env python3
"""Stage 2 — LLM-vs-human distance figure (close-up companion to the
14-leaf dendrogram).

The era-expanded dendrogram (``render_dendrogram.py --include-human``)
shows the *macro* picture: where the Webis-CMV-20 human corpus sits in
the model×model clustering. This script renders the *close-up*: for one
distance metric, each of the 13 LLM corpora's distance to the human
corpus, ranked low→high, era-coloured, with two reference lines —

* the **LLM↔LLM median** (median of all 78 between-corpus distances
  among the canonical 13-corpus roster), and
* the **Mistral engine-bridge** value (``phase0_mistral-7b ↔
  run5_lmstudio`` — same weights, 2023 vs our 2026 LM Studio re-gen):
  the chapter's noise scale-bar.

Reads:
* ``stage2_distance_matrix_with_human/matrices/<metric>__<scope>.csv``
  (15×15: 13 LLM + run6 + webis_cmv_20 — produced by
  ``build_distance_matrix.py --include-human``). The ``webis_cmv_20`` row
  gives the LLM→human distances; ``run6_lmstudio`` is dropped here so the
  ranking matches the canonical 13-corpus roster.
* ``stage2_distance_matrix/matrices/<metric>__<scope>.csv`` (the canonical
  13×13) for the two reference lines.

Output (PNG; ``--jpg`` also writes JPGs):
* ``llm_vs_human__<metric>__pooled.png`` — ranked bars, pooled across
  topics.
* ``llm_vs_human__<metric>__by_topic.png`` — 1×4 small-multiples, one
  ranked panel per Improta topic. Note Webis ``math_anxiety`` is only
  n≈37 — the bar is drawn (not suppressed) but read it with that caveat.

Usage::

    python3 scripts/figures/render_human_distance.py            # burrows_delta
    python3 scripts/figures/render_human_distance.py --metric cosine_delta
    python3 scripts/figures/render_human_distance.py --all      # every metric, pooled

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
from aiidiolects.visualize_stage1 import ERA_COLORS, MODEL_LABELS, MODEL_ORDER, era_of

HUMAN_KEY = "webis_cmv_20"
BRIDGE_A, BRIDGE_B = "phase0_mistral-7b", "run5_lmstudio"

# Pretty titles — kept in sync with render_dendrogram.METRIC_TITLES.
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
SCOPE_TITLES = {
    "pooled":         "pooled across topics",
    "climate":        "climate (broad)",
    "global_warming": "global warming",
    "math_anxiety":   "math (broad) — n≈37, noisy",
    "misinfo_health": "health misinfo",
}
TOPIC_SCOPES = ["climate", "global_warming", "math_anxiety", "misinfo_health"]


def _short(m: str) -> str:
    return MODEL_LABELS.get(m, m).replace("\n", " ")


def _ref_values(matrices_dir: Path, metric: str, scope: str
                ) -> tuple[float | None, float | None]:
    """(median LLM↔LLM distance, Mistral engine-bridge distance) read from
    the *same* (with-human) matrix the bars come from — so the reference
    lines sit on exactly the metric scale of the bars. The median is taken
    over the canonical 13-corpus LLM roster only (Webis and run6 excluded);
    the engine bridge is phase0_mistral-7b ↔ run5_lmstudio."""
    path = matrices_dir / f"{metric}__{scope}.csv"
    if not path.exists():
        return None, None
    M = pd.read_csv(path, index_col=0)
    llm = [m for m in MODEL_ORDER if m in M.index]   # canonical 13, no Webis/run6
    med = None
    if len(llm) >= 2:
        sub = M.loc[llm, llm].to_numpy(dtype=float)
        iu = np.triu_indices_from(sub, k=1)
        med = float(np.nanmedian(sub[iu]))
    bridge = None
    if BRIDGE_A in M.index and BRIDGE_B in M.columns:
        bridge = float(M.loc[BRIDGE_A, BRIDGE_B])
    return med, bridge


def _load_human_ci(matrices_dir: Path, metric: str, scope: str
                   ) -> dict[str, tuple[float, float]]:
    """{llm model_dir -> (boot_2.5, boot_97.5)} for its distance to the human
    corpus, from the bootstrap summary in the sibling ``bootstrap/`` dir.
    Empty {} if the bootstrap hasn't been run (e.g. it's a Lambda follow-on)."""
    path = matrices_dir.parent / "bootstrap" / "bootstrap_summary.csv"
    if not path.exists():
        return {}
    try:
        bs = pd.read_csv(path)
    except Exception:
        return {}
    need = {"scope", "metric", "model_a", "model_b", "boot_2.5", "boot_97.5"}
    if not need.issubset(bs.columns):
        return {}
    sub = bs[(bs["scope"] == scope) & (bs["metric"] == metric)]
    out: dict[str, tuple[float, float]] = {}
    for _, r in sub.iterrows():
        a, b = r["model_a"], r["model_b"]
        other = a if b == HUMAN_KEY else (b if a == HUMAN_KEY else None)
        if other is None:
            continue
        out[other] = (float(r["boot_2.5"]), float(r["boot_97.5"]))
    return out


def _ranked_panel(ax: plt.Axes, dist: pd.Series, metric: str, scope: str,
                  llm_median: float | None, bridge: float | None,
                  show_value_labels: bool = True,
                  ci: dict[str, tuple[float, float]] | None = None) -> None:
    dist = dist.dropna().sort_values()
    models = list(dist.index)
    ys = np.arange(len(models))
    colors = [ERA_COLORS[era_of(m)] for m in models]

    ax.barh(ys, dist.values, color=colors, edgecolor="white",
            linewidth=0.5, height=0.66, zorder=3)
    if ci:
        for y, m, v in zip(ys, models, dist.values):
            lohi = ci.get(m)
            if lohi is None:
                continue
            lo, hi = lohi
            ax.errorbar([v], [y], xerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                        fmt="none", ecolor="#222", elinewidth=1.0, capsize=2.5,
                        zorder=5)
    if show_value_labels:
        for y, v in zip(ys, dist.values):
            ax.text(v, y, f" {v:.3f}", va="center", ha="left",
                    fontsize=7, color="#333", zorder=4)

    if bridge is not None:
        ax.axvline(bridge, color="#d97706", linewidth=1.8, linestyle="-",
                   zorder=2, alpha=0.9)
    if llm_median is not None:
        ax.axvline(llm_median, color="#555", linewidth=1.2, linestyle="--",
                   zorder=2, alpha=0.8)

    ax.set_yticks(ys)
    ax.set_yticklabels([_short(m) for m in models], fontsize=8)
    for tick, m in zip(ax.get_yticklabels(), models):
        tick.set_color(ERA_COLORS[era_of(m)])
    ax.invert_yaxis()
    ax.set_xlabel(f"distance to human corpus — {METRIC_TITLES.get(metric, metric)}",
                  fontsize=8.5)
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(SCOPE_TITLES.get(scope, scope), fontsize=10, loc="left", pad=4)


def _legend_handles(have_bridge: bool, have_median: bool,
                    have_ci: bool = False) -> list:
    h = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["2023"],
                   label="2023 cohort"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["bridge"],
                   label="2026 engine bridge (Mistral)"),
        plt.Line2D([0], [0], marker="s", linestyle="", color=ERA_COLORS["2026"],
                   label="2026 cohort"),
    ]
    if have_bridge:
        h.append(plt.Line2D([0], [0], color="#d97706", linewidth=1.8,
                            label="Mistral engine-bridge distance (noise scale-bar)"))
    if have_median:
        h.append(plt.Line2D([0], [0], color="#555", linewidth=1.2, linestyle="--",
                            label="median LLM↔LLM distance"))
    if have_ci:
        h.append(plt.Line2D([0], [0], color="#222", linewidth=1.0,
                            marker="|", markersize=8,
                            label="95% bootstrap CI"))
    return h


def _llm_to_human(matrices_dir: Path, metric: str, scope: str) -> pd.Series | None:
    path = matrices_dir / f"{metric}__{scope}.csv"
    if not path.exists():
        return None
    M = pd.read_csv(path, index_col=0)
    if HUMAN_KEY not in M.index:
        return None
    llm_models = [m for m in MODEL_ORDER if m in M.index]   # the canonical 13
    return M.loc[HUMAN_KEY, llm_models].astype(float)


def _save_jpg(png: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(png) as im:
        im.convert("RGB").save(png.with_suffix(".jpg"), quality=88,
                               optimize=True, progressive=True)
    print(f"  wrote {png.with_suffix('.jpg')}")


def render_pooled(matrices_dir: Path, out_dir: Path,
                  metric: str, write_jpg: bool) -> bool:
    dist = _llm_to_human(matrices_dir, metric, "pooled")
    if dist is None:
        print(f"  (skip {metric}: no webis row in {metric}__pooled.csv)")
        return False
    llm_med, bridge = _ref_values(matrices_dir, metric, "pooled")
    ci = _load_human_ci(matrices_dir, metric, "pooled")

    fig, ax = plt.subplots(figsize=(10, 5.4))
    _ranked_panel(ax, dist, metric, "pooled", llm_med, bridge, ci=ci)
    fig.legend(handles=_legend_handles(bridge is not None, llm_med is not None,
                                       have_ci=bool(ci)),
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Stage 2: each LLM corpus' distance to human prose (Webis-CMV-20)\n"
        f"{METRIC_TITLES.get(metric, metric)}, pooled across topics — "
        "ranked closest→furthest from human writing",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = out_dir / f"llm_vs_human__{metric}__pooled.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    if write_jpg:
        _save_jpg(out)
    return True


def render_by_topic(matrices_dir: Path, out_dir: Path,
                    metric: str, write_jpg: bool) -> bool:
    panels = []
    for sc in TOPIC_SCOPES:
        d = _llm_to_human(matrices_dir, metric, sc)
        if d is not None:
            panels.append((sc, d))
    if not panels:
        return False

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5.6),
                             squeeze=False)
    have_bridge = have_median = have_ci = False
    for ax, (sc, d) in zip(axes[0], panels):
        llm_med, bridge = _ref_values(matrices_dir, metric, sc)
        ci = _load_human_ci(matrices_dir, metric, sc)
        have_bridge |= bridge is not None
        have_median |= llm_med is not None
        have_ci |= bool(ci)
        _ranked_panel(ax, d, metric, sc, llm_med, bridge,
                      show_value_labels=False, ci=ci)
    fig.legend(handles=_legend_handles(have_bridge, have_median, have_ci=have_ci),
               loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Stage 2: LLM-to-human ({METRIC_TITLES.get(metric, metric)}) "
        "by topic — reference lines are per-topic (Mistral bridge solid amber, "
        "median LLM↔LLM dashed grey)",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = out_dir / f"llm_vs_human__{metric}__by_topic.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    if write_jpg:
        _save_jpg(out)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matrices-dir",
        default=str(RESULTS_DIR / "phase2_mapping"
                    / "stage2_distance_matrix_with_human" / "matrices"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "phase2_mapping"
                    / "stage2_distance_matrix_with_human" / "figures"),
    )
    ap.add_argument("--metric", default="burrows_delta",
                    help="distance metric to plot (default burrows_delta)")
    ap.add_argument("--all", action="store_true",
                    help="render every metric found (pooled), plus the "
                         "burrows_delta by-topic panel")
    ap.add_argument("--jpg", action="store_true",
                    help="also write JPG snapshots")
    args = ap.parse_args()

    matrices_dir = Path(args.matrices_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        metrics = sorted({
            p.stem.split("__", 1)[0]
            for p in matrices_dir.glob("*__pooled.csv")
        })
        print(f"Rendering pooled LLM-vs-human for {len(metrics)} metric(s)")
        for m in metrics:
            render_pooled(matrices_dir, out_dir, m, args.jpg)
        render_by_topic(matrices_dir, out_dir, "burrows_delta", args.jpg)
    else:
        render_pooled(matrices_dir, out_dir, args.metric, args.jpg)
        render_by_topic(matrices_dir, out_dir, args.metric, args.jpg)


if __name__ == "__main__":
    main()
