#!/usr/bin/env python3
"""Stage 3A render — syntactic-feature figures.

Reads ``per_text_syntactic.csv`` / ``summary_syntactic.csv`` /
``per_length_syntactic.csv`` / ``dl_by_function__<scope>.csv`` (from
``syntactic_features.py``) and writes:

* ``fig_syntactic_complexity.png`` — 6-panel per-corpus boxplots: mean
  dependency distance, mean hierarchical distance (MHD), mean per-sentence
  tree depth, passive-construction rate, subordinate-clause ratio, and mean
  noun-chunk length. Era-coloured row labels (2023 grey / bridge amber /
  2026 blue / human green), reusing ``visualize_stage1.boxplot_by_model``.
* ``fig_pos_distribution.png`` — heatmap of Universal-POS-tag relative
  frequencies per corpus (pooled scope).
* ``fig_dl_vs_length.png`` — per-sentence-length mean dependency length,
  one curve per era. The Juzek/Krielke/Teich (UDW 2020) Figure 2 idiom — a
  length-controlled view that decides whether cross-era MDD differences
  are genuine or length-driven.
* ``fig_mhd_vs_length.png`` — same idiom for MHD (the depth analogue).
* ``fig_dl_by_function.png`` — pooled mean DL per UD dep label, ranked
  long→short, with bars darkened when the label's frequency decreases from
  the 2023 to the 2026 cohort (UDW 2020 Figure 4 idiom).
* ``fig_clause_type_rates.png`` — per-corpus rates for the seven long-DL
  clause functions (relcl / advcl / ccomp / xcomp / acl / conj / parataxis),
  small-multiple panels with era-coloured bars.

The POS-ngram and dep-label JSD distance matrices are rendered with the
existing ``render_dendrogram.py`` (generic over ``<metric>__<scope>.csv``)::

    python3 scripts/figures/render_dendrogram.py --include-human \\
        --in-dir  results/phase2_mapping/stage3_syntactic/matrices \\
        --out-dir results/phase2_mapping/stage3_syntactic/figures \\
        --exclude run6_lmstudio

Methodological template: Juzek, Krielke & Teich (2020), *"Exploring
diachronic syntactic shifts with dependency length: the case of scientific
English"*, Proceedings of UDW 2020. We follow their Figure-2 length-control
and Figure-4 by-function decomposition; MHD is the depth measure they
flagged as future work (their §5).

Usage::

    python3 scripts/figures/render_syntactic.py
    python3 scripts/figures/render_syntactic.py --in-dir <stage3_syntactic dir>

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

from aiidiolects.visualize_stage1 import (
    ERA_COLORS,
    MODEL_LABELS,
    MODEL_ORDER_WITH_HUMAN,
    boxplot_by_model,
    era_of,
    heatmap,
    topic_legend_handles,
)
from aiidiolects.paths import RESULTS_DIR

# Universal POS tags, in a readable order (content → function → other).
UPOS_ORDER = [
    "NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV",
    "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "NUM", "INTJ",
    "PUNCT", "SYM", "X",
]

# Sentence-length window for the Figure-2 idiom curves (matches Juzek 2020
# Figure 2 x-range; the per_length_syntactic.csv caps at 80).
LEN_MIN, LEN_MAX = 5, 60

# Per-corpus colour palette for the all-corpora overlay figures. Designed so
# each era family stays visually distinct (2023 = greys, bridge = amber,
# 2026 closed-API = blues, 2026 open-weights = purples, human = greens) while
# every corpus has a unique colour. Defaults to ERA_COLORS['other'] if a
# corpus is missing here (shouldn't happen for the canonical roster).
MODEL_COLORS = {
    # 2023 cohort — greys (light → dark)
    "phase0_gpt-3.5":        "#9ca3af",
    "phase0_gpt-4o":         "#6b7280",
    "phase0_haiku":          "#374151",
    "phase0_llama-3-8b":     "#bdbdbd",
    "phase0_llama-3.1-70b":  "#737373",
    "phase0_mistral-7b":     "#525252",
    # 2026 engine bridge — amber
    "run5_lmstudio":         "#d97706",
    # 2026 closed-API cohort — blues
    "gemini-3-flash":        "#3b82f6",
    "gpt-5-4-mini":          "#1d4ed8",
    "haiku-4-5":             "#0c2769",
    # 2026 open-weights cohort — purples
    "olmo3-7b":              "#c084fc",
    "nemo-12b":              "#9333ea",
    "qwen3-14b":             "#581c87",
    # Human (Phase 2b) — greens
    "webis_cmv_20":          "#16a34a",
    "reuters_50":            "#064e3b",
}

# Clause-type rates to display in fig_clause_type_rates (UDW 2020 §4.2
# long-DL functions); order = long-DL first.
CLAUSE_TYPES = [
    ("parataxis_rate", "parataxis"),
    ("advcl_rate", "adverbial clause (advcl)"),
    ("ccomp_rate", "complement clause (ccomp)"),
    ("conj_rate", "coordination (conj)"),
    ("relcl_rate", "relative clause (relcl / acl:relcl)"),
    ("acl_rate", "non-finite modifier (acl)"),
    ("xcomp_rate", "open complement (xcomp)"),
]


def _order(present: set[str]) -> list[str]:
    return [m for m in MODEL_ORDER_WITH_HUMAN if m in present]


def _save(fig: plt.Figure, png_path: Path, write_jpg: bool) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png_path}")
    if write_jpg:
        try:
            from PIL import Image
        except ImportError:
            return
        with Image.open(png_path) as im:
            im.convert("RGB").save(png_path.with_suffix(".jpg"), quality=88,
                                   optimize=True, progressive=True)
        print(f"  wrote {png_path.with_suffix('.jpg')}")


# ---------------------------------------------------------------------------
# Complexity boxplots (6 panels: MDD, MHD, tree_depth, passive, subord, NP-len)
# ---------------------------------------------------------------------------

def fig_complexity(pt: pd.DataFrame, out_path: Path, write_jpg: bool) -> None:
    models = _order(set(pt["model_dir"]))
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5), sharey=True)

    boxplot_by_model(axes[0, 0], pt, "mdd",
                     "Mean dependency distance (MDD)",
                     "|i − head.i| (tokens)", models)
    axes[0, 0].set_xlim(left=0)

    boxplot_by_model(axes[0, 1], pt, "mhd",
                     "Mean hierarchical distance (MHD)",
                     "arcs to root (tokens)", models)
    axes[0, 1].set_xlim(left=0)

    boxplot_by_model(axes[0, 2], pt, "tree_depth",
                     "Mean per-sentence tree depth",
                     "max ancestor count", models)
    axes[0, 2].set_xlim(left=0)

    boxplot_by_model(axes[1, 0], pt, "passive_rate",
                     "Passive-construction rate",
                     "share of sentences", models)
    axes[1, 0].set_xlim(0, 1)

    boxplot_by_model(axes[1, 1], pt, "subord_ratio",
                     "Subordinate-clause ratio",
                     "embedded clauses / sentence", models)
    axes[1, 1].set_xlim(left=0)

    boxplot_by_model(axes[1, 2], pt, "mean_noun_chunk_len",
                     "Mean noun-chunk length",
                     "tokens / NP", models)
    axes[1, 2].set_xlim(left=0)

    # boxplot_by_model calls invert_yaxis() per panel; with sharey=True and an
    # even panel count those cancel out, so set the direction explicitly on
    # the shared y so the first model in the order sits at the top.
    axes[0, 0].set_ylim(len(models) + 0.5, 0.5)

    fig.suptitle(
        "Stage 3A: syntactic complexity by corpus "
        "(spaCy en_core_web_lg parse; quality-clean texts; "
        "human corpora = Phase 2b, green)\n"
        "MDD = Liu 2008 (punct-excluded); MHD = Liu 2010 / Jiang & Liu 2015. "
        "Methodological template: Juzek/Krielke/Teich, UDW 2020.",
        fontsize=12, y=1.0,
    )
    fig.legend(handles=topic_legend_handles(), loc="lower center", ncol=4,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    _save(fig, out_path, write_jpg)


# ---------------------------------------------------------------------------
# POS distribution heatmap (unchanged from prior Stage 3A)
# ---------------------------------------------------------------------------

def fig_pos(summary: pd.DataFrame, out_path: Path, write_jpg: bool) -> None:
    pooled = summary[summary["scope"] == "pooled"].set_index("model_dir")
    models = _order(set(pooled.index))
    tags = [t for t in UPOS_ORDER if f"pos_{t}" in pooled.columns]
    mat = pooled.loc[models, [f"pos_{t}" for t in tags]].copy()
    mat.columns = tags
    fig, ax = plt.subplots(figsize=(13, 6.5))
    im = heatmap(ax, mat, "Universal-POS-tag relative frequency (pooled)",
                 cmap="viridis", vmin=0.0,
                 vmax=float(np.nanmax(mat.values)), annotate_fmt="{:.2f}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="relative frequency")
    fig.suptitle(
        "Stage 3A: POS-tag profile by corpus — note the human rows' "
        "low NOUN / high PRON share vs the 2023 cohort's nominal, impersonal "
        "profile (2026 models sit in between)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_path, write_jpg)


# ---------------------------------------------------------------------------
# Length-binned curves (Juzek 2020 Figure 2 idiom)
# ---------------------------------------------------------------------------

def _era_weighted_mean(df: pd.DataFrame, value_col: str) -> dict[str, pd.Series]:
    """For each era, return a Series indexed by sent_len holding the
    n_sents-weighted mean of ``value_col`` across corpora in that era."""
    df = df.copy()
    df["era"] = df["model_dir"].map(era_of)
    out: dict[str, pd.Series] = {}
    for era_key in ("2023", "bridge", "2026", "human"):
        sel = df[df["era"] == era_key]
        if sel.empty:
            continue
        sel = sel[(sel["sent_len"] >= LEN_MIN) & (sel["sent_len"] <= LEN_MAX)]
        sel = sel.assign(weighted=sel[value_col] * sel["n_sents"])
        agg = sel.groupby("sent_len").agg(
            num=("weighted", "sum"), den=("n_sents", "sum")
        )
        # Drop very thin length-cells (era-level support < 30 sentences); they
        # are noisy and produce visual artefacts.
        agg = agg[agg["den"] >= 30]
        out[era_key] = agg["num"] / agg["den"]
    return out


def _length_curve_fig(
    df_len: pd.DataFrame, value_col: str, ylabel: str, title: str,
    out_path: Path, write_jpg: bool,
) -> None:
    pooled = df_len[df_len["scope"] == "pooled"]
    if pooled.empty:
        print(f"  (skipping {out_path.name}: no pooled length data)")
        return
    era_curves = _era_weighted_mean(pooled, value_col)
    fig, ax = plt.subplots(figsize=(10, 6.0))
    labels = {
        "2023": "2023 cohort", "bridge": "2026 engine bridge",
        "2026": "2026 cohort", "human": "human (Phase 2b)",
    }
    for era_key, series in era_curves.items():
        ax.plot(series.index, series.values, marker="o", markersize=3,
                linewidth=1.5, color=ERA_COLORS[era_key], label=labels[era_key])
    ax.set_xlim(LEN_MIN, LEN_MAX)
    ax.set_xlabel("sentence length (number of non-punct dep edges)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.suptitle(
        "Stage 3A: length-controlled syntactic complexity — does the era "
        "ordering hold at every sentence length?\n"
        "Method follows Juzek/Krielke/Teich (UDW 2020) Figure 2.",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out_path, write_jpg)


def fig_dl_vs_length(df_len: pd.DataFrame, out_path: Path, write_jpg: bool) -> None:
    _length_curve_fig(
        df_len, "dl_mean",
        ylabel="mean per-token dependency distance (tokens)",
        title="mean DL per sentence length, by era",
        out_path=out_path, write_jpg=write_jpg,
    )


def fig_mhd_vs_length(df_len: pd.DataFrame, out_path: Path, write_jpg: bool) -> None:
    _length_curve_fig(
        df_len, "mhd_mean",
        ylabel="mean per-token hierarchical distance (arcs to root)",
        title="mean MHD per sentence length, by era",
        out_path=out_path, write_jpg=write_jpg,
    )


def _per_corpus_length_fig(
    df_len: pd.DataFrame, value_col: str, ylabel: str, title: str,
    out_path: Path, write_jpg: bool,
) -> None:
    """Small-multiples grid: one panel per corpus showing its DL/MHD vs
    sentence-length curve, with the all-corpora-pooled median curve drawn
    in light grey as a reference. Panels are in MODEL_ORDER_WITH_HUMAN
    order so era grouping is visually consistent across the figures."""
    pooled = df_len[df_len["scope"] == "pooled"].copy()
    if pooled.empty:
        print(f"  (skipping {out_path.name}: no pooled length data)")
        return
    pooled = pooled[(pooled["sent_len"] >= LEN_MIN) & (pooled["sent_len"] <= LEN_MAX)]
    pooled = pooled[pooled["n_sents"] >= 20]

    # Reference curve: n_sents-weighted mean across all corpora per sent_len.
    weighted = pooled.assign(w=pooled[value_col] * pooled["n_sents"])
    ref = (weighted.groupby("sent_len")
                   .agg(num=("w", "sum"), den=("n_sents", "sum")))
    ref_series = ref["num"] / ref["den"]

    models = [m for m in MODEL_ORDER_WITH_HUMAN if m in set(pooled["model_dir"])]
    n = len(models)
    n_cols = 5
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 2.7 * n_rows),
                              sharex=True, sharey=True)
    axes_flat = np.array(axes).reshape(-1)

    # Shared y-limits derived from the corpora data + reference.
    all_vals = np.concatenate([
        pooled[value_col].values, ref_series.values,
    ])
    ymin = float(np.nanmin(all_vals)) - 0.1
    ymax = float(np.nanmax(all_vals)) + 0.1

    for i, m in enumerate(models):
        ax = axes_flat[i]
        sub = pooled[pooled["model_dir"] == m].sort_values("sent_len")
        if sub.empty:
            ax.axis("off")
            continue
        # Reference curve (all corpora pooled, n_sents-weighted).
        ax.plot(ref_series.index, ref_series.values, color="#bbbbbb",
                linewidth=1.0, linestyle="--", label="all-corpora mean", zorder=1)
        # This corpus's curve.
        ax.plot(sub["sent_len"], sub[value_col], marker="o", markersize=2,
                linewidth=1.4, color=ERA_COLORS[era_of(m)], zorder=2)
        label = MODEL_LABELS.get(m, m).replace("\n", " ")
        ax.set_title(label, fontsize=9, loc="left",
                     color=ERA_COLORS[era_of(m)])
        ax.set_xlim(LEN_MIN, LEN_MAX)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Only label outer axes.
        if i % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel("sentence length (non-punct dep edges)", fontsize=8)

    # Hide any trailing empty cells; place a small legend in the first hidden one.
    for j in range(len(models), len(axes_flat)):
        axes_flat[j].axis("off")
    if len(models) < len(axes_flat):
        leg_ax = axes_flat[len(models)]
        leg_ax.legend(handles=[
            plt.Line2D([0], [0], color=ERA_COLORS["2023"], lw=1.6, label="2023 cohort"),
            plt.Line2D([0], [0], color=ERA_COLORS["bridge"], lw=1.6, label="2026 engine bridge"),
            plt.Line2D([0], [0], color=ERA_COLORS["2026"], lw=1.6, label="2026 cohort"),
            plt.Line2D([0], [0], color=ERA_COLORS["human"], lw=1.6, label="human (Phase 2b)"),
            plt.Line2D([0], [0], color="#bbbbbb", lw=1.0, linestyle="--",
                       label="all-corpora mean"),
        ], loc="center", fontsize=9, frameon=False)

    fig.suptitle(
        title + "\nEach panel: one corpus's curve over its own sentences "
        "(n ≥ 20 per length bin); dashed grey = n-sents-weighted mean over all corpora.\n"
        "Method follows Juzek/Krielke/Teich (UDW 2020) Figure 2.",
        fontsize=11, y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_path, write_jpg)


def fig_dl_vs_length_per_corpus(
    df_len: pd.DataFrame, out_path: Path, write_jpg: bool,
) -> None:
    _per_corpus_length_fig(
        df_len, "dl_mean",
        ylabel="mean per-token DL (tokens)",
        title="mean DL per sentence length, one panel per corpus",
        out_path=out_path, write_jpg=write_jpg,
    )


def fig_mhd_vs_length_per_corpus(
    df_len: pd.DataFrame, out_path: Path, write_jpg: bool,
) -> None:
    _per_corpus_length_fig(
        df_len, "mhd_mean",
        ylabel="mean per-token MHD (arcs to root)",
        title="mean MHD per sentence length, one panel per corpus",
        out_path=out_path, write_jpg=write_jpg,
    )


def _all_corpora_length_fig(
    df_len: pd.DataFrame, value_col: str, ylabel: str, title: str,
    out_path: Path, write_jpg: bool,
) -> None:
    """Single-panel overlay: every corpus's curve in one axes, per-model
    colours grouped by era family (2023 = greys, bridge = amber, 2026
    closed-API = blues, 2026 open-weights = purples, human = greens) so
    variation / non-variation across corpora is visible at a glance."""
    pooled = df_len[df_len["scope"] == "pooled"].copy()
    if pooled.empty:
        print(f"  (skipping {out_path.name}: no pooled length data)")
        return
    pooled = pooled[(pooled["sent_len"] >= LEN_MIN) & (pooled["sent_len"] <= LEN_MAX)]
    pooled = pooled[pooled["n_sents"] >= 20]
    models = [m for m in MODEL_ORDER_WITH_HUMAN if m in set(pooled["model_dir"])]

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    for m in models:
        sub = pooled[pooled["model_dir"] == m].sort_values("sent_len")
        if sub.empty:
            continue
        col = MODEL_COLORS.get(m, ERA_COLORS["other"])
        ax.plot(sub["sent_len"], sub[value_col], marker="o", markersize=2.5,
                linewidth=1.5, color=col,
                label=MODEL_LABELS.get(m, m).replace("\n", " "))
    ax.set_xlim(LEN_MIN, LEN_MAX)
    ax.set_xlabel("sentence length (number of non-punct dep edges)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Build a structured legend: era headers + corpus entries. We do this with
    # a sequence of empty proxy handles for the headers (an invisible Line2D
    # with a bolder label).
    handles, labels = [], []

    def _add_header(text: str):
        handles.append(plt.Line2D([], [], color="none", label=text))
        labels.append(text)

    def _add_model(m: str):
        col = MODEL_COLORS.get(m, ERA_COLORS["other"])
        handles.append(plt.Line2D([0], [0], color=col, lw=2.0))
        labels.append("  " + MODEL_LABELS.get(m, m).replace("\n", " "))

    eras_groups = [
        ("2023 cohort", [m for m in models if era_of(m) == "2023"]),
        ("2026 engine bridge", [m for m in models if era_of(m) == "bridge"]),
        ("2026 closed-API", ["gemini-3-flash", "gpt-5-4-mini", "haiku-4-5"]),
        ("2026 open-weights", ["olmo3-7b", "nemo-12b", "qwen3-14b"]),
        ("human (Phase 2b)", [m for m in models if era_of(m) == "human"]),
    ]
    for header, group in eras_groups:
        group = [m for m in group if m in models]
        if not group:
            continue
        _add_header(header)
        for m in group:
            _add_model(m)
    leg = ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=9, handlelength=1.5, borderaxespad=0,
    )
    # Bold the era headers (they're the entries whose handle has color="none").
    for txt, hdl in zip(leg.get_texts(), leg.legend_handles):
        if hasattr(hdl, "get_color") and hdl.get_color() == "none":
            txt.set_fontweight("bold")
    fig.suptitle(
        title + " — all corpora overlaid (one line per corpus)\n"
        "Colour groups: 2023 = greys, bridge = amber, 2026 closed-API = blues, "
        "2026 open-weights = purples, human = greens.",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 0.78, 0.96])
    _save(fig, out_path, write_jpg)


def fig_dl_vs_length_all_corpora(
    df_len: pd.DataFrame, out_path: Path, write_jpg: bool,
) -> None:
    _all_corpora_length_fig(
        df_len, "dl_mean",
        ylabel="mean per-token dependency distance (tokens)",
        title="mean DL per sentence length",
        out_path=out_path, write_jpg=write_jpg,
    )


def fig_mhd_vs_length_all_corpora(
    df_len: pd.DataFrame, out_path: Path, write_jpg: bool,
) -> None:
    _all_corpora_length_fig(
        df_len, "mhd_mean",
        ylabel="mean per-token hierarchical distance (arcs to root)",
        title="mean MHD per sentence length",
        out_path=out_path, write_jpg=write_jpg,
    )


# ---------------------------------------------------------------------------
# DL by dep label (Juzek 2020 Figure 4 idiom)
# ---------------------------------------------------------------------------

def fig_dl_by_function(
    dl_pooled: pd.DataFrame, out_path: Path, write_jpg: bool,
    top_k: int = 28,
) -> None:
    """Pooled mean DL per dep label, ranked long→short, with bars darkened
    when the label's freq_per_1k_tok decreases from the 2023 to the 2026
    cohort (UDW 2020 Figure 4 idiom)."""
    if dl_pooled.empty:
        print(f"  (skipping {out_path.name}: no dl_by_function data)")
        return
    df = dl_pooled.copy()
    df["era"] = df["model_dir"].map(era_of)
    # Era-level frequency & mean DL: weight by n.
    grp = df.groupby(["era", "dep_label"]).agg(
        n=("n", "sum"),
        sum_dl=("mean_dl", "mean"),  # placeholder; recomputed via weighted mean
    )
    # Recompute era-weighted mean DL = sum(mean_dl * n) / sum(n).
    df["weighted_dl"] = df["mean_dl"] * df["n"]
    grp = df.groupby(["era", "dep_label"]).agg(
        n=("n", "sum"), weighted_dl=("weighted_dl", "sum"),
    )
    grp["mean_dl"] = grp["weighted_dl"] / grp["n"]
    grp["freq_per_1k_tok"] = grp["n"] / grp["n"].groupby(level="era").transform("sum") * 1000

    # Pooled-across-eras mean DL per label (overall ranking).
    overall = df.groupby("dep_label").apply(
        lambda g: pd.Series({
            "n": g["n"].sum(),
            "mean_dl": float((g["mean_dl"] * g["n"]).sum() / max(g["n"].sum(), 1)),
        })
    )
    overall = overall.sort_values("mean_dl", ascending=False).head(top_k)

    # Frequency direction 2023 → 2026.
    freq_2023 = grp.xs("2023", level="era")["freq_per_1k_tok"] if "2023" in grp.index.get_level_values("era") else pd.Series(dtype=float)
    freq_2026 = grp.xs("2026", level="era")["freq_per_1k_tok"] if "2026" in grp.index.get_level_values("era") else pd.Series(dtype=float)
    direction = {}
    for dep in overall.index:
        f23 = float(freq_2023.get(dep, 0.0))
        f26 = float(freq_2026.get(dep, 0.0))
        direction[dep] = "decreasing" if f26 < f23 else "increasing"

    labels = list(overall.index)[::-1]   # longest DL at top
    vals = [float(overall.loc[lab, "mean_dl"]) for lab in labels]
    colors = ["#333333" if direction[lab] == "decreasing" else "#bbbbbb"
              for lab in labels]

    fig, ax = plt.subplots(figsize=(10, 0.32 * len(labels) + 1.5))
    ys = np.arange(len(labels))
    ax.barh(ys, vals, color=colors, edgecolor="white", linewidth=0.5, height=0.7)
    for y, v in zip(ys, vals):
        ax.text(v, y, f" {v:.2f}", va="center", ha="left", fontsize=8, color="#333")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.set_xlabel("pooled mean dependency length (tokens)", fontsize=10)
    ax.set_xlim(left=0)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Legend: dark = decreasing 2023→2026, light = increasing.
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#333333", label="decreasing 2023 → 2026"),
        plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", label="increasing 2023 → 2026"),
    ], loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(
        "Stage 3A: dependency length by UD function (top "
        f"{len(labels)} most-frequent dep labels)\n"
        "Bars darkened when the label's frequency decreases from the "
        "2023 to the 2026 LLM cohort (UDW 2020 Figure 4 idiom).",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, out_path, write_jpg)


# ---------------------------------------------------------------------------
# Clause-type rates per corpus (small multiples)
# ---------------------------------------------------------------------------

def fig_clause_type_rates(pt: pd.DataFrame, out_path: Path, write_jpg: bool) -> None:
    models = _order(set(pt["model_dir"]))
    fig, axes = plt.subplots(2, 4, figsize=(19, 10.5), sharey=True)
    axes_flat = axes.flatten()
    for idx, (col, title) in enumerate(CLAUSE_TYPES):
        ax = axes_flat[idx]
        if col not in pt.columns:
            ax.axis("off")
            continue
        medians = [
            float(pt.loc[pt["model_dir"] == m, col].median()) for m in models
        ]
        ys = np.arange(len(models))
        colors = [ERA_COLORS[era_of(m)] for m in models]
        ax.barh(ys, medians, color=colors, edgecolor="white",
                linewidth=0.5, height=0.72)
        ax.set_yticks(ys)
        if idx % 4 == 0:
            ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models],
                               fontsize=8)
            for tick, m in zip(ax.get_yticklabels(), models):
                tick.set_color(ERA_COLORS[era_of(m)])
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("median rate (per sentence)", fontsize=9)
        ax.set_xlim(left=0)
        ax.grid(True, axis="x", alpha=0.25, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # Hide the unused 8th panel; use it for the era legend instead.
    leg_ax = axes_flat[len(CLAUSE_TYPES)]
    leg_ax.axis("off")
    leg_ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=ERA_COLORS["2023"], label="2023 cohort"),
        plt.Rectangle((0, 0), 1, 1, color=ERA_COLORS["bridge"], label="2026 engine bridge"),
        plt.Rectangle((0, 0), 1, 1, color=ERA_COLORS["2026"], label="2026 cohort"),
        plt.Rectangle((0, 0), 1, 1, color=ERA_COLORS["human"], label="human (Phase 2b)"),
    ], loc="center", fontsize=10, frameon=False, title="era", title_fontsize=10)
    fig.suptitle(
        "Stage 3A: clause-type rates per corpus (UDW 2020 §4.2 long-DL "
        "functions)\n"
        "Each panel shows the median per-sentence rate of one clause type; "
        "era-coloured bars.",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_path, write_jpg)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage3_syntactic"),
    )
    ap.add_argument("--out-dir", default=None,
                    help="default: <in-dir>/figures")
    ap.add_argument("--jpg", action="store_true",
                    help="also write JPG snapshots next to the PNGs")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pt = pd.read_csv(in_dir / "per_text_syntactic.csv")
    summary = pd.read_csv(in_dir / "summary_syntactic.csv")
    print(f"Loaded {len(pt):,} per-text rows; "
          f"models: {sorted(pt['model_dir'].unique())}")

    fig_complexity(pt, out_dir / "fig_syntactic_complexity.png", args.jpg)
    fig_pos(summary, out_dir / "fig_pos_distribution.png", args.jpg)

    len_path = in_dir / "per_length_syntactic.csv"
    if len_path.exists():
        df_len = pd.read_csv(len_path)
        fig_dl_vs_length(df_len, out_dir / "fig_dl_vs_length.png", args.jpg)
        fig_mhd_vs_length(df_len, out_dir / "fig_mhd_vs_length.png", args.jpg)
        fig_dl_vs_length_per_corpus(
            df_len, out_dir / "fig_dl_vs_length_per_corpus.png", args.jpg)
        fig_mhd_vs_length_per_corpus(
            df_len, out_dir / "fig_mhd_vs_length_per_corpus.png", args.jpg)
        fig_dl_vs_length_all_corpora(
            df_len, out_dir / "fig_dl_vs_length_all_corpora.png", args.jpg)
        fig_mhd_vs_length_all_corpora(
            df_len, out_dir / "fig_mhd_vs_length_all_corpora.png", args.jpg)
    else:
        print(f"  (skipping length-curve figures: {len_path.name} not found)")

    dl_path = in_dir / "dl_by_function__pooled.csv"
    if dl_path.exists():
        dl_pooled = pd.read_csv(dl_path)
        fig_dl_by_function(dl_pooled, out_dir / "fig_dl_by_function.png", args.jpg)
    else:
        print(f"  (skipping fig_dl_by_function: {dl_path.name} not found)")

    fig_clause_type_rates(pt, out_dir / "fig_clause_type_rates.png", args.jpg)


if __name__ == "__main__":
    main()
