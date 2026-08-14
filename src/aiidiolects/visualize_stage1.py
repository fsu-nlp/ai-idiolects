#!/usr/bin/env python3
"""Render Stage 1 (descriptives) figures for the AI idiolects chapter.

Reads ``results/phase2_mapping/stage1_descriptives/per_text.csv``
(produced by ``descriptive_stats.py``) and writes paper-ready PNGs to
``results/phase2_mapping/stage1_descriptives/figures/``.

Figures:

1. ``fig1_length.png`` — word count, sentence count, mean sentence length per
   model (boxes with topic-colored strip overlay). Quality-clean texts only;
   truncated Gemini texts are excluded for length-sensitive comparisons.
2. ``fig2_lexical_diversity.png`` — TTR per text + corpus-level vocabulary
   size per (model × topic). Quality-clean only.
3. ``fig3_opener_style.png`` — heatmaps of disclaimer-opener %, AI-self-ref %,
   and refusal % across model × topic.
4. ``fig4_truncation.png`` — heatmap of truncation rate per model × topic
   (the Gemini methodological caveat).

Usage::

    python3 -m aiidiolects.visualize_stage1
    python3 -m aiidiolects.visualize_stage1 --in-csv path/to/per_text.csv \\
                                             --out-dir path/to/figures

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

# ---------- presentation order and palettes ----------

# Era-grouped 13-corpus roster: 6 Improta 2023 originals → 1 engine bridge
# (Run 5 Mistral-7B, our 2026 LM Studio re-generation of the same weights as
# the 2023 Mistral-7B) → 6 Phase 2 production models. The bridge sits between
# cohorts as a scale bar for "engine + time noise on identical weights"
# (Phase 1 Run 5↔Run 6 + Run 5↔originals quantify it as ~JSD 0.11).
MODEL_ORDER = [
    # 2023 cohort (Improta originals)
    "phase0_gpt-3.5",
    "phase0_gpt-4o",
    "phase0_haiku",
    "phase0_llama-3-8b",
    "phase0_llama-3.1-70b",
    "phase0_mistral-7b",
    # 2023 → 2026 engine bridge: same weights as phase0_mistral-7b
    "run5_lmstudio",
    # 2026 cohort (production roster)
    "gemini-3-flash",
    "gpt-5-4-mini",
    "haiku-4-5",
    "olmo3-7b",
    "nemo-12b",
    "qwen3-14b",
]

MODEL_LABELS = {
    # 2023
    "phase0_gpt-3.5":       "GPT-3.5\n(2023)",
    "phase0_gpt-4o":        "GPT-4o\n(2023)",
    "phase0_haiku":         "Claude 3.5 Haiku\n(2023)",
    "phase0_llama-3-8b":    "Llama-3\n8B (2023)",
    "phase0_llama-3.1-70b": "Llama-3.1\n70B (2023)",
    "phase0_mistral-7b":    "Mistral-7B\n(2023)",
    # bridge
    "run5_lmstudio":        "Mistral-7B\n(2026 bridge)",
    # 2026
    "gemini-3-flash":       "Gemini-3\nFlash (2026)",
    "gpt-5-4-mini":         "GPT-5.4\nMini (2026)",
    "haiku-4-5":            "Claude 4.5 Haiku\n(2026)",
    "olmo3-7b":             "OLMo-3\n7B (2026)",
    "nemo-12b":             "Mistral-Nemo\n12B (2026)",
    "qwen3-14b":            "Qwen-3\n14B (2026)",
    # human baselines (opt-in via --include-human)
    "webis_cmv_20":         "Humans\n(r/CMV)",
    "reuters_50":           "Humans\n(Reuters-50)",
}

# Opt-in extension: append the human baselines after the 13-corpus roster.
# Webis-CMV-20 (r/ChangeMyView OPs, keyword-filtered to the 4 Improta topics)
# participates in every figure; Reuters-50/C50 (corporate news, single
# "reuters_ccat" pseudo-topic) appears only on per-model boxplots — it has no
# per-topic structure to populate the topic×model heatmaps, and 50 authors
# averaged into one centroid never joins the cross-corpus distance matrix
# (see build_distance_matrix.py / within_author_baseline.py).
MODEL_ORDER_WITH_HUMAN = MODEL_ORDER + ["webis_cmv_20", "reuters_50"]

# Reuters has no LLM-topic data, so it is dropped from topic×model heatmaps.
HEATMAP_EXCLUDE = {"reuters_50"}

# Era classification + colour cue for row labels. Bridge row is the only
# 2026 corpus we can pair with a 2023 original at the same weights.
ERA_2023 = {
    "phase0_gpt-3.5", "phase0_gpt-4o", "phase0_haiku",
    "phase0_llama-3-8b", "phase0_llama-3.1-70b", "phase0_mistral-7b",
}
ERA_BRIDGE = {"run5_lmstudio", "run6_lmstudio"}
ERA_2026 = {
    "gemini-3-flash", "gpt-5-4-mini", "haiku-4-5",
    "olmo3-7b", "nemo-12b", "qwen3-14b",
}
# Naturally-occurring human text (opt-in via --include-human). Not a "model";
# kept in its own cohort so the figures read it as an outgroup, not as one
# more LLM row.
ERA_HUMAN = {"webis_cmv_20", "reuters_50"}
ERA_COLORS = {
    "2023":   "#666666",   # neutral grey
    "bridge": "#d97706",   # amber — visually flags the bridge anchor
    "2026":   "#1f3a93",   # deep blue
    "human":  "#2e7d32",   # green — human baseline corpora
    "other":  "#000000",
}


def era_of(model_dir: str) -> str:
    if model_dir in ERA_2023:
        return "2023"
    if model_dir in ERA_BRIDGE:
        return "bridge"
    if model_dir in ERA_2026:
        return "2026"
    if model_dir in ERA_HUMAN:
        return "human"
    return "other"


def era_separator_indices(models: list[str]) -> list[int]:
    """Indices i for which models[i] and models[i+1] belong to different
    eras — i.e. where a horizontal separator should be drawn."""
    return [
        i for i in range(len(models) - 1)
        if era_of(models[i]) != era_of(models[i + 1])
    ]

TOPIC_ORDER = ["climate", "global_warming", "math_anxiety", "misinfo_health"]
TOPIC_LABELS = {
    "climate": "climate",
    "global_warming": "global warming",
    "math_anxiety": "math anxiety",
    "misinfo_health": "health misinfo",
}
TOPIC_COLORS = {
    "climate": "#1f77b4",
    "global_warming": "#ff7f0e",
    "math_anxiety": "#2ca02c",
    "misinfo_health": "#d62728",
}


# ---------- shared helpers ----------

def quality_clean_mask(df: pd.DataFrame, exclude_truncated: bool = False) -> pd.Series:
    """Return mask of quality-clean rows. Repetition / code / non-English are
    always excluded; truncation is only excluded when explicitly requested
    (length-sensitive metrics)."""
    mask = ~(
        df["flag_repetition_loop"].astype(bool)
        | df["flag_code_contam"].astype(bool)
        | df["flag_non_english"].astype(bool)
    )
    if exclude_truncated:
        mask &= ~df["flag_truncation"].astype(bool)
    return mask


# Active presentation order. main() swaps this to MODEL_ORDER_WITH_HUMAN when
# --include-human is passed; every fig_* helper goes through order_models().
_ACTIVE_ORDER: list[str] = MODEL_ORDER


def order_models(present: set[str], *, for_heatmap: bool = False) -> list[str]:
    out = [m for m in _ACTIVE_ORDER if m in present]
    if for_heatmap:
        out = [m for m in out if m not in HEATMAP_EXCLUDE]
    return out


def boxplot_by_model(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    xlabel: str,
    models: list[str],
):
    """Horizontal boxplot, models on Y, with a thin topic-colored strip overlay."""
    data = [df.loc[df["model_dir"] == m, value_col].dropna().values for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    ax.boxplot(
        data,
        vert=False,
        showfliers=False,
        widths=0.55,
        patch_artist=True,
        boxprops=dict(facecolor="#dddddd", edgecolor="#666666"),
        medianprops=dict(color="#cc0000", linewidth=2),
        whiskerprops=dict(color="#666666"),
        capprops=dict(color="#666666"),
    )

    rng = np.random.default_rng(0)
    for i, m in enumerate(models, start=1):
        sub = df[df["model_dir"] == m]
        for tp in TOPIC_ORDER:
            sub_tp = sub[sub["topic"] == tp]
            if sub_tp.empty:
                continue
            jitter = rng.normal(0, 0.07, size=len(sub_tp))
            ax.scatter(
                sub_tp[value_col].values,
                np.full(len(sub_tp), i) + jitter,
                s=4,
                alpha=0.18,
                color=TOPIC_COLORS[tp],
                linewidths=0,
            )

    ax.set_yticks(range(1, len(models) + 1))
    ax.set_yticklabels(labels, fontsize=9)
    for tick, m in zip(ax.get_yticklabels(), models):
        tick.set_color(ERA_COLORS[era_of(m)])
    ax.invert_yaxis()
    # Boxplot positions are 1-based; an era boundary between models[i] and
    # models[i+1] sits at y = (i+1) + 0.5.
    for i in era_separator_indices(models):
        ax.axhline(i + 1.5, color="#999999", linewidth=0.5, linestyle="--")
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def topic_legend_handles():
    return [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=TOPIC_COLORS[tp], label=TOPIC_LABELS[tp])
        for tp in TOPIC_ORDER
    ]


def heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    title: str,
    cmap: str = "magma_r",
    vmin: float | None = 0,
    vmax: float | None = 100,
    annotate_fmt: str = "{:.0f}",
):
    im = ax.imshow(
        matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
    )
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        [TOPIC_LABELS.get(t, t) for t in matrix.columns],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(
        [MODEL_LABELS.get(m, m).replace("\n", " ") for m in matrix.index],
        fontsize=8,
    )
    row_models = list(matrix.index)
    for tick, m in zip(ax.get_yticklabels(), row_models):
        tick.set_color(ERA_COLORS[era_of(m)])
    # Heatmap rows are 0-based; an era boundary between row i and row i+1
    # sits at y = i + 0.5.
    for i in era_separator_indices(row_models):
        ax.axhline(i + 0.5, color="#999999", linewidth=0.5, linestyle="--")
    # Annotation contrast: switch to white when the cell value is in the
    # upper half of the colour scale. Falls back to absolute > 60 when the
    # caller didn't pass an explicit scale (preserves existing behaviour
    # for percentage heatmaps).
    if vmin is not None and vmax is not None:
        midpoint = vmin + 0.5 * (vmax - vmin)
    else:
        midpoint = 60
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            if pd.isna(v):
                continue
            color = "white" if v > midpoint else "black"
            ax.text(j, i, annotate_fmt.format(v),
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title, fontsize=10, loc="left", pad=4)
    return im


# ---------- figures ----------

def fig_length(
    df: pd.DataFrame, out_path: Path, exclude_truncated: bool = True
) -> None:
    """3-panel: word count / sentence count / words per sentence.

    With ``exclude_truncated=False`` the same plot is drawn on all
    quality-clean texts including truncated ones — useful for visualizing
    Gemini's behaviour against the 2048-token output cap.
    """
    sub = df[quality_clean_mask(df, exclude_truncated=exclude_truncated)].copy()
    models = order_models(set(sub["model_dir"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    boxplot_by_model(
        axes[0], sub, "n_words",
        "Word count per text",
        "words", models,
    )
    boxplot_by_model(
        axes[1], sub, "n_sents",
        "Sentence count per text",
        "sentences", models,
    )
    boxplot_by_model(
        axes[2], sub, "mean_sent_len",
        "Mean sentence length",
        "words / sentence", models,
    )
    # Cap the mean-sentence-length axis: empirically all medians fall well
    # under 30 words/sentence and the long tail of fliers is suppressed by
    # boxplot(showfliers=False); without a cap, half the panel is whitespace.
    axes[2].set_xlim(0, 35)
    if exclude_truncated:
        title = (
            "Stage 1: output length and rhythm by model "
            "(quality-clean, non-truncated texts)"
        )
    else:
        title = (
            "Stage 1: output length and rhythm by model "
            "(truncated texts INCLUDED — surfaces GPT-5.4 Mini's cap-hit subset)"
        )
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.legend(
        handles=topic_legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_lexical_diversity(
    df: pd.DataFrame, out_path: Path, summary_path: Path | None = None
) -> None:
    """2-panel: per-text TTR, vocab size per (model × topic).

    ``summary_path`` overrides the default lookup location for ``summary.csv``
    (corpus-level vocab size); falls back to ``<out_path>.parent.parent /
    summary.csv`` for callers that put figures and summary together.
    """
    sub = df[quality_clean_mask(df, exclude_truncated=False)].copy()
    models = order_models(set(sub["model_dir"]))
    models_heat = order_models(set(sub["model_dir"]), for_heatmap=True)

    # Vocab size requires recomputation across docs; per_text doesn't store
    # corpus-level vocab. Read summary.csv for that:
    if summary_path is None:
        summary_path = out_path.parent.parent / "summary.csv"
    summary = pd.read_csv(summary_path)
    vocab_matrix = (
        summary.pivot(index="model_dir", columns="topic", values="vocab_size")
        .reindex(models_heat)
        .reindex(columns=TOPIC_ORDER)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    boxplot_by_model(
        axes[0], sub, "ttr",
        "Type-token ratio per text",
        "TTR", models,
    )
    axes[0].set_xlim(0.25, 0.65)

    im = heatmap(
        axes[1], vocab_matrix,
        "Vocabulary size per (model × topic)",
        cmap="viridis",
        vmin=vocab_matrix.values[~np.isnan(vocab_matrix.values)].min(),
        vmax=vocab_matrix.values[~np.isnan(vocab_matrix.values)].max(),
        annotate_fmt="{:.0f}",
    )
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="unique lemmas")

    fig.suptitle(
        "Stage 1: lexical diversity (per text) and vocabulary breadth (per group)",
        fontsize=12, y=1.0,
    )
    fig.legend(
        handles=topic_legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _flag_rate_matrix(df: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    g = (
        df.groupby(["model_dir", "topic"])[flag_col]
        .mean()
        .mul(100)
        .unstack("topic")
    )
    return (
        g.reindex(order_models(set(df["model_dir"]), for_heatmap=True))
        .reindex(columns=TOPIC_ORDER)
    )


def fig_opener_style(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmaps of disclaimer / AI self-ref / refusal opener rates."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    m1 = _flag_rate_matrix(df, "flag_disclaimer")
    m2 = _flag_rate_matrix(df, "flag_ai_self_ref")
    m3 = _flag_rate_matrix(df, "flag_refusal")

    for ax, m, title in [
        (axes[0], m1, "Disclaimer opener (%)"),
        (axes[1], m2, "AI self-reference (%)"),
        (axes[2], m3, "Refusal opener (%)"),
    ]:
        heatmap(ax, m, title, cmap="magma_r", vmin=0, vmax=100)

    fig.suptitle(
        "Stage 1: opener style is sharply model-specific and topic-conditional",
        fontsize=12, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_topic_verbosity(
    df: pd.DataFrame,
    out_path: Path,
    exclude_truncated: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
    subtitle: str | None = None,
) -> None:
    """Heatmap of median word count per (model × topic).

    Surfaces the topic-conditional verbosity finding (e.g., Haiku and GPT
    expand on math_anxiety), which is hard to read from per-text strip plots.
    With ``exclude_truncated=False`` the same heatmap uses all quality-clean
    texts including truncated ones — useful for comparing Gemini's apparent
    median against the 2048-token cap.

    ``vmin``/``vmax`` override the colour-scale endpoints. Default is to
    auto-fit to the data range, which works for Phase 2 (where models span
    ~350–1350 words median); for narrower trios (e.g. Phase 1 anchoring,
    where all three corpora sit within ~350–450) the auto-fit visually
    exaggerates a within-noise gap, so callers can pass a wider scale to
    keep cross-figure comparability.
    """
    sub = df[quality_clean_mask(df, exclude_truncated=exclude_truncated)].copy()
    g = (
        sub.groupby(["model_dir", "topic"])["n_words"]
        .median()
        .unstack("topic")
        .reindex(order_models(set(sub["model_dir"]), for_heatmap=True))
        .reindex(columns=TOPIC_ORDER)
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = heatmap(
        ax, g, "",
        cmap="viridis",
        vmin=vmin if vmin is not None else g.values[~np.isnan(g.values)].min(),
        vmax=vmax if vmax is not None else g.values[~np.isnan(g.values)].max(),
        annotate_fmt="{:.0f}",
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="median words per text")
    if subtitle is None:
        if exclude_truncated:
            subtitle = (
                "(quality-clean, non-truncated texts; "
                "GPT-5.4 Mini cap-hits excluded — see fig5)"
            )
        else:
            subtitle = (
                "(truncated texts INCLUDED — surfaces GPT-5.4 Mini's cap-hit subset)"
            )
    fig.suptitle(
        f"Stage 1: topic-conditional verbosity\n{subtitle}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_truncation(df: pd.DataFrame, out_path: Path) -> None:
    """Truncation rate per (model × topic) — the Gemini methodological caveat."""
    m = _flag_rate_matrix(df, "flag_truncation")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    im = heatmap(ax, m, "", cmap="magma_r", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, shrink=0.8, label="% of texts truncated")
    fig.suptitle(
        "Stage 1: text truncation rate by model × topic\n"
        "(threshold: word count >= 1483 AND no terminal punctuation, "
        "i.e. close to the 2048-token / 1.1047-word-per-token ceiling)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    canon_in = str(
        RESULTS_DIR / "phase2_mapping" / "stage1_descriptives" / "per_text.csv"
    )
    canon_out = str(
        RESULTS_DIR / "phase2_mapping" / "stage1_descriptives" / "figures"
    )
    human_in = str(
        RESULTS_DIR / "phase2_mapping"
        / "stage1_descriptives_with_human" / "per_text.csv"
    )
    human_out = str(
        RESULTS_DIR / "phase2_mapping"
        / "stage1_descriptives_with_human" / "figures"
    )
    ap.add_argument("--in-csv", default=None,
                    help=f"default: {canon_in} (or the *_with_human path "
                         "under --include-human)")
    ap.add_argument("--out-dir", default=None,
                    help=f"default: {canon_out} (or *_with_human under "
                         "--include-human)")
    ap.add_argument(
        "--include-human", action="store_true",
        help="add the Webis-CMV-20 + Reuters-50 human baselines. Boxplots "
             "show both human corpora; topic×model heatmaps show Webis only "
             "(Reuters has no per-topic structure). Switches the --in-csv / "
             "--out-dir defaults to the stage1_descriptives_with_human paths.",
    )
    args = ap.parse_args()

    global _ACTIVE_ORDER
    if args.include_human:
        _ACTIVE_ORDER = MODEL_ORDER_WITH_HUMAN
        in_path = Path(args.in_csv or human_in)
        out_dir = Path(args.out_dir or human_out)
    else:
        in_path = Path(args.in_csv or canon_in)
        out_dir = Path(args.out_dir or canon_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df):,} rows from {in_path}")
    print(f"Models present: {sorted(df['model_dir'].unique())}")
    print(f"Topics present: {sorted(df['topic'].unique())}")

    figures = [
        ("fig1_length.png", fig_length),
        (
            "fig1b_length_truncated_included.png",
            lambda df, p: fig_length(df, p, exclude_truncated=False),
        ),
        ("fig2_lexical_diversity.png", fig_lexical_diversity),
        ("fig3_opener_style.png", fig_opener_style),
        ("fig4_topic_verbosity.png", fig_topic_verbosity),
        (
            "fig4b_topic_verbosity_truncated_included.png",
            lambda df, p: fig_topic_verbosity(df, p, exclude_truncated=False),
        ),
        ("fig5_truncation.png", fig_truncation),
    ]
    for name, fn in figures:
        out_path = out_dir / name
        print(f"  rendering {name}")
        fn(df, out_path)
        print(f"    wrote {out_path}")


if __name__ == "__main__":
    main()
