#!/usr/bin/env python3
"""Stage 3A: syntactic features from the parsed corpus.

Loads the spaCy ``DocBin`` siblings produced by ``parse_corpus.py`` (one
``<Model>.spacy`` next to each ``<Model>.cleaned.csv``, for every Phase 0 /
1 / 2a corpus *and* the Phase 2b human baselines) — **no re-parse** — and
derives several kinds of syntactic descriptor:

* **POS-tag distributions** (Universal POS tags, ``token.pos_``) per corpus
  → POS-unigram / -bigram / -trigram **Jensen–Shannon divergence** distance
  matrices, parallel to the lexical n-gram JSDs in
  ``build_distance_matrix.py``. The roster here adds Reuters-50/C50: a POS
  distribution over 5,000 news docs is a meaningful corpus descriptor
  (unlike a single lexical centroid, which is why Reuters stays out of the
  lexical matrix). Matrices are written for each scope (pooled + the four
  Improta topics; Reuters contributes only to ``pooled``).
* **Per-text dependency-tree statistics** —
    - ``mdd`` (mean dependency distance, Liu 2008: mean ``|i − head.i|``
      over non-root, non-space, **non-punct** tokens; punctuation excluded
      to align with Juzek/Krielke/Teich UDW 2020 §3.2).
    - ``mdd_per_sent`` (Juzek/Krielke/Teich 2020 §3.3 formulation: per
      sentence, summed distances normalised by sentence length in deps;
      then averaged over the doc's sentences — equal weight per sentence).
    - ``mhd`` (mean hierarchical distance / per-token mean ancestor count;
      Liu 2010, Jiang & Liu 2015 — the dependency-depth analogue of MDD;
      flagged as future work in Juzek/Krielke/Teich 2020 §5).
    - ``tree_depth`` (mean per-sentence max ancestor count; worst-case
      embedding reach per sentence, doc-averaged).
    - ``passive_rate`` (share of sentences with a passive marker via the
      ``*pass`` deps or ``Voice=Pass`` morph).
    - ``subord_ratio`` (existing rollup: clause-introducing edges per
      sentence) plus separate per-sentence rates for each clause type the
      UDW 2020 paper §4.2 flags as long-DL: ``relcl_rate, advcl_rate,
      ccomp_rate, xcomp_rate, acl_rate, conj_rate, parataxis_rate``.
    - ``mean_noun_chunk_len`` (mean length in tokens of ``doc.noun_chunks``;
      the nominalisation-creep measure named in UDW 2020 §5).

  Written to ``per_text_syntactic.csv`` and aggregated into
  ``summary_syntactic.csv``.
* **Per-dep-label DL** (UDW 2020 Figure 4 ingredient) — for every UD dep
  label, the relative frequency per 1k tokens and the mean DL of edges with
  that label. Written to ``dl_by_function__<scope>.csv``.
* **Dep-label JSD matrix** — per-corpus distribution over the ~45 spaCy
  English dep labels as a fourth distance matrix beside POS-uni/bi/tri.
  Written to ``matrices/dep_label_jsd__<scope>.csv``.
* **Length-binned per-sentence DL & MHD** (UDW 2020 Figure 2 ingredient) —
  for each integer sentence length (capped at 80; longer pooled), the mean
  of (per-token DL) and (per-token MHD) over all sentences of that length
  in the corpus. Written to ``per_length_syntactic.csv``.

The Stage-1a quality filter is applied (drop rows flagged
``repetition_loop`` / ``code_contam`` / ``non_english`` / ``truncation``)
so the syntactic numbers line up with the lexical ones. DocBins are loaded
one corpus at a time and freed before the next — the full corpus of ~63 K
parsed docs would not fit in 16 GB at once.

Render side: ``render_syntactic.py`` (complexity boxplots + length-binned
DL/MHD curves + DL-by-function and clause-type-rate bars) and
``render_dendrogram.py --include-human --in-dir <…>/stage3_syntactic/matrices
--out-dir <…>/stage3_syntactic/figures --exclude run6_lmstudio`` (POS-JSD +
dep-label JSD dendrograms + MDS — it's already generic over
``<metric>__<scope>.csv``).

Methodological template: Juzek, Krielke & Teich (2020), *"Exploring
diachronic syntactic shifts with dependency length: the case of scientific
English"*, Proceedings of UDW 2020.

Usage::

    python3 scripts/analyse/syntactic_features.py
    python3 scripts/analyse/syntactic_features.py --base data \\
        --out-dir results/phase2_mapping/stage3_syntactic

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import DocBin

from aiidiolects.build_distance_matrix import (
    MODEL_ORDER,
    QUALITY_FLAGS_DROP,
    TOPICS,
    find_cleaned_csvs,
    model_dir_for,
)
from aiidiolects.compare_fingerprints import jensen_shannon_divergence
from aiidiolects.parse_corpus import cleaned_to_spacy_path
from aiidiolects.paths import DATA_DIR, RESULTS_DIR

# Roster for the syntactic distance matrices: the 14-corpus build roster
# (13 + run6) plus the two Phase 2b human corpora. run6 is excluded again at
# the render step (``--exclude run6_lmstudio``), like the canonical figures.
SYNTACTIC_ORDER: list[str] = MODEL_ORDER + ["webis_cmv_20", "reuters_50"]

# Reuters has a single pseudo-topic; it participates only in the pooled scope.
SCOPES = ["pooled"] + TOPICS

POS_NGRAM_METRICS = ["pos_unigram_jsd", "pos_bigram_jsd", "pos_trigram_jsd"]

# Dependency labels that flag a passive construction in spaCy's English
# parser scheme (ClearNLP-derived). Morph "Voice=Pass" is checked as a
# fallback for models that don't emit the *pass labels.
_PASSIVE_DEPS = {"nsubjpass", "auxpass", "csubjpass"}
# Dependency labels that introduce a (subordinate / embedded) clause — kept
# verbatim so ``subord_ratio`` remains back-compatible with the previous
# Stage-3A definition. The seven clause-type rates below decompose this set
# (and add ``conj`` / ``parataxis``, which UDW 2020 §4.2 identifies as the
# long-DL functions whose frequency shifts drive diachronic DL change).
_CLAUSE_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl", "acl:relcl", "csubj"}

# Per-clause-type buckets (UDW 2020 §4.2 long-DL functions). ``acl`` is kept
# distinct from ``acl:relcl`` to avoid double-counting with ``relcl``.
_RELCL = {"relcl", "acl:relcl"}
_ACL_ONLY = {"acl"}

# Sentence-length cap for per-length aggregation (sentences longer than 80
# non-punct deps are pooled into the ``80`` bucket; UDW 2020 Figure 2's x-axis
# goes to ~60).
SENT_LEN_CAP = 80


# ---------------------------------------------------------------------------
# Per-corpus extraction
# ---------------------------------------------------------------------------

def _quality_mask(df: pd.DataFrame) -> pd.Series:
    keep = pd.Series(True, index=df.index)
    for flag in QUALITY_FLAGS_DROP:
        if flag in df.columns:
            keep &= ~df[flag].astype(bool)
    return keep


def _doc_syntax(doc) -> tuple[
    dict[str, float] | None,
    list[tuple[int, float, float]],
    Counter,
    Counter,
]:
    """Per-doc syntactic stats.

    Returns ``(stats_dict, per_sent_rows, dep_label_count, dep_label_dist_sum)``.

    * ``stats_dict`` — per-text columns or ``None`` if no tokens.
    * ``per_sent_rows`` — list of ``(sent_len_deps, mean_dl, mean_mhd)``
      tuples for length-binned aggregation; ``sent_len_deps`` is the count
      of non-root, non-space, non-punct tokens (= number of dep edges).
    * ``dep_label_count`` / ``dep_label_dist_sum`` — Counters keyed by dep
      label across this doc's edges (non-root, non-space, non-punct).
    """
    content = [t for t in doc if not t.is_space]
    n_tok = len(content)
    if n_tok == 0:
        return None, [], Counter(), Counter()

    # ---------- per-edge accumulators (non-root, non-space, non-punct) ----------
    edge_tokens = [t for t in content if t.head.i != t.i and t.dep_ != "punct"]
    dep_dists = [abs(t.i - t.head.i) for t in edge_tokens]
    mdd = float(np.mean(dep_dists)) if dep_dists else 0.0

    dep_count: Counter = Counter()
    dep_dist_sum: Counter = Counter()
    for t, d in zip(edge_tokens, dep_dists):
        dep_count[t.dep_] += 1
        dep_dist_sum[t.dep_] += d

    # ---------- per-token MHD (over non-root, non-space, non-punct) ----------
    # Ancestor count gives depth: root → 0, direct child of root → 1, etc.
    # We average over the same set used for MDD so they are directly comparable.
    if edge_tokens:
        depths = [len(list(t.ancestors)) for t in edge_tokens]
        mhd = float(np.mean(depths))
    else:
        mhd = 0.0

    # ---------- per-sentence walk ----------
    sent_depths_max: list[int] = []
    n_sents = 0
    n_passive_sents = 0
    # Clause-type sums (counts of qualifying tokens across all sentences).
    n_subord = 0     # back-compat rollup over _CLAUSE_DEPS
    n_relcl = 0
    n_advcl = 0
    n_ccomp = 0
    n_xcomp = 0
    n_acl = 0
    n_conj = 0
    n_parataxis = 0
    # Per-sentence rows for length-binned aggregation.
    per_sent_rows: list[tuple[int, float, float]] = []
    sum_sent_norm_dl = 0.0
    n_sent_with_deps = 0

    for sent in doc.sents:
        sent_toks = [t for t in sent if not t.is_space]
        if not sent_toks:
            continue
        n_sents += 1

        # Sentence-level tree-depth max (over non-punct content for cleanliness).
        sent_content = [t for t in sent_toks if t.dep_ != "punct"]
        if sent_content:
            sent_depths_max.append(max(len(list(t.ancestors)) for t in sent_content))
        else:
            sent_depths_max.append(0)

        if any(
            (t.dep_ in _PASSIVE_DEPS) or (t.morph.get("Voice") == ["Pass"])
            for t in sent_toks
        ):
            n_passive_sents += 1

        # Clause-type counts inside this sentence.
        for t in sent_toks:
            d = t.dep_
            if d in _CLAUSE_DEPS:
                n_subord += 1
            if d in _RELCL:
                n_relcl += 1
            elif d == "advcl":
                n_advcl += 1
            elif d == "ccomp":
                n_ccomp += 1
            elif d == "xcomp":
                n_xcomp += 1
            elif d in _ACL_ONLY:
                n_acl += 1
            elif d == "conj":
                n_conj += 1
            elif d == "parataxis":
                n_parataxis += 1

        # Sentence-level DL & MHD for length-binned aggregation.
        sent_edges = [t for t in sent_content if t.head.i != t.i]
        n_edges = len(sent_edges)
        if n_edges > 0:
            sent_dl_sum = sum(abs(t.i - t.head.i) for t in sent_edges)
            sent_dl_mean = sent_dl_sum / n_edges
            sent_mhd_mean = sum(len(list(t.ancestors)) for t in sent_edges) / n_edges
            per_sent_rows.append((n_edges, sent_dl_mean, sent_mhd_mean))
            sum_sent_norm_dl += sent_dl_mean
            n_sent_with_deps += 1

    tree_depth = float(np.mean(sent_depths_max)) if sent_depths_max else 0.0
    passive_rate = (n_passive_sents / n_sents) if n_sents else 0.0
    subord_ratio = (n_subord / n_sents) if n_sents else 0.0
    mdd_per_sent = (sum_sent_norm_dl / n_sent_with_deps) if n_sent_with_deps else 0.0

    # ---------- NP complexity (noun_chunks; spaCy iterator) ----------
    chunk_lens = [len(chunk) for chunk in doc.noun_chunks]
    mean_noun_chunk_len = float(np.mean(chunk_lens)) if chunk_lens else 0.0

    rates = {}
    if n_sents:
        rates = {
            "relcl_rate": n_relcl / n_sents,
            "advcl_rate": n_advcl / n_sents,
            "ccomp_rate": n_ccomp / n_sents,
            "xcomp_rate": n_xcomp / n_sents,
            "acl_rate": n_acl / n_sents,
            "conj_rate": n_conj / n_sents,
            "parataxis_rate": n_parataxis / n_sents,
        }
    else:
        rates = {k: 0.0 for k in (
            "relcl_rate", "advcl_rate", "ccomp_rate", "xcomp_rate",
            "acl_rate", "conj_rate", "parataxis_rate",
        )}

    stats = {
        "n_tokens_spacy": n_tok,
        "n_sents": n_sents,
        "mdd": mdd,
        "mdd_per_sent": mdd_per_sent,
        "mhd": mhd,
        "tree_depth": tree_depth,
        "passive_rate": passive_rate,
        "subord_ratio": subord_ratio,
        "mean_noun_chunk_len": mean_noun_chunk_len,
        **rates,
    }
    return stats, per_sent_rows, dep_count, dep_dist_sum


def _pos_seq(doc) -> list[str]:
    return [t.pos_ for t in doc if not t.is_space and t.pos_]


def _ngram_counter(seq: list[str], n: int) -> Counter:
    c: Counter = Counter()
    for i in range(len(seq) - n + 1):
        c[tuple(seq[i:i + n])] += 1
    return c


def extract_corpus(
    cleaned_csv: Path, vocab,
) -> tuple[
    str, str, list[dict],                # model_dir, topic, per_text_rows
    list[tuple[int, float, float]],      # per_sent_rows
    dict[int, Counter],                  # n -> POS-ngram Counter for this corpus
    Counter,                             # dep-label count Counter
    Counter,                             # dep-label distance-sum Counter
]:
    """Return aggregates for one (model_dir, topic) corpus.

    All accumulators sum over docs that pass the quality filter.
    """
    model_dir = model_dir_for(cleaned_csv)
    topic = cleaned_csv.parent.name
    spacy_path = cleaned_to_spacy_path(cleaned_csv)
    if not spacy_path.exists():
        print(f"  !! no DocBin for {cleaned_csv} — skipping")
        return (model_dir, topic, [], [],
                {1: Counter(), 2: Counter(), 3: Counter()},
                Counter(), Counter())

    df = pd.read_csv(cleaned_csv)
    keep = _quality_mask(df).to_numpy()
    db = DocBin().from_disk(spacy_path)
    docs = list(db.get_docs(vocab))
    if len(docs) != len(df):
        print(f"  !! DocBin/CSV length mismatch for {cleaned_csv} "
              f"({len(docs)} vs {len(df)}) — skipping")
        return (model_dir, topic, [], [],
                {1: Counter(), 2: Counter(), 3: Counter()},
                Counter(), Counter())

    pos_counters = {1: Counter(), 2: Counter(), 3: Counter()}
    dep_count: Counter = Counter()
    dep_dist_sum: Counter = Counter()
    rows: list[dict] = []
    sent_rows: list[tuple[int, float, float]] = []

    for doc, k in zip(docs, keep):
        if not k:
            continue
        st, per_sent, dlc, dlds = _doc_syntax(doc)
        if st is None:
            continue
        rows.append({"model_dir": model_dir, "topic": topic, **st})
        sent_rows.extend(per_sent)
        dep_count.update(dlc)
        dep_dist_sum.update(dlds)
        seq = _pos_seq(doc)
        pos_counters[1].update(seq)
        pos_counters[2].update(_ngram_counter(seq, 2))
        pos_counters[3].update(_ngram_counter(seq, 3))

    del docs, db
    return model_dir, topic, rows, sent_rows, pos_counters, dep_count, dep_dist_sum


# ---------------------------------------------------------------------------
# Distance-matrix assembly
# ---------------------------------------------------------------------------

def _matrix_from_counters(
    counters: dict[str, Counter], order: list[str],
) -> pd.DataFrame:
    present = [m for m in order if m in counters and sum(counters[m].values())]
    n = len(present)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jensen_shannon_divergence(counters[present[i]], counters[present[j]])
            M[i, j] = M[j, i] = d
    return pd.DataFrame(M, index=present, columns=present)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage3_syntactic"),
    )
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir)
    (out_dir / "matrices").mkdir(parents=True, exist_ok=True)

    vocab = spacy.blank("en").vocab
    csvs = find_cleaned_csvs(base, include_human=True)
    print(f"Found {len(csvs)} cleaned CSV(s); loading DocBins one corpus at a time")

    # Accumulators keyed by model_dir, then by scope.
    per_text_rows: list[dict] = []
    # pos_counters[n][scope][model_dir] -> Counter
    pos_counters: dict[int, dict[str, dict[str, Counter]]] = {
        n: {sc: {} for sc in SCOPES} for n in (1, 2, 3)
    }
    # Dep-label distribution per (scope, model_dir): Counter of labels (counts).
    dep_label_counts: dict[str, dict[str, Counter]] = {sc: {} for sc in SCOPES}
    # Dep-label distance sum per (scope, model_dir): Counter of label -> total
    # |i - head.i| over edges with that label.
    dep_label_dist_sum: dict[str, dict[str, Counter]] = {sc: {} for sc in SCOPES}
    # Total non-root non-punct edges per (scope, model_dir) for freq_per_1k.
    dep_label_tot: dict[str, dict[str, int]] = {sc: defaultdict(int) for sc in SCOPES}
    # Length-binned (sum_dl, sum_mhd, count) per (scope, model_dir, sent_len_cap).
    length_acc: dict[tuple[str, str, int], list[float]] = {}

    def _bump_length(scope: str, model: str, sent_len: int,
                     dl: float, mhd: float) -> None:
        key = (scope, model, min(sent_len, SENT_LEN_CAP))
        if key not in length_acc:
            length_acc[key] = [0.0, 0.0, 0]
        length_acc[key][0] += dl
        length_acc[key][1] += mhd
        length_acc[key][2] += 1

    t0 = time.time()
    for csv in csvs:
        m, tp, rows, sent_rows, cdict, dlc, dlds = extract_corpus(csv, vocab)
        if not rows:
            continue
        per_text_rows.extend(rows)
        for n in (1, 2, 3):
            if tp in SCOPES:
                pos_counters[n][tp].setdefault(m, Counter())
                pos_counters[n][tp][m].update(cdict[n])
            pos_counters[n]["pooled"].setdefault(m, Counter())
            pos_counters[n]["pooled"][m].update(cdict[n])
        # Dep-label accumulators per scope.
        for sc in ("pooled", tp if tp in SCOPES else None):
            if sc is None:
                continue
            dep_label_counts[sc].setdefault(m, Counter())
            dep_label_counts[sc][m].update(dlc)
            dep_label_dist_sum[sc].setdefault(m, Counter())
            dep_label_dist_sum[sc][m].update(dlds)
            dep_label_tot[sc][m] += sum(dlc.values())
        # Length-binned aggregation.
        for sent_len, dl, mhd in sent_rows:
            _bump_length("pooled", m, sent_len, dl, mhd)
            if tp in SCOPES:
                _bump_length(tp, m, sent_len, dl, mhd)
        kept = len(rows)
        print(f"  {m:<22} {tp:<16} kept={kept:>5}  "
              f"({time.time() - t0:5.1f}s elapsed)")

    # --- per-text + summary CSVs ---
    pt = pd.DataFrame(per_text_rows)
    pt_path = out_dir / "per_text_syntactic.csv"
    pt.to_csv(pt_path, index=False)
    print(f"\nWrote {pt_path}  ({len(pt):,} rows)")

    # POS distribution per (model_dir, topic) + pooled, from the unigram counters.
    upos_tags = sorted({
        t for sc in SCOPES for cs in pos_counters[1][sc].values() for t in cs
    })
    summ_rows: list[dict] = []
    metric_cols = [
        "mdd", "mdd_per_sent", "mhd", "tree_depth",
        "passive_rate", "subord_ratio",
        "relcl_rate", "advcl_rate", "ccomp_rate", "xcomp_rate",
        "acl_rate", "conj_rate", "parataxis_rate",
        "mean_noun_chunk_len",
        "n_tokens_spacy", "n_sents",
    ]
    for sc in SCOPES:
        sub_df = pt if sc == "pooled" else pt[pt["topic"] == sc]
        for m, g in sub_df.groupby("model_dir"):
            row = {"model_dir": m, "scope": sc, "n_texts": len(g)}
            for c in metric_cols:
                if c in g.columns:
                    row[f"{c}_median"] = float(g[c].median())
                    row[f"{c}_mean"] = float(g[c].mean())
            uni = pos_counters[1][sc].get(m, Counter())
            tot = sum(uni.values()) or 1
            for tag in upos_tags:
                row[f"pos_{tag}"] = uni.get(tag, 0) / tot
            summ_rows.append(row)
    summ = pd.DataFrame(summ_rows)
    summ_path = out_dir / "summary_syntactic.csv"
    summ.to_csv(summ_path, index=False)
    print(f"Wrote {summ_path}  ({len(summ):,} rows)")

    # --- POS-ngram JSD matrices ---
    n_to_metric = {1: "pos_unigram_jsd", 2: "pos_bigram_jsd", 3: "pos_trigram_jsd"}
    for n, metric in n_to_metric.items():
        for sc in SCOPES:
            mat = _matrix_from_counters(pos_counters[n][sc], SYNTACTIC_ORDER)
            if mat.empty:
                continue
            p = out_dir / "matrices" / f"{metric}__{sc}.csv"
            mat.to_csv(p)
            print(f"  wrote {p}  ({mat.shape[0]}×{mat.shape[1]})")

    # --- Dep-label JSD matrix (4th distance family) ---
    for sc in SCOPES:
        mat = _matrix_from_counters(dep_label_counts[sc], SYNTACTIC_ORDER)
        if mat.empty:
            continue
        p = out_dir / "matrices" / f"dep_label_jsd__{sc}.csv"
        mat.to_csv(p)
        print(f"  wrote {p}  ({mat.shape[0]}×{mat.shape[1]})")

    # --- Per-dep-label DL tables (Juzek 2020 Fig.4 ingredient) ---
    dl_rows: dict[str, list[dict]] = {sc: [] for sc in SCOPES}
    for sc in SCOPES:
        for m, counts in dep_label_counts[sc].items():
            tot = dep_label_tot[sc][m] or 1
            for dep, n_ in counts.items():
                if n_ == 0:
                    continue
                mean_dl = dep_label_dist_sum[sc][m][dep] / n_
                dl_rows[sc].append({
                    "model_dir": m,
                    "dep_label": dep,
                    "n": n_,
                    "freq_per_1k_tok": 1000.0 * n_ / tot,
                    "mean_dl": float(mean_dl),
                })
    for sc, rows_ in dl_rows.items():
        if not rows_:
            continue
        dfp = pd.DataFrame(rows_).sort_values(
            ["model_dir", "freq_per_1k_tok"], ascending=[True, False]
        )
        p = out_dir / f"dl_by_function__{sc}.csv"
        dfp.to_csv(p, index=False)
        print(f"  wrote {p}  ({len(dfp):,} rows)")

    # --- Length-binned DL & MHD (Juzek 2020 Fig.2 ingredient) ---
    len_rows: list[dict] = []
    for (sc, m, sl), (sum_dl, sum_mhd, n) in length_acc.items():
        if n == 0:
            continue
        len_rows.append({
            "scope": sc,
            "model_dir": m,
            "sent_len": sl,
            "n_sents": n,
            "dl_mean": sum_dl / n,
            "mhd_mean": sum_mhd / n,
        })
    if len_rows:
        df_len = pd.DataFrame(len_rows).sort_values(
            ["scope", "model_dir", "sent_len"]
        )
        p = out_dir / "per_length_syntactic.csv"
        df_len.to_csv(p, index=False)
        print(f"  wrote {p}  ({len(df_len):,} rows)")

    print(f"\nDone in {time.time() - t0:.1f}s. UPOS tags seen: {upos_tags}")


if __name__ == "__main__":
    main()
