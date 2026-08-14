"""
Linguistic fingerprint comparison between LLMMap originals and replicated texts.

Computes distributional, lexical, and stylometric metrics on two corpora
and tests whether they are linguistically equivalent.

Usage:
    python compare_fingerprints.py \
        --original data/phase0_textualllmap/climate/Mistral-7b.csv \
        --replicated data/phase1_anchoring/run5_lmstudio/climate/Mistral-7b.csv \
        --output results/phase1_anchoring/run5_lmstudio_vs_originals/climate/

Requires: pip install numpy scipy matplotlib pandas

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (`claude-opus-4-7`) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

import argparse
import csv
import re
import string
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from aiidiolects.paths import RESULTS_DIR


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def load_texts(path: str, text_col: str = "text") -> list[str]:
    """Load texts from a CSV, return list of strings."""
    df = pd.read_csv(path)
    texts = df[text_col].dropna().astype(str).tolist()
    print(f"  Loaded {len(texts)} texts from {path}")
    return texts


# ---------------------------------------------------------------------------
# Tokenisation (simple whitespace + punctuation strip — no heavy deps)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokeniser with punctuation removal."""
    text = _PUNCT_RE.sub(" ", text.lower())
    return text.split()


def sentencize(text: str) -> list[str]:
    """Naive sentence splitter on .!? followed by space/newline."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if len(s) > 0]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def corpus_tokens(texts: list[str]) -> list[list[str]]:
    return [tokenize(t) for t in texts]


def text_lengths(token_lists: list[list[str]]) -> np.ndarray:
    return np.array([len(toks) for toks in token_lists])


def sentence_lengths(texts: list[str]) -> np.ndarray:
    lengths = []
    for t in texts:
        for s in sentencize(t):
            lengths.append(len(tokenize(s)))
    return np.array(lengths)


def ttr(token_lists: list[list[str]]) -> float:
    """Global type-token ratio."""
    all_tokens = [t for tl in token_lists for t in tl]
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def mattr(token_lists: list[list[str]], window: int = 500) -> float:
    """Moving-average type-token ratio."""
    all_tokens = [t for tl in token_lists for t in tl]
    if len(all_tokens) < window:
        return ttr(token_lists)
    ratios = []
    for i in range(len(all_tokens) - window + 1):
        chunk = all_tokens[i:i + window]
        ratios.append(len(set(chunk)) / window)
    return float(np.mean(ratios))


def hapax_ratio(token_lists: list[list[str]]) -> float:
    """Proportion of words that occur exactly once."""
    all_tokens = [t for tl in token_lists for t in tl]
    freq = Counter(all_tokens)
    hapax = sum(1 for v in freq.values() if v == 1)
    return hapax / len(freq) if freq else 0.0


def yules_k(token_lists: list[list[str]]) -> float:
    """Yule's K measure of vocabulary richness."""
    all_tokens = [t for tl in token_lists for t in tl]
    freq = Counter(all_tokens)
    N = len(all_tokens)
    if N == 0:
        return 0.0
    spectrum = Counter(freq.values())  # m -> number of types with freq m
    M2 = sum(i * i * vi for i, vi in spectrum.items())
    K = 10000 * (M2 - N) / (N * N) if N > 0 else 0.0
    return K


def word_freq_vector(token_lists: list[list[str]], top_n: int = 200) -> tuple[list[str], np.ndarray]:
    """Return top-N words and their relative frequencies."""
    all_tokens = [t for tl in token_lists for t in tl]
    freq = Counter(all_tokens)
    total = len(all_tokens)
    most_common = freq.most_common(top_n)
    words = [w for w, _ in most_common]
    freqs = np.array([c / total for _, c in most_common])
    return words, freqs


def jaccard_top_n(words_a: list[str], words_b: list[str]) -> float:
    """Jaccard similarity of two word lists."""
    a, b = set(words_a), set(words_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def ngram_distribution(token_lists: list[list[str]], n: int = 2, top_k: int = 500) -> Counter:
    """Frequency distribution of n-grams across the corpus."""
    ngrams = Counter()
    for toks in token_lists:
        for i in range(len(toks) - n + 1):
            ngrams[tuple(toks[i:i + n])] += 1
    return Counter(dict(ngrams.most_common(top_k)))


def jensen_shannon_divergence(counter_a: Counter, counter_b: Counter) -> float:
    """JSD between two frequency distributions (aligned on shared keys)."""
    keys = set(counter_a.keys()) | set(counter_b.keys())
    total_a = sum(counter_a.values())
    total_b = sum(counter_b.values())
    if total_a == 0 or total_b == 0:
        return 1.0
    p = np.array([counter_a.get(k, 0) / total_a for k in keys])
    q = np.array([counter_b.get(k, 0) / total_b for k in keys])
    # Add small epsilon to avoid log(0)
    eps = 1e-12
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    return float(jsd)


def burrows_delta(token_lists_a, token_lists_b, n_features: int = 100,
                  n_chunks: int = 20):
    """
    Burrows' Delta between two sub-corpora.

    Splits each corpus into n_chunks pseudo-documents, computes relative
    word frequencies per chunk for the top-N MFW of the combined corpus,
    z-scores across all chunks, then returns the mean absolute difference
    between the two corpus centroids.

    For two halves of the same corpus, Delta should be near 0.
    """
    all_a = [t for tl in token_lists_a for t in tl]
    all_b = [t for tl in token_lists_b for t in tl]
    combined = Counter(all_a) + Counter(all_b)
    features = [w for w, _ in combined.most_common(n_features)]

    def chunk_freqs(tokens, n_chunks):
        """Split tokens into chunks and compute relative freq vectors."""
        chunk_size = max(len(tokens) // n_chunks, 1)
        chunks = [tokens[i:i + chunk_size]
                  for i in range(0, len(tokens), chunk_size)]
        if len(chunks) > n_chunks:
            chunks = chunks[:n_chunks]
        vecs = []
        for chunk in chunks:
            total = len(chunk)
            freq = Counter(chunk)
            vecs.append([freq.get(w, 0) / total if total else 0 for w in features])
        return np.array(vecs)

    vecs_a = chunk_freqs(all_a, n_chunks)
    vecs_b = chunk_freqs(all_b, n_chunks)
    all_vecs = np.vstack([vecs_a, vecs_b])

    # Z-score each feature across all chunks
    mu = all_vecs.mean(axis=0)
    sigma = all_vecs.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1e-12

    z_a = ((vecs_a - mu) / sigma).mean(axis=0)  # centroid of corpus A
    z_b = ((vecs_b - mu) / sigma).mean(axis=0)  # centroid of corpus B

    delta = np.mean(np.abs(z_a - z_b))
    return float(delta)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def ks_test(dist_a: np.ndarray, dist_b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p-value)."""
    stat, p = stats.ks_2samp(dist_a, dist_b)
    return float(stat), float(p)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_overlaid_histogram(a, b, xlabel, title, path, label_a="Original", label_b="Replicated"):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.histogram_bin_edges(np.concatenate([a, b]), bins="auto")
    ax.hist(a, bins=bins, alpha=0.5, density=True, label=label_a, color="#2196F3")
    ax.hist(b, bins=bins, alpha=0.5, density=True, label=label_b, color="#FF9800")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_qq(a, b, title, path, label_a="Original", label_b="Replicated"):
    """QQ plot comparing quantiles of two distributions."""
    q = np.linspace(0, 100, 200)
    qa = np.percentile(a, q)
    qb = np.percentile(b, q)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(qa, qb, s=8, alpha=0.6)
    lim = [min(qa.min(), qb.min()), max(qa.max(), qb.max())]
    ax.plot(lim, lim, "k--", alpha=0.4, lw=1)
    ax.set_xlabel(f"{label_a} quantiles")
    ax.set_ylabel(f"{label_b} quantiles")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_rank_frequency(words_a, freqs_a, words_b, freqs_b, path,
                        label_a="Original", label_b="Replicated"):
    """Log-log rank-frequency plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ranks = np.arange(1, len(freqs_a) + 1)
    ax.loglog(ranks, freqs_a, "o-", markersize=3, alpha=0.7, label=label_a)
    ranks_b = np.arange(1, len(freqs_b) + 1)
    ax.loglog(ranks_b, freqs_b, "s-", markersize=3, alpha=0.7, label=label_b)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative frequency")
    ax.set_title("Word rank-frequency (Zipf) plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare linguistic fingerprints")
    parser.add_argument("--original", required=True, help="Path to original LLMMap CSV")
    parser.add_argument("--replicated", required=True, help="Path to replicated CSV")
    parser.add_argument("--output", default=str(RESULTS_DIR),
                        help="Output directory for results and plots")
    parser.add_argument("--text-col", default="text", help="Column name containing text")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading corpora...")
    texts_orig = load_texts(args.original, args.text_col)
    texts_repl = load_texts(args.replicated, args.text_col)

    print("\nTokenising...")
    toks_orig = corpus_tokens(texts_orig)
    toks_repl = corpus_tokens(texts_repl)

    results = {}

    # --- Distributional ---
    print("\n=== Distributional metrics ===")

    tl_orig = text_lengths(toks_orig)
    tl_repl = text_lengths(toks_repl)
    results["text_length_mean_orig"] = float(np.mean(tl_orig))
    results["text_length_mean_repl"] = float(np.mean(tl_repl))
    results["text_length_std_orig"] = float(np.std(tl_orig))
    results["text_length_std_repl"] = float(np.std(tl_repl))
    ks_stat, ks_p = ks_test(tl_orig, tl_repl)
    results["text_length_ks_stat"] = ks_stat
    results["text_length_ks_p"] = ks_p
    results["text_length_cohens_d"] = cohens_d(tl_orig, tl_repl)
    print(f"  Text length: orig={np.mean(tl_orig):.1f}±{np.std(tl_orig):.1f}, "
          f"repl={np.mean(tl_repl):.1f}±{np.std(tl_repl):.1f}")
    print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4f} | Cohen's d={results['text_length_cohens_d']:.4f}")

    sl_orig = sentence_lengths(texts_orig)
    sl_repl = sentence_lengths(texts_repl)
    results["sent_length_mean_orig"] = float(np.mean(sl_orig))
    results["sent_length_mean_repl"] = float(np.mean(sl_repl))
    ks_stat, ks_p = ks_test(sl_orig, sl_repl)
    results["sent_length_ks_stat"] = ks_stat
    results["sent_length_ks_p"] = ks_p
    results["sent_length_cohens_d"] = cohens_d(sl_orig, sl_repl)
    print(f"  Sentence length: orig={np.mean(sl_orig):.1f}±{np.std(sl_orig):.1f}, "
          f"repl={np.mean(sl_repl):.1f}±{np.std(sl_repl):.1f}")
    print(f"  KS test: stat={ks_stat:.4f}, p={ks_p:.4f} | Cohen's d={results['sent_length_cohens_d']:.4f}")

    # --- Lexical ---
    print("\n=== Lexical metrics ===")

    ttr_orig = ttr(toks_orig)
    ttr_repl = ttr(toks_repl)
    results["ttr_orig"] = ttr_orig
    results["ttr_repl"] = ttr_repl
    print(f"  TTR: orig={ttr_orig:.4f}, repl={ttr_repl:.4f}")

    mattr_orig = mattr(toks_orig)
    mattr_repl = mattr(toks_repl)
    results["mattr_orig"] = mattr_orig
    results["mattr_repl"] = mattr_repl
    print(f"  MATTR (w=500): orig={mattr_orig:.4f}, repl={mattr_repl:.4f}")

    hap_orig = hapax_ratio(toks_orig)
    hap_repl = hapax_ratio(toks_repl)
    results["hapax_ratio_orig"] = hap_orig
    results["hapax_ratio_repl"] = hap_repl
    print(f"  Hapax ratio: orig={hap_orig:.4f}, repl={hap_repl:.4f}")

    yk_orig = yules_k(toks_orig)
    yk_repl = yules_k(toks_repl)
    results["yules_k_orig"] = yk_orig
    results["yules_k_repl"] = yk_repl
    print(f"  Yule's K: orig={yk_orig:.2f}, repl={yk_repl:.2f}")

    # Word frequencies
    words_orig, freqs_orig = word_freq_vector(toks_orig, top_n=200)
    words_repl, freqs_repl = word_freq_vector(toks_repl, top_n=200)
    jaccard = jaccard_top_n(words_orig[:100], words_repl[:100])
    results["jaccard_top100"] = jaccard
    print(f"  Jaccard (top-100 words): {jaccard:.4f}")

    # --- N-gram level ---
    print("\n=== N-gram metrics ===")

    bi_orig = ngram_distribution(toks_orig, n=2)
    bi_repl = ngram_distribution(toks_repl, n=2)
    jsd_bi = jensen_shannon_divergence(bi_orig, bi_repl)
    results["bigram_jsd"] = jsd_bi
    print(f"  Bigram JSD: {jsd_bi:.4f}")

    tri_orig = ngram_distribution(toks_orig, n=3)
    tri_repl = ngram_distribution(toks_repl, n=3)
    jsd_tri = jensen_shannon_divergence(tri_orig, tri_repl)
    results["trigram_jsd"] = jsd_tri
    print(f"  Trigram JSD: {jsd_tri:.4f}")

    # --- Stylometric ---
    print("\n=== Stylometric metrics ===")

    delta = burrows_delta(toks_orig, toks_repl)
    results["burrows_delta"] = delta
    print(f"  Burrows' Delta (100 MFW): {delta:.4f}")

    # --- Plots ---
    print("\n=== Generating plots ===")

    plot_overlaid_histogram(
        tl_orig, tl_repl,
        "Text length (tokens)", "Text length distribution",
        out / "text_length_hist.png",
    )
    plot_overlaid_histogram(
        sl_orig, sl_repl,
        "Sentence length (tokens)", "Sentence length distribution",
        out / "sent_length_hist.png",
    )
    plot_qq(
        tl_orig, tl_repl,
        "Text length QQ plot",
        out / "text_length_qq.png",
    )
    plot_qq(
        sl_orig, sl_repl,
        "Sentence length QQ plot",
        out / "sent_length_qq.png",
    )
    plot_rank_frequency(
        words_orig, freqs_orig, words_repl, freqs_repl,
        out / "rank_frequency.png",
    )

    # --- Save results ---
    results_df = pd.DataFrame([results])
    results_path = out / "fingerprint_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results table saved to {results_path}")

    # --- Summary verdict ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    warnings = []
    if results["text_length_ks_p"] < 0.05:
        warnings.append(f"Text length distributions differ (KS p={results['text_length_ks_p']:.4f})")
    if results["sent_length_ks_p"] < 0.05:
        warnings.append(f"Sentence length distributions differ (KS p={results['sent_length_ks_p']:.4f})")
    if abs(results["text_length_cohens_d"]) > 0.2:
        warnings.append(f"Text length effect size non-trivial (d={results['text_length_cohens_d']:.3f})")
    if results["jaccard_top100"] < 0.7:
        warnings.append(f"Top-100 word overlap low (Jaccard={results['jaccard_top100']:.3f})")
    if results["bigram_jsd"] > 0.15:
        warnings.append(f"Bigram distributions divergent (JSD={results['bigram_jsd']:.4f})")
    if results["burrows_delta"] > 1.0:
        warnings.append(f"Burrows' Delta high ({results['burrows_delta']:.3f}) — may indicate different 'author'")

    if not warnings:
        print("All metrics within expected ranges — corpora appear linguistically equivalent.")
    else:
        print("Potential differences detected:")
        for w in warnings:
            print(f"  ⚠ {w}")
        print("\nThese may be due to inference setup differences (quantization, sampling).")
        print("Investigate before concluding non-equivalence.")

    print(f"\nAll outputs in: {out}/")


if __name__ == "__main__":
    main()
