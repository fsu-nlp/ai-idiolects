#!/usr/bin/env python3
"""Stage 4 (Lambda-GPU): sentence-embedding semantic map.

Encodes every cleaned text (all Phase 0/1/2a corpora + the Phase 2b human
baselines) with a sentence encoder, projects to 2-D with UMAP, and records
per-corpus centroids / spread / internal redundancy. A supplementary view
of the corpus collection in *semantic* space (vs the lexical/stylometric
distances of §4.1): do the eras / topics / models separate?

Default encoder: ``intfloat/e5-large-v2`` (ungated, ~335 M params, 1024-dim;
512-token cap → long texts are truncated to their first ~400 words, which
is still topic-representative — noted). E5 wants a ``"passage: "`` prefix on
documents; that's applied. Embeddings are L2-normalised (cosine geometry).
Runs on CUDA if available, CPU otherwise (~minutes on an A100, ~45–90 min
on CPU). The Stage-1a quality filter is applied (drop ``repetition_loop`` /
``code_contam`` / ``non_english`` / ``truncation`` rows).

Output → ``results/phase2_mapping/stage4_embedding_map/``:
``embeddings_2d.csv`` (model_dir, topic, x, y — one row per text;
**not committed**, regenerable) and ``summary.csv`` (per model_dir × scope:
n_texts, centroid_x, centroid_y, spread = mean cosine distance from the
centroid in 1024-d, mean_internal_cosine = mean pairwise cosine of the
corpus's text embeddings — high ⇒ semantically homogeneous / templated).
Render with ``render_embedding_map.py``.

(A within-model near-duplicate-*sentence* "template detection" pass — split
texts into sentences, embed, cluster — is a noted TODO; v1 does the
text-level ``mean_internal_cosine`` proxy instead.)

Usage::

    python3 scripts/analyse/embedding_map.py
    python3 scripts/analyse/embedding_map.py --encoder intfloat/e5-large-v2 --limit 50

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aiidiolects.build_distance_matrix import QUALITY_FLAGS_DROP, TOPICS, find_cleaned_csvs, model_dir_for
from aiidiolects.paths import DATA_DIR, RESULTS_DIR

SCOPES = ["pooled"] + TOPICS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(DATA_DIR))
    ap.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "phase2_mapping" / "stage4_embedding_map"),
    )
    ap.add_argument("--encoder", default="intfloat/e5-large-v2")
    ap.add_argument("--passage-prefix", default="passage: ",
                    help="prefix prepended to each text before encoding "
                         "(e5 family expects 'passage: '; set '' for plain encoders)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-flagged", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N texts/corpus (smoke test)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading cleaned corpora …")
    keys: list[tuple[str, str]] = []     # (model_dir, topic) per text, aligned with `texts`
    texts: list[str] = []
    for csv in find_cleaned_csvs(Path(args.base), include_human=True):
        m = model_dir_for(csv)
        tp = csv.parent.name
        df = pd.read_csv(csv)
        if not args.keep_flagged:
            mask = pd.Series(True, index=df.index)
            for flag in QUALITY_FLAGS_DROP:
                if flag in df.columns:
                    mask &= ~df[flag].astype(bool)
            df = df.loc[mask]
        rows = [t for t in df["text_clean"].fillna("").astype(str).tolist() if t.strip()]
        if args.limit:
            rows = rows[: args.limit]
        keys.extend((m, tp) for _ in rows)
        texts.extend(rows)
    print(f"  {len(texts):,} texts; encoder={args.encoder}")

    # --- encode ---
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(args.encoder, device=device)
    inp = [args.passage_prefix + t for t in texts] if args.passage_prefix else texts
    print(f"  encoding on {device} …")
    emb = enc.encode(inp, batch_size=args.batch_size, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True)
    print(f"  embeddings: {emb.shape}  ({time.time() - t0:.1f}s)")

    # --- UMAP to 2-D ---
    import umap
    reducer = umap.UMAP(n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
                        metric="cosine", random_state=args.seed)
    coords = reducer.fit_transform(emb)
    print(f"  UMAP done  ({time.time() - t0:.1f}s)")

    mds = pd.DataFrame({"model_dir": [k[0] for k in keys],
                        "topic": [k[1] for k in keys],
                        "x": coords[:, 0], "y": coords[:, 1]})
    mds_path = out_dir / "embeddings_2d.csv"
    mds.to_csv(mds_path, index=False)
    print(f"Wrote {mds_path}  ({len(mds):,} rows)")

    # --- per-corpus summary (centroid / spread / internal redundancy) in 1024-d ---
    key_arr = np.array([f"{m}\t{tp}" for m, tp in keys])
    summ_rows: list[dict] = []
    for scope in SCOPES:
        for m in sorted({k[0] for k in keys}):
            if scope == "pooled":
                idx = np.array([i for i, (mm, _) in enumerate(keys) if mm == m])
            else:
                idx = np.array([i for i, (mm, tp) in enumerate(keys)
                                if mm == m and tp == scope])
            if idx.size < 2:
                continue
            E = emb[idx]                                  # (k, 1024), already L2-norm
            centroid = E.mean(axis=0)
            cn = np.linalg.norm(centroid)
            cos_to_centroid = E @ (centroid / cn) if cn > 0 else np.zeros(len(E))
            # mean pairwise cosine among the corpus's texts (off-diagonal mean of E E^T)
            k = len(E)
            sum_all = float((E @ E.T).sum())
            mean_pair_cos = (sum_all - k) / (k * (k - 1)) if k > 1 else float("nan")
            cx = float(coords[idx, 0].mean())
            cy = float(coords[idx, 1].mean())
            summ_rows.append({
                "model_dir": m, "scope": scope, "n_texts": int(k),
                "centroid_x": cx, "centroid_y": cy,
                "spread": float(1.0 - cos_to_centroid.mean()),     # mean cosine *distance* from centroid
                "mean_internal_cosine": mean_pair_cos,
            })
    summ = pd.DataFrame(summ_rows)
    summ_path = out_dir / "summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"Wrote {summ_path}  ({len(summ):,} rows)")

    if not summ.empty:
        ic = summ[summ.scope == "pooled"].sort_values("mean_internal_cosine", ascending=False)
        print("\nMean internal cosine (pooled), ranked high→low "
              "(high ⇒ semantically homogeneous / templated):")
        for _, r in ic.iterrows():
            print(f"  {r.model_dir:<22} {r['mean_internal_cosine']:.4f}  (n={int(r['n_texts'])})")
    print(f"\nDone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
