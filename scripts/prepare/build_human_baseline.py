"""Build the human-text baseline corpora into Improta-schema CSVs.

Two modes:

* ``--survey`` — **Pass 1, the "see the data first" reconnaissance.**
  Stream-parses every r/ChangeMyView original post in
  ``data/human_baseline/webis_cmv_20/raw/posts_malleability.jsonl.bz2``,
  runs a deliberately broad topic-keyword sweep, and writes
  ``data/human_baseline/webis_cmv_20/survey/``:

    - ``topic_match_stats.md``   — per-topic OP counts at several min-length
      thresholds, length percentiles, distinct-author / repeat-author counts,
      delta-award rate, link-flair distribution.
    - ``keyword_hits.csv``       — raw hit count per candidate keyword.
    - ``samples_<topic>.txt``    — ~30 random matched OPs, full title+body, for
      register/relevance eyeballing.

  Nothing under the corpus dirs is written in survey mode. The output is the
  *evidence* the authors use to lock ``FILTER_CONFIG`` below; it is
  gitignored. **There is a human review gate here** — do not run the default
  build mode until the keyword lists and thresholds in ``FILTER_CONFIG`` have
  been reviewed against this survey.

* (default) — **Pass 2, the deterministic build.** Applies the locked
  ``FILTER_CONFIG`` and writes:

    - ``webis_cmv_20/<topic>/Webis-CMV-20.csv`` — cols
      ``topic,model,text,language,author_id,source_id``
      (``model="Webis-CMV-20"``, ``language="en"``, ``author_id`` =
      salted-SHA-256 of the Reddit username, ``source_id`` = submission id).
    - ``reuters_50/reuters_ccat/Reuters-50.csv`` — cols
      ``topic,model,text,language,author_id`` (one combined file; ``topic``
      is the constant ``"reuters_ccat"``, ``author_id`` = the C50 author
      directory name).
    - ``manifest/webis_selected.csv``  — one row per emitted Webis text:
      ``topic,submission_id,author_id,sha256,n_words``.
    - ``manifest/reuters_manifest.csv`` — one row per emitted Reuters text:
      ``author_id,source_file,sha256,n_words``.
    - ``manifest/webis_filter_config.json`` — resolved ``FILTER_CONFIG`` +
      build timestamp + langdetect version, for provenance.

  Same ``raw/`` + same ``FILTER_CONFIG`` → byte-identical CSVs and manifest.
  Rows are emitted in a stable sort order; only the manifest JSON carries a
  timestamp. This is the artifact that lets a fresh checkout regenerate the
  working subsets without us redistributing any text.

Run from the repository root::

    python3 download_human_baseline.py            # first — fetch raw/
    python3 build_human_baseline.py --survey      # Pass 1 — then review the gate
    # ... review the survey, then lock FILTER_CONFIG ...
    python3 build_human_baseline.py               # Pass 2 — build + manifest

Requires: ``langdetect`` (English filter; same opportunistic use as
``clean_corpus.py``). Everything else is stdlib (``bz2``, ``json``, ``html``,
``hashlib``, ``csv``).

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import html
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from aiidiolects.paths import DATA_DIR as REPO_DATA_DIR

DATA_DIR = REPO_DATA_DIR / "human_baseline"
WEBIS_DIR = DATA_DIR / "webis_cmv_20"
WEBIS_RAW = WEBIS_DIR / "raw"
WEBIS_SURVEY = WEBIS_DIR / "survey"
REUTERS_DIR = DATA_DIR / "reuters_50"
REUTERS_RAW = REUTERS_DIR / "raw"
MANIFEST_DIR = DATA_DIR / "manifest"

# Preference order for the Webis OP source file (first one present wins).
WEBIS_SOURCE_CANDIDATES = ("posts_malleability.jsonl.bz2", "threads.jsonl.bz2")

IMPROTA_TOPICS = ["climate", "global_warming", "math_anxiety", "misinfo_health"]

DELETED_MARKERS = {"[deleted]", "[removed]", "", None}
KNOWN_BOTS = {"DeltaBot", "AutoModerator", "BotDefense", "Mod_Approved_Source",
              "transcribersofreddit", "RemindMeBot", "MAGIC_EYE_BOT"}

# Stable per-run salt for hashing Reddit usernames into author_ids. Fixed so
# the manifest is reproducible; not secret (it only obfuscates the join back
# to live Reddit accounts in the committed manifest).
AUTHOR_SALT = "ai-idiolects/webis-cmv-20/v1"


# ---------------------------------------------------------------------------
# FILTER_CONFIG — the locked Pass-2 parameters.
#
# Locked 2026-05-11 after the Stage-A survey (webis_cmv_20/survey/) was
# reviewed by the authors. Decisions taken:
#   - Source = posts_malleability.jsonl.bz2 only (the ~16 K-OP malleability
#     subset). The resulting working-subset sizes (~360 / 136 / 38 / 146 OPs
#     at ≥100w for climate / global_warming / math_anxiety / misinfo_health)
#     were accepted as a "good enough" baseline — threads.jsonl not pulled.
#   - climate keyword list kept deliberately BROAD (it also includes "global
#     warming", so global_warming ⊆ climate). Label it "climate (broad)" in
#     prose / figures — `labels` below carries that. The topic *directory* is
#     still "climate" so the existing pipeline compares it against the LLM
#     `climate` corpus in the same scope; the breadth is documented in this
#     config, in data/human_baseline/README.md, and in the committed manifest.
#   - math_anxiety: same pattern — topic dir stays "math_anxiety" (so it lines
#     up with the LLM `math_anxiety` corpus), but it is LABELLED "math (broad)"
#     (parallel to "climate (broad)"). CMV has essentially no genuine "fear of
#     math" content; the ~37 matches are math-EDUCATION posts ("calculators in
#     math class", "should HS require math"). `labels` carries this; §3/§4
#     report it. (Resolved 2026-05-11: keep + relabel "math (broad)".)
#   - Topic assignment is NON-disjoint: a post lands in every topic whose
#     keyword list it matches (e.g. a "climate misinformation" post → both
#     `climate` and `misinfo_health`). The manifest reports per-topic counts
#     and pairwise overlap. This mirrors the LLM side (separate per-prompt
#     corpora) without artificially carving up naturally co-occurring topics.
# To re-tighten: edit `keywords` / `min_words` / `max_words` /
# `target_n_per_topic`, bump the date in `review_note`, re-run the build.
# ---------------------------------------------------------------------------

FILTER_CONFIG: dict = {
    "locked": True,                  # reviewed/greenlit 2026-05-11; build mode warns if False
    "review_note": ("locked 2026-05-11 — the posts_malleability subset sizes were accepted as a "
                    "'good enough' baseline; the `climate` and `math_anxiety` topic dirs keep their "
                    "names (so the pipeline compares them against the LLM `climate` / `math_anxiety` "
                    "corpora) but are LABELLED 'climate (broad)' / 'math (broad)' — both are broad "
                    "keyword filters, and the `math_anxiety` matches are math-education CMVs, not "
                    "'fear of math' content. Full transparency via README + manifest; the ~37 "
                    "'math (broad)' OPs are reported as such rather than treated as a matched sample."),
    "min_words": 100,                # OP body length floor (CMV's own rule is ~500 chars ≈ 80-100 words)
    "max_words": 2000,               # drop pathological essays/copypasta above this
    "include_title_in_text": False,  # text = selftext only; title kept as provenance, not corpus text
    "target_n_per_topic": None,      # None = keep all; or an int to random-sample (seeded) down to N
    "sample_seed": 20260511,
    "langdetect_seed": 0,            # matches clean_corpus.py
    # Prose / figure labels — make the broad keyword filters explicit wherever
    # the topics are named (the topic *directory* names stay aligned with the
    # LLM corpora: `climate`, `global_warming`, `math_anxiety`, `misinfo_health`).
    "labels": {
        "climate": "climate (broad keyword filter — see `keywords`)",
        "global_warming": "global warming",
        "math_anxiety": "math (broad keyword filter — see `keywords`; CMV math-education posts, ≈no genuine 'fear of math' content)",
        "misinfo_health": "health misinformation",
    },
    # Non-disjoint topic assignment: a post is emitted into every topic whose
    # keyword list it matches. `climate` deliberately includes "global warming"
    # so global_warming ⊆ climate.
    "keywords": {
        "global_warming": [
            "global warming",
        ],
        "climate": [
            "climate change", "climate crisis", "climate emergency", "climate policy",
            "climate science", "climate denial", "climate action", "the climate",
            "global warming",
            "carbon emissions", "carbon footprint", "carbon tax", "carbon dioxide",
            "greenhouse gas", "greenhouse effect", "fossil fuels", "fossil fuel",
            "renewable energy", "clean energy", "paris agreement", "paris accord",
            "net zero", "net-zero", "decarboni", "ipcc", "sea level rise",
            "extreme weather", "rising temperatures",
        ],
        "math_anxiety": [
            "math anxiety", "maths anxiety", "mathematics anxiety", "math phobia",
            "fear of math", "afraid of math", "scared of math", "anxious about math",
            "bad at math", "terrible at math", "hate math", "i hate maths",
            "math is hard", "maths is hard", "struggle with math", "struggling with math",
            "math education", "maths education", "mathematics education",
            "math class", "maths class", "math curriculum", "learning math",
            "teaching math", "math teacher", "maths teacher", "dyscalculia",
            "math requirement", "should math be", "why do we learn math",
            "stem anxiety",
        ],
        "misinfo_health": [
            "health misinformation", "medical misinformation", "health disinformation",
            "anti-vax", "anti vax", "antivax", "anti-vaxx", "antivaxx", "anti-vaxxer",
            "vaccine", "vaccines", "vaccinated", "vaccination", "vaccine hesitancy",
            "vaccine misinformation", "mmr vaccine", "flu shot", "covid vaccine",
            "covid misinformation", "pandemic misinformation", "covid conspiracy",
            "alternative medicine", "natural medicine", "natural remedies", "naturopath",
            "homeopathy", "homeopathic", "essential oils", "detox tea", "detoxify",
            "miracle cure", "snake oil", "big pharma", "medical conspiracy",
            "fake health", "health hoax", "wellness industry", "wellness influencer",
            "fluoride in water", "raw milk", "anti-science", "pseudoscience",
            "faith healing", "chiropractic", "essential oil", "supplement industry",
        ],
    },
}


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _author_id(username: str) -> str:
    return hashlib.sha256((AUTHOR_SALT + "\x00" + username).encode("utf-8")).hexdigest()[:16]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalise_text(raw: str) -> str:
    """HTML-unescape, normalise newlines, collapse runs of blank lines, strip."""
    t = html.unescape(raw or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _word_count(text: str) -> int:
    return len(text.split())


def _resolve_webis_source() -> Path:
    for name in WEBIS_SOURCE_CANDIDATES:
        p = WEBIS_RAW / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No Webis source file found in {WEBIS_RAW}/ — run `python3 download_human_baseline.py` first.\n"
        f"Looked for: {', '.join(WEBIS_SOURCE_CANDIDATES)}"
    )


def _iter_ops(source: Path):
    """Yield (record_dict) for each CMV original post in the source file.

    For ``posts_malleability.jsonl.bz2`` each line *is* an OP record.
    For ``threads.jsonl.bz2`` each line is a thread; the root submission is the
    same dict minus its comment tree — same top-level keys, so we yield it too.
    Either way the caller reads ``id/name/author/title/selftext/...``.
    """
    with bz2.open(source, "rt", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            yield rec


def _clean_op(rec: dict) -> tuple[str, str, str, str] | None:
    """Return (source_id, author, title, body) for a usable OP, else None.

    Filters: must be a self-post in r/changemyview, non-deleted body & author,
    not a known bot. No length/topic filtering here — callers do that.
    """
    if str(rec.get("subreddit", "")).lower() not in ("changemyview", ""):
        return None
    author = rec.get("author")
    if author in DELETED_MARKERS or (isinstance(author, str) and author in KNOWN_BOTS):
        return None
    body_raw = rec.get("selftext")
    if body_raw in DELETED_MARKERS:
        return None
    body = _normalise_text(body_raw)
    if not body or body in ("[deleted]", "[removed]"):
        return None
    title = _normalise_text(rec.get("title") or "")
    source_id = str(rec.get("id") or rec.get("name") or "").strip()
    if not source_id:
        return None
    return source_id, str(author), title, body


def _match_topics(haystack: str) -> dict[str, list[str]]:
    """Return {topic: [keywords that hit]} over the lowercased haystack."""
    hay = haystack.lower()
    hits: dict[str, list[str]] = {}
    for topic, kws in FILTER_CONFIG["keywords"].items():
        got = [kw for kw in kws if kw in hay]
        if got:
            hits[topic] = got
    return hits


def _assign_topics(hits: dict[str, list[str]]) -> list[str]:
    """Non-disjoint: every topic whose keyword list matched, in IMPROTA_TOPICS order."""
    return [t for t in IMPROTA_TOPICS if t in hits]


# ---------------------------------------------------------------------------
# Survey mode (Pass 1)
# ---------------------------------------------------------------------------

def run_survey() -> None:
    source = _resolve_webis_source()
    WEBIS_SURVEY.mkdir(parents=True, exist_ok=True)
    print(f"Survey: reading {source} ...")

    n_lines = n_ops = 0
    kw_hits: Counter = Counter()                       # keyword -> raw OP hit count
    # per-topic: list of (source_id, author, n_words, delta_bool, flair, title, body)
    per_topic: dict[str, list[tuple]] = defaultdict(list)
    overall_wc: list[int] = []

    for rec in _iter_ops(source):
        n_lines += 1
        cleaned = _clean_op(rec)
        if cleaned is None:
            continue
        n_ops += 1
        source_id, author, title, body = cleaned
        wc = _word_count(body)
        overall_wc.append(wc)
        hits = _match_topics(title + "\n" + body)
        for topic, got in hits.items():
            for kw in got:
                kw_hits[kw] += 1
        for topic in hits:
            per_topic[topic].append(
                (source_id, author, wc, bool(rec.get("delta")),
                 rec.get("link_flair_text"), title, body)
            )

    # ---- keyword_hits.csv ----
    with (WEBIS_SURVEY / "keyword_hits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topic", "keyword", "op_hits"])
        for topic, kws in FILTER_CONFIG["keywords"].items():
            for kw in kws:
                w.writerow([topic, kw, kw_hits.get(kw, 0)])

    # ---- samples_<topic>.txt ----
    rng = random.Random(FILTER_CONFIG["sample_seed"])
    for topic, rows in per_topic.items():
        # only sample from rows that clear the provisional min_words, so the
        # samples reflect what the build would actually keep-ish
        eligible = [r for r in rows if r[2] >= FILTER_CONFIG["min_words"]]
        pick = eligible if len(eligible) <= 30 else rng.sample(eligible, 30)
        with (WEBIS_SURVEY / f"samples_{topic}.txt").open("w", encoding="utf-8") as f:
            f.write(f"# Survey samples — topic '{topic}' — {len(pick)} of {len(rows)} matched OPs "
                    f"({len(eligible)} clear the provisional {FILTER_CONFIG['min_words']}-word floor)\n")
            f.write("# (raw broad-keyword matches; NOT the final corpus — review for register/relevance)\n\n")
            for source_id, author, wc, delta, flair, title, body in pick:
                f.write("=" * 100 + "\n")
                f.write(f"id={source_id}  author={author}  words={wc}  op_awarded_delta={delta}  flair={flair!r}\n")
                f.write(f"TITLE: {title}\n\n{body}\n\n")

    # ---- topic_match_stats.md ----
    thresholds = [50, 100, 200, 400]
    lines: list[str] = []
    lines.append("# Webis-CMV-20 — topic-keyword survey\n")
    lines.append(f"Generated {_now()} from `{source.name}`.\n")
    lines.append(f"- Lines read: **{n_lines:,}**  ·  usable OPs (self-post, non-deleted, non-bot): **{n_ops:,}**")
    if overall_wc:
        q = statistics.quantiles(overall_wc, n=20)  # 5th, 10th, ... 95th
        lines.append(f"- OP body word count — min {min(overall_wc)}, "
                     f"p5 {q[0]:.0f}, p25 {q[4]:.0f}, median {statistics.median(overall_wc):.0f}, "
                     f"p75 {q[14]:.0f}, p95 {q[18]:.0f}, max {max(overall_wc)}")
    lines.append("")
    lines.append("## Matched OPs per topic, by minimum body word count\n")
    lines.append("| topic | any | " + " | ".join(f"≥{t}w" for t in thresholds)
                 + " | distinct authors (≥100w) | authors with ≥2 OPs (≥100w) | OP-delta rate (≥100w) |")
    lines.append("|---|" + "---|" * (len(thresholds) + 1) + "---|---|---|")
    for topic in IMPROTA_TOPICS:
        rows = per_topic.get(topic, [])
        counts = {t: sum(1 for r in rows if r[2] >= t) for t in thresholds}
        rows100 = [r for r in rows if r[2] >= 100]
        auths = Counter(r[1] for r in rows100)
        n_authors = len(auths)
        n_repeat = sum(1 for c in auths.values() if c >= 2)
        delta_rate = (sum(1 for r in rows100 if r[3]) / len(rows100)) if rows100 else 0.0
        lines.append(f"| {topic} | {len(rows)} | "
                     + " | ".join(str(counts[t]) for t in thresholds)
                     + f" | {n_authors} | {n_repeat} | {delta_rate:.1%} |")
    lines.append("")
    lines.append("## Top link-flair among matched OPs (≥100w)\n")
    for topic in IMPROTA_TOPICS:
        rows100 = [r for r in per_topic.get(topic, []) if r[2] >= 100]
        flairs = Counter(str(r[4]) for r in rows100).most_common(8)
        lines.append(f"- **{topic}**: " + (", ".join(f"{fl}×{n}" for fl, n in flairs) if flairs else "—"))
    lines.append("")
    lines.append("## Keyword hit counts\n")
    lines.append("See `keyword_hits.csv` (per-keyword OP hit counts) for narrowing the lists.\n")
    lines.append("Top 15 keywords overall:\n")
    for kw, n in kw_hits.most_common(15):
        lines.append(f"- `{kw}` — {n}")
    lines.append("")
    lines.append("---\n")
    lines.append("**Review gate.** Narrow the keyword lists in `build_human_baseline.py` → `FILTER_CONFIG['keywords']`, "
                 "fix `min_words` / `max_words` / `target_n_per_topic`, set `locked = True` with a dated note, then run "
                 "`python3 build_human_baseline.py` (no `--survey`). Sparse topics (likely `math_anxiety`, possibly "
                 "`misinfo_health`) should be reported as-is in §4 Results, not padded with off-topic posts.\n")
    (WEBIS_SURVEY / "topic_match_stats.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Survey written to {WEBIS_SURVEY}/ :")
    print(f"  topic_match_stats.md  ·  keyword_hits.csv  ·  samples_<topic>.txt "
          f"({', '.join(t for t in IMPROTA_TOPICS if per_topic.get(t))})")
    print("\n>>> REVIEW GATE: read topic_match_stats.md + samples_*.txt, then lock FILTER_CONFIG before the build. <<<")


# ---------------------------------------------------------------------------
# Build mode (Pass 2)
# ---------------------------------------------------------------------------

def _ensure_langdetect():
    try:
        from langdetect import DetectorFactory, detect  # noqa
        DetectorFactory.seed = FILTER_CONFIG["langdetect_seed"]
        import langdetect as _ld
        return detect, getattr(_ld, "__version__", "unknown")
    except ImportError:
        sys.exit("ERROR: `langdetect` is required for the build (English filter). "
                 "Install it: pip3 install langdetect")


def build_webis() -> list[dict]:
    """Filter the Webis OPs per FILTER_CONFIG, write per-topic CSVs, return manifest rows."""
    detect, ld_version = _ensure_langdetect()
    source = _resolve_webis_source()
    print(f"Build (Webis): reading {source} ...")

    min_w, max_w = FILTER_CONFIG["min_words"], FILTER_CONFIG["max_words"]
    include_title = FILTER_CONFIG["include_title_in_text"]

    # bucket -> list of (source_id, author, text, n_words)
    buckets: dict[str, list[tuple]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_author_text: set[tuple[str, str]] = set()  # (author_id, text-prefix hash) dedup
    n_seen = n_kept = 0

    for rec in _iter_ops(source):
        cleaned = _clean_op(rec)
        if cleaned is None:
            continue
        source_id, author, title, body = cleaned
        if source_id in seen_ids:
            continue
        text = (title + "\n\n" + body) if include_title else body
        wc = _word_count(text)
        if wc < min_w or wc > max_w:
            continue
        topics = _assign_topics(_match_topics(title + "\n" + body))
        if not topics:
            continue
        n_seen += 1
        # language filter
        try:
            if detect(text[:4000]) != "en":
                continue
        except Exception:
            continue
        aid = _author_id(author)
        dkey = (aid, hashlib.sha256(text[:300].encode("utf-8")).hexdigest())
        if dkey in seen_author_text:
            continue
        seen_author_text.add(dkey)
        seen_ids.add(source_id)
        for topic in topics:                 # non-disjoint: a post may land in several topic CSVs
            buckets[topic].append((source_id, aid, text, wc))
        n_kept += 1

    # optional per-topic downsampling
    target = FILTER_CONFIG["target_n_per_topic"]
    if target:
        rng = random.Random(FILTER_CONFIG["sample_seed"])
        for topic, rows in buckets.items():
            if len(rows) > target:
                buckets[topic] = sorted(rng.sample(rows, target))

    # write per-topic CSVs (sorted by source_id for determinism)
    manifest_rows: list[dict] = []
    for topic in IMPROTA_TOPICS:
        rows = sorted(buckets.get(topic, []), key=lambda r: r[0])
        out_dir = WEBIS_DIR / topic
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "Webis-CMV-20.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["topic", "model", "text", "language", "author_id", "source_id"])
            for source_id, aid, text, wc in rows:
                w.writerow([topic, "Webis-CMV-20", text, "en", aid, source_id])
                manifest_rows.append({
                    "topic": topic, "submission_id": source_id, "author_id": aid,
                    "sha256": _sha256_text(text), "n_words": wc,
                })
        print(f"  {topic:>14}: {len(rows)} OPs → {out_csv}")
    n_topics = len([t for t in IMPROTA_TOPICS if buckets.get(t)])
    print(f"Build (Webis): {n_seen:,} unique OPs matched ≥1 topic + length window → {n_kept:,} kept after "
          f"EN filter & dedup → {sum(len(v) for v in buckets.values()):,} topic-rows written across {n_topics} topics "
          f"(non-disjoint — see manifest for per-topic counts & overlap).")
    return manifest_rows, ld_version


def build_reuters() -> list[dict]:
    """Walk the C50 raw dirs, write one combined CSV, return manifest rows."""
    train, test = REUTERS_RAW / "C50train", REUTERS_RAW / "C50test"
    if not (train.is_dir() and test.is_dir()):
        raise FileNotFoundError(
            f"Reuters C50 not extracted under {REUTERS_RAW}/ — run `python3 download_human_baseline.py` first."
        )
    print(f"Build (Reuters): walking {REUTERS_RAW}/C50train + C50test ...")
    out_dir = REUTERS_DIR / "reuters_ccat"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "Reuters-50.csv"

    # collect (author_id, source_file, text) sorted for determinism
    records: list[tuple[str, str, str]] = []
    for split_dir in (train, test):
        for author_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            author_id = author_dir.name
            for txt in sorted(author_dir.glob("*.txt")):
                raw = txt.read_text(encoding="latin-1")  # RCV1/Reuters dumps are latin-1
                text = _normalise_text(raw)
                if not text:
                    continue
                rel = f"{split_dir.name}/{author_id}/{txt.name}"
                records.append((author_id, rel, text))
    records.sort(key=lambda r: (r[0], r[1]))

    manifest_rows: list[dict] = []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["topic", "model", "text", "language", "author_id"])
        for author_id, rel, text in records:
            w.writerow(["reuters_ccat", "Reuters-50", text, "en", author_id])
            manifest_rows.append({
                "author_id": author_id, "source_file": rel,
                "sha256": _sha256_text(text), "n_words": _word_count(text),
            })
    n_authors = len({r[0] for r in records})
    print(f"  {len(records)} docs from {n_authors} authors → {out_csv}")
    return manifest_rows


def _write_webis_manifest(webis_rows: list[dict], ld_version: str) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with (MANIFEST_DIR / "webis_selected.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["topic", "submission_id", "author_id", "sha256", "n_words"])
        w.writeheader()
        for row in sorted(webis_rows, key=lambda r: (r["topic"], r["submission_id"])):
            w.writerow(row)
    # per-topic counts, distinct authors, and pairwise topic overlap (a post can
    # be in several topic CSVs — see the non-disjoint assignment rule).
    ids_by_topic: dict[str, set[str]] = defaultdict(set)
    auth_by_topic: dict[str, set[str]] = defaultdict(set)
    for r in webis_rows:
        ids_by_topic[r["topic"]].add(r["submission_id"])
        auth_by_topic[r["topic"]].add(r["author_id"])
    per_topic = {t: {"n_texts": len(ids_by_topic.get(t, set())),
                     "n_authors": len(auth_by_topic.get(t, set())),
                     "label": FILTER_CONFIG.get("labels", {}).get(t, t)}
                 for t in IMPROTA_TOPICS}
    overlap = {f"{a}∩{b}": len(ids_by_topic.get(a, set()) & ids_by_topic.get(b, set()))
               for i, a in enumerate(IMPROTA_TOPICS) for b in IMPROTA_TOPICS[i + 1:]}
    snapshot = {
        "built_at": _now(),
        "langdetect_version": ld_version,
        "author_salt": AUTHOR_SALT,
        "webis_source_candidates": list(WEBIS_SOURCE_CANDIDATES),
        "n_webis_texts_total_rows": len(webis_rows),
        "n_webis_unique_posts": len({r["submission_id"] for r in webis_rows}),
        "per_topic": per_topic,
        "pairwise_topic_overlap": overlap,
        "filter_config": FILTER_CONFIG,
    }
    (MANIFEST_DIR / "webis_filter_config.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    counts = "  ".join(f"{t}={per_topic[t]['n_texts']}" for t in IMPROTA_TOPICS)
    print(f"Manifest: webis_selected.csv ({len(webis_rows)} topic-rows)  ·  webis_filter_config.json   [{counts}]")


def _write_reuters_manifest(reuters_rows: list[dict]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with (MANIFEST_DIR / "reuters_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["author_id", "source_file", "sha256", "n_words"])
        w.writeheader()
        for row in sorted(reuters_rows, key=lambda r: (r["author_id"], r["source_file"])):
            w.writerow(row)
    print(f"Manifest: reuters_manifest.csv ({len(reuters_rows)} rows)")


def run_build(only: str | None = None) -> None:
    do_webis = only in (None, "webis")
    do_reuters = only in (None, "reuters")
    if do_webis and not FILTER_CONFIG.get("locked"):
        print("WARNING: FILTER_CONFIG['locked'] is False — these keyword lists / thresholds are the broad survey "
              "candidates, not a reviewed selection. The Webis corpus this produces should NOT be treated as final "
              "until the Stage-A review gate is closed. (Use --only reuters to build just the review-free corpus.)\n")
    if do_webis:
        webis_rows, ld_version = build_webis()
        _write_webis_manifest(webis_rows, ld_version)
    if do_reuters:
        reuters_rows = build_reuters()
        _write_reuters_manifest(reuters_rows)
    print("\nDone. Next: python3 clean_corpus.py  (picks up the human_baseline/ corpora)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Build the human-baseline corpora (Webis-CMV-20 + Reuters-50/C50).")
    p.add_argument("--survey", action="store_true",
                   help="Pass-1 reconnaissance over the Webis OPs (writes webis_cmv_20/survey/, no corpus output).")
    p.add_argument("--only", choices=["webis", "reuters"],
                   help="Build only one corpus (default: both). Reuters needs no review gate; "
                        "Webis should wait until FILTER_CONFIG is locked.")
    args = p.parse_args()
    if args.survey:
        if args.only == "reuters":
            p.error("--survey only applies to Webis; drop --only reuters")
        run_survey()
    else:
        run_build(only=args.only)


if __name__ == "__main__":
    main()
