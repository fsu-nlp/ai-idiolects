"""Download the human-text baseline corpora.

Two corpora, fetched from their original public hosts (we do not redistribute
them — this script + ``build_human_baseline.py`` + the committed manifest are
what reproduce the working subsets):

* **Webis-CMV-20** — r/ChangeMyView posts & comments, Jan 2013 – Sep 2017,
  CC-BY 4.0, Zenodo record 3778298 (DOI 10.5281/zenodo.3778298).
  Default download is ``posts_malleability.jsonl.bz2`` (~427 MB) — one record
  per CMV submission in the original Reddit-crawl format, which is the right
  granularity for the *original post* essays we want. ``--with-threads`` also
  pulls ``threads.jsonl.bz2`` (~661 MB, full submission+comment trees) as a
  fallback source of OPs if the survey shows topic matches are too sparse.
  ``--with-author-meta`` pulls the small per-author LIWC / subreddit-stats
  files (~28 MB total) in case a later L1/persona heuristic wants them.
  (The 1.65 GB ``author_entity_category.jsonl.bz2`` is never fetched — it is
  Wikipedia-entity mentions, irrelevant here.)
* **Reuters-50 / C50** — 50 authors × 100 docs, Reuters CCAT, Aug 1996 – Aug
  1997, CC-BY 4.0, UCI ML Repository dataset 217 (~7.8 MB zip). Extracted to
  ``raw/C50train/<author>/*.txt`` and ``raw/C50test/<author>/*.txt``.

Idempotent: a file already present in ``raw/`` is skipped. Streams large
downloads to disk in chunks; retries transient failures. ``.bz2`` files are
left compressed — ``build_human_baseline.py`` stream-reads them.

Run from the repository root::

    python3 download_human_baseline.py                 # Webis OPs + Reuters
    python3 download_human_baseline.py --with-threads   # + full thread trees
    python3 download_human_baseline.py --only reuters   # Reuters only
    python3 download_human_baseline.py --only webis     # Webis only

Requires: no extra dependencies (urllib + zipfile + bz2 + hashlib from stdlib).

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (``claude-opus-4-7``) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

ZENODO_RECORD = "3778298"  # Webis-CMV-20, DOI 10.5281/zenodo.3778298
_ZENODO_FILE = (
    "https://zenodo.org/api/records/" + ZENODO_RECORD + "/files/{name}/content"
)

# (filename on Zenodo, approx size for the log) — tier decides what gets fetched.
WEBIS_FILES = {
    "core": [("posts_malleability.jsonl.bz2", 427_000_000)],
    "threads": [("threads.jsonl.bz2", 661_000_000)],
    "author_meta": [
        ("author_liwc.jsonl.bz2", 14_700_000),
        ("author_subreddit.jsonl.bz2", 7_200_000),
        ("author_subreddit_category.jsonl.bz2", 6_000_000),
    ],
}

REUTERS_ZIP_URL = "https://archive.ics.uci.edu/static/public/217/reuter+50+50.zip"

from aiidiolects.paths import DATA_DIR as REPO_DATA_DIR

DATA_DIR = REPO_DATA_DIR / "human_baseline"
WEBIS_RAW = DATA_DIR / "webis_cmv_20" / "raw"
REUTERS_RAW = DATA_DIR / "reuters_50" / "raw"

_CHUNK = 1 << 20  # 1 MiB
_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def _append_manifest(raw_dir: Path, fname: str, size: int, sha256: str, url: str) -> None:
    line = f"{_now()}\t{fname}\t{size}\t{sha256}\t{url}\n"
    (raw_dir / "DOWNLOAD_MANIFEST.tsv").open("a", encoding="utf-8").write(line)


def _stream_download(url: str, dest: Path, approx: int | None = None) -> tuple[int, str]:
    """Download ``url`` to ``dest`` (via a .part file), returning (size, sha256)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, _RETRIES + 1):
        try:
            h = hashlib.sha256()
            written = 0
            req = urllib.request.Request(url, headers={"User-Agent": "ai-idiolects/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
                total = approx
                clen = resp.headers.get("Content-Length")
                if clen and clen.isdigit():
                    total = int(clen)
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100.0 * written / total
                        print(f"\r    {dest.name}: {_human(written)} / ~{_human(total)} ({pct:4.1f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r    {dest.name}: {_human(written)}", end="", flush=True)
            print()
            tmp.replace(dest)
            return written, h.hexdigest()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:  # transient
            last_err = e
            print(f"\n    attempt {attempt}/{_RETRIES} failed: {e}; retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)
        finally:
            if tmp.exists() and not dest.exists():
                # leave the .part for a possible manual resume; do not auto-delete
                pass
    raise RuntimeError(f"download failed after {_RETRIES} attempts: {url}") from last_err


# ---------------------------------------------------------------------------
# Webis-CMV-20
# ---------------------------------------------------------------------------

def download_webis(tiers: list[str]) -> None:
    WEBIS_RAW.mkdir(parents=True, exist_ok=True)
    wanted: list[tuple[str, int]] = []
    for tier in tiers:
        wanted.extend(WEBIS_FILES[tier])
    print(f"Webis-CMV-20 (Zenodo {ZENODO_RECORD}) → {WEBIS_RAW.resolve()}")
    for fname, approx in wanted:
        dest = WEBIS_RAW / fname
        if dest.exists():
            print(f"  {fname}: already present ({_human(dest.stat().st_size)}), skipping")
            continue
        url = _ZENODO_FILE.format(name=fname)
        print(f"  {fname}: downloading (~{_human(approx)})...")
        size, sha = _stream_download(url, dest, approx)
        print(f"  {fname}: done — {_human(size)}, sha256={sha[:16]}…")
        _append_manifest(WEBIS_RAW, fname, size, sha, url)


# ---------------------------------------------------------------------------
# Reuters-50 / C50
# ---------------------------------------------------------------------------

def download_reuters() -> None:
    REUTERS_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Reuters-50 / C50 (UCI dataset 217) → {REUTERS_RAW.resolve()}")
    # "already present" check: both split dirs populated.
    train_dir, test_dir = REUTERS_RAW / "C50train", REUTERS_RAW / "C50test"
    if train_dir.is_dir() and test_dir.is_dir() and any(train_dir.iterdir()) and any(test_dir.iterdir()):
        n = sum(1 for _ in REUTERS_RAW.rglob("*.txt"))
        print(f"  already extracted ({n} .txt files under {REUTERS_RAW}), skipping")
        return
    zip_dest = REUTERS_RAW / "reuter+50+50.zip"
    if not zip_dest.exists():
        print(f"  downloading {REUTERS_ZIP_URL} (~7.8 MB)...")
        size, sha = _stream_download(REUTERS_ZIP_URL, zip_dest, 7_800_000)
        print(f"  done — {_human(size)}, sha256={sha[:16]}…")
        _append_manifest(REUTERS_RAW, "reuter+50+50.zip", size, sha, REUTERS_ZIP_URL)
    print("  extracting C50train/ and C50test/...")
    with zipfile.ZipFile(zip_dest) as zf:
        count = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            # Normalise paths; keep only the C50train/<author>/<file>.txt structure.
            parts = Path(info.filename).parts
            if "C50train" in parts:
                idx = parts.index("C50train")
            elif "C50test" in parts:
                idx = parts.index("C50test")
            else:
                continue
            rel = Path(*parts[idx:])
            if rel.suffix.lower() != ".txt":
                continue
            target = REUTERS_RAW / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info.filename))
            count += 1
    print(f"  extracted {count} .txt files")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Download the human-baseline corpora (Webis-CMV-20 + Reuters-50/C50).")
    p.add_argument("--only", choices=["webis", "reuters"], help="Download only one corpus (default: both).")
    p.add_argument("--with-threads", action="store_true",
                   help="Also fetch threads.jsonl.bz2 (~661 MB) as a fallback OP source.")
    p.add_argument("--with-author-meta", action="store_true",
                   help="Also fetch the small per-author LIWC / subreddit-stats files (~28 MB).")
    args = p.parse_args()

    print("Human-baseline corpus downloader\n")
    do_webis = args.only in (None, "webis")
    do_reuters = args.only in (None, "reuters")

    if do_webis:
        tiers = ["core"]
        if args.with_threads:
            tiers.append("threads")
        if args.with_author_meta:
            tiers.append("author_meta")
        download_webis(tiers)
        print()
    if do_reuters:
        download_reuters()
        print()
    print("Done. Next: python3 build_human_baseline.py --survey")


if __name__ == "__main__":
    sys.exit(main())
