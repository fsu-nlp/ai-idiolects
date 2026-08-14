"""
Download TextualLLMap original data from OSF (Improta, Veltri & Stella, 2024).

Downloads zip archives for each topic and extracts CSVs into
data/phase0_textualllmap/<topic>/.
Skips downloads if files already exist.

Usage:
    python download_data.py                  # download all topics
    python download_data.py --topic climate  # download one topic

Requires: no extra dependencies (uses urllib + zipfile from stdlib)

AI-assistance disclosure: developed with substantial AI assistance from
Anthropic Claude Opus 4.7 (`claude-opus-4-7`) via Claude Code at "max"
reasoning effort (extended thinking). Reviewed by the human authors before
commit. See the project README for the full disclosure.
"""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

# OSF download URLs for each topic zip
# Source: https://osf.io/uwv59
TOPICS = {
    "climate": "https://osf.io/download/rz8fx/",
    "global_warming": "https://osf.io/download/drzm8/",
    "math_anxiety": "https://osf.io/download/3cm6v/",
    "misinfo_health": "https://osf.io/download/wbj73/",
}

from aiidiolects.paths import DATA_DIR as REPO_DATA_DIR

DATA_DIR = REPO_DATA_DIR / "phase0_textualllmap"


def download_and_extract(topic: str, url: str):
    out_dir = DATA_DIR / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (look for at least one CSV)
    existing = list(out_dir.glob("*.csv"))
    if existing:
        print(f"  {topic}: already have {len(existing)} CSVs in {out_dir}, skipping")
        return

    print(f"  {topic}: downloading from OSF...", end=" ", flush=True)
    resp = urllib.request.urlopen(url)
    data = resp.read()
    print(f"({len(data) / 1e6:.1f} MB)")

    print(f"  {topic}: extracting...", end=" ", flush=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # Extract only CSVs, flatten any subdirectory structure
        count = 0
        for info in zf.infolist():
            if info.filename.endswith(".csv") and not info.is_dir():
                # Use just the filename, ignore any directory prefix
                fname = Path(info.filename).name
                target = out_dir / fname
                target.write_bytes(zf.read(info.filename))
                count += 1
        print(f"{count} CSVs extracted to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download TextualLLMap data from OSF")
    parser.add_argument(
        "--topic", choices=list(TOPICS.keys()),
        help="Download a single topic (default: all)",
    )
    args = parser.parse_args()

    print("TextualLLMap data downloader")
    print(f"Target directory: {DATA_DIR.resolve()}\n")

    if args.topic:
        topics = {args.topic: TOPICS[args.topic]}
    else:
        topics = TOPICS

    for topic, url in topics.items():
        download_and_extract(topic, url)

    print("\nDone.")


if __name__ == "__main__":
    main()
