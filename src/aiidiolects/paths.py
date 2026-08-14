"""Repository-root-relative paths, resolved from the installed package.

Every script writes under ``data/`` and ``results/`` at the repository root.
Importing ``REPO_ROOT`` from here means a script works the same whether it is
launched from the repository root, from its own directory, or through
``python3 -m``.

The layout this assumes::

    <repo root>/
        src/aiidiolects/paths.py   <- this file
        data/                      <- corpora, fetched or generated (gitignored)
        results/                   <- analysis outputs (gitignored)

AI-assistance disclosure: written with substantial AI assistance from Anthropic
Claude Opus 5 (``claude-opus-5``) via Claude Code at "max" reasoning effort.
Reviewed by the human authors. Full statement in the project README.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root — three parents up from ``src/aiidiolects/paths.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Corpora: the 2024 cohort as fetched, the 2026 cohort as generated.
DATA_DIR = REPO_ROOT / "data"

#: Analysis outputs: matrices, tables, figures.
RESULTS_DIR = REPO_ROOT / "results"

__all__ = ["REPO_ROOT", "DATA_DIR", "RESULTS_DIR"]
