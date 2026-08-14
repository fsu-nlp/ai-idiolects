"""Shared modules for the ai-idiolects generation and analysis pipeline.

This package holds the parts of the pipeline that other code imports:

``compare_fingerprints``
    The metric library — length distributions, TTR/MATTR, hapax ratio,
    Yule's K, Jaccard overlap, bigram/trigram JSD, Burrows' Delta.
``build_distance_matrix``
    Corpus discovery (``find_cleaned_csvs``, ``model_dir_for``), the topic and
    quality-flag constants, and cross-corpus distance matrices.
``visualize_stage1``
    The shared plotting conventions — model order, display labels, cohort
    colours — plus the Stage 1 descriptive figures.
``parse_corpus``
    spaCy tagging and dependency parsing, and the cleaned-to-parsed path map.
``keyness``, ``discourse_features``
    Log-likelihood keyness and discourse-marker counting; each is imported by
    its renderer under ``scripts/figures/``.
``paths``
    ``REPO_ROOT``, ``DATA_DIR``, ``RESULTS_DIR``.

Each of these is also runnable directly::

    python3 -m aiidiolects.build_distance_matrix --help

Scripts that nothing else imports live under ``scripts/`` instead.

AI-assistance disclosure: this package was developed with substantial AI
assistance from Anthropic Claude Opus 4.7 (``claude-opus-4-7``) for the
pipeline itself and Claude Opus 5 (``claude-opus-5``) for the packaging, via
Claude Code at "max" reasoning effort. Reviewed by the human authors. Full
statement in the project README.
"""

__version__ = "1.0.0"
