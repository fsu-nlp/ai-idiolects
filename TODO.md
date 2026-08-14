# Planned additions

Open items for this repository, in rough priority order.

## 1. Corpus deposit

The generated 2026-cohort corpora and the LM Studio replication controls are to be deposited on OSF and linked from the "Data availability" section of the [README](README.md#data-availability). The chapter states that the datasets will be made publicly available on publication.

- [ ] Create the OSF deposit and mint a DOI
- [ ] Add the DOI and download instructions to the README
- [ ] Note the deposit in `scripts/generate/download_data.py` so the fetch path is uniform with the 2024 cohort

## 2. R-side analyses

Chapter §5 (stylometric PCA via the `stylo` package, v0.7.5) and §6 (contraction frequencies across six contracted-form categories, normalised per million words) were carried out in R, alongside the R corpus-cleaning script described in §5.1. That code is to be added here.

- [ ] Add the R cleaning script (CSV artefact stripping, instruction-token removal)
- [ ] Add the `stylo` PCA driver, including MFW/culling settings and the sampling scheme
- [ ] Add the contraction counting and normalisation code
- [ ] Record R and package versions alongside them

## 3. Model identity and attribution

The 2024-cohort inventory has been reconciled against the deposit; the results are in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) and the mapping table in [`docs/MODELS.md`](docs/MODELS.md). Two items remain open, and both need the depositors rather than us.

- [ ] Confirm with the TextualLLMap authors whether the 70B stream is Llama 3 70B (as its `model` column says) or Llama 3.1 70B (as its filename says)
- [ ] Ask whether generation parameters exist for the two undocumented English streams (GPT-4o, the 70B)
- [ ] Once settled, correct the labels here and in the manuscript revision, and cut a new release

Note that this release deliberately keeps the labels as published — see the preamble to `KNOWN_ISSUES.md`.

## 4. Environment

- [ ] Pin dependency versions (`pip freeze > requirements.lock.txt`) so the reported numbers are reproducible against a fixed stack
- [ ] Record the spaCy model version (`en_core_web_lg`) used for the published parse

> **AI-assistance** -- this document was written with substantial AI assistance (Claude Opus 5) and the final version was approved by the authors.
