# Project Changelog

## 2026-11-09

- Resumed B3 from 83/100 fold checkpoints without retraining completed folds.
- Finished the formal B3 5-seed LOSO run.
- B3 site-macro AUC: 0.6148 +/- 0.0282; balanced accuracy: 0.5695.
- Verified B0-B3 together with `src.verify_stage2`.
- Full test suite passed: 19 tests.

## 2026-11-04

- Created branch `codex/stage3-b3`.
- Installed and pinned `mambapy==1.2.0` as the pure-PyTorch Mamba backend.
- Added the B3 unidirectional temporal Mamba runner with training-only
  standardisation, LOSO checkpoints, provenance, and shared Stage 2 metrics.
- Added B3 forward, checkpoint, and LOSO smoke tests.

## 2026-11-02

- Created branch `codex/project-cleanup`.
- Removed the unused local `.venv`, IDE metadata, pytest cache, and source
  bytecode caches.
- Removed 19 obsolete smoke, seed0, and pre-v2 experiment directories.
- Kept the three formal B0-B2 result directories and all raw data/cache files.
- Rewrote `README.md` as the project entry point and synchronized the
  experiment TODO with the completed B1/B2 results.
- Reran B0 under the current output contract.
- Verified all three formal B0-B2 outputs with `src.verify_stage2` and ran the
  complete test suite (16 passed).

## 2026-10-27

- Created branch `codex/stage2-pre-b3`.
- Archived the pre-optimization B0-B2 implementation at commit `4d6b843`.
- Added the Stage 2 evaluation contract in `METRICS.md`.
- Added PR-AUC, F1, and confusion-matrix counts to the shared metric layer.
- Routed B0, B1, and B2 through the shared Stage 2 summary.
- Changed B1 to strict inductive evaluation: held-out sites are absent from
  training and early-stopping graphs, and each test subject is scored without
  edges to other held-out subjects.
- Added per-fold B1 checkpoints and data/fold/config fingerprints for B1 and B2.
- Added environment, Git commit, dependency version, and data-hash provenance
  to experiment summaries.
- Added regression tests for strict inductive inference, checkpoint reuse,
  metric fields, and provenance hashing.
