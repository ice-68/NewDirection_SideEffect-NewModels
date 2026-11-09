# B3 Run Status

Updated: 2026-11-09

## Current State

- Branch: `codex/stage3-b3`
- Last committed model revision: `5dfea5e` (`Add temporal Mamba baseline`)
- Formal output: `outputs/stage2_b3_v1/`
- Status: complete.
- Completed folds: 100/100 with one JSON checkpoint per fold.
- Site-macro AUC: 0.6148 +/- 0.0282.
- Site-macro balanced accuracy: 0.5695.
- PR-AUC: 0.6189 +/- 0.0247.
- Result verification: B0-B3 passed.

## Implemented

- Unidirectional temporal Mamba using `mambapy==1.2.0`.
- Training-only ROI standardisation and training-site-only validation.
- Shared Stage 2 metrics, provenance, prediction exports, and per-fold
  checkpoint resume.
- B3 forward, checkpoint, and LOSO smoke tests.
- Full test suite passed before the formal run: 19 passed.
- `src/verify_stage2.py` verifies B3 when its output exists.

## Resume Command

```powershell
& 'C:\Users\28439\miniconda3\python.exe' -m src.b3 --device cuda
```

The command targets `outputs/stage2_b3_v1/` by default and resumes only folds
whose cache, LOSO indices, config, software version, and Git commit match.

## Next Step

Freeze the B4 graph/fusion design before implementation.
