# B3 Run Status

Updated: 2026-11-04

## Current State

- Branch: `codex/stage3-b3`
- Last committed model revision: `5dfea5e` (`Add temporal Mamba baseline`)
- Formal output: `outputs/stage2_b3_v1/`
- Last observed progress: 82/100 fold checkpoints written.
- Current process: `python -m src.b3 --device cuda` was still running when this
  note was written.
- The last console checkpoint seen before interruption: 80/100 folds at
  seed=3, site=YALE, fold runtime 1616.6 seconds.
- Completed folds are stored as one JSON checkpoint per fold under
  `outputs/stage2_b3_v1/checkpoints/`.

## Implemented

- Unidirectional temporal Mamba using `mambapy==1.2.0`.
- Training-only ROI standardisation and training-site-only validation.
- Shared Stage 2 metrics, provenance, prediction exports, and per-fold
  checkpoint resume.
- B3 forward, checkpoint, and LOSO smoke tests.
- Full test suite passed before the formal run: 19 passed.
- `src/verify_stage2.py` was extended locally to verify B3 when its output
  exists; this verifier edit is not yet committed.

## Resume Command

```powershell
& 'C:\Users\28439\miniconda3\python.exe' -m src.b3 --device cuda
```

The command targets `outputs/stage2_b3_v1/` by default and resumes only folds
whose cache, LOSO indices, config, software version, and Git commit match.

## Remaining Work

1. Finish folds through 100/100.
2. Run `python -m src.verify_stage2` and confirm B0-B3 all pass.
3. Update `README.md` and `实验TODOLIST.md` with B3 results.
4. Commit the B3 verifier and documentation updates.
