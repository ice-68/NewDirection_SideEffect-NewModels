# Stage 2 Runtime Environment

Recorded: 2026-10-27

## Canonical Interpreter

The verified Stage 2 interpreter is:

```text
C:\Users\28439\miniconda3\python.exe
```

Verified versions:

- Python 3.13.12
- PyTorch 2.12.0.dev20260315+cu128
- CUDA 12.8
- scikit-learn 1.8.0
- NumPy 2.3.5
- pandas 3.0.1

The repository-local `.venv` does not contain PyTorch and must not be used for
Stage 2 runs. Experiment summaries also record the interpreter and package
versions through `src/provenance.py`.

## Run Commands

```powershell
& 'C:\Users\28439\miniconda3\python.exe' -m src.b0
& 'C:\Users\28439\miniconda3\python.exe' -m src.b1
& 'C:\Users\28439\miniconda3\python.exe' -m src.b2
```

Long B1 and B2 runs write one atomic checkpoint per LOSO fold. A rerun with the
same output directory resumes only when the data hashes, fold signature,
training configuration, software versions, and Git commit match.
