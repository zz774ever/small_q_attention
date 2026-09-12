# Local Development Setup

## Current machine assessment

The current Windows host exposes an RTX 3050 and `nvidia-smi` works. The installed toolkit is CUDA 11.3, but the available Python 3.8 environment has CPU-only `torch 1.12.0`. No Visual Studio C++ host compiler was detected in PATH. This is enough for the reference harness, but not a reliable FlashInfer build environment.

## Intended local path

Use a WSL2 Ubuntu environment with the NVIDIA WSL driver integration. Keep the Windows Python and toolkit out of the project environment.

Recommended starting point for the Linux environment:

```text
Python 3.10 or 3.11
PyTorch with CUDA wheels matching the selected toolkit/runtime
CUDA toolkit compatible with the pinned FlashInfer revision
gcc/g++ and cmake/ninja
pytest, ruff, and black for project checks
```

Before installing anything, capture the environment with:

```bash
python scripts/env_report.py
nvidia-smi
nvcc --version
```

Then clone FlashInfer as a separate checkout and record its commit. Do not vendor the full upstream repository into this project. The integration work will be represented by a patch or a small branch against that pinned checkout.

## Server path

Run the same environment report after pushing the project. An A100 is useful for SM80 cross-architecture measurements, but direct validation of the XQA path from issue #3420 and PR #3859 requires SM90 or newer hardware such as H100/H20.
