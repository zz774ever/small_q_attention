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

The project now runs primarily on the H20 server. Recorded facts:

```text
GPU:        NVIDIA H20 (SM90), 97871 MiB
driver:     595.71.05
toolkit:    CUDA 13.0 (nvcc V13.0.88)
python:     3.12.11 in the conda env `py312` (/usr/local/miniconda3/envs/py312)
torch:      2.14.0+cu130
flashinfer: 0.7.0, in-tree checkout third_party/flashinfer @ a69ad808f8ff4095df4460cbda86ebcf815aa31d
```

Two setup steps are required before any JIT-backed run, because the conda env
only activates in login shells and the upstream submodules are not vendored:

```bash
source /usr/local/miniconda3/bin/activate py312
git -C third_party/flashinfer submodule update --init --recursive \
  3rdparty/cutlass 3rdparty/cccl 3rdparty/spdlog
```

Nsight Compute cannot read performance counters on this host
(`ERR_NVGPUCTRPERM`, a driver-level restriction), so kernel-level evidence comes
from CUDA events plus `scripts/profile_v1_v2.py --torch-profiler`.

An A100 would be useful for SM80 cross-architecture measurements, but direct
validation of the XQA path from issue #3420 and PR #3859 needs SM90 or newer,
which the H20 already provides.
