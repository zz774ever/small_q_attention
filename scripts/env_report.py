"""Print a compact, JSON-like environment report for reproducible runs."""

import json
import platform
import shutil
import subprocess
import sys


def command_version(command, args):
    path = shutil.which(command)
    if path is None:
        return {"path": None, "version": None}
    try:
        output = subprocess.check_output([path, *args], stderr=subprocess.STDOUT, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        output = str(exc)
    return {"path": path, "version": output.strip().splitlines()[-1] if output.strip() else None}


def main():
    report = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "tools": {
            "nvcc": command_version("nvcc", ["--version"]),
            "nvidia_smi": command_version("nvidia-smi", ["--query-gpu=name,compute_cap", "--format=csv,noheader"]),
            "gcc": command_version("gcc", ["--version"]),
            "cmake": command_version("cmake", ["--version"]),
        },
    }
    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        }
    except ImportError:
        report["torch"] = None
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
