# Windows CUDA 12.8 development environment

This fork intentionally targets Windows on NVIDIA CUDA 12.8 hardware. It is
tested against an RTX 5070; the supported Python version is 3.12.

All dependencies are declared in `pyproject.toml` and resolved by `uv`. Do not
install packages with `pip` into the project environment.

## Create or update the runtime environment

From PowerShell at the repository root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-cuda128"
uv sync --locked
```

Run the GUI with the same environment selection:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-cuda128"
uv run python gui.py
```

To verify the accelerator:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-cuda128"
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Packaging environment

The QPT packager is intentionally excluded from the normal runtime environment.
Create a separate build environment when producing a Windows archive:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-build-cuda128"
uv sync --locked --group build
uv export --locked --no-hashes --no-emit-project --no-emit-package torch --no-emit-package torchvision --output-file requirements-qpt.txt
uv run --group build python backend/tools/makedist.py --cuda 12.8
```

`requirements-qpt.txt` is generated from `uv.lock` for QPT and is intentionally
ignored by Git. Do not recreate a hand-maintained `requirements.txt`.

## GPU subtitle detection environment

Paddle's RTX 50-series GPU wheel uses CUDA 12.9, whereas ProPainter uses
PyTorch CUDA 12.8. They cannot safely share one Python process. This fork runs
PaddleOCR in a short-lived worker from a separate uv environment and returns
its subtitle boxes to the CUDA 12.8 runtime.

Create it once from the repository root:

```powershell
Push-Location ocr-gpu
$env:UV_PROJECT_ENVIRONMENT = "..\\.venv-ocr-gpu"
uv sync --locked
Pop-Location
```

When that environment exists and **Hardware acceleration** is enabled in the
GUI, precise subtitle detection uses the GPU worker automatically. If the
environment is absent, or hardware acceleration is disabled, detection falls
back to the CPU Paddle runtime.

Regenerate the lock file after a deliberate dependency change:

```powershell
uv lock
```
