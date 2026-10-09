# Windows NVIDIA GPU setup

## Verified environment

The following local configuration has been verified:

| Component | Verified value |
|---|---|
| Operating system | Windows |
| Python | 3.12 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8 GB |
| NVIDIA driver capability reported by `nvidia-smi` | CUDA 13.1 |
| PyTorch | `2.8.0+cu128` |
| PyTorch CUDA runtime | 12.8 |
| Transformers | `4.57.6` |
| scikit-learn | `1.9.1` |

The NVIDIA driver's reported CUDA capability may be newer than PyTorch's bundled
runtime. This is expected: the driver can run the CUDA 12.8 runtime bundled with the
PyTorch wheel. The separately installed CUDA Toolkit is not used by ordinary model
inference and does not need to match the wheel.

## Why PyTorch is pinned

The tested machine produced `WinError 1114` while loading `c10.dll` with newer
PyTorch Windows wheels. PyTorch `2.8.0+cu128` imports successfully and detects the
GPU. The project therefore constrains PyTorch to `>=2.8,<2.9` until a later Windows
build is explicitly retested. Do not upgrade PyTorch independently without rerunning
the verification below.

## Create a short-path environment

Windows long-path support was disabled on the tested machine, and installing PyTorch
inside the project's deeply nested `.venv` exceeded the legacy filename limit. Keep
the GPU environment at a short path outside the repository. From the project root:

```powershell
$FpiGpuEnv = Join-Path $env:USERPROFILE "fpi-gpu"
& "$env:USERPROFILE\anaconda3\python.exe" -m venv $FpiGpuEnv
& "$FpiGpuEnv\Scripts\python.exe" -m pip install --upgrade pip
```

If Python is not installed under `anaconda3`, replace that executable with the full
path to the available Python 3.11 or 3.12 interpreter.

## Install GPU and project dependencies

Install the CUDA wheel first because the standard Python package index may otherwise
select a CPU-only PyTorch build:

```powershell
& "$FpiGpuEnv\Scripts\python.exe" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& "$FpiGpuEnv\Scripts\python.exe" -m pip install -e ".[dev,nlp]"
& "$FpiGpuEnv\Scripts\python.exe" -m pip check
```

`pyproject.toml` remains the authoritative dependency specification. A separate
`requirements.txt` is intentionally not maintained because duplicate dependency lists
can drift.

## Verify CUDA

```powershell
nvidia-smi
& "$FpiGpuEnv\Scripts\python.exe" -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The Python command should report `2.8.0+cu128`, CUDA `12.8`, `True`, and the NVIDIA
GPU name. If `torch.cuda.is_available()` is `False`, do not force GPU execution.

## Run the project on GPU

Device selection is machine-specific and is not committed to `config/config.yaml`.
Set it in each new PowerShell session:

```powershell
$env:FPI_DEVICE = "cuda"
& "$FpiGpuEnv\Scripts\python.exe" -m fed_policy_intelligence.nlp.pipeline --config config/config.yaml --limit 5
```

The first real inference run downloads the FinBERT and emotion-model weights into the
local Hugging Face cache. Start with five documents, inspect GPU memory with
`nvidia-smi`, and review `data/processed/nlp_predictions.csv` before increasing the
limit.

For editor integration, select
`%USERPROFILE%\fpi-gpu\Scripts\python.exe` as the project interpreter.
