$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv-voxcpm"
$pythonExe = Join-Path $venvDir "Scripts\\python.exe"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required but was not found on PATH."
}

if (-not (Test-Path $venvDir)) {
    uv venv $venvDir --python 3.12
}

$torchIndex = "https://download.pytorch.org/whl/cu128"

# Install CUDA-enabled PyTorch from the official PyTorch wheel index first.
uv pip install --python $pythonExe --index-url $torchIndex --upgrade --reinstall torch torchaudio

# Then install VoxCPM and the local web UI deps from the normal index.
uv pip install --python $pythonExe voxcpm gradio soundfile

Write-Host ""
Write-Host "VoxCPM env ready."
Write-Host "Python: $pythonExe"
Write-Host "Run with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\\run_voxcpm_web.ps1"
