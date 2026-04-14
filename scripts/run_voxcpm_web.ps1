$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv-voxcpm\\Scripts\\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Missing .venv-voxcpm. Run scripts\\setup_voxcpm_env.ps1 first."
}

Push-Location $repoRoot
try {
    & $pythonExe scripts\voxcpm_local_web.py --host 127.0.0.1 --port 8808
}
finally {
    Pop-Location
}
