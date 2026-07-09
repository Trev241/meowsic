param(
    [string]$Python = "python",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $RepoRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

Set-Location $RepoRoot

if ($Recreate -and (Test-Path $VenvPath)) {
    Remove-Item -Recurse -Force -LiteralPath $VenvPath
}

if (-not (Test-Path $PythonExe)) {
    & $Python -m venv $VenvPath
}

& $PythonExe -m pip install --upgrade pip setuptools wheel

# Install the project and all optional dependency groups used by development and the dashboard.
& $PythonExe -m pip install -e ".[demucs,youtube,dashboard,dev]"

# yt-dlp changes frequently as providers change. Always force the newest available build.
& $PythonExe -m pip install --upgrade --upgrade-strategy eager yt-dlp

# Keep the local Windows Demucs stack on the tested Torch/TorchAudio line and ensure WAV saving works.
& $PythonExe -m pip install "torch==2.6.0" "torchaudio==2.6.0" "soundfile>=0.12"

Write-Host ""
Write-Host "Meowsic environment ready."
Write-Host "Python: $PythonExe"
Write-Host ""
Write-Host "Activate it with:"
Write-Host ".\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Run the dashboard with:"
Write-Host ".\.venv\Scripts\python.exe run.py"
