$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Missing .venv. Run scripts\setup.ps1 first."
}

$Code = @'
import importlib

packages = [
    ("meowsic", "meowsic"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("demucs", "demucs"),
    ("soundfile", "soundfile"),
    ("yt-dlp", "yt_dlp"),
    ("gradio", "gradio"),
]

for label, module_name in packages:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if label == "yt-dlp":
        version = module.version.__version__
    print(f"{label}: {version or 'installed'}")

import torchaudio
print(f"torchaudio backends: {torchaudio.list_audio_backends()}")
'@

$Code | & $PythonExe -
