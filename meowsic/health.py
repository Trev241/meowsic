from __future__ import annotations

import importlib
import sys


def check_health() -> dict[str, str]:
    """Check the health and presence of critical dependencies."""
    status: dict[str, str] = {}

    status["Python"] = sys.version.split()[0]

    packages = ["torch", "torchaudio", "demucs", "soundfile", "yt_dlp", "gradio"]
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "installed")
            status[pkg] = version
        except ImportError:
            status[pkg] = "missing"

    if status.get("torchaudio") != "missing":
        import torchaudio

        status["torchaudio_backends"] = str(torchaudio.list_audio_backends())

    return status
