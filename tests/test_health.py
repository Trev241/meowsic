from __future__ import annotations

import sys
from unittest import mock

from meowsic.health import check_health


def test_check_health_all_installed():
    status = check_health()
    assert status["Python"] == sys.version.split()[0]
    assert "missing" not in status["torch"]
    assert "missing" not in status["demucs"]
    assert "torchaudio_backends" in status


@mock.patch("meowsic.health.importlib.import_module")
def test_check_health_missing_deps(mock_import):
    mock_import.side_effect = ImportError("Mock missing")
    status = check_health()
    assert status["torch"] == "missing"
    assert status["gradio"] == "missing"
    assert "torchaudio_backends" not in status
