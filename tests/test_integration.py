from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np

from meowsic.pipeline import process_song
from meowsic.types import AudioBuffer, MeowsicConfig


def test_integration_default_meow_and_process(tmp_path: Path):
    """End-to-end integration test with mocked default meow and Demucs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    source_path = workspace / "test_source.wav"
    from meowsic.io import write_wav

    # Create dummy source
    test_audio = AudioBuffer(np.random.uniform(-0.1, 0.1, (22050, 2)).astype(np.float32), 22050)
    write_wav(source_path, test_audio)

    # We mock fetch_default_meow so we don't actually hit the network in unit tests
    dummy_meow = AudioBuffer(np.random.uniform(-0.1, 0.1, (10000, 1)).astype(np.float32), 22050)
    from meowsic.types import MeowSample, MeowSampleMetadata
    mock_meow = MeowSample(
        audio=dummy_meow,
        metadata=MeowSampleMetadata(
            local_path=workspace / "mock_meow.wav",
            source_type="mocked",
        ),
    )

    with mock.patch("meowsic.samples.fetch_default_meow", return_value=mock_meow), \
         mock.patch("meowsic.stems.run_demucs") as mock_demucs, \
         mock.patch("meowsic.pipeline.estimate_pitch_contour") as mock_contour, \
         mock.patch("meowsic.pipeline.detect_syllable_events") as mock_events:
         
        from meowsic.types import PitchContour, SyllableEvent
        mock_contour.return_value = PitchContour(
            times=np.array([0.0, 0.1]),
            f0_hz=np.array([440.0, 440.0]),
            voiced=np.array([True, True]),
            energy=np.array([1.0, 1.0]),
        )
        mock_events.return_value = [SyllableEvent(start=0.0, end=0.1, energy=1.0, pitch_hz=440.0)]

        # Mock demucs return stems
        from meowsic.stems import StemSet
        mock_demucs.return_value = StemSet(
            vocal=test_audio,
            instrumental=test_audio,
            strategy="mock_demucs",
            metadata={},
        )

        config = MeowsicConfig(
            demucs_cache_dir=workspace / "demucs",
            meow_sample_cache_dir=workspace / "meows",
        )

        output_path = workspace / "output.wav"
        result = process_song(
            source_path=source_path,
            output_path=output_path,
            config=config,
        )

        assert output_path.exists()
        assert result.output_path == output_path
        assert result.vocal_stem is not None
        assert result.event_count >= 0
