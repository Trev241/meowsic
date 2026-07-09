from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from meowsic import (
    AudioBuffer,
    MeowsicConfig,
    fetch_public_meow_sample,
    fetch_youtube_audio,
    load_wav,
    process_song,
    run_demucs,
    write_wav,
)


def sine_audio(frequency: float, sample_rate: int = 22050, duration: float = 1.0, gain: float = 0.2) -> AudioBuffer:
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    samples = gain * np.sin(2 * np.pi * frequency * t)
    return AudioBuffer(samples, sample_rate)


class CoreTests(unittest.TestCase):
    def test_wav_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            audio = sine_audio(220.0)
            write_wav(path, audio, overwrite=True)
            loaded = load_wav(path)
            self.assertEqual(loaded.sample_rate, audio.sample_rate)
            self.assertEqual(loaded.channels, 1)
            self.assertGreater(float(np.max(np.abs(loaded.samples))), 0.05)

    def test_pipeline_with_provided_stems_and_meow_sample(self) -> None:
        sample_rate = 22050
        duration = 1.2
        t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
        vocal = 0.25 * np.sin(2 * np.pi * 220.0 * t)
        vocal *= (np.sin(2 * np.pi * 3.0 * t) > -0.35).astype(np.float32)
        instrumental = np.column_stack(
            [
                0.08 * np.sin(2 * np.pi * 110.0 * t),
                -0.08 * np.sin(2 * np.pi * 110.0 * t),
            ]
        ).astype(np.float32)
        source = instrumental + vocal[:, None]
        meow = sine_audio(330.0, sample_rate=sample_rate, duration=0.22, gain=0.5)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.wav"
            vocal_path = tmp_path / "vocal.wav"
            instrumental_path = tmp_path / "instrumental.wav"
            meow_path = tmp_path / "meow.wav"
            output_path = tmp_path / "output.wav"
            write_wav(source_path, AudioBuffer(source, sample_rate), overwrite=True)
            write_wav(vocal_path, AudioBuffer(vocal, sample_rate), overwrite=True)
            write_wav(instrumental_path, AudioBuffer(instrumental, sample_rate), overwrite=True)
            write_wav(meow_path, meow, overwrite=True)

            result = process_song(
                source_path=source_path,
                vocal_path=vocal_path,
                instrumental_path=instrumental_path,
                meow_sample_path=meow_path,
                output_path=output_path,
                config=MeowsicConfig(overwrite=True),
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(result.event_count, 0)
            self.assertEqual(result.stem_strategy, "provided_vocal_and_instrumental")
            rendered = load_wav(output_path)
            self.assertEqual(rendered.channels, 2)
            self.assertGreater(float(np.max(np.abs(rendered.samples))), 0.01)

    def test_demucs_adapter_with_mocked_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.wav"
            write_wav(source_path, sine_audio(220.0), overwrite=True)

            demucs_module = types.ModuleType("demucs")
            separate_module = types.ModuleType("demucs.separate")

            def fake_main(args):
                output_dir = Path(args[args.index("-o") + 1])
                model = args[args.index("-n") + 1]
                stem_dir = output_dir / model / source_path.stem
                stem_dir.mkdir(parents=True, exist_ok=True)
                write_wav(stem_dir / "vocals.wav", sine_audio(220.0), overwrite=True)
                write_wav(stem_dir / "no_vocals.wav", sine_audio(110.0), overwrite=True)

            separate_module.main = fake_main
            old_demucs = sys.modules.get("demucs")
            old_separate = sys.modules.get("demucs.separate")
            sys.modules["demucs"] = demucs_module
            sys.modules["demucs.separate"] = separate_module
            try:
                stems = run_demucs(source_path, config=MeowsicConfig(demucs_cache_dir=tmp_path / "demucs"))
            finally:
                if old_demucs is None:
                    sys.modules.pop("demucs", None)
                else:
                    sys.modules["demucs"] = old_demucs
                if old_separate is None:
                    sys.modules.pop("demucs.separate", None)
                else:
                    sys.modules["demucs.separate"] = old_separate

            self.assertEqual(stems.strategy, "demucs_two_stems_vocals")
            self.assertGreater(stems.vocal.duration, 0)

    def test_youtube_adapter_with_mocked_ytdlp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            class FakeYDL:
                def __init__(self, options):
                    self.options = options

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def extract_info(self, url, download):
                    target = tmp_path / "abc123.wav"
                    write_wav(target, sine_audio(220.0), overwrite=True)
                    return {"id": "abc123", "title": "Demo", "extractor": "youtube"}

                def prepare_filename(self, info):
                    return str(tmp_path / "abc123.webm")

            fake_module = types.ModuleType("yt_dlp")
            fake_module.YoutubeDL = FakeYDL
            old = sys.modules.get("yt_dlp")
            sys.modules["yt_dlp"] = fake_module
            try:
                metadata = fetch_youtube_audio(
                    "https://www.youtube.com/watch?v=abc123",
                    config=MeowsicConfig(enable_youtube_fetch=True, youtube_cache_dir=tmp_path),
                )
            finally:
                if old is None:
                    sys.modules.pop("yt_dlp", None)
                else:
                    sys.modules["yt_dlp"] = old

            self.assertEqual(metadata.source_type, "youtube")
            self.assertTrue(metadata.local_path.exists())
            self.assertEqual(metadata.title, "Demo")

    def test_public_meow_fetch_with_file_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "public_meow.wav"
            write_wav(source, sine_audio(440.0), overwrite=True)
            sample = fetch_public_meow_sample(
                config=MeowsicConfig(
                    enable_public_sample_fetch=True,
                    meow_sample_url=source.resolve().as_uri(),
                    meow_sample_license="CC0",
                    meow_sample_attribution="Test fixture",
                    meow_sample_cache_dir=tmp_path / "cache",
                )
            )
            self.assertEqual(sample.metadata.source_type, "fetched_public_resource")
            self.assertEqual(sample.metadata.license, "CC0")
            self.assertGreater(sample.audio.duration, 0)


if __name__ == "__main__":
    unittest.main()
