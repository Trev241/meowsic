from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from meowsic import (
    AudioBuffer,
    MeowSample,
    MeowSampleMetadata,
    MeowsicConfig,
    PitchContour,
    SyllableEvent,
    fetch_public_meow_sample,
    fetch_youtube_audio,
    load_wav,
    process_song,
    render_meow_vocal,
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

    def test_render_maps_melody_into_cat_register(self) -> None:
        sample_rate = 22050
        t = np.arange(int(sample_rate * 1.0), dtype=np.float32) / sample_rate
        times = np.linspace(0.0, 1.0, 24, dtype=np.float32)
        contour = PitchContour(
            times=times,
            f0_hz=np.linspace(150.0, 900.0, times.size, dtype=np.float32),
            voiced=np.ones(times.size, dtype=bool),
            energy=np.ones(times.size, dtype=np.float32),
        )
        events = [
            SyllableEvent(start=0.05, end=0.35, energy=1.0, pitch_hz=150.0),
            SyllableEvent(start=0.55, end=0.85, energy=1.0, pitch_hz=900.0),
        ]
        sample_audio = 0.6 * np.sin(2 * np.pi * 260.0 * t[: int(sample_rate * 0.25)])
        sample = MeowSample(
            audio=AudioBuffer(sample_audio, sample_rate),
            metadata=MeowSampleMetadata(local_path=Path("fixture.wav"), source_type="user_provided"),
        )

        rendered = render_meow_vocal(
            events,
            contour,
            sample,
            sample_rate=sample_rate,
            duration=1.0,
            config=MeowsicConfig(cat_min_pitch_hz=220.0, cat_max_pitch_hz=520.0, render_mode="granular"),
        )

        low_pitch = _dominant_frequency(rendered.mono()[int(0.08 * sample_rate) : int(0.30 * sample_rate)], sample_rate)
        high_pitch = _dominant_frequency(rendered.mono()[int(0.58 * sample_rate) : int(0.80 * sample_rate)], sample_rate)
        self.assertGreater(high_pitch, low_pitch)
        self.assertGreaterEqual(low_pitch, 180.0)
        self.assertLessEqual(high_pitch, 620.0)


class InstrumentRenderTests(unittest.TestCase):
    def test_instrument_mode_tracks_rising_melody(self) -> None:
        try:
            import pytsmod  # noqa: F401
        except ImportError:
            self.skipTest("pytsmod not installed")

        sample_rate = 22050
        duration = 2.0
        # A clearly voiced meow-like vowel (harmonic tone) to build the instrument.
        st = np.arange(int(sample_rate * 0.5), dtype=np.float32) / sample_rate
        vowel = sum(0.4 / k * np.sin(2 * np.pi * 300.0 * k * st) for k in (1, 2, 3)).astype(np.float32)
        sample = MeowSample(
            audio=AudioBuffer(vowel, sample_rate),
            metadata=MeowSampleMetadata(local_path=Path("fixture.wav"), source_type="user_provided"),
        )
        n = 40
        times = np.linspace(0.0, duration, n, dtype=np.float32)
        contour = PitchContour(
            times=times,
            f0_hz=np.linspace(180.0, 700.0, n, dtype=np.float32),
            voiced=np.ones(n, dtype=bool),
            energy=np.ones(n, dtype=np.float32),
        )
        events = [SyllableEvent(start=0.0, end=duration, energy=1.0, pitch_hz=None)]
        rendered = render_meow_vocal(
            events, contour, sample, sample_rate=sample_rate, duration=duration,
            config=MeowsicConfig(render_mode="instrument"),
        )
        self.assertGreater(float(np.max(np.abs(rendered.samples))), 0.05)
        early = _dominant_frequency(rendered.mono()[int(0.2 * sample_rate) : int(0.6 * sample_rate)], sample_rate)
        late = _dominant_frequency(rendered.mono()[int(1.4 * sample_rate) : int(1.8 * sample_rate)], sample_rate)
        self.assertGreater(late, early)

class DspTests(unittest.TestCase):
    def test_wsola_preserves_length_and_pitch(self) -> None:
        from meowsic.dsp import time_stretch_wsola

        sample_rate = 22050
        freq = 300.0
        t = np.arange(int(sample_rate * 0.5), dtype=np.float32) / sample_rate
        tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

        for factor in (0.5, 1.7, 2.5):
            target = int(round(tone.shape[0] * factor))
            stretched = time_stretch_wsola(tone, target)
            self.assertEqual(stretched.shape[0], target)
            self.assertGreater(float(np.max(np.abs(stretched))), 0.05)
            dominant = _dominant_frequency(stretched, sample_rate)
            # Pitch must be preserved (WSOLA), not scaled by the stretch factor.
            self.assertLess(abs(dominant - freq), 30.0)

    def test_brighten_lifts_high_frequencies(self) -> None:
        from meowsic.dsp import brighten

        sample_rate = 22050
        t = np.arange(int(sample_rate * 0.2), dtype=np.float32) / sample_rate
        low = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
        high = np.sin(2 * np.pi * 3000.0 * t).astype(np.float32)
        mixed = (0.5 * low + 0.5 * high).astype(np.float32)

        def high_ratio(sig: np.ndarray) -> float:
            spectrum = np.abs(np.fft.rfft(sig * np.hanning(sig.shape[0])))
            freqs = np.fft.rfftfreq(sig.shape[0], 1 / sample_rate)
            highs = spectrum[freqs > 1500].sum()
            lows = spectrum[freqs <= 1500].sum()
            return float(highs / (lows + 1e-9))

        self.assertGreater(high_ratio(brighten(mixed, 0.5)), high_ratio(mixed))


class EventDetectionTests(unittest.TestCase):
    def _burst_train(self, count: int, sample_rate: int = 22050, gap: float = 0.3) -> AudioBuffer:
        burst = int(sample_rate * 0.12)
        silence = int(sample_rate * (gap - 0.12))
        t = np.arange(burst, dtype=np.float32) / sample_rate
        tone = (0.5 * np.sin(2 * np.pi * 220.0 * t) * np.hanning(burst)).astype(np.float32)
        chunks = []
        for _ in range(count):
            chunks.append(tone)
            chunks.append(np.zeros(silence, dtype=np.float32))
        return AudioBuffer(np.concatenate(chunks), sample_rate)

    def test_detect_onsets_finds_bursts(self) -> None:
        from meowsic import detect_onsets

        audio = self._burst_train(6)
        onsets = detect_onsets(audio, MeowsicConfig())
        self.assertGreaterEqual(onsets.size, 4)
        self.assertLessEqual(onsets.size, 8)

    def test_onset_mode_is_denser_than_sparse_energy(self) -> None:
        from meowsic import detect_syllable_events, estimate_pitch_contour

        audio = self._burst_train(8)
        config = MeowsicConfig()
        contour = estimate_pitch_contour(audio, config)
        onset_events = detect_syllable_events(contour, config, audio=audio)
        energy_events = detect_syllable_events(contour, config)  # no audio -> energy mode
        self.assertGreaterEqual(len(onset_events), 4)
        # Onset detection should not collapse the rhythm into a single blob.
        self.assertGreaterEqual(len(onset_events), max(1, len(energy_events) // 2))


class NoteDetectionTests(unittest.TestCase):
    def test_detect_notes_segments_and_quantizes(self) -> None:
        from meowsic import detect_notes

        n = 60
        times = np.linspace(0.0, 2.0, n, dtype=np.float32)
        f0 = np.where(np.arange(n) < n // 2, 220.0, 330.0).astype(np.float32)
        contour = PitchContour(
            times=times, f0_hz=f0, voiced=np.ones(n, dtype=bool), energy=np.ones(n, dtype=np.float32)
        )
        notes = detect_notes(contour, MeowsicConfig(cat_min_pitch_hz=220.0, cat_max_pitch_hz=520.0))
        self.assertGreaterEqual(len(notes), 2)
        for note in notes:
            self.assertIsNotNone(note.pitch_hz)
            self.assertGreaterEqual(note.pitch_hz, 200.0)
            self.assertLessEqual(note.pitch_hz, 560.0)
        # The two halves are different pitches, so at least two distinct note pitches.
        self.assertGreaterEqual(len({round(note.pitch_hz) for note in notes}), 2)


class NoteRenderTests(unittest.TestCase):
    def test_notes_engine_renders_distinct_pitched_meows(self) -> None:
        sample_rate = 22050
        st = np.arange(int(sample_rate * 0.5), dtype=np.float32) / sample_rate
        vowel = sum(0.4 / k * np.sin(2 * np.pi * 300.0 * k * st) for k in (1, 2, 3)).astype(np.float32)
        sample = MeowSample(
            audio=AudioBuffer(vowel, sample_rate),
            metadata=MeowSampleMetadata(local_path=Path("fixture.wav"), source_type="user_provided"),
        )
        events = [
            SyllableEvent(start=0.1, end=0.5, energy=1.0, pitch_hz=260.0),
            SyllableEvent(start=0.7, end=1.1, energy=1.0, pitch_hz=440.0),
        ]
        contour = PitchContour(
            times=np.linspace(0, 1.2, 10, dtype=np.float32),
            f0_hz=np.full(10, 300.0, dtype=np.float32),
            voiced=np.ones(10, dtype=bool),
            energy=np.ones(10, dtype=np.float32),
        )
        rendered = render_meow_vocal(
            events, contour, sample, sample_rate=sample_rate, duration=1.2,
            config=MeowsicConfig(render_mode="notes", meow_lowpass_hz=0.0, pitch_shift_method="resample"),
        )
        self.assertGreater(float(np.max(np.abs(rendered.samples))), 0.05)
        low = _dominant_frequency(rendered.mono()[int(0.15 * sample_rate) : int(0.45 * sample_rate)], sample_rate)
        high = _dominant_frequency(rendered.mono()[int(0.75 * sample_rate) : int(1.05 * sample_rate)], sample_rate)
        self.assertGreater(high, low)


class EvaluateTests(unittest.TestCase):
    def test_self_comparison_scores_high(self) -> None:
        from meowsic import evaluate_render

        sample_rate = 22050
        t = np.arange(int(sample_rate * 2.0), dtype=np.float32) / sample_rate
        melody = 0.4 * np.sin(2 * np.pi * (220.0 + 60.0 * np.sin(2 * np.pi * 1.5 * t)) * t)
        gate = (np.sin(2 * np.pi * 3.0 * t) > -0.2).astype(np.float32)
        vocal = AudioBuffer((melody * gate).astype(np.float32), sample_rate)

        quality = evaluate_render(vocal, vocal, MeowsicConfig())
        self.assertGreaterEqual(quality.onset_f_measure, 0.95)
        self.assertGreaterEqual(quality.pitch_correlation, 0.95)
        self.assertGreaterEqual(quality.meow_score, 0.8)
        self.assertIn("meow_score", quality.as_dict())


def _dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    windowed = samples * np.hanning(samples.shape[0])
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(samples.shape[0], 1 / sample_rate)
    usable = (freqs >= 80) & (freqs <= 1000)
    index = int(np.argmax(spectrum[usable]))
    return float(freqs[usable][index])


if __name__ == "__main__":
    unittest.main()
