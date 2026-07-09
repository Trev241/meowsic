from __future__ import annotations

import numpy as np

from .analysis import estimate_sample_pitch
from .dsp import crossfade_loop, ensure_sample_rate, match_length, peak_normalize, resample_linear, time_stretch_linear, time_stretch_ola, to_channels
from .types import AudioBuffer, MeowSample, MeowsicConfig, PitchContour, SyllableEvent


def render_meow_vocal(
    events: list[SyllableEvent],
    contour: PitchContour,
    meow_sample: MeowSample,
    *,
    sample_rate: int,
    duration: float,
    config: MeowsicConfig | None = None,
) -> AudioBuffer:
    """Render a mono meow vocal by transforming recorded sample audio."""

    config = config or MeowsicConfig()
    output = np.zeros(max(1, int(round(duration * sample_rate))), dtype=np.float32)
    source_sample = meow_sample.audio.mono()
    if meow_sample.audio.sample_rate != sample_rate:
        source_sample = resample_linear(source_sample, meow_sample.audio.sample_rate, sample_rate)
    source_sample = _trim_silence(source_sample)
    source_pitch = estimate_sample_pitch(AudioBuffer(source_sample, sample_rate), config) or 220.0
    pitch_mapper = _CatPitchMapper(contour, config)
    energy_reference = max(float(np.percentile(contour.energy, 90)) if contour.energy.size else 0.0, 1e-6)

    for event in events:
        start_sample = max(0, int(round(event.start * sample_rate)))
        end_sample = min(output.shape[0], int(round(event.end * sample_rate)))
        if end_sample <= start_sample:
            continue
        original_pitch = event.pitch_hz or _pitch_at_time(contour, event.start) or source_pitch
        target_pitch = pitch_mapper.map(original_pitch)
        segment = _render_event_sample(source_sample, source_pitch, target_pitch, end_sample - start_sample)
        segment *= _event_envelope(segment.shape[0])
        gain = min(1.4, max(0.2, event.energy / energy_reference)) * config.meow_gain
        output[start_sample:end_sample] += segment * gain

    return AudioBuffer(peak_normalize(output[:, None], peak=0.98), sample_rate)


def mix_tracks(
    instrumental: AudioBuffer,
    meow_vocal: AudioBuffer,
    config: MeowsicConfig | None = None,
) -> AudioBuffer:
    """Mix rendered meow vocal into instrumental audio."""

    config = config or MeowsicConfig()
    ensure_sample_rate(instrumental, meow_vocal)
    length = max(instrumental.samples.shape[0], meow_vocal.samples.shape[0])
    channels = instrumental.channels
    inst = match_length(instrumental.samples, length) * config.instrumental_gain
    vocal = to_channels(match_length(meow_vocal.samples, length), channels)
    mixed = inst + vocal
    return AudioBuffer(peak_normalize(mixed, peak=config.output_gain), instrumental.sample_rate)


def _render_event_sample(
    source_sample: np.ndarray,
    source_pitch: float,
    target_pitch: float,
    target_length: int,
) -> np.ndarray:
    ratio = float(np.clip(target_pitch / max(source_pitch, 1e-6), 0.25, 4.0))
    pitched_length = max(1, int(round(source_sample.shape[0] / ratio)))
    pitched = time_stretch_linear(source_sample, pitched_length)
    return _fit_length_preserve_pitch(pitched, target_length, source_pitch, target_pitch)


def _fit_length_preserve_pitch(samples: np.ndarray, target_length: int, source_pitch: float, target_pitch: float) -> np.ndarray:
    """Fit samples to target_length using the best strategy depending on stretch ratio.

    - If the target fits within the sample: simple trim.
    - If stretch is < 3x: OLA time-stretch (good quality, preserves pitch texture).
    - If stretch is >= 3x (long held note): crossfade loop (avoids robotic artifacts).
    """
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if samples.shape[0] >= target_length:
        return samples[:target_length].astype(np.float32, copy=False)
    stretch_ratio = target_length / samples.shape[0]
    if stretch_ratio < 3.0:
        return time_stretch_ola(samples, target_length)
    # For very long holds, OLA alone causes buzzing — crossfade-loop instead
    return crossfade_loop(samples, target_length)


class _CatPitchMapper:
    def __init__(self, contour: PitchContour, config: MeowsicConfig) -> None:
        self.config = config
        valid = contour.f0_hz[contour.voiced & (contour.f0_hz > 0)]
        if valid.size:
            self.source_low = float(np.percentile(valid, 10))
            self.source_high = float(np.percentile(valid, 90))
            self.source_center = float(np.median(valid))
        else:
            self.source_low = config.min_pitch_hz
            self.source_high = config.max_pitch_hz
            self.source_center = (config.min_pitch_hz + config.max_pitch_hz) / 2

        if self.source_high <= self.source_low:
            self.source_high = self.source_low + 1.0
        self.target_low = min(config.cat_min_pitch_hz, config.cat_max_pitch_hz)
        self.target_high = max(config.cat_min_pitch_hz, config.cat_max_pitch_hz)
        self.target_center = (self.target_low + self.target_high) / 2.0

    def map(self, pitch_hz: float) -> float:
        normalized = (pitch_hz - self.source_low) / (self.source_high - self.source_low)
        normalized = float(np.clip(normalized, 0.0, 1.0))
        mapped = self.target_low + normalized * (self.target_high - self.target_low)
        strength = float(np.clip(self.config.cat_pitch_contour_strength, 0.0, 1.0))
        blended = self.target_center + (mapped - self.target_center) * strength
        return float(np.clip(blended, self.target_low, self.target_high))


def _pitch_at_time(contour: PitchContour, time: float) -> float | None:
    valid = contour.voiced & (contour.f0_hz > 0)
    if np.count_nonzero(valid) < 2:
        return None
    return float(np.interp([time], contour.times[valid], contour.f0_hz[valid])[0])


def _event_envelope(length: int) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    progress = np.linspace(0.0, 1.0, length, endpoint=False)
    attack = np.clip(progress / 0.08, 0.0, 1.0)
    release = np.clip((1.0 - progress) / 0.14, 0.0, 1.0)
    return (attack * release).astype(np.float32)


def _trim_silence(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples
    threshold = max(1e-4, float(np.max(np.abs(samples))) * 0.03)
    active = np.flatnonzero(np.abs(samples) > threshold)
    if active.size == 0:
        return samples
    return samples[max(0, active[0] - 32) : min(samples.shape[0], active[-1] + 33)]
