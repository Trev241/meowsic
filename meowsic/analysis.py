from __future__ import annotations

import numpy as np

from .types import AudioBuffer, MeowsicConfig, PitchContour, SyllableEvent


def estimate_pitch_contour(audio: AudioBuffer, config: MeowsicConfig | None = None) -> PitchContour:
    """Estimate frame-level F0 and energy for a vocal stem."""

    config = config or MeowsicConfig()
    y = audio.mono()
    frame_length = config.frame_length
    hop = config.hop_length
    if y.shape[0] < frame_length:
        y = np.pad(y, (0, frame_length - y.shape[0]))

    starts = np.arange(0, max(1, y.shape[0] - frame_length + 1), hop)
    window = np.hanning(frame_length).astype(np.float32)
    min_lag = max(1, int(audio.sample_rate / config.max_pitch_hz))
    max_lag = min(frame_length - 2, int(audio.sample_rate / config.min_pitch_hz))
    f0 = np.zeros(starts.shape[0], dtype=np.float32)
    voiced = np.zeros(starts.shape[0], dtype=bool)
    energy = np.zeros(starts.shape[0], dtype=np.float32)

    for index, start in enumerate(starts):
        frame = y[start : start + frame_length]
        frame = (frame - np.mean(frame)) * window
        rms = float(np.sqrt(np.mean(frame * frame)))
        energy[index] = rms
        if rms < 1e-5:
            continue
        corr = np.correlate(frame, frame, mode="full")[frame_length - 1 :]
        if corr[0] <= 1e-9:
            continue
        search = corr[min_lag:max_lag]
        if search.size == 0:
            continue
        lag = int(np.argmax(search) + min_lag)
        confidence = float(corr[lag] / corr[0])
        if confidence < 0.25:
            continue
        f0[index] = float(audio.sample_rate / _refine_lag(corr, lag))
        voiced[index] = True

    times = (starts + frame_length / 2) / audio.sample_rate
    return PitchContour(times=times.astype(np.float32), f0_hz=f0, voiced=voiced, energy=energy)


def detect_syllable_events(
    contour: PitchContour,
    config: MeowsicConfig | None = None,
) -> list[SyllableEvent]:
    """Detect syllable-like voiced regions from pitch and energy frames."""

    config = config or MeowsicConfig()
    if contour.times.size == 0:
        return []
    active = _active_frames(contour, config.energy_threshold_ratio)
    regions = _contiguous_regions(active)
    events: list[SyllableEvent] = []
    frame_step = _median_frame_step(contour.times)

    for start_idx, end_idx in regions:
        for event_start_idx, event_end_idx in _split_region(contour, start_idx, end_idx, config):
            start = max(0.0, float(contour.times[event_start_idx] - frame_step / 2))
            end = float(contour.times[min(event_end_idx, contour.times.size - 1)] + frame_step / 2)
            if end - start < config.min_event_duration:
                continue
            event_energy = float(np.mean(contour.energy[event_start_idx:event_end_idx]))
            voiced_f0 = contour.f0_hz[event_start_idx:event_end_idx]
            voiced_f0 = voiced_f0[voiced_f0 > 0]
            pitch = float(np.median(voiced_f0)) if voiced_f0.size else None
            events.append(SyllableEvent(start=start, end=end, energy=event_energy, pitch_hz=pitch))
    return events


def estimate_sample_pitch(sample: AudioBuffer, config: MeowsicConfig | None = None) -> float | None:
    contour = estimate_pitch_contour(sample, config)
    valid = contour.f0_hz[contour.voiced & (contour.f0_hz > 0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _refine_lag(corr: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= corr.shape[0] - 1:
        return float(lag)
    left, center, right = float(corr[lag - 1]), float(corr[lag]), float(corr[lag + 1])
    denom = left - 2 * center + right
    if abs(denom) < 1e-9:
        return float(lag)
    return float(lag + 0.5 * (left - right) / denom)


def _active_frames(contour: PitchContour, threshold_ratio: float) -> np.ndarray:
    voiced_energy = contour.energy[contour.voiced]
    if voiced_energy.size == 0:
        threshold = float(np.max(contour.energy) * threshold_ratio) if contour.energy.size else 0.0
    else:
        threshold = float(np.percentile(voiced_energy, 70) * threshold_ratio)
    return contour.voiced & (contour.energy >= threshold)


def _contiguous_regions(active: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            regions.append((start, index))
            start = None
    if start is not None:
        regions.append((start, active.shape[0]))
    return regions


def _split_region(
    contour: PitchContour,
    start_idx: int,
    end_idx: int,
    config: MeowsicConfig,
) -> list[tuple[int, int]]:
    frame_step = _median_frame_step(contour.times)
    max_frames = max(1, int(round(config.max_event_duration / frame_step)))
    if end_idx - start_idx <= max_frames:
        return [(start_idx, end_idx)]
    ranges: list[tuple[int, int]] = []
    cursor = start_idx
    while cursor < end_idx:
        split = min(cursor + max_frames, end_idx)
        local_energy = contour.energy[cursor:split]
        if local_energy.size > 4:
            offset = int(np.argmin(local_energy[local_energy.size // 2 :]) + local_energy.size // 2)
            split = min(end_idx, max(cursor + 1, cursor + offset))
        ranges.append((cursor, split))
        cursor = split
    return ranges


def _median_frame_step(times: np.ndarray) -> float:
    if times.size < 2:
        return 0.02
    return float(np.median(np.diff(times)))
