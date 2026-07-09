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


def onset_envelope(audio: AudioBuffer, config: MeowsicConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, normalized spectral-flux onset envelope) for a vocal stem.

    Spectral flux (sum of positive magnitude changes between STFT frames) responds
    to note attacks and syllable onsets, which is what we want to align meows to.
    """

    config = config or MeowsicConfig()
    y = audio.mono()
    win = config.onset_frame_length
    hop = config.onset_hop_length
    if y.shape[0] < win:
        y = np.pad(y, (0, win - y.shape[0]))
    window = np.hanning(win).astype(np.float32)
    starts = np.arange(0, y.shape[0] - win + 1, hop)
    flux = np.zeros(starts.shape[0], dtype=np.float32)
    prev_mag: np.ndarray | None = None
    for index, start in enumerate(starts):
        mag = np.abs(np.fft.rfft(y[start : start + win] * window))
        if prev_mag is not None:
            flux[index] = float(np.sum(np.maximum(0.0, mag - prev_mag)))
        prev_mag = mag
    times = (starts + win / 2) / audio.sample_rate
    peak = float(flux.max()) if flux.size else 0.0
    if peak > 0:
        flux = flux / peak
    return times.astype(np.float32), flux


def detect_onsets(audio: AudioBuffer, config: MeowsicConfig | None = None) -> np.ndarray:
    """Pick onset times from the spectral-flux envelope with adaptive thresholding."""

    config = config or MeowsicConfig()
    times, flux = onset_envelope(audio, config)
    if flux.size < 3:
        return np.zeros(0, dtype=np.float32)
    threshold = float(flux.mean() + config.onset_threshold * flux.std())
    onsets: list[float] = []
    last = -1e9
    for index in range(1, flux.size - 1):
        value = flux[index]
        if value > flux[index - 1] and value >= flux[index + 1] and value > threshold:
            time = float(times[index])
            if time - last >= config.onset_min_interval:
                onsets.append(time)
                last = time
    return np.array(onsets, dtype=np.float32)


def detect_syllable_events(
    contour: PitchContour,
    config: MeowsicConfig | None = None,
    *,
    audio: AudioBuffer | None = None,
) -> list[SyllableEvent]:
    """Detect syllable-like events from the vocal.

    In the default "onset" mode the vocal is segmented at note attacks and long
    held notes are re-articulated so the meows follow the song's rhythm. The older
    "energy" mode (contiguous voiced regions) is kept as a fallback and is used
    automatically when no audio is supplied or no onsets are found.
    """

    config = config or MeowsicConfig()
    if contour.times.size == 0:
        return []
    if config.event_mode == "onset" and audio is not None:
        events = _events_from_onsets(contour, audio, config)
        if events:
            return events
    return _events_from_energy(contour, config)


def detect_notes(contour: PitchContour, config: MeowsicConfig | None = None) -> list[SyllableEvent]:
    """Segment the melody into discrete, semitone-quantized notes (cat-cover formula).

    The vocal contour is transposed by whole octaves into the cat register (preserving
    the tune's intervals so it stays in key), snapped to the nearest semitone, and
    grouped into contiguous same-pitch notes. Short notes are merged into neighbours so
    each note has room to be a recognizable meow. Each returned event's ``pitch_hz`` is
    the final cat-register target pitch in Hz.
    """

    config = config or MeowsicConfig()
    times, f0, voiced, energy = contour.times, contour.f0_hz, contour.voiced, contour.energy
    vmask = voiced & (f0 > 0)
    if int(np.count_nonzero(vmask)) < 2:
        return []

    cat_low = min(config.cat_min_pitch_hz, config.cat_max_pitch_hz)
    cat_high = max(config.cat_min_pitch_hz, config.cat_max_pitch_hz)
    median = float(np.median(f0[vmask]))
    center = float(np.sqrt(cat_low * cat_high))
    octaves = round(np.log2(max(center, 1e-6) / max(median, 1e-6)))
    step = _median_frame_step(times)

    midi = np.full(times.size, -999, dtype=int)
    for index in range(times.size):
        if vmask[index]:
            hz = _fold_into_range(float(f0[index]) * (2.0 ** octaves), cat_low, cat_high)
            midi[index] = int(round(69 + 12 * np.log2(max(hz, 1e-6) / 440.0)))

    raw: list[SyllableEvent] = []
    index = 0
    total = times.size
    while index < total:
        if midi[index] == -999:
            index += 1
            continue
        end = index
        while end + 1 < total and midi[end + 1] == midi[index]:
            end += 1
        start_t = float(times[index])
        end_t = float(times[end] + step)
        pitch = float(440.0 * 2.0 ** ((midi[index] - 69) / 12.0))
        note_energy = float(np.mean(energy[index : end + 1]))
        raw.append(SyllableEvent(start=start_t, end=end_t, energy=note_energy, pitch_hz=pitch))
        index = end + 1

    return _merge_short_notes(raw, config)


def _merge_short_notes(notes: list[SyllableEvent], config: MeowsicConfig) -> list[SyllableEvent]:
    if not notes:
        return notes
    merged: list[SyllableEvent] = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        gap = note.start - prev.end
        if gap < 0.06 and (prev.end - prev.start) < config.note_merge_duration:
            merged[-1] = SyllableEvent(
                start=prev.start, end=note.end,
                energy=max(prev.energy, note.energy), pitch_hz=note.pitch_hz,
            )
        else:
            merged.append(note)
    return [n for n in merged if (n.end - n.start) >= config.note_min_duration]


def _fold_into_range(hz: float, low: float, high: float) -> float:
    if hz <= 0:
        return low
    while hz > high:
        hz /= 2.0
    while hz < low:
        hz *= 2.0
    return hz


def _events_from_onsets(
    contour: PitchContour,
    audio: AudioBuffer,
    config: MeowsicConfig,
) -> list[SyllableEvent]:
    onsets = detect_onsets(audio, config)
    if onsets.size == 0:
        return []
    times = contour.times
    active = _active_frames(contour, config.energy_threshold_ratio)
    step = _median_frame_step(times)
    duration = audio.duration

    bounds = [float(b) for b in onsets if 0.0 <= b < duration]
    if not bounds:
        return []
    bounds.append(duration)

    events: list[SyllableEvent] = []
    for index in range(len(bounds) - 1):
        interval_start = bounds[index]
        interval_end = bounds[index + 1]
        mask = (times >= interval_start) & (times < interval_end) & active
        if not np.any(mask):
            continue
        idxs = np.flatnonzero(mask)
        note_start = max(0.0, interval_start)
        note_end = float(times[idxs[-1]] + step / 2)
        if note_end <= note_start:
            continue
        for sub_start, sub_end in _rearticulate(note_start, note_end, config):
            if sub_end - sub_start < config.min_event_duration:
                continue
            events.append(_build_event(contour, sub_start, sub_end))
    return events


def _events_from_energy(contour: PitchContour, config: MeowsicConfig) -> list[SyllableEvent]:
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


def _rearticulate(start: float, end: float, config: MeowsicConfig) -> list[tuple[float, float]]:
    if not config.rearticulate:
        return [(start, end)]
    interval = max(0.05, config.rearticulate_interval)
    span = end - start
    if span <= interval * 1.5:
        return [(start, end)]
    count = max(1, int(round(span / interval)))
    edges = np.linspace(start, end, count + 1)
    return [(float(edges[i]), float(edges[i + 1])) for i in range(count)]


def _build_event(contour: PitchContour, start: float, end: float) -> SyllableEvent:
    mask = (contour.times >= start) & (contour.times < end)
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(contour.times - start)))
        mask = np.zeros(contour.times.shape[0], dtype=bool)
        mask[nearest] = True
    energy = float(np.mean(contour.energy[mask]))
    voiced_f0 = contour.f0_hz[mask & contour.voiced & (contour.f0_hz > 0)]
    pitch = float(np.median(voiced_f0)) if voiced_f0.size else None
    return SyllableEvent(start=start, end=end, energy=energy, pitch_hz=pitch)


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
