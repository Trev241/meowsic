from __future__ import annotations

import numpy as np

from .analysis import estimate_sample_pitch
from .dsp import (
    brighten,
    crossfade_loop,
    ensure_sample_rate,
    extract_loud_region,
    fade_edges,
    lowpass_fft,
    match_length,
    normalize_peak,
    peak_normalize,
    remove_dc,
    resample_linear,
    time_stretch_linear,
    time_stretch_wsola,
    to_channels,
)
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
    """Render a mono meow vocal from recorded sample audio.

    Uses the "instrument" engine (TD-PSOLA, formant-preserving melodic singing) when
    ``config.render_mode == "instrument"`` and pytsmod is available; otherwise falls
    back to the "granular" per-note splicing engine.
    """

    config = config or MeowsicConfig()
    if config.render_mode == "notes":
        rendered = _render_notes(events, meow_sample, sample_rate, duration, config)
        if rendered is not None:
            return rendered
    if config.render_mode == "instrument":
        rendered = _render_instrument(events, contour, meow_sample, sample_rate, duration, config)
        if rendered is not None:
            return rendered
    return _render_granular(events, contour, meow_sample, sample_rate=sample_rate, duration=duration, config=config)


def _render_notes(
    events: list[SyllableEvent],
    meow_sample: MeowSample,
    sample_rate: int,
    duration: float,
    config: MeowsicConfig,
) -> AudioBuffer | None:
    """Cat-cover engine: one whole pitched meow per note.

    Each event carries a target melody pitch (``pitch_hz``). We shift a clean meow to
    that pitch and fit it to the note: for short notes ``compress`` plays the complete
    meow gesture scaled down (so it stays a recognizable "meow"), while ``trim`` plays
    just the onset; long notes loop the meow. This is what makes it read as a cat
    singing the tune rather than a drone or chopped chips.
    """

    if not events:
        return None
    meow = _prepare_meow_body(meow_sample, sample_rate, config)
    if meow.size < int(0.04 * sample_rate):
        return None

    total = max(1, int(round(duration * sample_rate)))
    output = np.zeros(total, dtype=np.float32)
    median = estimate_sample_pitch(AudioBuffer(meow, sample_rate), config) or 500.0

    use_psola = config.pitch_shift_method == "psola"
    tsm = None
    src_f0 = None
    hop = win = 0
    if use_psola:
        try:
            import pytsmod as tsm  # type: ignore

            hop, win = max(1, sample_rate // 100), max(2, sample_rate // 30)
            src_f0 = _meow_f0_contour(meow, sample_rate, hop, win, median)
        except ImportError:
            use_psola = False

    energies = [e.energy for e in events] or [1.0]
    energy_reference = max(float(np.percentile(energies, 90)), 1e-6)

    for event in events:
        start_sample = max(0, int(round(event.start * sample_rate)))
        dur = min(total, int(round(event.end * sample_rate))) - start_sample
        if dur < int(0.04 * sample_rate):
            continue
        target = event.pitch_hz or median
        ratio = float(np.clip(target / max(median, 1e-6), 0.25, 4.0))
        if use_psola and tsm is not None:
            try:
                shifted = np.asarray(
                    tsm.tdpsola(meow.astype(np.float64), sample_rate, src_f0, beta=ratio, p_hop_size=hop, p_win_size=win),
                    dtype=np.float32,
                )
            except Exception:
                shifted = resample_linear(meow, sample_rate, max(1, int(sample_rate / ratio)))
        else:
            shifted = resample_linear(meow, sample_rate, max(1, int(sample_rate / ratio)))

        segment = _fit_note(shifted, dur, config)
        segment = fade_edges(segment, max(1, int(0.008 * sample_rate)))
        gain = min(1.3, max(0.35, event.energy / energy_reference)) * config.meow_gain
        end = min(total, start_sample + segment.shape[0])
        output[start_sample:end] += segment[: end - start_sample] * gain

    if float(np.max(np.abs(output))) < 1e-6:
        return None
    return AudioBuffer(peak_normalize(output[:, None], peak=0.98), sample_rate)


def _fit_note(shifted: np.ndarray, dur: int, config: MeowsicConfig) -> np.ndarray:
    if dur <= 0:
        return np.zeros(0, dtype=np.float32)
    if shifted.size == 0:
        return np.zeros(dur, dtype=np.float32)
    if config.note_fit == "compress":
        # Play the complete meow gesture scaled to the note; loop very long notes.
        if dur <= shifted.shape[0]:
            return time_stretch_wsola(shifted, dur)
        return crossfade_loop(shifted, dur)
    # "trim": onset for short notes, loop for long.
    if shifted.shape[0] >= dur:
        return shifted[:dur].astype(np.float32, copy=True)
    return crossfade_loop(shifted, dur)


def _prepare_meow_body(meow_sample: MeowSample, sample_rate: int, config: MeowsicConfig) -> np.ndarray:
    """Clean the meow for the note engine: resample, de-noise (low-pass), body-trim, normalize."""

    raw = meow_sample.audio.mono()
    if meow_sample.audio.sample_rate != sample_rate:
        raw = resample_linear(raw, meow_sample.audio.sample_rate, sample_rate)
    raw = remove_dc(raw)
    if config.meow_lowpass_hz > 0:
        raw = lowpass_fft(raw, sample_rate, config.meow_lowpass_hz)
    raw = extract_loud_region(raw, sample_rate)
    if raw.size == 0:
        return raw
    raw = normalize_peak(raw, 0.97)
    return fade_edges(raw, max(1, int(0.008 * sample_rate)))


def _meow_f0_contour(meow: np.ndarray, sample_rate: int, hop: int, win: int, median: float) -> np.ndarray:
    frames = max(1, len(meow) // hop + 1)
    f0 = np.zeros(frames, dtype=np.float64)
    window = np.hanning(win)
    lo, hi = int(sample_rate / 900), int(sample_rate / 200)
    for idx in range(frames):
        seg = meow[idx * hop : idx * hop + win]
        if seg.size < win:
            seg = np.pad(seg, (0, win - seg.size))
        seg = (seg - seg.mean()) * window
        corr = np.correlate(seg, seg, "full")[win - 1 :]
        if corr[0] <= 1e-9 or hi <= lo:
            continue
        lag = int(np.argmax(corr[lo:hi]) + lo)
        if lag > 0:
            f0[idx] = sample_rate / lag
    f0[f0 <= 0] = median
    return np.clip(f0, median * 0.5, median * 2.0)


def _render_granular(
    events: list[SyllableEvent],
    contour: PitchContour,
    meow_sample: MeowSample,
    *,
    sample_rate: int,
    duration: float,
    config: MeowsicConfig,
) -> AudioBuffer:
    """Per-note splicing engine: pitch-shift a meow per detected event."""

    output = np.zeros(max(1, int(round(duration * sample_rate))), dtype=np.float32)
    source_sample = meow_sample.audio.mono()
    if meow_sample.audio.sample_rate != sample_rate:
        source_sample = resample_linear(source_sample, meow_sample.audio.sample_rate, sample_rate)
    source_sample = _prepare_source_sample(source_sample, sample_rate, config)
    # Estimate pitch on the *raw* trimmed sample (brightening can bias autocorrelation).
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
        segment = _render_event_sample(source_sample, source_pitch, target_pitch, end_sample - start_sample, config)
        segment *= _event_envelope(segment.shape[0], sample_rate)
        gain = min(1.4, max(0.2, event.energy / energy_reference)) * config.meow_gain
        output[start_sample:end_sample] += segment * gain

    return AudioBuffer(peak_normalize(output[:, None], peak=0.98), sample_rate)


def _render_instrument(
    events: list[SyllableEvent],
    contour: PitchContour,
    meow_sample: MeowSample,
    sample_rate: int,
    duration: float,
    config: MeowsicConfig,
) -> AudioBuffer | None:
    """Play the meow as a formant-preserving voice that sings the melody.

    The meow's vowel is turned into a sustained "oscillator" and driven along the
    song's (octave-transposed) pitch contour with TD-PSOLA, so pitch glides
    continuously and formants stay put. Phrases are synthesized legato and only
    broken where the singer pauses; note onsets are marked by loudness articulation
    rather than silence, so the result tracks the rhythm without sounding chopped.

    Returns ``None`` (so the caller falls back to the granular engine) if pytsmod is
    unavailable or the sample cannot be turned into a usable vowel.
    """

    try:
        import pytsmod as tsm  # type: ignore
    except ImportError:
        return None

    raw = meow_sample.audio.mono()
    if meow_sample.audio.sample_rate != sample_rate:
        raw = resample_linear(raw, meow_sample.audio.sample_rate, sample_rate)
    raw = _trim_silence(remove_dc(raw))
    if raw.size < int(0.1 * sample_rate):
        return None

    vowel = _extract_tonal_core(raw, sample_rate, config)
    vowel = normalize_peak(vowel, 0.95)
    vowel = brighten(vowel, config.meow_brightness)
    vowel = normalize_peak(vowel, 0.95)
    vowel = fade_edges(vowel, max(1, int(0.01 * sample_rate)))
    if vowel.size < int(0.06 * sample_rate):
        return None
    vowel_f0 = estimate_sample_pitch(AudioBuffer(vowel, sample_rate), config) or 500.0

    attack = _attack_grain(raw, sample_rate, config) if config.attack_blend else np.zeros(0, dtype=np.float32)

    hop = max(1, int(sample_rate / 100))
    win = max(hop * 2, int(sample_rate / 30))
    n_out = max(1, int(round(duration * sample_rate)))
    grid_t = np.arange(0, duration, hop / sample_rate)
    if grid_t.size < 2:
        return None

    target_f0 = _build_target_f0(contour, config, grid_t, vowel_f0)
    target_f0 = _smooth(target_f0, max(1, int((config.portamento_ms / 1000.0) * sample_rate / hop)))
    amp = _build_amplitude(contour, grid_t)
    active = _active_grid(contour, grid_t, config)
    phrases = _phrases_from_mask(active, hop, sample_rate, config)
    onset_samples = sorted(int(round(e.start * sample_rate)) for e in events)

    output = np.zeros(n_out, dtype=np.float32)
    vowel64 = vowel.astype(np.float64)
    vowel_frames = max(1, len(vowel) // hop + 1)
    vowel_src_f0 = np.full(vowel_frames, vowel_f0, dtype=np.float64)

    for start_sample, end_sample in phrases:
        start_sample = max(0, start_sample)
        end_sample = min(n_out, end_sample)
        span = end_sample - start_sample
        if span < int(0.06 * sample_rate):
            continue
        try:
            seamless = tsm.tdpsola(vowel64, sample_rate, vowel_src_f0, alpha=span / len(vowel), p_hop_size=hop, p_win_size=win)
            frames = max(1, len(seamless) // hop + 1)
            grid_idx = np.clip(np.arange(start_sample, end_sample, hop) // hop, 0, target_f0.size - 1)
            phrase_tgt = np.interp(
                np.linspace(0, 1, frames),
                np.linspace(0, 1, grid_idx.size),
                target_f0[grid_idx],
            ).astype(np.float64)
            src_f0 = np.full(frames, vowel_f0, dtype=np.float64)
            voice = tsm.tdpsola(seamless.astype(np.float64), sample_rate, src_f0, tgt_f0=phrase_tgt, p_hop_size=hop, p_win_size=win)
        except Exception:
            return None
        voice = np.asarray(voice, dtype=np.float32)[:span]
        if voice.size < span:
            voice = np.pad(voice, (0, span - voice.size))

        env = _phrase_envelope(amp, grid_t, start_sample, end_sample, sample_rate)
        env = _apply_articulation(env, onset_samples, start_sample, sample_rate, config)
        voice = voice * env
        if attack.size:
            blend = min(attack.size, voice.size)
            voice[:blend] += attack[:blend] * 0.7
        voice = fade_edges(voice, max(1, int(0.01 * sample_rate)))
        output[start_sample : start_sample + voice.size] += voice[: n_out - start_sample]

    if float(np.max(np.abs(output))) < 1e-6:
        return None
    output *= config.meow_gain
    return AudioBuffer(peak_normalize(output[:, None], peak=0.98), sample_rate)


def _attack_grain(raw: np.ndarray, sample_rate: int, config: MeowsicConfig) -> np.ndarray:
    """The meow's natural onset (~80 ms), used to articulate phrase starts."""

    grain = raw[: int(0.08 * sample_rate)].astype(np.float32, copy=True)
    if grain.size == 0:
        return grain
    grain = normalize_peak(grain, 0.9)
    grain = brighten(grain, config.meow_brightness)
    return fade_edges(grain, max(1, int(0.006 * sample_rate)))


def _build_target_f0(contour: PitchContour, config: MeowsicConfig, grid_t: np.ndarray, vowel_f0: float) -> np.ndarray:
    """Continuous target pitch (Hz) on the grid.

    In adaptive mode the vocal melody is transposed by whole octaves into the cat
    register, which preserves the tune's intervals (so it stays in key) rather than
    compressing the range and detuning it. Otherwise the configured linear
    cat-register mapping is used.
    """

    voiced = contour.voiced & (contour.f0_hz > 0)
    cat_low = min(config.cat_min_pitch_hz, config.cat_max_pitch_hz)
    cat_high = max(config.cat_min_pitch_hz, config.cat_max_pitch_hz)
    if int(np.count_nonzero(voiced)) < 2:
        return np.full(grid_t.shape, float(np.sqrt(cat_low * cat_high)), dtype=np.float64)

    source = np.interp(grid_t, contour.times[voiced], contour.f0_hz[voiced],
                       left=contour.f0_hz[voiced][0], right=contour.f0_hz[voiced][-1])

    if config.adaptive:
        source_center = float(np.median(contour.f0_hz[voiced]))
        target_center = float(np.sqrt(cat_low * cat_high))
        octaves = round(np.log2(max(target_center, 1e-6) / max(source_center, 1e-6)))
        transposed = source * (2.0 ** octaves)
        strength = float(np.clip(config.cat_pitch_contour_strength, 0.0, 1.0))
        center = float(np.median(transposed))
        target = center + (transposed - center) * (0.5 + 0.5 * strength)
        return np.clip(target, cat_low * 0.75, cat_high * 1.25).astype(np.float64)

    mapper = _CatPitchMapper(contour, config)
    return np.array([mapper.map(float(f)) for f in source], dtype=np.float64)


def _build_amplitude(contour: PitchContour, grid_t: np.ndarray) -> np.ndarray:
    if contour.energy.size < 2:
        return np.ones(grid_t.shape, dtype=np.float32)
    amp = np.interp(grid_t, contour.times, contour.energy)
    peak = float(amp.max())
    if peak > 1e-9:
        amp = amp / peak
    return amp.astype(np.float32)


def _active_grid(contour: PitchContour, grid_t: np.ndarray, config: MeowsicConfig) -> np.ndarray:
    if contour.energy.size < 2:
        return np.ones(grid_t.shape, dtype=bool)
    energy = np.interp(grid_t, contour.times, contour.energy)
    voiced_energy = contour.energy[contour.voiced] if np.any(contour.voiced) else contour.energy
    threshold = float(np.percentile(voiced_energy, 60) * config.energy_threshold_ratio) if voiced_energy.size else 0.0
    return energy >= threshold


def _phrases_from_mask(active: np.ndarray, hop: int, sample_rate: int, config: MeowsicConfig) -> list[tuple[int, int]]:
    """Group active grid frames into phrases, bridging short unvoiced gaps."""

    gap_frames = max(1, int((config.phrase_gap * sample_rate) / hop))
    phrases: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for index, flag in enumerate(active):
        if flag:
            if start is None:
                start = index
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap > gap_frames:
                    phrases.append((start * hop, (index - gap + 1) * hop))
                    start = None
                    gap = 0
    if start is not None:
        phrases.append((start * hop, active.size * hop))
    return phrases


def _phrase_envelope(amp: np.ndarray, grid_t: np.ndarray, start_sample: int, end_sample: int, sample_rate: int) -> np.ndarray:
    span = end_sample - start_sample
    grid_start = grid_t[0] if grid_t.size else 0.0
    idx = np.clip(np.arange(start_sample, end_sample) / sample_rate, grid_start, grid_t[-1])
    env = np.interp(idx, grid_t, amp).astype(np.float32)
    # Keep some floor so the voice never fully drops mid-phrase.
    return np.clip(0.35 + 0.65 * env, 0.0, 1.0)


def _apply_articulation(env: np.ndarray, onset_samples: list[int], phrase_start: int, sample_rate: int, config: MeowsicConfig) -> np.ndarray:
    depth = float(np.clip(config.articulation_depth, 0.0, 1.0))
    if depth <= 0.0 or not onset_samples:
        return env
    dip = max(1, int(0.012 * sample_rate))
    rise = max(1, int(0.03 * sample_rate))
    out = env.copy()
    for onset in onset_samples:
        local = onset - phrase_start
        if local <= 0 or local >= env.size:
            continue
        d0 = max(0, local - dip)
        out[d0:local] *= np.linspace(1.0, 1.0 - depth, local - d0, dtype=np.float32)
        r1 = min(env.size, local + rise)
        out[local:r1] *= np.linspace(1.0 - depth, 1.0, r1 - local, dtype=np.float32)
    return out


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


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
    config: MeowsicConfig,
) -> np.ndarray:
    # Pitch-shift by resampling: this changes both pitch and length. A ratio > 1
    # shifts up (shorter), < 1 shifts down (longer).
    ratio = float(np.clip(target_pitch / max(source_pitch, 1e-6), 0.25, 4.0))
    pitched_length = max(1, int(round(source_sample.shape[0] / ratio)))
    pitched = time_stretch_linear(source_sample, pitched_length)
    return _fit_length_preserve_pitch(pitched, target_length, config)


def _fit_length_preserve_pitch(samples: np.ndarray, target_length: int, config: MeowsicConfig) -> np.ndarray:
    """Fit a pitched meow to target_length while preserving its (already shifted) pitch.

    Strategy (chosen for quality + a natural cat-cover feel):
    - target shorter than the meow: play the natural meow onset (trim), unless
      ``meow_full_syllable`` is set, which time-compresses the whole meow into the note.
    - moderate stretch: WSOLA time-stretch (phase-coherent, no buzzing).
    - long held note (beyond ``meow_max_stretch``): crossfade-loop the meow so a
      sustained note becomes repeated meows rather than one unnaturally slow meow.
    """

    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.size == 0:
        return np.zeros(target_length, dtype=np.float32)

    length = samples.shape[0]
    if length >= target_length:
        if config.meow_full_syllable:
            return time_stretch_wsola(samples, target_length)
        return samples[:target_length].astype(np.float32, copy=False)

    stretch_ratio = target_length / length
    if stretch_ratio <= max(1.0, config.meow_max_stretch):
        return time_stretch_wsola(samples, target_length)
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


def _event_envelope(length: int, sample_rate: int) -> np.ndarray:
    """Short, fixed click-free fades.

    Fixed millisecond fades (rather than large proportional ones) avoid clicks at
    note boundaries without swallowing the meow's own attack/decay shape.
    """

    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    env = np.ones(length, dtype=np.float32)
    attack = min(length // 2, max(1, int(0.006 * sample_rate)))
    release = min(length - attack, max(1, int(0.020 * sample_rate)))
    if attack > 0:
        env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    if release > 0:
        env[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
    return env


def _prepare_source_sample(samples: np.ndarray, sample_rate: int, config: MeowsicConfig) -> np.ndarray:
    """Clean up the raw meow sample before it is used for rendering.

    Removes DC, trims leading/trailing silence, normalizes level, applies a gentle
    brightness lift (helps the muffled 8 kHz CatMeows recordings), and fades the
    edges so looping/stretching never introduces boundary clicks.
    """

    cleaned = remove_dc(samples)
    cleaned = _trim_silence(cleaned)
    if cleaned.size == 0:
        return cleaned
    if config.meow_use_core:
        cleaned = _extract_tonal_core(cleaned, sample_rate, config)
    cleaned = normalize_peak(cleaned, peak=0.95)
    cleaned = brighten(cleaned, config.meow_brightness)
    cleaned = normalize_peak(cleaned, peak=0.95)
    fade = max(1, int(config.meow_core_attack * sample_rate))
    cleaned = fade_edges(cleaned, fade_samples=fade)
    return cleaned.astype(np.float32)


def _extract_tonal_core(samples: np.ndarray, sample_rate: int, config: MeowsicConfig) -> np.ndarray:
    """Return the meow's stable, steadily-pitched vowel region.

    A meow glides through many pitches; if we shift the whole sample to a target
    note and then only play a short slice, the slice usually is not at the target
    pitch. Isolating the longest run of frames whose pitch stays near the sample's
    median gives a tonal "core" that pitch-shifts predictably, so short notes land
    on the intended melody note.
    """

    from .analysis import estimate_pitch_contour

    if samples.size < config.frame_length:
        return samples
    contour = estimate_pitch_contour(AudioBuffer(samples, sample_rate), config)
    voiced = contour.voiced & (contour.f0_hz > 0)
    if int(np.count_nonzero(voiced)) < 3:
        return samples
    median = float(np.median(contour.f0_hz[voiced]))
    if median <= 0:
        return samples
    tolerance = max(0.05, config.meow_core_tolerance)
    stable = voiced & (np.abs(contour.f0_hz - median) <= tolerance * median)

    best_start, best_end, run_start = 0, 0, None
    for index, flag in enumerate(stable):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            if index - run_start > best_end - best_start:
                best_start, best_end = run_start, index
            run_start = None
    if run_start is not None and stable.size - run_start > best_end - best_start:
        best_start, best_end = run_start, stable.size
    if best_end <= best_start:
        return samples

    start_sample = max(0, int(contour.times[best_start] * sample_rate) - config.hop_length)
    end_idx = min(best_end, contour.times.size - 1)
    end_sample = min(samples.shape[0], int(contour.times[end_idx] * sample_rate) + config.hop_length)
    core = samples[start_sample:end_sample]
    # Require a minimum usable core; otherwise keep the full sample.
    if core.shape[0] < int(0.05 * sample_rate):
        return samples
    return core.astype(np.float32)


def _trim_silence(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples
    threshold = max(1e-4, float(np.max(np.abs(samples))) * 0.03)
    active = np.flatnonzero(np.abs(samples) > threshold)
    if active.size == 0:
        return samples
    return samples[max(0, active[0] - 32) : min(samples.shape[0], active[-1] + 33)]
