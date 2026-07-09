from __future__ import annotations

import numpy as np

from .types import AudioBuffer


def ensure_sample_rate(*buffers: AudioBuffer) -> int:
    rates = {buffer.sample_rate for buffer in buffers}
    if len(rates) != 1:
        raise ValueError(f"Sample rates must match, got {sorted(rates)}")
    return rates.pop()


def match_length(samples: np.ndarray, length: int) -> np.ndarray:
    if samples.shape[0] == length:
        return samples
    if samples.shape[0] > length:
        return samples[:length]
    pad_width = [(0, length - samples.shape[0])]
    if samples.ndim == 2:
        pad_width.append((0, 0))
    return np.pad(samples, pad_width)


def to_channels(samples: np.ndarray, channels: int) -> np.ndarray:
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.shape[1] == channels:
        return samples
    if samples.shape[1] == 1:
        return np.repeat(samples, channels, axis=1)
    if channels == 1:
        return samples.mean(axis=1, keepdims=True)
    raise ValueError(f"Cannot convert {samples.shape[1]} channels to {channels} channels")


def peak_normalize(samples: np.ndarray, peak: float = 0.95) -> np.ndarray:
    current = float(np.max(np.abs(samples))) if samples.size else 0.0
    if current <= 1e-9 or current <= peak:
        return samples
    return samples * (peak / current)


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    if samples.shape[0] == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.shape[0] / source_rate
    target_count = max(1, int(round(duration * target_rate)))
    source_times = np.linspace(0.0, duration, samples.shape[0], endpoint=False)
    target_times = np.linspace(0.0, duration, target_count, endpoint=False)
    if samples.ndim == 1:
        return np.interp(target_times, source_times, samples).astype(np.float32)
    channels = [
        np.interp(target_times, source_times, samples[:, channel])
        for channel in range(samples.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def time_stretch_linear(samples: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    if samples.shape[0] == target_length:
        return samples.astype(np.float32, copy=False)
    if samples.shape[0] == 0:
        return np.zeros(target_length, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, samples.shape[0], endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_length, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def crossfade_loop(samples: np.ndarray, target_length: int, xfade_samples: int = 1024) -> np.ndarray:
    """Loop a sample to target_length with crossfades at loop boundaries to avoid clicks.

    For vocal events longer than the source sample, this smoothly blends the
    tail of one loop iteration into the head of the next rather than hard-splicing.
    """
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    src = samples.astype(np.float32, copy=False)
    n = src.shape[0]
    if n == 0:
        return np.zeros(target_length, dtype=np.float32)
    if target_length <= n:
        return src[:target_length].copy()

    xfade = min(xfade_samples, n // 4)
    out = np.zeros(target_length, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)

    pos = 0
    while pos < target_length:
        remaining = target_length - pos
        chunk_end = min(n, remaining)
        out[pos : pos + chunk_end] += src[:chunk_end]
        if pos + n < target_length and xfade > 0:
            join_start = pos + n - xfade
            join_end = min(pos + n, target_length)
            blend_len = join_end - join_start
            if blend_len > 0:
                out[join_start:join_end] = (
                    out[join_start:join_end] * fade_out[:blend_len]
                    + src[:blend_len] * fade_in[:blend_len]
                )
            pos = pos + n - xfade
        else:
            pos += chunk_end

    return out[:target_length]


def time_stretch_wsola(
    samples: np.ndarray,
    target_length: int,
    frame_size: int = 1024,
    tolerance: int = 256,
) -> np.ndarray:
    """Pitch-preserving time-scaling via WSOLA (Waveform Similarity Overlap-Add).

    Unlike a naive OLA, WSOLA searches for the analysis frame that best matches
    the *natural continuation* of the previous frame (via normalized
    cross-correlation) before overlap-adding. This keeps successive grains phase
    coherent, which avoids the hollow/buzzy comb-filter artifacts a fixed-hop OLA
    produces. Works for both time compression and expansion.
    """

    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    src = samples.astype(np.float32, copy=False)
    n = src.shape[0]
    if n == 0:
        return np.zeros(target_length, dtype=np.float32)
    if n == target_length:
        return src.copy()

    # Shrink the frame for short samples so WSOLA still has room to work.
    frame_size = int(min(frame_size, max(64, n // 4)))
    if frame_size < 64:
        return time_stretch_linear(src, target_length)
    tolerance = int(min(tolerance, max(0, (n - frame_size) // 2)))

    synth_hop = frame_size // 2
    alpha = target_length / n
    analysis_hop = max(1, int(round(synth_hop / alpha)))
    window = np.hanning(frame_size).astype(np.float32)

    out = np.zeros(target_length + frame_size, dtype=np.float32)
    norm = np.zeros(target_length + frame_size, dtype=np.float32)

    def grab(start: int) -> np.ndarray:
        start = int(start)
        frame = np.zeros(frame_size, dtype=np.float32)
        lo = max(0, start)
        hi = min(n, start + frame_size)
        if hi > lo:
            frame[lo - start : hi - start] = src[lo:hi]
        return frame

    analysis_pos = 0
    out_pos = 0
    while out_pos < target_length:
        frame = grab(analysis_pos) * window
        end = min(out_pos + frame_size, out.shape[0])
        length = end - out_pos
        out[out_pos:end] += frame[:length]
        norm[out_pos:end] += window[:length]
        out_pos += synth_hop
        if out_pos >= target_length:
            break

        natural = grab(analysis_pos + synth_hop)
        nominal = analysis_pos + analysis_hop
        if tolerance > 0:
            lo = nominal - tolerance
            region = np.zeros(2 * tolerance + frame_size, dtype=np.float32)
            r_lo = max(0, lo)
            r_hi = min(n, lo + region.shape[0])
            if r_hi > r_lo:
                region[r_lo - lo : r_hi - lo] = src[r_lo:r_hi]
            corr = np.correlate(region, natural, mode="valid")
            energy = np.sqrt(np.convolve(region * region, np.ones(frame_size, dtype=np.float32), mode="valid"))
            score = corr / (energy + 1e-6)
            best = int(np.argmax(score))
            analysis_pos = int(np.clip(lo + best, 0, max(0, n - 1)))
        else:
            analysis_pos = nominal

    safe_norm = np.where(norm > 1e-6, norm, 1.0)
    out /= safe_norm
    return out[:target_length].astype(np.float32)


def remove_dc(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    return (samples - float(np.mean(samples))).astype(np.float32)


def normalize_peak(samples: np.ndarray, peak: float = 0.95) -> np.ndarray:
    current = float(np.max(np.abs(samples))) if samples.size else 0.0
    if current <= 1e-9:
        return samples.astype(np.float32, copy=False)
    return (samples * (peak / current)).astype(np.float32)


def fade_edges(samples: np.ndarray, fade_samples: int) -> np.ndarray:
    out = samples.astype(np.float32, copy=True)
    n = out.shape[0]
    if n == 0 or fade_samples <= 0:
        return out
    fade = min(fade_samples, n // 2)
    if fade <= 0:
        return out
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[:fade] *= ramp
    out[-fade:] *= ramp[::-1]
    return out


def lowpass_fft(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    """Zero-phase low-pass via FFT with a soft cosine roll-off (numpy-only).

    Low-sample-rate meow recordings (e.g. the 8 kHz CatMeows dataset) carry no real
    content above ~4 kHz, so normalizing them just amplifies hiss into audible grain.
    Removing that band makes the meow clean without touching its real timbre.
    """

    if cutoff_hz <= 0 or samples.size < 4:
        return samples.astype(np.float32, copy=False)
    spectrum = np.fft.rfft(samples.astype(np.float64))
    freqs = np.fft.rfftfreq(samples.shape[0], 1.0 / sample_rate)
    mask = np.ones(freqs.shape[0], dtype=np.float64)
    start = cutoff_hz * 0.85
    taper = (freqs >= start) & (freqs <= cutoff_hz)
    mask[taper] = 0.5 * (1.0 + np.cos(np.pi * (freqs[taper] - start) / max(cutoff_hz - start, 1e-6)))
    mask[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(spectrum * mask, n=samples.shape[0]).astype(np.float32)


def extract_loud_region(samples: np.ndarray, sample_rate: int, *, ratio: float = 0.2) -> np.ndarray:
    """Return the loud 'body' of a sample, discarding quiet/noisy lead-in and tail.

    Many meow recordings have seconds of low-level room noise around the actual
    meow; playing a note from the file start would play that noise instead of the
    meow. This isolates the contiguous region above ``ratio`` of the peak RMS.
    """

    if samples.size == 0:
        return samples
    win = max(1, int(0.03 * sample_rate))
    hop = max(1, int(0.01 * sample_rate))
    starts = np.arange(0, max(1, samples.shape[0] - win), hop)
    rms = np.array([np.sqrt(np.mean(samples[s : s + win] ** 2)) for s in starts])
    if rms.size == 0 or float(rms.max()) <= 1e-9:
        return samples
    active = np.flatnonzero(rms > ratio * float(rms.max()))
    if active.size == 0:
        return samples
    start = max(0, int(starts[active[0]]) - int(0.02 * sample_rate))
    end = min(samples.shape[0], int(starts[active[-1]]) + win + int(0.05 * sample_rate))
    return samples[start:end].astype(np.float32, copy=False)


def brighten(samples: np.ndarray, amount: float) -> np.ndarray:
    """Gentle high-frequency lift (high-shelf) to counter muffled low-rate samples.

    Adds a scaled high-passed copy of the signal back onto itself. The high-pass
    is a simple one-pole difference, so this is cheap and deterministic. `amount`
    is roughly the extra weight given to high frequencies (0 disables it).
    """

    if amount <= 0.0 or samples.size < 2:
        return samples.astype(np.float32, copy=False)
    x = samples.astype(np.float32, copy=False)
    # One-pole high-pass: y[n] = x[n] - a * x[n-1] approximates a shelf tilt.
    high = np.empty_like(x)
    high[0] = 0.0
    high[1:] = x[1:] - x[:-1]
    return (x + float(amount) * high).astype(np.float32)

