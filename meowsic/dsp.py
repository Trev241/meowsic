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


def time_stretch_ola(
    samples: np.ndarray,
    target_length: int,
    frame_size: int = 512,
    hop_size: int = 128,
) -> np.ndarray:
    """OLA (Overlap-Add) time-stretching — better quality than linear interpolation.

    Preserves pitch while changing duration, avoiding the robotic sound of
    linear resampling when stretching beyond 2x.
    """
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    src = samples.astype(np.float32, copy=False)
    n = src.shape[0]
    if n == 0:
        return np.zeros(target_length, dtype=np.float32)
    if n == target_length:
        return src.copy()

    ratio = target_length / n
    output_hop = max(1, int(round(hop_size * ratio)))
    window = np.hanning(frame_size).astype(np.float32)
    out = np.zeros(target_length + frame_size, dtype=np.float32)
    norm = np.zeros(target_length + frame_size, dtype=np.float32)

    out_pos = 0
    src_pos = 0
    while out_pos < target_length:
        src_start = int(round(src_pos)) % max(1, n - frame_size + 1) if n > frame_size else 0
        src_end = src_start + frame_size
        if src_end <= n:
            frame = src[src_start:src_end] * window
        else:
            frame_data = np.zeros(frame_size, dtype=np.float32)
            available = n - src_start
            if available > 0:
                frame_data[:available] = src[src_start:]
            frame = frame_data * window
        end = min(out_pos + frame_size, out.shape[0])
        length = end - out_pos
        out[out_pos:end] += frame[:length]
        norm[out_pos:end] += window[:length]
        out_pos += output_hop
        src_pos += hop_size

    safe_norm = np.where(norm > 1e-6, norm, 1.0)
    out /= safe_norm
    return out[:target_length]

