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
