from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class AudioBuffer:
    """Floating-point audio with shape (samples, channels)."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.ndim != 2:
            raise ValueError("AudioBuffer samples must be a 1D or 2D array")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "samples", samples)

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        return float(self.samples.shape[0] / self.sample_rate)

    def mono(self) -> np.ndarray:
        return self.samples.mean(axis=1).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SourceMetadata:
    source_type: Literal["local_file", "youtube"]
    local_path: Path
    original_url: str | None = None
    title: str | None = None
    extractor: str | None = None
    retrieval_date: str | None = None


@dataclass(frozen=True)
class StemSet:
    vocal: AudioBuffer
    instrumental: AudioBuffer
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PitchContour:
    times: np.ndarray
    f0_hz: np.ndarray
    voiced: np.ndarray
    energy: np.ndarray


@dataclass(frozen=True)
class SyllableEvent:
    start: float
    end: float
    energy: float
    pitch_hz: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class MeowSampleMetadata:
    local_path: Path
    source_type: Literal["user_provided", "fetched_public_resource", "default_zenodo", "mocked"]
    source_url: str | None = None
    license: str | None = None
    attribution: str | None = None
    retrieval_date: str | None = None


@dataclass(frozen=True)
class MeowSample:
    audio: AudioBuffer
    metadata: MeowSampleMetadata


@dataclass(frozen=True)
class MeowsicConfig:
    frame_length: int = 2048
    hop_length: int = 512
    min_pitch_hz: float = 75.0
    max_pitch_hz: float = 1200.0
    min_event_duration: float = 0.08
    max_event_duration: float = 5.0
    energy_threshold_ratio: float = 0.22
    cat_min_pitch_hz: float = 220.0
    cat_max_pitch_hz: float = 520.0
    cat_pitch_contour_strength: float = 0.75
    meow_gain: float = 0.85
    instrumental_gain: float = 0.9
    output_gain: float = 0.95
    overwrite: bool = False
    enable_demucs: bool = True
    demucs_model: str = "htdemucs"
    demucs_device: str | None = None
    demucs_cache_dir: Path = Path("cache/demucs")
    enable_youtube_fetch: bool = False
    youtube_cache_dir: Path = Path("cache/youtube")
    ytdlp_options: dict[str, Any] = field(default_factory=dict)
    enable_public_sample_fetch: bool = True
    meow_sample_url: str | None = None
    meow_sample_license: str | None = None
    meow_sample_attribution: str | None = None
    meow_sample_cache_dir: Path = Path("cache/meow_samples")


@dataclass(frozen=True)
class MeowsicResult:
    output_path: Path
    sample_rate: int
    duration: float
    event_count: int
    stem_strategy: str
    meow_sample: MeowSampleMetadata
    source: SourceMetadata
    vocal_stem: AudioBuffer | None = None
    pitch_contour: PitchContour | None = None

