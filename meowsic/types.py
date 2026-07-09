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
    min_event_duration: float = 0.06
    max_event_duration: float = 5.0
    energy_threshold_ratio: float = 0.15
    # Event detection. "onset" segments the vocal at note attacks (spectral flux)
    # so the meows track the song's rhythm; "energy" is the older voiced-region mode.
    event_mode: str = "onset"
    onset_frame_length: int = 1024
    onset_hop_length: int = 256
    # Higher onset_threshold = fewer, stronger onsets; lower = more, busier meows.
    onset_threshold: float = 0.35
    onset_min_interval: float = 0.07
    # Re-articulation: a held note longer than ~rearticulate_interval is re-struck
    # as repeated meows so sustained singing stays upbeat instead of one long meow.
    rearticulate: bool = True
    rearticulate_interval: float = 0.28
    cat_min_pitch_hz: float = 220.0
    cat_max_pitch_hz: float = 520.0
    cat_pitch_contour_strength: float = 0.75
    meow_gain: float = 0.85
    # Rendering quality controls.
    # meow_brightness: gentle high-shelf lift (0..1) to counter the muffled sound
    # of low-sample-rate meow recordings (the CatMeows dataset is 8 kHz).
    meow_brightness: float = 0.35
    # meow_full_syllable: when True, time-scale the entire meow into every note so
    # each note is a complete "meow"; when False, short notes play the natural
    # meow onset (steadier, closer to the original behaviour).
    meow_full_syllable: bool = False
    # meow_use_core: render from the meow's stable, steadily-pitched vowel "core"
    # instead of the raw sample. This makes short notes actually land on the target
    # pitch (the raw onset glides through many pitches), improving melody tracking.
    meow_use_core: bool = True
    # Fraction of median pitch a frame may deviate and still count as "core".
    meow_core_tolerance: float = 0.35
    # Length of the click-avoidance attack/release on the core (seconds).
    meow_core_attack: float = 0.012
    # Rendering engine:
    #   "notes"      - the cat-cover formula: melody -> discrete semitone-quantized
    #                  notes, one whole recognizable meow per note (the default).
    #   "instrument" - continuous TD-PSOLA glide (experimental; can sound like a drone).
    #   "granular"   - per-onset splicing (experimental; can sound chopped).
    render_mode: str = "notes"
    # Note engine controls.
    meow_lowpass_hz: float = 3800.0        # remove HF grain/hiss (0 disables)
    note_min_duration: float = 0.09        # drop notes shorter than this
    note_merge_duration: float = 0.22      # merge notes shorter than this into neighbours
    note_fit: str = "compress"             # "compress" (whole meow scaled) or "trim" (onset)
    pitch_shift_method: str = "resample"   # "resample" (brighter/cat-cover) or "psola" (formant-preserving)
    # Portamento: milliseconds of smoothing applied to the target melody contour so
    # the cat glides between notes instead of stepping.
    portamento_ms: float = 45.0
    # Articulation: how much the loudness dips at each note onset inside a phrase
    # (0 = pure legato, 1 = fully re-struck). Keeps rhythm without silent gaps.
    articulation_depth: float = 0.45
    # Unvoiced gap (seconds) bridged within a phrase; longer gaps split phrases so
    # the cat "breathes" where the singer does.
    phrase_gap: float = 0.16
    # Blend the meow's natural onset ("m") at phrase starts so it reads as a meow.
    attack_blend: bool = True
    # Adaptive per-song calibration (auto-centre the cat register on the vocal).
    adaptive: bool = True
    # Stretch factor beyond which a held note loops the meow instead of stretching
    # a single one (avoids an unnaturally slow, drawn-out meow on long holds).
    meow_max_stretch: float = 2.5
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
    default_meow_sample_name: str = "B_CAN01_EU_FN_GIA01_205.wav"
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
    meow_vocal: AudioBuffer | None = None
    events: list[SyllableEvent] = field(default_factory=list)

