"""Importable API for Meowsic."""

from .analysis import detect_syllable_events, estimate_pitch_contour
from .dashboard import launch_dashboard
from .io import load_wav, write_wav
from .pipeline import process_song
from .render import mix_tracks, render_meow_vocal
from .samples import fetch_public_meow_sample, load_meow_sample
from .source import fetch_youtube_audio
from .stems import prepare_stems, run_demucs
from .types import (
    AudioBuffer,
    MeowSample,
    MeowSampleMetadata,
    MeowsicConfig,
    MeowsicResult,
    PitchContour,
    SourceMetadata,
    StemSet,
    SyllableEvent,
)

__all__ = [
    "AudioBuffer",
    "MeowSample",
    "MeowSampleMetadata",
    "MeowsicConfig",
    "MeowsicResult",
    "PitchContour",
    "SourceMetadata",
    "StemSet",
    "SyllableEvent",
    "detect_syllable_events",
    "estimate_pitch_contour",
    "fetch_public_meow_sample",
    "fetch_youtube_audio",
    "load_meow_sample",
    "load_wav",
    "launch_dashboard",
    "mix_tracks",
    "prepare_stems",
    "process_song",
    "render_meow_vocal",
    "run_demucs",
    "write_wav",
]
