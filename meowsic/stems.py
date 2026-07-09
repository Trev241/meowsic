from __future__ import annotations

from pathlib import Path

import numpy as np

from .dsp import ensure_sample_rate, match_length, to_channels
from .errors import OptionalDependencyError, StemSeparationError
from .io import load_wav
from .types import AudioBuffer, MeowsicConfig, StemSet


def prepare_stems(
    source: AudioBuffer,
    *,
    source_path: str | Path | None = None,
    vocal: AudioBuffer | None = None,
    instrumental: AudioBuffer | None = None,
    config: MeowsicConfig | None = None,
) -> StemSet:
    """Prepare vocal and instrumental stems, preferring user-provided stems."""

    config = config or MeowsicConfig()
    if vocal is not None or instrumental is not None:
        return _provided_or_difference_stems(source, vocal=vocal, instrumental=instrumental)
    if not config.enable_demucs:
        raise StemSeparationError("No stems were provided and Demucs is disabled")
    if source_path is None:
        raise StemSeparationError("No stems were provided and source_path is required for Demucs")
    return run_demucs(source_path, config=config)


def run_demucs(source_path: str | Path, *, config: MeowsicConfig | None = None) -> StemSet:
    """Run Demucs in two-stem vocals mode and load the resulting stems."""

    config = config or MeowsicConfig()
    try:
        from demucs.separate import main as demucs_main  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "Demucs is not installed. Install the optional demucs dependencies or provide stems."
        ) from exc

    source_path = Path(source_path)
    output_dir = Path(config.demucs_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--two-stems",
        "vocals",
        "-n",
        config.demucs_model,
        "-o",
        str(output_dir),
        str(source_path),
    ]
    if config.demucs_device:
        args[0:0] = ["-d", config.demucs_device]

    try:
        demucs_main(args)
    except Exception as exc:
        message = str(exc)
        if "TorchCodec is required" in message or "save_with_torchcodec" in message:
            raise StemSeparationError(
                "Demucs failed while saving stems because TorchCodec is missing. "
                "Install it with `python -m pip install torchcodec`, or reinstall this project with "
                "`python -m pip install -e .[demucs]`."
            ) from exc
        raise StemSeparationError(f"Demucs failed to separate stems: {exc}") from exc

    stem_dir = output_dir / config.demucs_model / source_path.stem
    vocal_path = stem_dir / "vocals.wav"
    instrumental_path = stem_dir / "no_vocals.wav"
    if not vocal_path.exists() or not instrumental_path.exists():
        raise StemSeparationError(f"Demucs output was incomplete in {stem_dir}")

    return StemSet(
        vocal=load_wav(vocal_path),
        instrumental=load_wav(instrumental_path),
        strategy="demucs_two_stems_vocals",
        metadata={"vocal_path": str(vocal_path), "instrumental_path": str(instrumental_path)},
    )


def _provided_or_difference_stems(
    source: AudioBuffer,
    *,
    vocal: AudioBuffer | None,
    instrumental: AudioBuffer | None,
) -> StemSet:
    if vocal is not None and instrumental is not None:
        ensure_sample_rate(vocal, instrumental)
        length = max(vocal.samples.shape[0], instrumental.samples.shape[0])
        channels = max(vocal.channels, instrumental.channels)
        return StemSet(
            vocal=AudioBuffer(to_channels(match_length(vocal.samples, length), channels), vocal.sample_rate),
            instrumental=AudioBuffer(
                to_channels(match_length(instrumental.samples, length), channels),
                instrumental.sample_rate,
            ),
            strategy="provided_vocal_and_instrumental",
        )

    if vocal is not None:
        ensure_sample_rate(source, vocal)
        length = max(source.samples.shape[0], vocal.samples.shape[0])
        channels = max(source.channels, vocal.channels)
        vocal_samples = to_channels(match_length(vocal.samples, length), channels)
        source_samples = to_channels(match_length(source.samples, length), channels)
        return StemSet(
            vocal=AudioBuffer(vocal_samples, source.sample_rate),
            instrumental=AudioBuffer(source_samples - vocal_samples, source.sample_rate),
            strategy="provided_vocal_source_minus_vocal",
        )

    if instrumental is not None:
        ensure_sample_rate(source, instrumental)
        length = max(source.samples.shape[0], instrumental.samples.shape[0])
        channels = max(source.channels, instrumental.channels)
        instrumental_samples = to_channels(match_length(instrumental.samples, length), channels)
        source_samples = to_channels(match_length(source.samples, length), channels)
        return StemSet(
            vocal=AudioBuffer(source_samples - instrumental_samples, source.sample_rate),
            instrumental=AudioBuffer(instrumental_samples, source.sample_rate),
            strategy="provided_instrumental_source_minus_instrumental",
        )

    raise StemSeparationError("No stems were provided")


def center_channel_fallback(source: AudioBuffer) -> StemSet:
    """Low-quality helper for tests/experiments; not used by default pipeline."""

    if source.channels < 2:
        return StemSet(
            vocal=source,
            instrumental=AudioBuffer(np.zeros_like(source.samples), source.sample_rate),
            strategy="mono_no_separation_fallback",
        )
    mid = source.samples.mean(axis=1, keepdims=True)
    vocal = np.repeat(mid, source.channels, axis=1)
    return StemSet(
        vocal=AudioBuffer(vocal, source.sample_rate),
        instrumental=AudioBuffer(source.samples - vocal, source.sample_rate),
        strategy="center_channel_fallback",
    )
