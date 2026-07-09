from __future__ import annotations

from pathlib import Path

from .analysis import detect_syllable_events, estimate_pitch_contour
from .errors import MeowsicError
from .io import load_wav, write_wav
from .render import mix_tracks, render_meow_vocal
from .samples import resolve_meow_sample
from .source import resolve_source
from .stems import prepare_stems
from .types import MeowsicConfig, MeowsicResult


def process_song(
    source_path: str | Path | None = None,
    output_path: str | Path = "meowsic-output.wav",
    *,
    source_url: str | None = None,
    vocal_path: str | Path | None = None,
    instrumental_path: str | Path | None = None,
    meow_sample_path: str | Path | None = None,
    config: MeowsicConfig | None = None,
) -> MeowsicResult:
    """Run the full Meowsic MVP pipeline from Python."""

    config = config or MeowsicConfig()
    source_metadata = resolve_source(source_path=source_path, source_url=source_url, config=config)
    source = load_wav(source_metadata.local_path)
    vocal = load_wav(vocal_path) if vocal_path is not None else None
    instrumental = load_wav(instrumental_path) if instrumental_path is not None else None
    stems = prepare_stems(
        source,
        source_path=source_metadata.local_path,
        vocal=vocal,
        instrumental=instrumental,
        config=config,
    )
    sample = resolve_meow_sample(meow_sample_path=meow_sample_path, config=config)
    contour = estimate_pitch_contour(stems.vocal, config)
    events = detect_syllable_events(contour, config)
    if not events:
        raise MeowsicError("No detectable vocal events were found")

    duration = max(stems.vocal.duration, stems.instrumental.duration)
    meow_vocal = render_meow_vocal(
        events,
        contour,
        sample,
        sample_rate=stems.vocal.sample_rate,
        duration=duration,
        config=config,
    )
    mixed = mix_tracks(stems.instrumental, meow_vocal, config)
    written = write_wav(output_path, mixed, overwrite=config.overwrite)
    return MeowsicResult(
        output_path=written,
        sample_rate=mixed.sample_rate,
        duration=mixed.duration,
        event_count=len(events),
        stem_strategy=stems.strategy,
        meow_sample=sample.metadata,
        source=source_metadata,
    )
