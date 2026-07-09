from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import OptionalDependencyError, SourceFetchError
from .types import MeowsicConfig, SourceMetadata


def resolve_source(
    *,
    source_path: str | Path | None = None,
    source_url: str | None = None,
    config: MeowsicConfig | None = None,
) -> SourceMetadata:
    """Resolve local or YouTube source audio to a local path."""

    config = config or MeowsicConfig()
    if source_path and source_url:
        raise ValueError("Provide either source_path or source_url, not both")
    if source_path:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return SourceMetadata(source_type="local_file", local_path=path)
    if source_url:
        return fetch_youtube_audio(source_url, config=config)
    raise ValueError("source_path or source_url is required")


def fetch_youtube_audio(source_url: str, *, config: MeowsicConfig | None = None) -> SourceMetadata:
    """Fetch audio through yt-dlp when explicitly enabled."""

    config = config or MeowsicConfig()
    if not config.enable_youtube_fetch:
        raise SourceFetchError("YouTube fetching was requested but enable_youtube_fetch is False")
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise OptionalDependencyError(
            "yt-dlp is not installed. Install the optional youtube dependencies to enable URL ingestion."
        ) from exc

    cache_dir = Path(config.youtube_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(cache_dir / "%(id)s.%(ext)s")
    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }
    options.update(config.ytdlp_options)

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(source_url, download=True)
            prepared = Path(ydl.prepare_filename(info))
    except Exception as exc:  # yt-dlp raises several concrete runtime errors.
        raise SourceFetchError(f"yt-dlp failed to fetch source audio: {exc}") from exc

    local_path = prepared.with_suffix(".wav")
    if not local_path.exists():
        if prepared.exists():
            local_path = prepared
        else:
            raise SourceFetchError("yt-dlp completed but produced no usable local audio file")

    metadata = SourceMetadata(
        source_type="youtube",
        local_path=local_path,
        original_url=source_url,
        title=info.get("title"),
        extractor=info.get("extractor"),
        retrieval_date=datetime.now(timezone.utc).isoformat(),
    )
    _write_source_metadata(metadata)
    return metadata


def _write_source_metadata(metadata: SourceMetadata) -> None:
    payload = {
        "source_type": metadata.source_type,
        "local_path": str(metadata.local_path),
        "original_url": metadata.original_url,
        "title": metadata.title,
        "extractor": metadata.extractor,
        "retrieval_date": metadata.retrieval_date,
    }
    metadata.local_path.with_suffix(metadata.local_path.suffix + ".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
