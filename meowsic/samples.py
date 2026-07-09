from __future__ import annotations

import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .errors import SampleFetchError
from .io import load_wav
from .types import MeowSample, MeowSampleMetadata, MeowsicConfig

ALLOWED_SAMPLE_LICENSES = {"CC0", "CC-BY", "CC BY", "Creative Commons 0", "Creative Commons Attribution"}


def load_meow_sample(
    path: str | Path,
    *,
    license: str | None = None,
    attribution: str | None = None,
) -> MeowSample:
    """Load a user-provided meow sample."""

    path = Path(path)
    audio = load_wav(path)
    return MeowSample(
        audio=audio,
        metadata=MeowSampleMetadata(
            local_path=path,
            source_type="user_provided",
            license=license,
            attribution=attribution,
        ),
    )


def resolve_meow_sample(
    *,
    meow_sample_path: str | Path | None = None,
    config: MeowsicConfig | None = None,
) -> MeowSample:
    config = config or MeowsicConfig()
    if meow_sample_path:
        return load_meow_sample(
            meow_sample_path,
            license=config.meow_sample_license,
            attribution=config.meow_sample_attribution,
        )
    return fetch_public_meow_sample(config=config)


def fetch_public_meow_sample(*, config: MeowsicConfig | None = None) -> MeowSample:
    """Fetch a public meow sample when explicitly enabled.

    The MVP requires an explicit URL and compatible license metadata. Automated
    search can be layered on top later for specific provider APIs.
    """

    config = config or MeowsicConfig()
    if not config.enable_public_sample_fetch:
        raise SampleFetchError("No meow sample was provided and public sample fetching is disabled")
    if not config.meow_sample_url:
        raise SampleFetchError("Public sample fetching requires meow_sample_url")
    if not config.meow_sample_license:
        raise SampleFetchError("Public sample fetching requires meow_sample_license metadata")
    if not _is_allowed_license(config.meow_sample_license):
        raise SampleFetchError(f"Incompatible meow sample license: {config.meow_sample_license}")

    cache_dir = Path(config.meow_sample_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(config.meow_sample_url)
    filename = Path(parsed.path).name or "meow_sample.wav"
    local_path = cache_dir / filename
    if local_path.suffix.lower() != ".wav":
        raise SampleFetchError("MVP public sample fetching currently requires a direct WAV URL")

    if not local_path.exists():
        try:
            with urllib.request.urlopen(config.meow_sample_url, timeout=30) as response:
                with local_path.open("wb") as destination:
                    shutil.copyfileobj(response, destination)
        except Exception as exc:
            raise SampleFetchError(f"Failed to fetch meow sample: {exc}") from exc

    metadata = MeowSampleMetadata(
        local_path=local_path,
        source_type="fetched_public_resource",
        source_url=config.meow_sample_url,
        license=config.meow_sample_license,
        attribution=config.meow_sample_attribution,
        retrieval_date=datetime.now(timezone.utc).isoformat(),
    )
    _write_sample_metadata(metadata)
    return MeowSample(audio=load_wav(local_path), metadata=metadata)


def _is_allowed_license(license_name: str) -> bool:
    normalized = license_name.strip().upper().replace("_", "-")
    return any(normalized == allowed.upper().replace("_", "-") for allowed in ALLOWED_SAMPLE_LICENSES)


def _write_sample_metadata(metadata: MeowSampleMetadata) -> None:
    payload = {
        "local_path": str(metadata.local_path),
        "source_type": metadata.source_type,
        "source_url": metadata.source_url,
        "license": metadata.license,
        "attribution": metadata.attribution,
        "retrieval_date": metadata.retrieval_date,
    }
    metadata.local_path.with_suffix(metadata.local_path.suffix + ".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
