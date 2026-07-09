from __future__ import annotations

import io
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .errors import SampleFetchError
from .io import load_wav
from .types import MeowSample, MeowSampleMetadata, MeowsicConfig

ALLOWED_SAMPLE_LICENSES = {"CC0", "CC-BY", "CC BY", "Creative Commons 0", "Creative Commons Attribution"}

# How many individual named samples to extract from the default dataset
_DEFAULT_SAMPLE_COUNT = 12


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
    if config.meow_sample_url:
        return fetch_public_meow_sample(config=config)
    return fetch_default_meow(config=config)


def list_cached_samples(config: MeowsicConfig | None = None) -> list[Path]:
    """Return all cached meow WAV files available for selection."""
    config = config or MeowsicConfig()
    cache_dir = Path(config.meow_sample_cache_dir)
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("*.wav"))


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


def fetch_default_meow(config: MeowsicConfig | None = None) -> MeowSample:
    """Fetch the default CatMeows samples from Zenodo.

    Extracts up to _DEFAULT_SAMPLE_COUNT named WAV files from the dataset archive
    into the cache dir so the user can browse and pick from them in the dashboard.
    Returns the first sample as the default.
    """
    config = config or MeowsicConfig()
    cache_dir = Path(config.meow_sample_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = "https://zenodo.org/api/records/4008297/files/dataset.zip/content"
    marker = cache_dir / ".zenodo_fetched"

    preferred_name = config.default_meow_sample_name

    if not marker.exists():
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                wav_files = [n for n in z.namelist() if n.lower().endswith(".wav")]
                if not wav_files:
                    raise SampleFetchError("No WAV files found in default Zenodo dataset")
                # Make sure the preferred (uniform) meow is extracted even if it is
                # not among the first entries of the archive.
                selected = wav_files[:_DEFAULT_SAMPLE_COUNT]
                preferred_entry = next(
                    (n for n in wav_files if Path(n).name == preferred_name), None
                )
                if preferred_entry and preferred_entry not in selected:
                    selected.append(preferred_entry)
                for entry in selected:
                    name = Path(entry).name
                    dest = cache_dir / name
                    if not dest.exists():
                        with z.open(entry) as src, dest.open("wb") as tgt:
                            shutil.copyfileobj(src, tgt)
            marker.write_text(url, encoding="utf-8")
        except SampleFetchError:
            raise
        except Exception as exc:
            raise SampleFetchError(f"Failed to fetch default Zenodo meow: {exc}") from exc

    cached = sorted(cache_dir.glob("*.wav"))
    if not cached:
        raise SampleFetchError("Zenodo samples were not found in cache after fetch")
    # Prefer the configured uniform meow; fall back to the first cached sample.
    preferred_path = cache_dir / preferred_name
    local_path = preferred_path if preferred_path.exists() else cached[0]

    metadata = MeowSampleMetadata(
        local_path=local_path,
        source_type="default_zenodo",
        source_url=url,
        license="CC-BY 4.0",
        attribution="CatMeows Dataset (Zenodo 4008297)",
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
