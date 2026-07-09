# meowsic

Enjoy your favourite songs sung by a cat.

Meowsic is an importable Python MVP for replacing a song's lead vocal with
sample-based "meow" syllables while preserving the original melody contour and
backing track.

See [PLAN.md](PLAN.md), [SPEC.md](SPEC.md), and [AGENTS.md](AGENTS.md) for the
project contract and maintainer guidance.

## Current Shape

- Importable Python package.
- WAV-first audio I/O.
- User-provided vocal/instrumental stems are supported and preferred when present.
- Demucs can derive vocal and no-vocals stems when stems are omitted.
- User-provided meow WAV samples are supported.
- Public meow sample fetching is explicit and requires license metadata.
- YouTube ingestion is optional through `yt-dlp` and must be explicitly enabled.
- Meow pitch is mapped into a configurable cat-like register while preserving melodic contour.
- Gradio browser dashboard is available; no native GUI toolkit is used.
- Dashboard output can be played back in-browser and downloaded as a WAV.

## Setup

Use the setup script instead of installing dependencies one at a time:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

The script:

- creates `.venv`,
- installs all optional dependency groups,
- installs the tested Windows Demucs stack,
- installs SoundFile so Torchaudio can write Demucs WAV stems,
- upgrades `yt-dlp` to the latest available build.

Verify the environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_env.ps1
```

Expected local stack:

```text
torch: 2.6.0+cpu
torchaudio: 2.6.0+cpu
demucs: 4.0.1
soundfile: installed
yt-dlp: latest available
gradio: installed
torchaudio backends: ['soundfile']
```

## Dashboard

Run:

```powershell
.\.venv\Scripts\python.exe run.py
```

The dashboard uses Gradio and launches a local browser UI. It supports:

- source WAV upload,
- YouTube URL input with explicit `yt-dlp` enablement,
- optional vocal stem WAV,
- optional instrumental stem WAV,
- user-provided meow sample WAV,
- optional public meow sample URL plus license metadata,
- cat-register sliders,
- status output,
- in-browser playback,
- WAV download.

Cat register controls:

- **Lowest meow pitch:** lower values make the cat voice deeper and heavier; higher values keep even low notes more kitten-like.
- **Highest meow pitch:** lower values tame squeaky high notes; higher values allow brighter, sharper meows on melody peaks.
- **Melody contour strength:** lower values flatten the tune toward one cat register; higher values follow more of the song's original ups and downs.

## Library Usage

Provided stems:

```python
from meowsic import MeowsicConfig, process_song

result = process_song(
    source_path="song.wav",
    vocal_path="vocals.wav",
    instrumental_path="instrumental.wav",
    meow_sample_path="meow.wav",
    output_path="song-meowsic.wav",
    config=MeowsicConfig(overwrite=True),
)
```

Demucs-derived stems:

```python
result = process_song(
    source_path="song.wav",
    meow_sample_path="meow.wav",
    output_path="song-meowsic.wav",
    config=MeowsicConfig(
        overwrite=True,
        enable_demucs=True,
        demucs_device="cpu",
    ),
)
```

YouTube ingestion:

```python
result = process_song(
    source_url="https://www.youtube.com/watch?v=...",
    meow_sample_path="meow.wav",
    output_path="song-meowsic.wav",
    config=MeowsicConfig(
        overwrite=True,
        enable_youtube_fetch=True,
    ),
)
```

Only fetch tracks you have rights to download and transform, and comply with
platform terms.

Cat-register tuning:

```python
MeowsicConfig(
    cat_min_pitch_hz=220,
    cat_max_pitch_hz=520,
    cat_pitch_contour_strength=0.75,
)
```

## Optional Dependencies

Manual install, if you are not using `scripts/setup.ps1`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[demucs,youtube,dashboard,dev]"
.\.venv\Scripts\python.exe -m pip install --upgrade --upgrade-strategy eager yt-dlp
.\.venv\Scripts\python.exe -m pip install "torch==2.6.0" "torchaudio==2.6.0" "soundfile>=0.12"
```

Notes:

- `yt-dlp` should be kept current because provider extraction can break as sites change.
- On Windows, Demucs/Torchaudio needs the SoundFile backend for reliable WAV stem output.
- Avoid installing a much newer TorchAudio against an older Torch. That can trigger TorchCodec save paths and DLL errors.

## Tests And Validation

Run unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Current suite covers:

- WAV load/write round trip,
- provided-stem full pipeline,
- mocked Demucs adapter,
- mocked `yt-dlp` adapter,
- public meow sample fetch via local file URL,
- cat-register pitch mapping.

We also validated generated local E2E renders under ignored `.meowsic-dashboard/` workspace:

- provided-stems path produced a non-silent stereo WAV,
- Demucs path produced a non-silent stereo WAV with `stem_strategy='demucs_two_stems_vocals'`.

Do not commit generated audio, downloaded tracks, model files, dashboard cache, or `.venv`.
