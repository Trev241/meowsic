# meowsic

Enjoy your favourite songs sung by a cat.

Meowsic is an importable Python MVP for replacing a song's lead vocal with
sample-based "meow" syllables while preserving the original melody and backing
track.

See [PLAN.md](PLAN.md) and [SPEC.md](SPEC.md) for the product and implementation
contract.

## Current Shape

- Importable Python package.
- WAV-first audio I/O.
- User-provided vocal/instrumental stems are supported.
- Demucs is supported as an optional stem backend when installed.
- User-provided meow WAV samples are supported.
- Public meow sample fetching is explicit and requires license metadata.
- YouTube ingestion is optional through `yt-dlp` and must be explicitly enabled.
- Local browser dashboard is available; no native GUI toolkit is used.

## Library Usage

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

If stems are omitted, Meowsic will try Demucs when available:

```python
result = process_song(
    source_path="song.wav",
    meow_sample_path="meow.wav",
    output_path="song-meowsic.wav",
    config=MeowsicConfig(overwrite=True, enable_demucs=True),
)
```

YouTube ingestion requires `yt-dlp`, explicit enablement, and user responsibility
for rights/platform-term compliance:

```python
result = process_song(
    source_url="https://www.youtube.com/watch?v=...",
    meow_sample_path="meow.wav",
    output_path="song-meowsic.wav",
    config=MeowsicConfig(overwrite=True, enable_youtube_fetch=True),
)
```

## Browser Dashboard

```python
from meowsic import launch_dashboard

launch_dashboard()
```

Then open the printed local URL in a browser.

## Optional Dependencies

```powershell
pip install .[demucs]
pip install .[youtube]
pip install .[dashboard]
```

If Demucs fails with `TorchCodec is required for save_with_torchcodec`, your
TorchAudio version is probably too new for your installed PyTorch version. For
the local PyTorch 2.6 line, install the matching TorchAudio line:

```powershell
python -m pip install "torchaudio>=2.6,<2.7"
```

On Windows, Torchaudio may also need the SoundFile backend to write Demucs WAV
stems:

```powershell
python -m pip install soundfile
```

This project currently pins the Demucs extra for the PyTorch 2.6 line used in
local testing. If pip cannot resolve that cleanly, reinstall the Demucs extra:

```powershell
python -m pip install -e ".[demucs]"
```

## Tests

```powershell
python -m unittest discover -s tests
```
