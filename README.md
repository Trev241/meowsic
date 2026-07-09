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
- User-provided meow WAV samples are supported (default: `B_CAN01_EU_FN_GIA01_205.wav`).
- Public meow sample fetching is explicit and requires license metadata.
- YouTube ingestion is optional through `yt-dlp` and must be explicitly enabled.
- Default **note engine** (the "cat cover" formula): the melody is segmented into
  discrete, semitone-quantized notes (octave-transposed into the cat register so it
  stays in key), and one whole recognizable meow is rendered per note — pitch-shifted
  to the note and fit to its duration (compress short notes / loop long ones).
- Meow samples are cleaned for rendering: low-passed to remove HF hiss/grain, and the
  loud meow "body" is isolated so notes play the actual meow, not a noisy lead-in.
- Pitch shifting is `resample` (brighter, cat-cover feel) or `psola` (formant-preserving,
  needs the optional `pytsmod` extra).
- Two experimental engines remain available: `instrument` (continuous TD-PSOLA glide)
  and `granular` (per-onset splicing).
- `evaluate_render` gives an objective **MeowScore** — useful as a *guide* only; trust
  your ears for the final call (density metrics do not capture "catness").
- Gradio browser dashboard: pick/preview/upload meow samples, tune the engine, see a
  melody/rhythm plot, play back, and download the WAV.

## Setup

Use the `Makefile` (recommended):

```powershell
make setup     # creates .venv and installs all extras + the tested Windows Demucs stack
make verify    # prints installed versions and torchaudio backends
make test      # runs the test suite
make run       # launches the dashboard
```

Or install directly without `make`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[demucs,youtube,dashboard,psola,dev]"
.\.venv\Scripts\python.exe -m pip install --upgrade --upgrade-strategy eager yt-dlp
.\.venv\Scripts\python.exe -m pip install "torch==2.6.0" "torchaudio==2.6.0" "soundfile>=0.12"
```

This creates `.venv`, installs all optional dependency groups (including `pytsmod`
for formant-preserving pitch shifting), installs the tested Windows Demucs stack and
SoundFile, and upgrades `yt-dlp`.

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
.\.venv\Scripts\python.exe app.py
```

The dashboard uses Gradio and launches a local browser UI with a two-column layout
(inputs on the left, results on the right) and a floating System Health panel. It supports:

- source WAV upload,
- YouTube URL input with explicit `yt-dlp` enablement,
- optional vocal/instrumental stem WAVs,
- a cached meow-sample picker (defaults to `B_CAN01_EU_FN_GIA01_205.wav`) with preview,
  custom upload, and optional public sample URL plus license metadata,
- meow-voice controls (brightness, tonal core, full-syllable),
- cat-register sliders,
- rhythm & detection controls (event mode, onset sensitivity, re-articulation),
- mix-level controls,
- a MeowScore quality card plus a melody & rhythm tracking plot,
- extracted vocal stem, in-browser playback, and WAV download.

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

Cat-register and voice-quality tuning:

```python
MeowsicConfig(
    cat_min_pitch_hz=220,
    cat_max_pitch_hz=520,
    cat_pitch_contour_strength=0.75,
    # Voice quality
    meow_brightness=0.35,       # 0 disables; lifts highs to fight the muffled 8 kHz source
    meow_full_syllable=False,   # True = time-scale a whole meow into every note
    meow_max_stretch=2.5,       # held notes beyond this loop the meow instead of stretching
)
```

Note engine tuning (the default):

```python
MeowsicConfig(
    render_mode="notes",           # "notes" (default) | "instrument" | "granular"
    pitch_shift_method="resample", # "resample" (brighter) | "psola" (formant-preserving)
    note_fit="compress",           # "compress" (whole meow scaled) | "trim" (onset)
    note_min_duration=0.09,        # drop notes shorter than this
    note_merge_duration=0.22,      # merge short notes so each meow is recognizable
    meow_lowpass_hz=3800,          # remove HF grain from low-rate meows (0 = off)
    cat_min_pitch_hz=220,
    cat_max_pitch_hz=520,          # notes are octave-transposed into this range
)
```

Rhythm/detection tuning (only used by the experimental `instrument`/`granular` engines):

```python
MeowsicConfig(
    event_mode="onset",
    onset_threshold=0.35,
    onset_min_interval=0.07,
    rearticulate=True,
    rearticulate_interval=0.28,
    min_event_duration=0.06,
)
```

How the note engine works (based on the established "cat cover" recipe):

1. Estimate the vocal's pitch contour, segment it into **discrete notes**, and snap each
   note to the nearest **semitone** so the result is in tune.
2. **Octave-transpose** the melody into the cat register (`cat_min/max_pitch_hz`) — this
   preserves the tune's intervals rather than compressing/detuning them.
3. Clean the meow sample: **low-pass** (`meow_lowpass_hz`) to kill HF grain and isolate
   the loud meow **body** so notes play the actual meow, not a noisy lead-in.
4. For each note, pitch-shift one whole meow to the note and fit it to the duration:
   `compress` scales the complete meow gesture (stays recognizable on fast notes);
   `trim` plays just the onset; long notes loop.

Sample choice matters: the bundled CatMeows samples are 8 kHz and muffled. A clean
44.1 kHz meow with a clear vowel sounds noticeably better — use the dashboard picker to
try your own. Override the default with `default_meow_sample_name`.

## Measuring Quality (MeowScore)

The pipeline goal is that the meow vocal tracks the original lead vocal. `evaluate_render`
scores this objectively so parameter changes can be compared instead of only judged by ear:

```python
from meowsic import evaluate_render, process_song, MeowsicConfig

result = process_song(source_path="song.wav", vocal_path="vocals.wav",
                      instrumental_path="instrumental.wav",
                      output_path="out.wav", config=MeowsicConfig(overwrite=True))
quality = evaluate_render(result.vocal_stem, result.meow_vocal)
print(quality.as_dict())
# meow_score in [0,1] = 0.5*rhythm(onset F-measure) + 0.3*melody(pitch corr) + 0.2*dynamics(envelope corr)
```

- **Rhythm** – onset F-measure between the meows and the vocal note attacks.
- **Melody** – correlation of the meow's log-F0 contour with the vocal's.
- **Dynamics** – correlation of the loudness (RMS) envelopes.

The dashboard displays the MeowScore and its breakdown after every render.

## Optional Dependencies

Manual install, if you are not using the `Makefile`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[demucs,youtube,dashboard,dev]"
.\.venv\Scripts\python.exe -m pip install --upgrade --upgrade-strategy eager yt-dlp
.\.venv\Scripts\python.exe -m pip install "torch==2.6.0" "torchaudio==2.6.0" "soundfile>=0.12"
```

Notes:

- `yt-dlp` should be kept current because provider extraction can break as sites change.
- On Windows, Demucs/Torchaudio needs the SoundFile backend for reliable WAV stem output.
- Avoid installing a much newer TorchAudio against an older Torch. That can trigger TorchCodec save paths and DLL errors.

### FFmpeg Dependency (For YouTube Ingestion)

If you plan to use `yt-dlp` to fetch audio directly from YouTube URLs, you must install `ffmpeg` globally on your system.

**Windows:**
The easiest way is using Winget:
```powershell
winget install -e --id Gyan.FFmpeg
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt update
sudo apt install ffmpeg
```

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
- cat-register pitch mapping,
- WSOLA pitch-preserving time-stretch and sample brightening,
- onset detection and onset-vs-energy event density,
- MeowScore self-comparison sanity check.

We also validated generated local E2E renders under ignored `.meowsic-dashboard/` workspace:

- provided-stems path produced a non-silent stereo WAV,
- Demucs path produced a non-silent stereo WAV with `stem_strategy='demucs_two_stems_vocals'`.

Do not commit generated audio, downloaded tracks, model files, dashboard cache, or `.venv`.
