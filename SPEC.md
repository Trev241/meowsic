# Meowsic MVP Specification

## 1. Purpose

Meowsic is an importable Python audio-processing package that transforms a song so the lead vocal is replaced with cat-like "meow" syllables while preserving the original song's recognizable melody, rhythm, phrasing, and instrumental backing.

The MVP prioritizes a practical, inspectable pipeline over perfect realism. It should produce musically identifiable results when supplied with reasonably clean vocal and instrumental material, whether those stems are user-provided or derived with Demucs.

## 2. Product Scope

### In Scope

- A Python package that can be imported and used by another application.
- WAV-focused audio loading and writing.
- Optional YouTube track ingestion through `yt-dlp`.
- Use of provided vocal and instrumental stems when available.
- Built-in stem separation through Demucs when stems are not provided.
- A documented failure path when Demucs is unavailable.
- Vocal pitch and energy analysis.
- Syllable-like event detection from timing, pitch, and energy.
- Meow vocal rendering from user-provided or public-licensed meow sample audio.
- Public meow sample fetching with license metadata.
- Mixdown of the rendered meow vocal with the instrumental track.
- Optional Gradio-based browser dashboard with Visualization, System Health, and Stem outputs.
- Dashboard audio playback and output WAV download.
- Setup and environment verification scripts.

### Out of Scope

- Mandatory CLI interface.
- Unapproved model downloads.
- Automatic YouTube downloads without explicit user action.
- Unlicensed or unclearly licensed bundled meow samples.
- Real-time audio processing.
- Bundled copyrighted songs or large model artifacts.
- Celebrity, artist, or protected-identity voice imitation.
- Guaranteed clean source separation from arbitrary mixed tracks.
- Training a neural singing voice conversion model from scratch.
- Synthetic meow generation.

## 3. User Workflows

### 3.1 Library Workflow

An application imports Meowsic, calls a high-level processing function with source audio or a YouTube URL, optional stems, and optional meow sample audio, and receives a result object containing output metadata.

Expected shape:

```python
from meowsic import process_song

result = process_song(
    source_path="song.wav",
    output_path="song-meow.wav",
    vocal_path="vocals.wav",
    instrumental_path="instrumental.wav",
    meow_sample_path="meow.wav",
)
```

Expected YouTube-backed shape:

```python
from meowsic import process_song

result = process_song(
    source_url="https://www.youtube.com/watch?v=...",
    output_path="song-meow.wav",
    meow_sample_path="meow.wav",
    enable_youtube_fetch=True,
)
```

If stems are omitted, the high-level function may run Demucs, provided the Demucs integration has been installed and configured.

If `meow_sample_path` is omitted, the high-level function may fetch a sample from an approved public source, provided fetching is explicitly enabled by the caller.

### 3.2 Browser Dashboard Workflow

A user launches the optional local browser dashboard, selects or enters:

- source WAV,
- YouTube URL,
- optional vocal stem WAV,
- optional instrumental stem WAV,
- optional meow sample WAV,
- output WAV path,

then runs the same underlying pipeline used by the importable API.

If no meow sample is provided, the dashboard will automatically fetch a default CC-BY 4.0 meow sample from the Zenodo CatMeows dataset.

After rendering, the dashboard must expose the output for in-browser playback and file download.

## 4. Public API Requirements

The package must expose documented functions or classes for the following operations:

- Load WAV audio.
- Write WAV audio.
- Fetch source audio from YouTube through an optional `yt-dlp` adapter.
- Accept user-provided vocal and instrumental stems.
- Derive vocal and instrumental/no-vocals stems with Demucs.
- Estimate pitch contour and energy from the vocal stem.
- Detect syllable-like events.
- Load and validate meow sample audio.
- Fetch a meow sample from an approved public resource.
- Render a meow vocal from sample audio, events, and pitch data.
- Mix meow vocal with instrumental audio.
- Run the full pipeline from Python.

The high-level API must not require users to launch a CLI or browser dashboard.

## 5. Data Contracts

### 5.1 Audio Buffer

Represents decoded audio.

Required fields:

- `samples`: floating-point audio data.
- `sample_rate`: integer sample rate in Hz.

Requirements:

- Samples should use a normalized range of approximately `[-1.0, 1.0]`.
- Mono and multi-channel audio must be representable.
- Channel handling must be explicit when mixing or matching stems.

### 5.2 Stem Set

Represents paired vocal and instrumental material.

Required fields:

- `vocal`
- `instrumental`
- `strategy`

`strategy` must identify whether stems were provided, Demucs-derived, or produced through another explicit path.

### 5.3 Pitch Contour

Represents frame-level melody analysis.

Required fields:

- `times`
- `f0_hz`
- `voiced`
- `energy`

Requirements:

- Pitch values are in Hz.
- Unvoiced frames must be distinguishable from voiced frames.
- Energy must be available for event detection and vocal envelope shaping.

### 5.4 Syllable Event

Represents a time span that should be replaced by one meow.

Required fields:

- `start`
- `end`
- `energy`

Optional fields:

- `pitch_hz`

Requirements:

- `start` and `end` are seconds from the beginning of the track.
- `end` must be greater than `start`.
- Events should be suitable for rendering one sampled meow.

### 5.5 Meow Sample Metadata

Represents the source and licensing state of a meow sample.

Required fields:

- local path,
- source type: `user_provided` or `fetched_public_resource`,
- source URL when fetched,
- license,
- author/attribution when required,
- retrieval date when fetched.

### 5.6 Processing Result

Represents pipeline output metadata.

Required fields:

- output path,
- sample rate,
- duration,
- number of rendered events,
- stem strategy,
- meow sample source,
- meow sample license metadata when fetched,
- source ingestion metadata when fetched from YouTube.

## 6. Pipeline Behavior

### 6.1 Audio Loading

- The baseline implementation must support WAV files.
- Unsupported formats should fail with a clear error.
- Broader format support may be added later through optional dependencies.

### 6.1.1 YouTube Source Ingestion

The MVP should support optional YouTube ingestion through `yt-dlp`.

Requirements:

- `yt-dlp` must be isolated behind an adapter so the core package can import without it installed.
- Users must explicitly provide a URL and enable fetching.
- The API and dashboard must make clear that users are responsible for having rights to download and transform the track and for complying with platform terms.
- The adapter should download/extract audio and normalize it into the same local-audio path used by the rest of the pipeline.
- Downloads should be cached locally with metadata: original URL, extractor, title when available, retrieval date, and local path.
- Missing `yt-dlp`, failed extraction, unavailable/private videos, and unsupported output conversion must produce actionable errors.

### 6.2 Stem Handling

The system must support these cases:

- Source track plus vocal stem plus instrumental stem.
- Source track plus vocal stem only.
- Source track plus instrumental stem only.
- Source track only, using Demucs to derive stems.

When provided stems differ in length, the implementation should align them conservatively by trimming or padding rather than crashing unexpectedly.

When sample rates differ, the MVP may reject the input with a clear error rather than resampling.

Demucs integration requirements:

- Demucs must be isolated behind an adapter so the core package can still import without PyTorch/Demucs installed.
- The adapter should support vocals plus no-vocals/instrumental output.
- The adapter should expose clear model/device/config options.
- Missing dependency, missing model, GPU memory, and separation failures must produce actionable errors.
- User-provided stems must take precedence over running Demucs.

### 6.3 Pitch And Energy Analysis

The MVP must estimate:

- frame-level F0,
- voiced/unvoiced state,
- frame-level energy.

The default implementation should be deterministic and dependency-light. Higher-quality pitch trackers may be added as optional integrations.

### 6.4 Event Detection

The MVP must estimate syllable-like events from the vocal stem using available pitch and energy information.

The detector should:

- ignore low-energy unvoiced regions,
- split long voiced regions into smaller events,
- reject events shorter than a configurable minimum duration,
- preserve original timing closely enough for recognizable phrasing.

Lyrics-based alignment is a future enhancement, not an MVP requirement.

### 6.5 Meow Sample Handling

The MVP must use recorded/sample-based meow audio.

Supported sample sources:

- user-provided local audio file,
- fetched public resource with compatible license metadata.

The implementation must not synthesize meow audio for the MVP.

Fetched sample requirements:

- Prefer CC0 or CC BY audio.
- Reject or require explicit user confirmation for restrictive licenses.
- Store source URL, license, author/attribution, and retrieval date.
- Cache downloaded samples locally rather than refetching unnecessarily.
- Avoid committing downloaded samples to the repository unless the license and attribution have been reviewed.

Public-resource candidates:

- Freesound, because it hosts Creative Commons audio samples and provides a REST API.
- OpenGameArt, because it hosts free/open assets including sound effects and accepts CC0-like licensing.

### 6.6 Meow Rendering

The renderer must create one meow event from sample audio per detected vocal event.

Each rendered event should:

- follow the source vocal pitch contour where available,
- fit the event duration,
- use the event or vocal energy to shape loudness,
- fade in/out enough to avoid obvious clicks.

Rendering may use pitch shifting, time stretching, granular resampling, formant-preserving processing, or phase-vocoder-style processing. The input sound must still originate from a recorded meow sample.

Current pitch behavior:

- Original vocal pitch is analyzed as a contour.
- Absolute singer pitch is mapped into a configurable cat register.
- `cat_min_pitch_hz` defines the lowest rendered meow pitch.
- `cat_max_pitch_hz` defines the highest rendered meow pitch.
- `cat_pitch_contour_strength` controls how strongly the output follows the original melody shape.
- Lower contour strength flattens toward one cat register.
- Higher contour strength follows more of the original song's pitch movement.

### 6.7 Mixdown

The mixdown stage must combine:

- instrumental stem,
- rendered meow vocal.

The output should:

- avoid clipping through normalization or conservative gain,
- preserve the original instrumental timing,
- write a WAV file.

## 7. Configuration Requirements

The pipeline should expose configuration for:

- pitch-analysis frame length,
- pitch-analysis hop length,
- minimum and maximum pitch range,
- minimum event duration,
- maximum event duration,
- energy threshold,
- meow gain,
- cat minimum pitch,
- cat maximum pitch,
- cat melody contour strength,
- instrumental gain,
- output gain,
- overwrite behavior,
- source URL,
- YouTube fetch enable/disable flag,
- YouTube download/cache directory,
- `yt-dlp` backend configuration,
- `yt-dlp` should be upgraded to the newest available build by setup tooling,
- Demucs backend configuration,
- meow sample path,
- meow sample fetch source,
- meow sample cache directory.

Defaults should favor stable MVP behavior over aggressive transformation.

## 8. Optional Browser Dashboard Requirements

The dashboard must:

- be optional and separate from the core package import path,
- use Gradio as the browser dashboard UI library,
- run in a browser rather than a native GUI toolkit such as Tkinter,
- allow file selection for source, optional vocal stem, optional instrumental stem, optional meow sample, and output,
- allow entry of a YouTube URL as an alternative source input,
- require explicit user action before fetching from YouTube,
- allow fetching a licensed public meow sample when none is supplied (default enabled),
- expose cat-register sliders with plain-language notes,
- present System Health prominently via a floating panel at the top of the interface,
- output results in sequential order: Extracted Vocal -> Visualization -> Playback -> File,
- show progress/status,
- show clear, graceful error toasts via `gr.Error` without crashing the application state,
- avoid hiding processing failures.

The dashboard must not become the primary interface.

## 9. Error Handling

Errors should be explicit and actionable.

Required error cases:

- missing input file,
- unsupported audio format,
- mismatched sample rates,
- invalid or empty audio,
- output exists when overwrite is disabled,
- YouTube fetch requested but `yt-dlp` unavailable,
- YouTube fetch failed or produced no usable audio,
- no detectable vocal events,
- Demucs unavailable or failed,
- no meow sample provided and no approved public sample can be fetched,
- incompatible meow sample license.

Demucs and sample-fetch behavior must be identified in result metadata or logs.

## 10. Acceptance Criteria

The MVP is acceptable when:

- Another Python module can import and run the full pipeline without invoking a CLI.
- The full pipeline can process a short WAV fixture with provided stems plus a provided meow sample and write a WAV output.
- The full pipeline can ingest a YouTube URL through a mocked `yt-dlp` adapter and continue through the same local-audio pipeline.
- The full pipeline can derive stems through Demucs when the optional Demucs integration is installed and configured.
- The output contains rendered meow events rather than silence.
- The result reports event count, stem strategy, and meow sample source.
- The optional browser dashboard can call the same processing function.
- The dashboard dependency is optional and core imports work without Gradio installed.
- The dashboard can play back the rendered WAV in-browser.
- Setup and verification scripts provision `.venv` and report the dependency versions/backends.
- `yt-dlp` is upgraded to the latest available build by the setup script.
- Heavyweight ML dependencies are not required for core import.
- Generated audio and large artifacts are not committed.
- The implementation behavior is consistent with `PLAN.md` and `AGENTS.md`.

## 11. Testing Requirements

Minimum tests:

- WAV load/write round trip.
- YouTube ingestion behavior with mocked `yt-dlp`.
- Stem handling with provided stems.
- Demucs adapter behavior with mocked Demucs execution.
- Pitch contour generation on synthetic voiced audio.
- Event detection on synthetic voiced regions.
- Meow sample loading and validation.
- Public meow sample fetch behavior with mocked network/file cache.
- Meow sample rendering produces non-silent audio.
- Full pipeline smoke test with synthetic source, vocal, instrumental stems, and sample meow audio.
- Cat-register pitch mapping test to ensure melody contour is preserved inside the target register.

Tests should use generated synthetic audio fixtures and tiny test meow fixtures where license permits. Do not commit copyrighted music.

Local manual validation has also covered generated E2E renders through both provided stems and Demucs-derived stems.

## 12. Future Extensions

Potential future work:

- UVR interoperability for users who prefer UVR as an external stem preparation tool.
- WhisperX or Montreal Forced Aligner integration when lyrics are available.
- Curated meow sample bank support.
- Neural timbre conversion after deterministic sample-based guide-vocal rendering.
- Diagnostics for pitch confidence, clipping, event density, separation quality, and license validation.
- Optional support for MP3/FLAC through explicit audio backend dependencies.
- Explore a high-fidelity Neural Pipeline (BS-RoFormer, FCPE, ContentVec, SaMoye-SVC, DDSP-SVC) once the deterministic approach reaches its quality cap.
