# Meowsic MVP Plan

## Project Purpose

Meowsic transforms a song so the lead vocal sounds like it is sung with cat-like "meow" syllables while keeping the original melody, rhythm, phrasing, and backing track recognizable.

The MVP is not intended to be a perfect neural singing system. It is a practical, inspectable audio pipeline that can be imported from Python and improved over time.

## Baseline Objectives

- Provide an importable Python module, not a required CLI.
- Integrate Demucs as the built-in stem separation backend for deriving vocal and instrumental/no-vocals stems.
- Accept user-provided isolated vocal and/or instrumental stems when available, bypassing Demucs where appropriate.
- Preserve the original vocal melody by extracting or accepting a pitch contour from the source vocal.
- Replace lyric syllables with sampled "meow" events aligned to vocal timing.
- Let users provide their own meow audio file.
- If users do not provide a meow file, fetch a meow sample from an approved public resource with compatible license metadata.
- Support optional track ingestion from YouTube via `yt-dlp`, subject to user responsibility for rights and platform terms.
- Mix the rendered meow vocal back with the instrumental track.
- Include an optional Gradio-based browser dashboard with visualization and stem outputs.
- Let users play the rendered output directly in the dashboard and download the WAV.
- Provide setup and verification scripts that create `.venv`, install optional dependencies, keep `yt-dlp` current, and install the tested Windows Demucs/Torchaudio/SoundFile stack.
- Keep heavyweight ML integrations isolated so the core package can still be imported in a basic Python environment.

## MVP Requirements

### Importable API

The package must expose functions for:

- Loading audio from WAV files.
- Fetching and normalizing source audio from YouTube URLs via an optional `yt-dlp` integration.
- Separating vocal/instrumental stems with Demucs when stems are not supplied.
- Accepting user-provided vocal/instrumental stems.
- Estimating a vocal pitch contour.
- Estimating syllable-like vocal events from timing, energy, and pitch.
- Loading, validating, and using meow audio samples.
- Fetching meow samples from approved public resources.
- Rendering a meow vocal from sample events and a pitch contour.
- Mixing the rendered meow vocal with an instrumental stem.
- Running the full pipeline from Python.

### Optional Browser Dashboard

The dashboard should:

- Run as a local browser-based interface using Gradio rather than a native GUI toolkit such as Tkinter.
- Allow users to choose input audio, paste a YouTube URL, optional vocal stem, optional instrumental stem, optional meow sample, and output path.
- Allow users to fetch a meow sample from a configured public resource when they do not provide one (enabled by default).
- Expose simple cat-register controls with notes explaining low/high pitch and contour strength behavior.
- Show System Health section as a floating panel at the top right of the dashboard.
- Display clear, graceful error toasts using `gr.Error` on failures (e.g. missing dependencies).
- Expose outputs in order: Extracted Vocal Stem -> Visualization -> Rendered Cat Song -> Downloadable Output.

### Audio Quality Target

The MVP output should be recognizable as the original song when:

- The instrumental track is reasonably clean.
- Demucs or provided stems isolate the vocal well enough for pitch/event analysis.
- The vocal pitch tracker follows the lead melody.
- The vocal event detector finds phrase and syllable boundaries well enough for the song.
- The meow sample can be time-stretched and pitch-shifted without severe artifacts.

The result may sound processed. Musical recognizability is a higher priority than realism in the MVP.

## Non-Goals For MVP

- No bundled copyrighted music.
- No unlicensed or unclearly licensed bundled meow samples.
- No unapproved model downloads.
- No automatic YouTube downloads without explicit user action.
- No real-time processing.
- No celebrity or artist voice imitation features.
- No guarantee of clean vocal separation from arbitrary mixed songs; Demucs output quality depends on the input mix and available compute.
- No from-scratch neural singing voice conversion model.
- No synthetic meow generation.

## Recommended Technical Direction

### Near-Term Pipeline

1. Load source audio from a local file, or fetch/convert audio from a user-provided YouTube URL via optional `yt-dlp`.
2. Load optional user-provided stems.
3. If stems are not provided, run Demucs to derive vocal and instrumental/no-vocals stems.
4. Estimate pitch and energy from the vocal stem.
5. Segment the melody into discrete, semitone-quantized **notes** (default `notes` engine), octave-transposed into the cat register so the tune stays in key. (Experimental `instrument`/`granular` engines use onset/energy event detection instead.)
6. Automatically fetch the default CatMeows sample from Zenodo if no meow sample is provided (default: `B_CAN01_EU_FN_GIA01_205.wav`, a uniform meow).
7. Clean the meow (low-pass to remove HF grain, isolate the loud body) and render one whole recognizable meow per note, pitch-shifted (resample or PSOLA) and fit to the note's duration (compress short notes, loop long ones).
8. Mix with instrumental and write WAV output.
9. Optionally score the result with `evaluate_render` (MeowScore) — a guide, not a substitute for listening.

Pitch rendering should preserve the original melody contour while mapping the absolute pitches into a configurable cat-like register. This avoids forcing every meow to the singer's exact pitch when that pitch would sound unnaturally high or low for the selected sample.

The current implementation uses:

- `cat_min_pitch_hz` for the lowest rendered meow pitch.
- `cat_max_pitch_hz` for the highest rendered meow pitch.
- `cat_pitch_contour_strength` to blend between a flatter cat register and stronger original melody contour.

### Rhythm And Rendering Quality

"A cat singing a song" is a sequence of distinct, recognizable meows — one per musical note, in tune — not a continuous drone and not chopped tone fragments. The default note engine therefore:

- Segments the melody into discrete notes and snaps each to the nearest semitone (`note_min_duration`, `note_merge_duration`) so it is in tune and each meow has room to be recognizable.
- Octave-transposes the melody into the cat register (`cat_min/max_pitch_hz`), preserving intervals.
- Renders one whole meow per note, pitch-shifted (`pitch_shift_method`: `resample` or formant-preserving `psola`) and fit to the note (`note_fit`: `compress` scales the whole meow; `trim` uses the onset; long notes loop).
- Cleans the sample first: low-pass (`meow_lowpass_hz`) to remove HF grain and loud-body isolation so notes play the meow, not a noisy lead-in.

Sample quality is a hard limit: the 8 kHz CatMeows recordings are muffled; a clean 44.1 kHz meow sounds materially better. Users can audition samples through the dashboard picker.

### Measuring Quality

`evaluate_render` compares a rendered meow vocal against the source vocal and returns a **MeowScore** in `[0, 1]`:

- rhythm: onset F-measure between meow triggers and vocal note attacks,
- melody: correlation of log-F0 contours,
- dynamics: correlation of loudness envelopes.

This makes parameter changes measurable rather than purely subjective and is surfaced in the dashboard after each render.

### YouTube Track Ingestion

`yt-dlp` support should be optional and isolated behind an adapter.

Requirements:

- Users must explicitly provide a URL and initiate the fetch.
- The dashboard and API should make clear that users are responsible for having rights to download and transform the track and for complying with platform terms.
- Downloads should be cached as local source assets with metadata: original URL, extractor, title when available, retrieval date, and local path.
- The core package should still import when `yt-dlp` is not installed.
- The setup script should upgrade `yt-dlp` to the latest available build because extractor behavior changes frequently.
- Missing `yt-dlp`, failed extraction, private/unavailable videos, and unsupported output formats must produce actionable errors.

### Local Dependency Setup

The project includes:

- A `Makefile` (`make setup`) to create `.venv` and install all optional dependency groups.
- A `Makefile` (`make verify`) to print the installed versions and Torchaudio audio backends.

The tested Windows Demucs stack is:

- `torch==2.6.0`
- `torchaudio==2.6.0`
- `soundfile>=0.12`
- `demucs==4.0.1`

SoundFile is required so Torchaudio can write Demucs WAV stems reliably on Windows. Avoid mismatched newer TorchAudio/TorchCodec combinations; they can cause TorchCodec DLL loading failures.

### Public Meow Sample Resources

Approved resource candidates:

- Freesound, because it hosts Creative Commons audio samples and exposes a REST API.
- OpenGameArt, because it hosts free/open assets including sound effects and accepts licenses including CC0.

Fetched samples must record:

- source URL,
- license,
- author/attribution,
- retrieval date,
- local cache path.

If no specific sample URL is provided, the pipeline automatically fetches the default CC-BY 4.0 "CatMeows" dataset sample from Zenodo.

Prefer CC0. CC BY may be allowed when attribution metadata is preserved. Restrictive licenses should be rejected unless a future product requirement explicitly supports them.

### Future Improvements

- Add UVR interoperability for users who prefer UVR as an external stem preparation tool.
- Add WhisperX or Montreal Forced Aligner support when lyrics are available.
- Add a curated licensed meow sample bank and sample selection.
- Add neural timbre conversion after the deterministic sample-based meow guide vocal.
- Add objective diagnostics: pitch-tracking confidence, event count, clipping, stem quality checks, and sample license validation.
- **High-Fidelity Neural Pipeline Migration**: Once the deterministic pipeline reaches its performance cap, explore migrating to a state-of-the-art Deep Learning approach:
  - Isolate stems using BS-RoFormer (Mel-RoFormer).
  - Utilize FCPE (Fast Context-based Pitch Estimation) for sub-cent pitch tracking.
  - Apply ContentVec / Soft HuBERT for timbre-invariant linguistic feature extraction.
  - Adopt SaMoye-SVC paired with DDSP-SVC and a ReFlow diffusion step for zero-shot, realistic feline timbre generation.

## Developer Notes

- Prefer small, composable functions over a single opaque processor.
- Keep Demucs isolated behind an adapter so the core package can import without Demucs/PyTorch installed.
- Keep `yt-dlp` isolated behind an adapter so the core package can import without `yt-dlp` installed.
- Keep Gradio isolated to the dashboard entry point so the core package can import without dashboard dependencies installed.
- Demucs integration should fail with actionable errors when dependencies, models, or compute are unavailable.
- Prefer the `Makefile` targets (`make setup`, `make verify`) before troubleshooting missing dependency issues.
- Keep `yt-dlp` updated through the setup script.
- YouTube ingestion should require explicit user action and should not bypass rights/platform-term warnings.
- Preserve user-owned files and do not overwrite outputs unless explicitly requested by the calling code.
- The pipeline should work best on WAV files first; broader format support can be added through optional audio libraries.
- All public APIs should be documented enough for another app to import and call them.
- Meow samples must come from user-provided files or explicitly licensed public resources. Store source URL, license, author/attribution, and retrieval date with fetched samples.
