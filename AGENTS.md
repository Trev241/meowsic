# Agent Notes For Meowsic

Meowsic is a Python project for transforming songs so the lead vocal is replaced by melody-preserving cat-like "meow" syllables.

Use `PLAN.md` as the source of truth for MVP goals, requirements, non-goals, and future direction.

## Current Implementation Expectations

- This project should expose importable Python functions.
- Do not make a CLI the primary interface unless the user explicitly requests one.
- Integrate Demucs for built-in stem separation, but isolate it so the core package can import without Demucs/PyTorch installed.
- Add optional YouTube ingestion through `yt-dlp`, isolated behind an adapter so the core package can import without `yt-dlp` installed.
- Keep `yt-dlp` on the latest available build during setup because extractor behavior changes frequently.
- Continue to accept user-provided vocal/instrumental stems and prefer them over running Demucs.
- Prefer deterministic, debuggable DSP around analysis, sample handling, rendering, and mixdown.
- Do not synthesize the "meow" source for the MVP. Use user-provided meow audio or fetch a licensed public sample.
- Use a simple Gradio-based browser dashboard for manual use, not a native GUI toolkit such as Tkinter.
- The optional browser dashboard should be a convenience layer over the same importable functions, with in-browser playback and WAV download.
- Use the `Makefile` targets (`make setup`, `make verify`, `make test`, `make run`) for environment setup before piecemeal dependency troubleshooting.

## Engineering Constraints

- Avoid committing generated audio or large model files.
- Do not add heavyweight dependency downloads without explicit user approval.
- Do not add Demucs model downloads or public sample downloads without explicit user approval.
- Do not download tracks from YouTube without explicit user action.
- Surface that users are responsible for rights and platform-term compliance when fetching tracks from YouTube.
- The tested local Windows Demucs stack is `torch==2.6.0`, `torchaudio==2.6.0`, and `soundfile>=0.12`.
- Avoid mismatched newer TorchAudio/TorchCodec combinations unless deliberately updating the stack; they have caused TorchCodec DLL failures locally.
- Public meow samples must include source URL, license, author/attribution, and retrieval date metadata.
- Preserve any user changes in the working tree.
- Keep code paths clear about what is user-provided, Demucs-derived, fetched, cached, or fallback behavior.

## Useful Mental Model

The MVP is:

`local song or YouTube URL -> local source audio -> Demucs or provided stems -> melody -> discrete semitone-quantized NOTES -> one whole pitched meow per note -> mixdown -> MeowScore`

The target is the established "cat cover" formula: a sequence of distinct, recognizable meows — one per musical note, in tune — NOT a continuous drone and NOT chopped tone fragments. The default `notes` engine segments the melody into notes, snaps to semitones, octave-transposes into the cat register (preserving intervals), and renders one whole meow per note (pitch-shifted via resample or formant-preserving PSOLA; short notes compressed, long notes looped). Samples are low-passed and body-isolated first, because the 8 kHz CatMeows recordings are muffled/noisy — sample quality is a hard limit.

Two experimental engines exist but sounded worse and are not default: `instrument` (continuous TD-PSOLA glide -> drone) and `granular` (per-onset splicing -> chopped).

Use `meowsic.evaluate.evaluate_render` (MeowScore) only as a rough guide — it rewards on-beat density and does NOT capture "catness" or musicality, so trust listening for final decisions. The `.meowsic-dashboard/` workspace holds ad-hoc sweep/A-B/diagnostic scripts and generated audio; it is git-ignored and must not be committed.

It is not yet:

`arbitrary copyrighted song -> perfect neural cat singer cover`

(Note: A high-fidelity Deep Learning pipeline using BS-RoFormer, FCPE, ContentVec, and SaMoye-SVC will be explored as a future extension once the deterministic approach caps out.)
