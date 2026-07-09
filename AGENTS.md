# Agent Notes For Meowsic

Meowsic is a Python project for transforming songs so the lead vocal is replaced by melody-preserving cat-like "meow" syllables.

Use `PLAN.md` as the source of truth for MVP goals, requirements, non-goals, and future direction.

## Current Implementation Expectations

- This project should expose importable Python functions.
- Do not make a CLI the primary interface unless the user explicitly requests one.
- Integrate Demucs for built-in stem separation, but isolate it so the core package can import without Demucs/PyTorch installed.
- Add optional YouTube ingestion through `yt-dlp`, isolated behind an adapter so the core package can import without `yt-dlp` installed.
- Continue to accept user-provided vocal/instrumental stems and prefer them over running Demucs.
- Prefer deterministic, debuggable DSP around analysis, sample handling, rendering, and mixdown.
- Do not synthesize the "meow" source for the MVP. Use user-provided meow audio or fetch a licensed public sample.
- Use a simple Gradio-based browser dashboard for manual use, not a native GUI toolkit such as Tkinter.
- The optional browser dashboard should be a convenience layer over the same importable functions.

## Engineering Constraints

- Avoid committing generated audio or large model files.
- Do not add heavyweight dependency downloads without explicit user approval.
- Do not add Demucs model downloads or public sample downloads without explicit user approval.
- Do not download tracks from YouTube without explicit user action.
- Surface that users are responsible for rights and platform-term compliance when fetching tracks from YouTube.
- Public meow samples must include source URL, license, author/attribution, and retrieval date metadata.
- Preserve any user changes in the working tree.
- Keep code paths clear about what is user-provided, Demucs-derived, fetched, cached, or fallback behavior.

## Useful Mental Model

The MVP is:

`local song or YouTube URL -> local source audio -> Demucs or provided stems -> vocal pitch and event analysis -> sample-based meow rendering -> mixdown`

It is not yet:

`arbitrary copyrighted song -> perfect neural cat singer cover`
