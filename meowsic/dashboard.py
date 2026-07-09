from __future__ import annotations

import uuid
import socket
from pathlib import Path
from typing import Any

from .pipeline import process_song
from .types import MeowsicConfig


def launch_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace: str | Path = ".meowsic-dashboard",
    open_browser: bool = True,
) -> None:
    """Launch a simple local Gradio dashboard."""

    try:
        import gradio as gr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the dashboard. Install it with "
            "`python -m pip install -e .[dashboard]`."
        ) from exc

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    selected_port = _find_available_port(host, port)
    print(f"Meowsic dashboard running at http://{host}:{selected_port}", flush=True)

    with gr.Blocks(title="Meowsic") as app:
        gr.Markdown("# Meowsic")
        gr.Markdown(
            "Render a song with sample-based meow vocals. YouTube fetching is optional and must be explicitly enabled."
        )

        with gr.Group():
            gr.Markdown("## Source")
            source_file = gr.File(label="Source WAV", file_types=[".wav"], type="filepath")
            source_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            enable_youtube_fetch = gr.Checkbox(label="Enable YouTube fetch with yt-dlp")
            gr.Markdown(
                "Only fetch tracks you have rights to download and transform, and comply with platform terms."
            )

        with gr.Group():
            gr.Markdown("## Optional Stems")
            vocal_file = gr.File(label="Vocal stem WAV", file_types=[".wav"], type="filepath")
            instrumental_file = gr.File(label="Instrumental stem WAV", file_types=[".wav"], type="filepath")
            gr.Markdown("If stems are omitted, Meowsic will try Demucs if installed.")

        with gr.Group():
            gr.Markdown("## Meow Sample")
            meow_sample_file = gr.File(label="Meow sample WAV", file_types=[".wav"], type="filepath")
            meow_sample_url = gr.Textbox(label="Public sample URL", placeholder="https://.../meow.wav")
            meow_sample_license = gr.Textbox(label="License", placeholder="CC0 or CC BY")
            meow_sample_attribution = gr.Textbox(label="Attribution")
            enable_public_sample_fetch = gr.Checkbox(label="Enable public sample fetch")

        with gr.Group():
            gr.Markdown("## Cat Register")
            cat_min_pitch = gr.Slider(120, 500, value=220, step=5, label="Lowest meow pitch (Hz)")
            gr.Markdown(
                "Lowest meow pitch: lower values make the cat voice deeper and heavier; higher values keep even low notes more kitten-like."
            )
            cat_max_pitch = gr.Slider(220, 900, value=520, step=5, label="Highest meow pitch (Hz)")
            gr.Markdown(
                "Highest meow pitch: lower values tame squeaky high notes; higher values allow brighter, sharper meows on melody peaks."
            )
            cat_contour_strength = gr.Slider(
                0,
                1,
                value=0.75,
                step=0.05,
                label="Melody contour strength",
            )
            gr.Markdown(
                "Melody contour strength: lower values flatten the tune toward one cat register; higher values follow more of the song's original ups and downs."
            )

        render_button = gr.Button("Render", variant="primary")
        status = gr.Textbox(label="Status", lines=5)
        output_audio = gr.Audio(label="Playback", type="filepath")
        output_file = gr.File(label="Output WAV", file_types=[".wav"])

        render_button.click(
            fn=lambda *values: _render_from_dashboard(workspace, *values),
            inputs=[
                source_file,
                source_url,
                enable_youtube_fetch,
                vocal_file,
                instrumental_file,
                meow_sample_file,
                meow_sample_url,
                meow_sample_license,
                meow_sample_attribution,
                enable_public_sample_fetch,
                cat_min_pitch,
                cat_max_pitch,
                cat_contour_strength,
            ],
            outputs=[status, output_audio, output_file],
        )

    app.launch(server_name=host, server_port=selected_port, inbrowser=open_browser)


def _find_available_port(host: str, preferred_port: int, *, attempts: int = 25) -> int:
    for candidate in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
            return candidate
    raise OSError(f"Cannot find an empty port in range {preferred_port}-{preferred_port + attempts - 1}")


def _render_from_dashboard(
    workspace: Path,
    source_file: Any,
    source_url: str,
    enable_youtube_fetch: bool,
    vocal_file: Any,
    instrumental_file: Any,
    meow_sample_file: Any,
    meow_sample_url: str,
    meow_sample_license: str,
    meow_sample_attribution: str,
    enable_public_sample_fetch: bool,
    cat_min_pitch: float,
    cat_max_pitch: float,
    cat_contour_strength: float,
) -> tuple[str, str | None, str | None]:
    run_id = uuid.uuid4().hex[:10]
    output_dir = workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"meowsic-{run_id}.wav"

    config = MeowsicConfig(
        overwrite=True,
        enable_youtube_fetch=bool(enable_youtube_fetch),
        youtube_cache_dir=workspace / "cache" / "youtube",
        enable_public_sample_fetch=bool(enable_public_sample_fetch),
        meow_sample_url=_blank_to_none(meow_sample_url),
        meow_sample_license=_blank_to_none(meow_sample_license),
        meow_sample_attribution=_blank_to_none(meow_sample_attribution),
        meow_sample_cache_dir=workspace / "cache" / "meow_samples",
        demucs_cache_dir=workspace / "cache" / "demucs",
        cat_min_pitch_hz=float(cat_min_pitch),
        cat_max_pitch_hz=float(cat_max_pitch),
        cat_pitch_contour_strength=float(cat_contour_strength),
    )

    result = process_song(
        source_path=_file_path(source_file),
        source_url=_blank_to_none(source_url),
        output_path=output_path,
        vocal_path=_file_path(vocal_file),
        instrumental_path=_file_path(instrumental_file),
        meow_sample_path=_file_path(meow_sample_file),
        config=config,
    )

    message = "\n".join(
        [
            f"Wrote: {result.output_path}",
            f"Events: {result.event_count}",
            f"Stem strategy: {result.stem_strategy}",
            f"Meow sample: {result.meow_sample.source_type}",
        ]
    )
    output = str(result.output_path)
    return message, output, output


def _file_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return str(value)
    name = getattr(value, "name", None)
    if name:
        return str(name)
    path = getattr(value, "path", None)
    if path:
        return str(path)
    return None


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None
