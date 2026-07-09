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

    css = """
    #health-section {
        position: absolute;
        top: 10px;
        right: 10px;
        width: 300px;
        z-index: 1000;
        background: var(--background-fill-primary);
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 5px;
    }
    """

    with gr.Blocks(title="Meowsic", css=css) as app:
        with gr.Accordion("System Health", open=False, elem_id="health-section"):
            health_json = gr.JSON(label="Dependency Status")
            refresh_health_btn = gr.Button("Refresh Health")

            def get_health():
                from meowsic.health import check_health
                return check_health()

            refresh_health_btn.click(fn=get_health, outputs=health_json)
            app.load(fn=get_health, outputs=health_json)

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
            gr.Markdown(
                "Pick from cached samples below, upload your own WAV, or leave empty to auto-fetch from Zenodo."
            )
            with gr.Row():
                meow_sample_dropdown = gr.Dropdown(
                    choices=[],
                    label="Cached samples (auto-fetched from Zenodo)",
                    value=None,
                    allow_custom_value=False,
                    interactive=True,
                )
                refresh_samples_btn = gr.Button("↺ Refresh", scale=0, size="sm")
            meow_sample_preview = gr.Audio(label="Preview selected sample", type="filepath", interactive=False)
            meow_sample_file = gr.File(label="Upload custom meow WAV (overrides picker)", file_types=[".wav"], type="filepath")
            meow_sample_url = gr.Textbox(label="Public sample URL", placeholder="https://.../meow.wav")
            meow_sample_license = gr.Textbox(label="License", placeholder="CC0 or CC BY")
            meow_sample_attribution = gr.Textbox(label="Attribution")
            enable_public_sample_fetch = gr.Checkbox(label="Enable public sample fetch", value=True)

            def _get_sample_choices() -> list[str]:
                from meowsic.samples import list_cached_samples
                samples = list_cached_samples(MeowsicConfig(meow_sample_cache_dir=workspace / "cache" / "meow_samples"))
                return [str(p) for p in samples]

            refresh_samples_btn.click(
                fn=lambda: gr.update(choices=_get_sample_choices()),
                outputs=meow_sample_dropdown,
            )
            meow_sample_dropdown.change(
                fn=lambda v: v if v else None,
                inputs=meow_sample_dropdown,
                outputs=meow_sample_preview,
            )
            app.load(
                fn=lambda: gr.update(choices=_get_sample_choices()),
                outputs=meow_sample_dropdown,
            )

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
        vocal_stem_audio = gr.Audio(label="Extracted Vocal Stem", type="filepath")
        visualization = gr.Plot(label="Vocal Track & Cat Register Coverage")
        output_audio = gr.Audio(label="Cat Song Playback", type="filepath")
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
                meow_sample_dropdown,
                meow_sample_url,
                meow_sample_license,
                meow_sample_attribution,
                enable_public_sample_fetch,
                cat_min_pitch,
                cat_max_pitch,
                cat_contour_strength,
            ],
            outputs=[status, vocal_stem_audio, visualization, output_audio, output_file],
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
    meow_sample_dropdown: str | None,
    meow_sample_url: str,
    meow_sample_license: str,
    meow_sample_attribution: str,
    enable_public_sample_fetch: bool,
    cat_min_pitch: float,
    cat_max_pitch: float,
    cat_contour_strength: float,
) -> tuple[str, str | None, str | None, Any, str | None]:
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

    # Priority: uploaded file > dropdown selection > auto-fetch
    resolved_sample_path = _file_path(meow_sample_file) or meow_sample_dropdown or None

    try:
        import gradio as gr
        result = process_song(
            source_path=_file_path(source_file),
            source_url=_blank_to_none(source_url),
            output_path=output_path,
            vocal_path=_file_path(vocal_file),
            instrumental_path=_file_path(instrumental_file),
            meow_sample_path=resolved_sample_path,
            config=config,
        )
    except Exception as exc:
        import gradio as gr
        raise gr.Error(f"Rendering failed: {exc}")

    message = "\n".join(
        [
            f"Wrote: {result.output_path}",
            f"Events: {result.event_count}",
            f"Stem strategy: {result.stem_strategy}",
            f"Meow sample: {result.meow_sample.source_type}",
        ]
    )
    output = str(result.output_path)
    
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 4))
    if result.vocal_stem:
        samples = result.vocal_stem.mono()
        times = np.linspace(0, result.duration, len(samples))
        ax.plot(times, samples, color="lightgray", label="Vocal Waveform", alpha=0.7)
    
    if result.pitch_contour:
        ax2 = ax.twinx()
        valid = result.pitch_contour.voiced
        t = result.pitch_contour.times[valid]
        f0 = result.pitch_contour.f0_hz[valid]
        ax2.scatter(t, f0, color="blue", s=2, label="Original Pitch", alpha=0.5)
        
        ax2.axhspan(cat_min_pitch, cat_max_pitch, color="orange", alpha=0.2, label="Cat Register")
        ax2.set_ylabel("Pitch (Hz)")
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.title("Vocal Waveform & Pitch vs. Cat Register")
    plt.tight_layout()

    vocal_output_path = None
    if result.vocal_stem:
        from meowsic.io import write_wav
        vocal_output_path = str(output_dir / f"meowsic-{run_id}-vocals.wav")
        write_wav(vocal_output_path, result.vocal_stem)

    return message, vocal_output_path, fig, output, output


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
