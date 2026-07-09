from __future__ import annotations

import socket
import uuid
from pathlib import Path
from typing import Any

from .pipeline import process_song
from .types import MeowsicConfig

_CSS = """
#health-panel {
    position: fixed;
    top: 12px;
    right: 12px;
    width: 280px;
    z-index: 1000;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 6px 10px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}
#hero { text-align: center; padding: 8px 0 2px 0; }
#hero h1 { font-size: 2.1rem; margin-bottom: 0; }
#hero p { color: var(--body-text-color-subdued); margin-top: 4px; }
#render-btn { font-size: 1.05rem; }
#score-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 10px 16px;
    background: var(--background-fill-secondary);
}
.footnote { color: var(--body-text-color-subdued); font-size: 0.85rem; }
"""


def launch_dashboard(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace: str | Path = ".meowsic-dashboard",
    open_browser: bool = True,
) -> None:
    """Launch the local Gradio dashboard for Meowsic."""

    try:
        import gradio as gr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the dashboard. Install it with "
            "`python -m pip install -e .[dashboard]`."
        ) from exc

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    sample_cache_dir = workspace / "cache" / "meow_samples"
    default_sample_name = MeowsicConfig().default_meow_sample_name

    def get_sample_choices() -> list[str]:
        from .samples import list_cached_samples

        return [str(p) for p in list_cached_samples(MeowsicConfig(meow_sample_cache_dir=sample_cache_dir))]

    def default_sample_choice() -> str | None:
        choices = get_sample_choices()
        if not choices:
            return None
        preferred = str(sample_cache_dir / default_sample_name)
        return preferred if preferred in choices else choices[0]

    selected_port = _find_available_port(host, port)
    print(f"Meowsic dashboard running at http://{host}:{selected_port}", flush=True)

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate")

    with gr.Blocks(title="Meowsic") as app:
        # Floating dependency-health panel.
        with gr.Accordion("System Health", open=False, elem_id="health-panel"):
            health_json = gr.JSON(label="Dependencies")
            refresh_health_btn = gr.Button("Refresh", size="sm")

            def get_health():
                from .health import check_health

                return check_health()

            refresh_health_btn.click(fn=get_health, outputs=health_json)
            app.load(fn=get_health, outputs=health_json)

        gr.HTML(
            "<div id='hero'><h1>🐱 Meowsic</h1>"
            "<p>Turn any song into a cat singing the melody — deterministic, inspectable, tunable.</p></div>"
        )

        with gr.Row(equal_height=False):
            # ---------------- Left: inputs ----------------
            with gr.Column(scale=4):
                with gr.Accordion("1 · Source", open=True):
                    source_file = gr.File(label="Source WAV", file_types=[".wav"], type="filepath")
                    source_url = gr.Textbox(label="…or YouTube URL", placeholder="https://www.youtube.com/watch?v=…")
                    enable_youtube_fetch = gr.Checkbox(label="Enable YouTube fetch (yt-dlp)", value=False)
                    gr.Markdown(
                        "<span class='footnote'>Only fetch tracks you have the rights to download and transform, "
                        "and comply with the platform's terms.</span>"
                    )

                with gr.Accordion("2 · Stems (optional)", open=False):
                    vocal_file = gr.File(label="Vocal stem WAV", file_types=[".wav"], type="filepath")
                    instrumental_file = gr.File(label="Instrumental stem WAV", file_types=[".wav"], type="filepath")
                    gr.Markdown("<span class='footnote'>If omitted, Meowsic runs Demucs (when installed).</span>")

                with gr.Accordion("3 · Meow voice", open=True):
                    with gr.Row():
                        meow_sample_dropdown = gr.Dropdown(
                            choices=get_sample_choices(),
                            value=default_sample_choice(),
                            label="Cached sample",
                            allow_custom_value=False,
                            interactive=True,
                            scale=5,
                        )
                        refresh_samples_btn = gr.Button("↺", scale=0, size="sm")
                    meow_sample_preview = gr.Audio(
                        label="Preview", type="filepath", interactive=False, value=default_sample_choice()
                    )
                    meow_sample_file = gr.File(
                        label="Upload custom meow WAV (overrides picker)", file_types=[".wav"], type="filepath"
                    )
                    with gr.Accordion("Fetch a public sample", open=False):
                        meow_sample_url = gr.Textbox(label="Public sample URL", placeholder="https://…/meow.wav")
                        meow_sample_license = gr.Textbox(label="License", placeholder="CC0 or CC BY")
                        meow_sample_attribution = gr.Textbox(label="Attribution")
                        enable_public_sample_fetch = gr.Checkbox(label="Enable public sample fetch", value=True)
                    meow_brightness = gr.Slider(0, 1, value=0.35, step=0.05, label="Brightness")
                    meow_use_core = gr.Checkbox(
                        label="Use tonal core (recommended — locks meows to the melody note)", value=True
                    )
                    meow_full_syllable = gr.Checkbox(
                        label="Full syllable per note (legato; off = punchier staccato)", value=False
                    )

                with gr.Accordion("4 · Cat register", open=True):
                    cat_min_pitch = gr.Slider(120, 500, value=220, step=5, label="Lowest meow pitch (Hz)")
                    cat_max_pitch = gr.Slider(220, 900, value=520, step=5, label="Highest meow pitch (Hz)")
                    cat_contour_strength = gr.Slider(0, 1, value=0.75, step=0.05, label="Melody contour strength")
                    gr.Markdown(
                        "<span class='footnote'>Low/high pitch set the cat's range; contour strength controls how "
                        "strongly the meows follow the song's ups and downs.</span>"
                    )

                with gr.Accordion("5 · Engine & notes", open=True):
                    render_mode = gr.Radio(
                        ["notes", "instrument", "granular"], value="notes", label="Render engine",
                        info="notes = one meow per note (recommended); others are experimental",
                    )
                    pitch_shift_method = gr.Radio(
                        ["resample", "psola"], value="resample", label="Pitch shift",
                        info="resample = brighter/cat-cover; psola = formant-preserving",
                    )
                    note_fit = gr.Radio(
                        ["compress", "trim"], value="compress", label="Note fit",
                        info="compress = whole meow scaled to the note; trim = meow onset only",
                    )
                    note_merge_duration = gr.Slider(0.09, 0.5, value=0.22, step=0.01, label="Merge notes shorter than (s)")
                    meow_lowpass_hz = gr.Slider(0, 8000, value=3800, step=100, label="Meow low-pass (Hz, 0 = off)")

                with gr.Accordion("6 · Rhythm & detection (advanced)", open=False):
                    event_mode = gr.Radio(
                        ["onset", "energy"], value="onset", label="Event mode",
                        info="onset = meows land on note attacks (recommended); energy = older voiced-region mode",
                    )
                    onset_threshold = gr.Slider(
                        0.0, 2.0, value=0.35, step=0.05, label="Onset sensitivity",
                        info="Lower = more meows (busier); higher = fewer, stronger hits",
                    )
                    onset_min_interval = gr.Slider(
                        0.03, 0.3, value=0.07, step=0.01, label="Min time between meows (s)"
                    )
                    rearticulate = gr.Checkbox(label="Re-articulate held notes", value=True)
                    rearticulate_interval = gr.Slider(
                        0.12, 0.8, value=0.28, step=0.02, label="Held-note re-strike period (s)"
                    )
                    min_event_duration = gr.Slider(0.03, 0.4, value=0.06, step=0.01, label="Minimum meow length (s)")

                with gr.Accordion("7 · Mix levels (advanced)", open=False):
                    meow_gain = gr.Slider(0.0, 2.0, value=0.95, step=0.05, label="Cat voice level")
                    instrumental_gain = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Instrumental level")
                    output_gain = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Output ceiling")

                render_button = gr.Button("🎧 Render cat song", variant="primary", elem_id="render-btn")

            # ---------------- Right: results ----------------
            with gr.Column(scale=6):
                with gr.Group(elem_id="score-card"):
                    score_md = gr.Markdown("### MeowScore\nRender a song to see the quality metrics.")
                    metrics_json = gr.JSON(label="Quality metrics")
                status = gr.Textbox(label="Status", lines=4)
                output_audio = gr.Audio(label="🎶 Cat song", type="filepath")
                output_file = gr.File(label="Download WAV")
                visualization = gr.Plot(label="Melody & rhythm tracking")
                vocal_stem_audio = gr.Audio(label="Extracted vocal stem", type="filepath")

        # Sample picker wiring.
        def refresh_dropdown():
            return gr.update(choices=get_sample_choices(), value=default_sample_choice())

        refresh_samples_btn.click(fn=refresh_dropdown, outputs=meow_sample_dropdown)
        meow_sample_dropdown.change(
            fn=lambda v: v if v else None, inputs=meow_sample_dropdown, outputs=meow_sample_preview
        )
        app.load(fn=refresh_dropdown, outputs=meow_sample_dropdown)

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
                meow_brightness,
                meow_use_core,
                meow_full_syllable,
                render_mode,
                pitch_shift_method,
                note_fit,
                note_merge_duration,
                meow_lowpass_hz,
                event_mode,
                onset_threshold,
                onset_min_interval,
                rearticulate,
                rearticulate_interval,
                min_event_duration,
                meow_gain,
                instrumental_gain,
                output_gain,
            ],
            outputs=[status, score_md, metrics_json, output_audio, output_file, visualization, vocal_stem_audio],
        )

    app.launch(server_name=host, server_port=selected_port, inbrowser=open_browser, css=_CSS, theme=theme)


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
    meow_brightness: float,
    meow_use_core: bool,
    meow_full_syllable: bool,
    render_mode: str,
    pitch_shift_method: str,
    note_fit: str,
    note_merge_duration: float,
    meow_lowpass_hz: float,
    event_mode: str,
    onset_threshold: float,
    onset_min_interval: float,
    rearticulate: bool,
    rearticulate_interval: float,
    min_event_duration: float,
    meow_gain: float,
    instrumental_gain: float,
    output_gain: float,
):
    import gradio as gr

    run_id = uuid.uuid4().hex[:10]
    output_dir = workspace / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"meowsic-{run_id}.wav"

    config = MeowsicConfig(
        overwrite=True,
        min_event_duration=float(min_event_duration),
        event_mode=str(event_mode),
        onset_threshold=float(onset_threshold),
        onset_min_interval=float(onset_min_interval),
        rearticulate=bool(rearticulate),
        rearticulate_interval=float(rearticulate_interval),
        cat_min_pitch_hz=float(cat_min_pitch),
        cat_max_pitch_hz=float(cat_max_pitch),
        cat_pitch_contour_strength=float(cat_contour_strength),
        meow_brightness=float(meow_brightness),
        meow_use_core=bool(meow_use_core),
        meow_full_syllable=bool(meow_full_syllable),
        render_mode=str(render_mode),
        pitch_shift_method=str(pitch_shift_method),
        note_fit=str(note_fit),
        note_merge_duration=float(note_merge_duration),
        meow_lowpass_hz=float(meow_lowpass_hz),
        meow_gain=float(meow_gain),
        instrumental_gain=float(instrumental_gain),
        output_gain=float(output_gain),
        enable_youtube_fetch=bool(enable_youtube_fetch),
        youtube_cache_dir=workspace / "cache" / "youtube",
        enable_public_sample_fetch=bool(enable_public_sample_fetch),
        meow_sample_url=_blank_to_none(meow_sample_url),
        meow_sample_license=_blank_to_none(meow_sample_license),
        meow_sample_attribution=_blank_to_none(meow_sample_attribution),
        meow_sample_cache_dir=workspace / "cache" / "meow_samples",
        demucs_cache_dir=workspace / "cache" / "demucs",
    )

    resolved_sample_path = _file_path(meow_sample_file) or meow_sample_dropdown or None

    try:
        result = process_song(
            source_path=_file_path(source_file),
            source_url=_blank_to_none(source_url),
            output_path=output_path,
            vocal_path=_file_path(vocal_file),
            instrumental_path=_file_path(instrumental_file),
            meow_sample_path=resolved_sample_path,
            config=config,
        )
    except Exception as exc:  # surface a graceful toast rather than crashing state
        raise gr.Error(f"Rendering failed: {exc}")

    quality = None
    try:
        if result.vocal_stem is not None and result.meow_vocal is not None:
            from .evaluate import evaluate_render

            quality = evaluate_render(
                result.vocal_stem, result.meow_vocal, config, render_events=result.event_count
            )
    except Exception:
        quality = None

    status_message = "\n".join(
        [
            f"Wrote: {result.output_path}",
            f"Meows rendered: {result.event_count}",
            f"Stem strategy: {result.stem_strategy}",
            f"Meow sample: {result.meow_sample.source_type}",
        ]
    )

    if quality is not None:
        score_md = (
            f"### 🐱 MeowScore: {quality.meow_score:.2f} / 1.00\n"
            f"Rhythm **{quality.onset_f_measure:.2f}** · Melody **{max(0.0, quality.pitch_correlation):.2f}** · "
            f"Dynamics **{max(0.0, quality.envelope_correlation):.2f}** · {result.event_count} meows"
        )
        metrics = quality.as_dict()
    else:
        score_md = f"### 🐱 Rendered {result.event_count} meows\n(Quality metrics unavailable.)"
        metrics = {}

    vocal_output_path = None
    if result.vocal_stem is not None:
        from .io import write_wav

        vocal_output_path = str(output_dir / f"meowsic-{run_id}-vocals.wav")
        write_wav(vocal_output_path, result.vocal_stem)

    figure = _build_visualization(result, config)
    output = str(result.output_path)
    return status_message, score_md, metrics, output, output, figure, vocal_output_path


def _build_visualization(result, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from .analysis import estimate_pitch_contour

    fig, (ax_wave, ax_pitch) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    if result.vocal_stem is not None:
        samples = result.vocal_stem.mono()
        times = np.linspace(0, result.duration, len(samples))
        ax_wave.plot(times, samples, color="#9aa4b2", alpha=0.8, linewidth=0.6, label="Vocal")
    for event in result.events:
        ax_wave.axvline(event.start, color="#f59e0b", alpha=0.35, linewidth=0.7)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(f"Vocal waveform with {len(result.events)} meow triggers")
    ax_wave.legend(loc="upper right", fontsize=8)

    if result.pitch_contour is not None:
        voiced = result.pitch_contour.voiced
        ax_pitch.scatter(
            result.pitch_contour.times[voiced],
            result.pitch_contour.f0_hz[voiced],
            s=4,
            color="#2563eb",
            alpha=0.5,
            label="Original pitch",
        )
    if result.meow_vocal is not None:
        meow_contour = estimate_pitch_contour(result.meow_vocal, config)
        mv = meow_contour.voiced
        ax_pitch.scatter(
            meow_contour.times[mv],
            meow_contour.f0_hz[mv],
            s=6,
            color="#ea580c",
            alpha=0.6,
            label="Meow pitch",
        )
    ax_pitch.axhspan(config.cat_min_pitch_hz, config.cat_max_pitch_hz, color="#a78bfa", alpha=0.15, label="Cat register")
    ax_pitch.set_ylabel("Pitch (Hz)")
    ax_pitch.set_xlabel("Time (s)")
    ax_pitch.set_ylim(0, max(config.cat_max_pitch_hz * 1.5, 700))
    ax_pitch.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


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
