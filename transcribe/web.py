import atexit
import queue
import tempfile
import threading
from pathlib import Path

import gradio as gr

from transcribe.core import (
    SUPPORTED_EXTENSIONS,
    retry_diarize,
    save_txt,
    transcribe_media,
)

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]

# Single reusable temp directory — cleaned up on process exit
_tmp_dir = tempfile.TemporaryDirectory(prefix="transcribe_")
atexit.register(_tmp_dir.cleanup)

# ---------------------------------------------------------------------------
# Theme & styling
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#eef2ff", c100="#dbe4ff", c200="#bac8ff", c300="#91a7ff",
        c400="#748ffc", c500="#4f7df5", c600="#4263eb", c700="#3b5bdb",
        c800="#364fc7", c900="#2b44a8", c950="#1e3a8a",
    ),
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#f8f9fb",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 1px 3px rgba(0,0,0,0.06)",
    block_radius="12px",
    input_radius="8px",
    input_border_width="1px",
    button_primary_background_fill="#4f7df5",
    button_primary_text_color="white",
    button_large_radius="10px",
)

CUSTOM_CSS = """
.main-container {
    max-width: 820px !important;
    margin: 0 auto !important;
}
.app-header {
    text-align: center;
    padding: 16px 0 4px;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.app-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 2px;
}
.app-header p {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 0;
}
.format-hint {
    text-align: center;
    font-size: 0.78rem;
    color: #94a3b8;
    margin: -4px 0 8px;
}
.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin: 0 !important;
    padding: 0 !important;
}
.section-label p {
    margin: 0 !important;
    padding: 0 !important;
}
.transcribe-btn {
    margin-top: 4px !important;
}
.transcribe-btn button {
    transition: box-shadow 0.2s, transform 0.15s !important;
}
.transcribe-btn button:hover {
    box-shadow: 0 4px 12px rgba(79,125,245,0.35) !important;
    transform: translateY(-1px) !important;
}
.progress-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.transcript-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.6 !important;
    font-size: 0.88rem !important;
}
.results-group {
    border-top: 2px solid #eef2ff !important;
    padding-top: 12px !important;
}
footer {
    display: none !important;
}
.custom-footer {
    text-align: center;
    padding: 12px 0 8px;
    font-size: 0.78rem;
    color: #94a3b8;
}
"""


def _render_progress(fraction: float, message: str) -> str:
    """Render an HTML progress bar with status text."""
    pct = max(0, min(100, int(fraction * 100)))
    is_done = pct >= 100
    text_color = "#fff" if pct > 45 else "#475569"

    if is_done:
        fill = "linear-gradient(90deg, #22c55e, #16a34a)"
        icon = " &#10003;"
    else:
        fill = "linear-gradient(90deg, #4f7df5, #6b8cf7)"
        icon = ""

    shimmer = (
        ""
        if is_done
        else (
            "background-image: linear-gradient(90deg, transparent 0%, "
            "rgba(255,255,255,0.25) 50%, transparent 100%);"
            "background-size: 200% 100%;"
            "animation: shimmer 1.8s infinite;"
        )
    )

    return (
        f"<style>@keyframes shimmer {{0%{{background-position:200% 0}}"
        f"100%{{background-position:-200% 0}}}}</style>"
        f"<div style='background:#e8ecf1; border-radius:10px; overflow:hidden; "
        f"height:32px; position:relative; box-shadow:inset 0 1px 2px rgba(0,0,0,0.06)'>"
        f"<div style='width:{pct}%; background:{fill}; height:100%; "
        f"border-radius:10px; "
        f"transition:width 0.4s cubic-bezier(0.4,0,0.2,1); {shimmer}'></div>"
        f"<span style='position:absolute; top:0; left:0; right:0; height:100%; "
        f"display:flex; align-items:center; justify-content:center; "
        f"font-size:13px; color:{text_color}; font-weight:600'>"
        f"{message}{icon}</span>"
        f"</div>"
    )


def _format_result(result, media_file):
    """Build transcript texts, info string, and saved file path from a result dict.

    Returns (plain_text, speaker_text, txt_path, info).
    ``speaker_text`` is non-empty only when speaker labels are available.
    """
    tmp_path = Path(_tmp_dir.name)
    stem = Path(media_file).stem
    txt_path = save_txt(result, tmp_path / f"{stem}.txt")

    plain_text = result["text"]
    speaker_text = ""

    if result.get("speakers") and result["segments"]:
        lines = []
        for seg in result["segments"]:
            lines.append(f"{seg['speaker']}: {seg['text']}")
        speaker_text = "\n".join(lines)

    info = f"Language: {result['language']} | Segments: {len(result['segments'])}"
    if result.get("speakers"):
        info += f" | Speakers: {result['speakers']}"
    if result.get("diarize_error"):
        info += (
            f"\n⚠️ Speaker detection failed: {result['diarize_error']}"
            "\nFix your token above and click \"Retry Speaker Detection\"."
        )

    return plain_text, speaker_text, str(txt_path), info


def run_transcription(media_file, model_size, language, diarize_speakers, hf_token):
    if media_file is None:
        raise gr.Error("Please upload a video or audio file.")

    ext = Path(media_file).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise gr.Error(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if diarize_speakers and not hf_token:
        raise gr.Error(
            "Speaker detection requires a HuggingFace token. "
            "Enter your token or get one at huggingface.co/settings/tokens"
        )

    lang = language if language else None

    progress_queue: queue.Queue = queue.Queue()
    result_holder: list[dict] = []
    error_holder: list[Exception] = []

    def on_progress(fraction: float, message: str):
        progress_queue.put((fraction, message))

    def _run():
        try:
            result = transcribe_media(
                media_file,
                model_size=model_size,
                language=lang,
                progress_callback=on_progress,
                diarize=diarize_speakers,
                hf_token=hf_token if diarize_speakers else None,
            )
            result_holder.append(result)
        except Exception as exc:
            error_holder.append(exc)
        progress_queue.put(None)  # signal done

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Yield progress updates as HTML bar
    # Outputs: transcript, speaker_transcript, txt_file, info,
    #          progress_bar, cached_result, retry_btn, speaker_tab
    while True:
        try:
            item = progress_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            break
        fraction, message = item
        yield (gr.update(), gr.update(), gr.update(), gr.update(),
               _render_progress(fraction, message), gr.update(),
               gr.update(), gr.update())

    thread.join()

    if error_holder:
        raise gr.Error(str(error_holder[0]))

    result = result_holder[0]
    plain_text, speaker_text, txt_path, info = _format_result(result, media_file)

    show_retry = bool(result.get("diarize_error"))
    has_speakers = bool(speaker_text)

    yield (plain_text, speaker_text, txt_path, info, "",
           result, gr.update(visible=show_retry),
           gr.update(visible=has_speakers))


def run_retry_diarize(media_file, hf_token, cached_result):
    """Retry only the speaker diarization step using the cached transcription."""
    if cached_result is None or not cached_result.get("segments"):
        raise gr.Error("No transcription to retry. Please transcribe first.")

    if not hf_token:
        raise gr.Error(
            "Speaker detection requires a HuggingFace token. "
            "Enter your token or get one at huggingface.co/settings/tokens"
        )

    if media_file is None:
        raise gr.Error("Original media file is required for speaker detection retry.")

    progress_queue: queue.Queue = queue.Queue()
    result_holder: list[dict] = []
    error_holder: list[Exception] = []

    def on_progress(fraction: float, message: str):
        progress_queue.put((fraction, message))

    def _run():
        try:
            result = retry_diarize(
                media_file,
                cached_result,
                hf_token=hf_token,
                progress_callback=on_progress,
            )
            result_holder.append(result)
        except Exception as exc:
            error_holder.append(exc)
        progress_queue.put(None)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Outputs: transcript, speaker_transcript, txt_file, info,
    #          progress_bar, cached_result, retry_btn, speaker_tab
    while True:
        try:
            item = progress_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            break
        fraction, message = item
        yield (gr.update(), gr.update(), gr.update(), gr.update(),
               _render_progress(fraction, message), gr.update(),
               gr.update(), gr.update())

    thread.join()

    if error_holder:
        raise gr.Error(str(error_holder[0]))

    result = result_holder[0]
    plain_text, speaker_text, txt_path, info = _format_result(result, media_file)

    show_retry = bool(result.get("diarize_error"))
    has_speakers = bool(speaker_text)

    yield (plain_text, speaker_text, txt_path, info, "",
           result, gr.update(visible=show_retry),
           gr.update(visible=has_speakers))


def create_app():
    from pywhispercpp.constants import MODELS_DIR as _models_dir

    _file_types = sorted(SUPPORTED_EXTENSIONS)
    _format_list = ", ".join(
        ext.lstrip(".").upper() for ext in sorted(SUPPORTED_EXTENSIONS)
    )

    with gr.Blocks(title="Media Transcriber") as app:

        # --- Header ---
        gr.Markdown(
            "# Media Transcriber\nTranscribe video and audio using whisper.cpp",
            elem_classes=["app-header"],
        )

        # Hidden state to cache transcription result for diarization retry
        cached_result = gr.State(value=None)

        with gr.Column(elem_classes=["main-container"]):

            # --- Upload ---
            media_input = gr.File(
                label="Upload Media File",
                file_types=_file_types,
            )
            gr.Markdown(
                f"Supported formats: {_format_list}",
                elem_classes=["format-hint"],
            )

            # --- Settings ---
            with gr.Group():
                gr.Markdown("Settings", elem_classes=["section-label"])
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_CHOICES,
                        value="large-v3",
                        label="Model",
                        info="Larger models are more accurate but slower",
                        scale=2,
                    )
                    language_input = gr.Textbox(
                        label="Language",
                        placeholder="Auto-detect",
                        info="ISO 639-1 code (en, fr, de, ja, ...)",
                        scale=1,
                    )
                diarize_checkbox = gr.Checkbox(
                    label="Detect Speakers",
                    value=False,
                    info="Identify and label different speakers (requires HuggingFace token)",
                )
                hf_token_input = gr.Textbox(
                    label="HuggingFace Token",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                    info="Get a token at huggingface.co/settings/tokens",
                    type="password",
                    visible=False,
                )
                diarize_checkbox.change(
                    fn=lambda checked: gr.update(visible=checked),
                    inputs=[diarize_checkbox],
                    outputs=[hf_token_input],
                )

            # --- Actions ---
            with gr.Row():
                transcribe_btn = gr.Button(
                    "Transcribe",
                    variant="primary",
                    size="lg",
                    scale=2,
                    elem_classes=["transcribe-btn"],
                )
                retry_btn = gr.Button(
                    "Retry Speaker Detection",
                    variant="secondary",
                    size="lg",
                    scale=1,
                    visible=False,
                )

            # --- Progress ---
            progress_bar = gr.HTML(value="", elem_classes=["progress-container"])

            # --- Results ---
            with gr.Group(elem_classes=["results-group"]):
                info_text = gr.Textbox(
                    label="Info",
                    interactive=False,
                    lines=1,
                    max_lines=3,
                )
                with gr.Tabs():
                    with gr.Tab("Transcript"):
                        transcript_output = gr.Textbox(
                            label="Transcript",
                            lines=20,
                            max_lines=40,
                            interactive=False,
                            buttons=["copy"],
                            elem_classes=["transcript-box"],
                        )
                    with gr.Tab("Speakers", visible=False) as speaker_tab:
                        speaker_output = gr.Textbox(
                            label="Transcript with Speakers",
                            lines=20,
                            max_lines=40,
                            interactive=False,
                            buttons=["copy"],
                            elem_classes=["transcript-box"],
                        )
                txt_download = gr.File(label="Download Transcript")

        # --- Storage info (collapsible) ---
        with gr.Accordion("Storage & file locations", open=False):
            gr.Markdown(
                f"**Whisper models**\n\n"
                f"`{_models_dir}`\n\n"
                f"**Transcript files**\n\n"
                f"Stored in a temporary system directory (cleared on exit)"
            )

        # --- Footer ---
        gr.Markdown(
            "Media Transcriber — powered by whisper.cpp",
            elem_classes=["custom-footer"],
        )

        # --- Event wiring ---
        _outputs = [transcript_output, speaker_output, txt_download,
                     info_text, progress_bar, cached_result,
                     retry_btn, speaker_tab]

        transcribe_btn.click(
            fn=run_transcription,
            inputs=[media_input, model_dropdown, language_input,
                    diarize_checkbox, hf_token_input],
            outputs=_outputs,
        )

        retry_btn.click(
            fn=run_retry_diarize,
            inputs=[media_input, hf_token_input, cached_result],
            outputs=_outputs,
        )

    return app


def main():
    app = create_app()
    # Force local-only hosting by default for safer operation.
    app.launch(server_name="127.0.0.1", share=False, theme=THEME, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
