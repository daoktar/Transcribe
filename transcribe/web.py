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


def _render_progress(fraction: float, message: str) -> str:
    """Render an HTML progress bar with status text."""
    pct = max(0, min(100, int(fraction * 100)))
    return (
        f"<div style='background:#333; border-radius:4px; overflow:hidden; "
        f"height:28px; margin-top:8px; position:relative'>"
        f"<div style='width:{pct}%; background:#f97316; height:100%; "
        f"transition:width 0.3s ease'></div>"
        f"<span style='position:absolute; top:0; left:0; right:0; height:100%; "
        f"display:flex; align-items:center; justify-content:center; "
        f"font-size:13px; color:#fff; font-weight:500'>"
        f"{message}</span>"
        f"</div>"
    )


def _format_result(result, media_file):
    """Build transcript text, info string, and saved file path from a result dict."""
    tmp_path = Path(_tmp_dir.name)
    stem = Path(media_file).stem
    txt_path = save_txt(result, tmp_path / f"{stem}.txt")

    # Format transcript with speaker labels when available
    if result.get("speakers") and result["segments"]:
        lines = []
        for seg in result["segments"]:
            lines.append(f"{seg['speaker']}: {seg['text']}")
        transcript_text = "\n".join(lines)
    else:
        transcript_text = result["text"]

    info = f"Language: {result['language']} | Segments: {len(result['segments'])}"
    if result.get("speakers"):
        info += f" | Speakers: {result['speakers']}"
    if result.get("diarize_error"):
        info += (
            f"\n⚠️ Speaker detection failed: {result['diarize_error']}"
            "\nFix your token above and click \"Retry Speaker Detection\"."
        )

    return transcript_text, str(txt_path), info


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
    # Outputs: transcript, txt_file, info, progress_bar, cached_result, retry_btn
    while True:
        try:
            item = progress_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            break
        fraction, message = item
        yield (gr.update(), gr.update(), gr.update(),
               _render_progress(fraction, message), gr.update(), gr.update())

    thread.join()

    if error_holder:
        raise gr.Error(str(error_holder[0]))

    result = result_holder[0]
    transcript_text, txt_path, info = _format_result(result, media_file)

    # Show retry button if diarization failed (transcript is preserved)
    show_retry = bool(result.get("diarize_error"))

    # Final yield — populate results and clear progress bar
    yield (transcript_text, txt_path, info, "",
           result, gr.update(visible=show_retry))


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

    # Outputs: transcript, txt_file, info, progress_bar, cached_result, retry_btn
    while True:
        try:
            item = progress_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            break
        fraction, message = item
        yield (gr.update(), gr.update(), gr.update(),
               _render_progress(fraction, message), gr.update(), gr.update())

    thread.join()

    if error_holder:
        raise gr.Error(str(error_holder[0]))

    result = result_holder[0]
    transcript_text, txt_path, info = _format_result(result, media_file)

    show_retry = bool(result.get("diarize_error"))

    yield (transcript_text, txt_path, info, "",
           result, gr.update(visible=show_retry))


def create_app():
    _file_types = sorted(SUPPORTED_EXTENSIONS)

    with gr.Blocks(title="Media Transcriber") as app:
        gr.Markdown("# Media Transcriber\nUpload a video or audio file to transcribe using whisper.cpp.")

        # Hidden state to cache transcription result for diarization retry
        cached_result = gr.State(value=None)

        with gr.Row():
            with gr.Column():
                media_input = gr.File(
                    label="Upload Video or Audio",
                    file_types=_file_types,
                )
                model_dropdown = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="large-v3",
                    label="Model Size",
                )
                language_input = gr.Textbox(
                    label="Language Code (leave empty for auto-detect)",
                    placeholder="e.g. en, fr, de, ja",
                )
                diarize_checkbox = gr.Checkbox(
                    label="Detect Speakers",
                    value=False,
                    info="Identify and label different speakers (requires HuggingFace token)",
                )
                hf_token_input = gr.Textbox(
                    label="HuggingFace Token",
                    placeholder="Required for speaker detection",
                    type="password",
                    visible=False,
                )
                diarize_checkbox.change(
                    fn=lambda checked: gr.update(visible=checked),
                    inputs=[diarize_checkbox],
                    outputs=[hf_token_input],
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary")
                retry_btn = gr.Button(
                    "Retry Speaker Detection",
                    variant="secondary",
                    visible=False,
                )
                progress_bar = gr.HTML(value="")

            with gr.Column():
                info_text = gr.Textbox(label="Info", interactive=False)
                transcript_output = gr.Textbox(
                    label="Transcript",
                    lines=15,
                    interactive=False,
                )
                txt_download = gr.File(label="Download TXT")

        transcribe_btn.click(
            fn=run_transcription,
            inputs=[media_input, model_dropdown, language_input,
                    diarize_checkbox, hf_token_input],
            outputs=[transcript_output, txt_download, info_text,
                     progress_bar, cached_result, retry_btn],
        )

        retry_btn.click(
            fn=run_retry_diarize,
            inputs=[media_input, hf_token_input, cached_result],
            outputs=[transcript_output, txt_download, info_text,
                     progress_bar, cached_result, retry_btn],
        )

    return app


def main():
    app = create_app()
    # Force local-only hosting by default for safer operation.
    app.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
