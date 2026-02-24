import atexit
import queue
import tempfile
import threading
from pathlib import Path

import gradio as gr

from transcribe.core import save_txt, transcribe_video

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


def run_transcription(video_file, model_size, language):
    if video_file is None:
        raise gr.Error("Please upload a video file.")

    lang = language if language else None

    progress_queue: queue.Queue = queue.Queue()
    result_holder: list[dict] = []
    error_holder: list[Exception] = []

    def on_progress(fraction: float, message: str):
        progress_queue.put((fraction, message))

    def _run():
        try:
            result = transcribe_video(
                video_file,
                model_size=model_size,
                language=lang,
                progress_callback=on_progress,
            )
            result_holder.append(result)
        except Exception as exc:
            error_holder.append(exc)
        progress_queue.put(None)  # signal done

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Yield progress updates as HTML bar
    while True:
        try:
            item = progress_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            break
        fraction, message = item
        # Yield: transcript, txt_file, info, progress_bar
        yield gr.update(), gr.update(), gr.update(), _render_progress(fraction, message)

    thread.join()

    if error_holder:
        raise gr.Error(str(error_holder[0]))

    result = result_holder[0]
    tmp_path = Path(_tmp_dir.name)
    stem = Path(video_file).stem

    txt_path = save_txt(result, tmp_path / f"{stem}.txt")

    info = f"Language: {result['language']} | Segments: {len(result['segments'])}"

    # Final yield — populate results and clear progress bar
    yield result["text"], str(txt_path), info, ""


def create_app():
    with gr.Blocks(title="Video Transcriber") as app:
        gr.Markdown("# Video Transcriber\nUpload a video file to transcribe using whisper.cpp.")

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                model_dropdown = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="large-v3",
                    label="Model Size",
                )
                language_input = gr.Textbox(
                    label="Language Code (leave empty for auto-detect)",
                    placeholder="e.g. en, fr, de, ja",
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary")
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
            inputs=[video_input, model_dropdown, language_input],
            outputs=[transcript_output, txt_download, info_text, progress_bar],
        )

    return app


def main():
    app = create_app()
    # Force local-only hosting by default for safer operation.
    app.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
