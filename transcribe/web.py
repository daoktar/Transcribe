import atexit
import tempfile
from pathlib import Path

import gradio as gr

from transcribe.core import save_json, save_txt, transcribe_video

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]

# Single reusable temp directory — cleaned up on process exit
_tmp_dir = tempfile.TemporaryDirectory(prefix="transcribe_")
atexit.register(_tmp_dir.cleanup)


def run_transcription(video_file, model_size, language, progress=gr.Progress()):
    if video_file is None:
        raise gr.Error("Please upload a video file.")

    lang = language if language else None

    def on_progress(fraction: float, message: str):
        progress(fraction, desc=message)

    result = transcribe_video(
        video_file,
        model_size=model_size,
        language=lang,
        progress_callback=on_progress,
    )

    tmp_path = Path(_tmp_dir.name)
    stem = Path(video_file).stem

    txt_path = save_txt(result, tmp_path / f"{stem}.txt")
    json_path = save_json(result, tmp_path / f"{stem}.json")

    info = f"Language: {result['language']} | Segments: {len(result['segments'])}"

    return result["text"], str(txt_path), str(json_path), info


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

            with gr.Column():
                info_text = gr.Textbox(label="Info", interactive=False)
                transcript_output = gr.Textbox(
                    label="Transcript",
                    lines=15,
                    interactive=False,
                )
                txt_download = gr.File(label="Download TXT")
                json_download = gr.File(label="Download JSON")

        transcribe_btn.click(
            fn=run_transcription,
            inputs=[video_input, model_dropdown, language_input],
            outputs=[transcript_output, txt_download, json_download, info_text],
        )

    return app


def main():
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
