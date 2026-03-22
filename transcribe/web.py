import atexit
import html as html_mod
import json
import queue
import tempfile
import threading
import zipfile
from pathlib import Path

import gradio as gr

from transcribe.core import (
    SUPPORTED_EXTENSIONS,
    retry_diarize,
    save_txt,
    save_txt_alongside,
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
.batch-queue-item {
    display: flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 0.85rem;
    gap: 8px;
}
.batch-queue-item.pending { background: #f1f5f9; color: #64748b; }
.batch-queue-item.processing { background: #eef2ff; color: #3b5bdb; font-weight: 600; }
.batch-queue-item.done { background: #f0fdf4; color: #16a34a; }
.batch-queue-item.error { background: #fef2f2; color: #dc2626; }
.batch-queue-item.cancelled { background: #f9fafb; color: #9ca3af; }
"""


def _render_progress(fraction: float, message: str) -> str:
    """Render an HTML progress bar with status text."""
    pct = max(0, min(100, int(fraction * 100)))
    message = html_mod.escape(message)
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


def _render_batch_queue(filenames, statuses, errors=None):
    """Render the batch file queue as styled HTML."""
    if not filenames:
        return ""
    errors = errors or {}
    icons = {
        "pending": "&#9679;",      # bullet
        "processing": "&#9655;",   # play triangle
        "done": "&#10003;",        # checkmark
        "error": "&#10007;",       # x mark
        "cancelled": "&#8212;",    # em dash
    }
    items = []
    for i, name in enumerate(filenames):
        status = statuses[i]
        icon = icons.get(status, "")
        label = html_mod.escape(name)
        if status == "error" and i in errors:
            label += f" — {html_mod.escape(str(errors[i]))}"
        elif status == "done" and i in errors:
            label += f" — {html_mod.escape(str(errors[i]))}"
        items.append(
            f"<div class='batch-queue-item {status}'>"
            f"<span>{icon}</span><span>{label}</span></div>"
        )
    return "<div style='margin:8px 0'>" + "".join(items) + "</div>"


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


def run_transcription(
    media_files, model_size, language, diarize_speakers, hf_token,
    save_alongside, original_paths_json, cancel_state,
):
    """Transcribe one or more files sequentially with batch progress."""

    # Normalise: single file comes as a str, multiple as a list
    if media_files is None:
        raise gr.Error("Please upload at least one media file.")
    if isinstance(media_files, str):
        media_files = [media_files]
    if not media_files:
        raise gr.Error("Please upload at least one media file.")

    # Parse original paths supplied by the native file picker (if any)
    orig_paths: list[str | None] = [None] * len(media_files)
    if original_paths_json:
        try:
            parsed = json.loads(original_paths_json)
            if isinstance(parsed, list):
                for i, p in enumerate(parsed):
                    if i < len(orig_paths):
                        orig_paths[i] = p
        except (json.JSONDecodeError, TypeError):
            pass

    # Validate extensions up-front
    for mf in media_files:
        ext = Path(mf).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise gr.Error(
                f"Unsupported file format '{ext}' ({Path(mf).name}). "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

    if diarize_speakers and not hf_token:
        raise gr.Error(
            "Speaker detection requires a HuggingFace token. "
            "Enter your token or get one at huggingface.co/settings/tokens"
        )

    lang = language if language else None
    total = len(media_files)
    filenames = [Path(mf).name for mf in media_files]
    statuses = ["pending"] * total
    errors: dict[int, str] = {}
    all_results: dict[int, dict] = {}
    all_txt_paths: list[str] = []

    # Per-session cancel event — avoids cross-session interference
    cancel_event = threading.Event()
    if isinstance(cancel_state, dict):
        cancel_state.clear()
    cancel_state = {"event": cancel_event}

    # Outputs: transcript, speaker_transcript, txt_file, info,
    #          progress_bar, cached_results, retry_btn, speaker_tab,
    #          batch_queue_html, file_selector, cancel_btn, cancel_state
    _skip = gr.skip()

    for idx, media_file in enumerate(media_files):
        if cancel_event.is_set():
            for j in range(idx, total):
                statuses[j] = "cancelled"
            queue_html = _render_batch_queue(filenames, statuses, errors)
            yield (_skip, _skip, _skip, _skip, _skip, _skip, _skip, _skip,
                   queue_html, _skip, _skip, cancel_state)
            break

        statuses[idx] = "processing"
        queue_html = _render_batch_queue(filenames, statuses, errors)
        batch_msg = f"File {idx + 1} of {total}: {filenames[idx]}"

        yield (_skip, _skip, _skip, _skip,
               _render_progress(0, batch_msg), _skip, _skip, _skip,
               queue_html, _skip, gr.Button(visible=True), cancel_state)

        # Run transcription in a thread
        progress_queue: queue.Queue = queue.Queue()
        result_holder: list[dict] = []
        error_holder: list[Exception] = []

        def on_progress(fraction: float, message: str):
            progress_queue.put((fraction, message))

        def _run(_mf=media_file):
            try:
                result = transcribe_media(
                    _mf,
                    model_size=model_size,
                    language=lang,
                    progress_callback=on_progress,
                    diarize=diarize_speakers,
                    hf_token=hf_token if diarize_speakers else None,
                )
                result_holder.append(result)
            except Exception as exc:
                error_holder.append(exc)
            progress_queue.put(None)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        while True:
            try:
                item = progress_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            fraction, message = item
            combined_msg = f"[{idx + 1}/{total}] {message}"
            yield (_skip, _skip, _skip, _skip,
                   _render_progress(fraction, combined_msg), _skip, _skip, _skip,
                   _skip, _skip, _skip, _skip)

        thread.join()

        if error_holder:
            statuses[idx] = "error"
            errors[idx] = str(error_holder[0])
        else:
            statuses[idx] = "done"
            result = result_holder[0]
            all_results[idx] = result

            # Save to temp dir
            _, _, txt_path, _ = _format_result(result, media_file)
            all_txt_paths.append(txt_path)

            # Save alongside original if requested and path is known
            if save_alongside and orig_paths[idx]:
                try:
                    saved = save_txt_alongside(result, orig_paths[idx])
                    errors[idx] = f"Saved to {Path(saved).name}"
                except (PermissionError, OSError):
                    errors[idx] = "Could not save alongside source (permission denied)"

        queue_html = _render_batch_queue(filenames, statuses, errors)
        yield (_skip, _skip, _skip, _skip, _skip, _skip, _skip, _skip,
               queue_html, _skip, _skip, _skip)

    # --- Batch complete ---
    done_count = sum(1 for s in statuses if s == "done")
    err_count = sum(1 for s in statuses if s == "error")
    summary_parts = [f"{done_count} of {total} completed"]
    if err_count:
        summary_parts.append(f"{err_count} failed")

    # Show the first successful result in the transcript area
    first_result_idx = next((i for i in range(total) if i in all_results), None)

    if first_result_idx is not None:
        result = all_results[first_result_idx]
        plain_text, speaker_text, _, info = _format_result(
            result, media_files[first_result_idx]
        )
        show_retry = bool(result.get("diarize_error"))
        has_speakers = bool(speaker_text)
    else:
        plain_text = ""
        speaker_text = ""
        info = "All files failed."
        show_retry = False
        has_speakers = False

    # Build download file(s)
    if len(all_txt_paths) == 1:
        download_path = all_txt_paths[0]
    elif len(all_txt_paths) > 1:
        zip_path = Path(_tmp_dir.name) / "transcripts.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for tp in all_txt_paths:
                zf.write(tp, Path(tp).name)
        download_path = str(zip_path)
    else:
        download_path = None

    info = f"{' | '.join(summary_parts)}\n{info}" if first_result_idx is not None else info

    # Build file selector choices — use index prefix to disambiguate duplicates
    selector_choices = [
        f"{i + 1}. {filenames[i]}" for i in sorted(all_results.keys())
    ]

    queue_html = _render_batch_queue(filenames, statuses, errors)

    yield (
        plain_text, speaker_text, download_path, info,
        _render_progress(1.0, f"Done — {' | '.join(summary_parts)}"),
        all_results,
        gr.Button(visible=show_retry),
        gr.Tab(visible=has_speakers),
        queue_html,
        gr.Dropdown(choices=selector_choices,
                    value=selector_choices[0] if selector_choices else None,
                    visible=len(selector_choices) > 1),
        gr.Button(visible=False),  # hide cancel btn
        cancel_state,
    )


def _on_file_select(filename, all_results, media_files):
    """Switch displayed transcript when user picks a file from the dropdown."""
    if not filename or not all_results or not media_files:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()

    if isinstance(media_files, str):
        media_files = [media_files]

    # Parse index from "1. filename.mp4" format
    try:
        idx = int(filename.split(".", 1)[0]) - 1
    except (ValueError, IndexError):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip()

    if isinstance(all_results, dict) and idx in all_results and idx < len(media_files):
        result = all_results[idx]
        plain_text, speaker_text, _, info = _format_result(result, media_files[idx])
        has_speakers = bool(speaker_text)
        return plain_text, speaker_text, info, gr.Tab(visible=has_speakers)

    return gr.skip(), gr.skip(), gr.skip(), gr.skip()


def _on_cancel_click(cancel_state):
    """Set the cancellation flag for batch processing."""
    if isinstance(cancel_state, dict) and "event" in cancel_state:
        cancel_state["event"].set()
    return gr.Button(interactive=False, value="Cancelling...")


def run_retry_diarize(media_files, hf_token, cached_results):
    """Retry only the speaker diarization step using the cached transcription."""
    # Use the first file for retry (single-file behaviour preserved)
    if isinstance(media_files, list):
        media_file = media_files[0] if media_files else None
    else:
        media_file = media_files

    # Get the first cached result
    if isinstance(cached_results, dict) and cached_results:
        first_key = min(cached_results.keys(), key=int) if cached_results else None
        cached_result = cached_results[first_key] if first_key is not None else None
    else:
        cached_result = cached_results

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

    _skip = gr.skip()
    # Outputs: transcript, speaker_transcript, txt_file, info,
    #          progress_bar, cached_results, retry_btn, speaker_tab,
    #          batch_queue, file_selector, cancel_btn, cancel_state
    while True:
        try:
            item = progress_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            break
        fraction, message = item
        yield (_skip, _skip, _skip, _skip,
               _render_progress(fraction, message), _skip,
               _skip, _skip, _skip, _skip, _skip, _skip)

    thread.join()

    if error_holder:
        raise gr.Error(str(error_holder[0]))

    result = result_holder[0]
    plain_text, speaker_text, txt_path, info = _format_result(result, media_file)

    show_retry = bool(result.get("diarize_error"))
    has_speakers = bool(speaker_text)

    yield (plain_text, speaker_text, txt_path, info, "",
           {0: result}, gr.Button(visible=show_retry),
           gr.Tab(visible=has_speakers), _skip, _skip, _skip, _skip)


def create_app(native_mode=False):
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

        # Hidden state for batch results, original paths, and cancellation
        cached_results = gr.State(value=None)
        cancel_state = gr.State(value=None)
        original_paths = gr.Textbox(visible=False, elem_id="original_paths")

        with gr.Column(elem_classes=["main-container"]):

            # --- Upload ---
            media_input = gr.File(
                label="Upload Media Files",
                file_types=_file_types,
                file_count="multiple",
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
                    fn=lambda checked: gr.Textbox(visible=checked),
                    inputs=[diarize_checkbox],
                    outputs=[hf_token_input],
                )

                save_alongside_checkbox = gr.Checkbox(
                    label="Save transcripts alongside source files",
                    value=native_mode,
                    visible=native_mode,
                    info="Write .txt next to each original media file",
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
                cancel_btn = gr.Button(
                    "Cancel Remaining",
                    variant="stop",
                    size="lg",
                    scale=1,
                    visible=False,
                )

            # --- Batch queue display ---
            batch_queue_html = gr.HTML(value="", elem_classes=["progress-container"])

            # --- Progress ---
            progress_bar = gr.HTML(value="", elem_classes=["progress-container"])

            # --- Results ---
            with gr.Group(elem_classes=["results-group"]):
                file_selector = gr.Dropdown(
                    label="View results for",
                    choices=[],
                    visible=False,
                )
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
                     info_text, progress_bar, cached_results,
                     retry_btn, speaker_tab,
                     batch_queue_html, file_selector, cancel_btn,
                     cancel_state]

        transcribe_btn.click(
            fn=run_transcription,
            inputs=[media_input, model_dropdown, language_input,
                    diarize_checkbox, hf_token_input,
                    save_alongside_checkbox, original_paths, cancel_state],
            outputs=_outputs,
        )

        retry_btn.click(
            fn=run_retry_diarize,
            inputs=[media_input, hf_token_input, cached_results],
            outputs=_outputs,
        )

        cancel_btn.click(
            fn=_on_cancel_click,
            inputs=[cancel_state],
            outputs=[cancel_btn],
        )

        file_selector.change(
            fn=_on_file_select,
            inputs=[file_selector, cached_results, media_input],
            outputs=[transcript_output, speaker_output, info_text, speaker_tab],
        )

    return app


