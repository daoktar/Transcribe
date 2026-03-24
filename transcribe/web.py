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
# Theme & styling — Dark navy design from Figma
# ---------------------------------------------------------------------------

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#dae2fd", c100="#aac7ff", c200="#7fadff", c300="#5a94ff",
        c400="#3e90ff", c500="#3e90ff", c600="#3578d9", c700="#2c63b3",
        c800="#234e8d", c900="#1a3a67", c950="#003064",
    ),
    secondary_hue=gr.themes.Color(
        c50="#dae2fd", c100="#c1c6d7", c200="#414755", c300="#31394d",
        c400="#222a3d", c500="#1a2236", c600="#131b2e", c700="#0b1326",
        c800="#060e20", c900="#030a18", c950="#010510",
    ),
    neutral_hue=gr.themes.Color(
        c50="#dae2fd", c100="#c1c6d7", c200="#8a91a4", c300="#6b7280",
        c400="#414755", c500="#31394d", c600="#222a3d", c700="#131b2e",
        c800="#0b1326", c900="#060e20", c950="#030a18",
    ),
    font=["Inter", "-apple-system", "BlinkMacSystemFont", "system-ui", "sans-serif"],
    font_mono=["JetBrains Mono", "SF Mono", "Menlo", "monospace"],
).set(
    body_background_fill="#0b1326",
    body_text_color="#dae2fd",
    body_text_color_subdued="#c1c6d7",
    block_background_fill="#131b2e",
    block_border_width="1px",
    block_border_color="rgba(65,71,85,0.1)",
    block_shadow="none",
    block_radius="12px",
    block_label_text_color="#c1c6d7",
    block_title_text_color="#dae2fd",
    input_background_fill="#222a3d",
    input_border_color="rgba(65,71,85,0.2)",
    input_border_width="1px",
    input_radius="8px",
    input_placeholder_color="rgba(65,71,85,0.5)",
    button_primary_background_fill="linear-gradient(165deg, #aac7ff 0%, #3e90ff 100%)",
    button_primary_text_color="#003064",
    button_primary_border_color="transparent",
    button_secondary_background_fill="transparent",
    button_secondary_text_color="#c1c6d7",
    button_secondary_border_color="rgba(65,71,85,0.2)",
    button_cancel_background_fill="transparent",
    button_cancel_text_color="#ffb4ab",
    button_cancel_border_color="rgba(255,180,171,0.3)",
    button_large_radius="12px",
    checkbox_background_color="#222a3d",
    checkbox_border_color="rgba(65,71,85,0.2)",
    checkbox_label_text_color="#dae2fd",
    shadow_spread="0px",
    border_color_primary="rgba(65,71,85,0.1)",
)

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ===== Global ===== */
.gradio-container {
    max-width: 1280px !important;
    background: #0b1326 !important;
    overflow-x: hidden !important;
}
body, html { overflow-x: hidden !important; }
footer { display: none !important; }
.prose { color: #c1c6d7 !important; }
.prose h1, .prose h2, .prose h3 { color: #dae2fd !important; }

/* ===== App Header ===== */
.app-header {
    background: #0b1326 !important;
    border: none !important;
    box-shadow: none !important;
    padding: 14px 0 8px !important;
    margin-bottom: 0 !important;
}
.app-header h1 {
    font-family: 'Manrope', sans-serif !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #aac7ff !important;
    letter-spacing: 0.45px;
    margin: 0 !important;
}

/* ===== Tab Navigation (UPLOAD / PROCESS / REVIEW) ===== */
.dark-tabs {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.dark-tabs > .tab-wrapper {
    margin-top: -44px !important;
}
.dark-tabs > .tab-wrapper .tab-container[role="tablist"] {
    background: transparent !important;
    border: none !important;
    border-bottom: none !important;
    justify-content: flex-end !important;
    gap: 32px !important;
    padding: 0 !important;
}
.dark-tabs > .tab-wrapper .tab-container[role="tablist"] button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    color: #dae2fd !important;
    opacity: 0.6;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    letter-spacing: 1.4px !important;
    text-transform: uppercase !important;
    padding: 4px 0 6px !important;
    margin: 0 !important;
    transition: opacity 0.2s, border-color 0.2s !important;
}
.dark-tabs > .tab-wrapper .tab-container[role="tablist"] button.selected {
    opacity: 1 !important;
    color: #aac7ff !important;
    border-bottom-color: #aac7ff !important;
    font-weight: 600 !important;
}
.dark-tabs > .tab-wrapper .tab-container[role="tablist"] button:hover {
    opacity: 0.85 !important;
}
.dark-tabs > .tabitem {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ===== Upload Drop Zone ===== */
.upload-zone {
    background: #131b2e !important;
    border: 2px dashed rgba(65,71,85,0.3) !important;
    border-radius: 24px !important;
    min-height: 380px !important;
    box-shadow: none !important;
}
.upload-zone .wrap { color: #c1c6d7 !important; }
.upload-zone .or { color: #414755 !important; }

/* ===== Settings Panel ===== */
.settings-panel {
    background: #131b2e !important;
    border: 1px solid rgba(65,71,85,0.05) !important;
    border-radius: 24px !important;
    padding: 25px !important;
    box-shadow: none !important;
}
.settings-panel .section-title {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 700;
    font-size: 20px;
    color: #dae2fd;
}
.settings-panel label span {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #c1c6d7 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
}

/* ===== Format Tags ===== */
.format-tags {
    text-align: center;
    font-size: 0;
    padding: 8px 0;
    opacity: 0.4;
}
.format-tags span {
    display: inline-block;
    background: #222a3d;
    color: #dae2fd;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: -0.5px;
    padding: 4px 8px;
    border-radius: 8px;
    margin: 0 4px 4px 0;
}

/* ===== Supported Formats Box ===== */
.formats-box {
    background: #060e20 !important;
    border: 1px solid rgba(65,71,85,0.1) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
.formats-box .format-heading {
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: 12px;
    color: #c1c6d7;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}
.formats-box .format-list {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    line-height: 1.6;
    color: #c1c6d7;
}

/* ===== Transcribe Button ===== */
.transcribe-btn {
    margin-top: 8px !important;
}
.transcribe-btn button {
    background: linear-gradient(170deg, #aac7ff 0%, #3e90ff 100%) !important;
    color: #003064 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 18px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 20px 48px !important;
    box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25) !important;
    transition: box-shadow 0.2s, transform 0.15s, opacity 0.2s !important;
    max-width: 448px;
    margin: 0 auto;
}
.transcribe-btn button:hover {
    box-shadow: 0 25px 50px -8px rgba(62,144,255,0.3) !important;
    transform: translateY(-1px) !important;
}
.gpu-hint {
    text-align: center;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: #c1c6d7;
    opacity: 0.6;
    margin-top: 8px;
}

/* ===== Processing Card ===== */
.process-card {
    background: #131b2e !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: none !important;
    overflow: hidden;
    position: relative;
}
.process-card::after {
    content: '';
    position: absolute;
    top: -128px;
    right: -128px;
    width: 256px;
    height: 256px;
    background: #3e90ff;
    filter: blur(50px);
    opacity: 0.05;
    pointer-events: none;
}

/* ===== Log Section ===== */
.log-section {
    background: #060e20 !important;
    border: 1px solid rgba(65,71,85,0.1) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

/* ===== Job Queue ===== */
.job-queue {
    background: #131b2e !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: none !important;
}
.job-queue-header {
    font-family: 'Manrope', sans-serif;
    font-weight: 600;
    font-size: 18px;
    color: #dae2fd;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(65,71,85,0.1);
    margin-bottom: 16px;
}

/* ===== Cancel Button ===== */
.cancel-btn button {
    border: 1px solid rgba(255,180,171,0.3) !important;
    color: #ffb4ab !important;
    background: transparent !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 8px !important;
}

/* ===== Review Tab ===== */
.review-title {
    overflow: hidden !important;
}
.review-title h2 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
    font-size: 30px !important;
    color: #dae2fd !important;
    letter-spacing: -0.75px !important;
}
.review-subtitle p {
    font-family: 'Inter', sans-serif !important;
    color: #c1c6d7 !important;
    font-size: 16px !important;
}
.summary-box {
    background: #222a3d !important;
    border: 1px solid rgba(65,71,85,0.1) !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

/* ===== Transcript Canvas ===== */
.transcript-canvas {
    background: #060e20 !important;
    border: 1px solid rgba(65,71,85,0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25) !important;
    overflow: hidden;
}
.transcript-toolbar {
    background: #131b2e !important;
    border-bottom: 1px solid rgba(65,71,85,0.05) !important;
    border: none !important;
    box-shadow: none !important;
}
.transcript-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.6 !important;
    font-size: 14px !important;
    color: rgba(218,226,253,0.9) !important;
    background: #060e20 !important;
    max-height: 500px !important;
    overflow-y: auto !important;
}

/* ===== Download Button ===== */
.download-btn button {
    background: linear-gradient(7deg, #aac7ff 0%, #3e90ff 100%) !important;
    color: #003064 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
}
.retry-btn button {
    background: rgba(251,191,36,0.1) !important;
    color: #fbbf24 !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    border-radius: 8px !important;
}

/* ===== Transparent containers ===== */
.no-bg {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ===== File selector tabs (Review page) ===== */
.file-tabs {
    background: #131b2e !important;
    border-radius: 12px !important;
    padding: 6px !important;
    box-shadow: none !important;
    border: none !important;
}
.file-tabs > .tab-wrapper .tab-container[role="tablist"] {
    background: transparent !important;
    border: none !important;
    border-bottom: none !important;
    gap: 8px !important;
    padding: 0 !important;
}
.file-tabs > .tab-wrapper .tab-container[role="tablist"] button {
    background: transparent !important;
    border: none !important;
    color: #c1c6d7 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 16px !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    margin: 0 !important;
}
.file-tabs > .tab-wrapper .tab-container[role="tablist"] button.selected {
    background: #3e90ff !important;
    color: #002957 !important;
    box-shadow: 0 0 15px rgba(62,144,255,0.2) !important;
}
.file-tabs > .tabitem {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* ===== Info text styling ===== */
.info-display textarea {
    font-family: 'Inter', sans-serif !important;
    color: #c1c6d7 !important;
    background: transparent !important;
    border: none !important;
}
"""


def _render_progress(fraction: float, message: str) -> str:
    """Render a progress bar matching the Figma dark design."""
    pct = max(0, min(100, int(fraction * 100)))
    message = html_mod.escape(message)
    is_done = pct >= 100

    if is_done:
        fill = "linear-gradient(90deg, #3cddc7, #00a392)"
        glow = "0 0 15px rgba(60,221,199,0.4)"
    else:
        fill = "linear-gradient(90deg, #aac7ff, #3e90ff)"
        glow = "0 0 15px rgba(62,144,255,0.4)"

    shimmer = (
        ""
        if is_done
        else (
            "background-image: linear-gradient(90deg, transparent 0%, "
            "rgba(255,255,255,0.15) 50%, transparent 100%);"
            "background-size: 200% 100%;"
            "animation: shimmer 1.8s infinite;"
        )
    )

    return (
        f"<style>@keyframes shimmer {{0%{{background-position:200% 0}}"
        f"100%{{background-position:-200% 0}}}}</style>"
        f"<div style='text-align:center; padding:16px 0 8px'>"
        f"<div style='font-family:Inter,sans-serif; font-size:12px; "
        f"color:#c1c6d7; text-transform:uppercase; letter-spacing:1.2px; "
        f"margin-bottom:8px'>Current Activity</div>"
        f"<div style='font-family:Manrope,sans-serif; font-size:24px; "
        f"font-weight:600; color:#dae2fd; margin-bottom:24px'>{message}</div>"
        f"</div>"
        f"<div style='max-width:512px; margin:0 auto'>"
        f"<div style='background:#31394d; border-radius:9999px; overflow:hidden; "
        f"height:16px; position:relative'>"
        f"<div style='width:{pct}%; background:{fill}; height:100%; "
        f"border-radius:9999px; box-shadow:{glow}; "
        f"transition:width 0.4s cubic-bezier(0.4,0,0.2,1); {shimmer}'></div>"
        f"</div>"
        f"<div style='display:flex; justify-content:space-between; "
        f"margin-top:12px; font-family:monospace; font-size:14px; color:#c1c6d7'>"
        f"<span>0%</span>"
        f"<span style='font-weight:700; color:#aac7ff'>{pct}%</span>"
        f"<span>100%</span>"
        f"</div>"
        f"</div>"
    )


def _render_batch_queue(filenames, statuses, errors=None):
    """Render the batch file queue matching the Figma card design."""
    if not filenames:
        return ""
    errors = errors or {}

    badge_styles = {
        "processing": "background:#3e90ff; color:#002957;",
        "pending": "background:#31394d; color:#c1c6d7;",
        "done": "background:#00a392; color:#00302a;",
        "error": "background:#93000a; color:#ffdad6;",
        "cancelled": "background:#414755; color:#dae2fd;",
    }
    card_borders = {
        "processing": "border:none;",
        "pending": "border:1px solid rgba(65,71,85,0.1);",
        "done": "border:1px solid rgba(60,221,199,0.1);",
        "error": "border:1px solid rgba(255,180,171,0.1);",
        "cancelled": "border:none; opacity:0.4;",
    }

    items = []
    for i, name in enumerate(filenames):
        status = statuses[i]
        badge = badge_styles.get(status, badge_styles["pending"])
        border = card_borders.get(status, card_borders["pending"])
        label = html_mod.escape(name)
        note = ""
        if status in ("error", "done") and i in errors:
            note = f"<div style='font-size:11px; color:#c1c6d7; opacity:0.7; margin-top:4px'>{html_mod.escape(str(errors[i]))}</div>"

        progress_html = ""
        if status == "processing":
            card_bg = "#222a3d"
            progress_html = (
                "<div style='background:#171f33; height:4px; border-radius:9999px; "
                "overflow:hidden; margin-top:8px'>"
                "<div style='background:#aac7ff; width:45%; height:100%'></div>"
                "</div>"
            )
        elif status == "pending":
            card_bg = "#171f33"
        else:
            card_bg = "#171f33"

        items.append(
            f"<div style='background:{card_bg}; {border} border-radius:8px; "
            f"padding:16px; margin-bottom:8px'>"
            f"<div style='display:flex; align-items:center; justify-content:space-between'>"
            f"<span style='{badge} font-family:Inter,sans-serif; font-size:12px; "
            f"padding:2px 8px; border-radius:9999px'>{status.title()}</span>"
            f"</div>"
            f"<div style='font-family:Inter,sans-serif; font-weight:500; "
            f"font-size:14px; color:#dae2fd; margin-top:8px'>{label}</div>"
            f"{note}{progress_html}"
            f"</div>"
        )

    return "<div>" + "".join(items) + "</div>"


def _render_log_html(log_entries=None):
    """Render the log section matching the Figma dark design."""
    if not log_entries:
        return (
            "<div style='font-family:JetBrains Mono,monospace; font-size:12px; "
            "color:#c1c6d7; opacity:0.5; padding:8px 0'>Waiting for transcription to start...</div>"
        )
    lines = []
    for ts, msg in log_entries:
        lines.append(
            f"<div style='display:flex; gap:16px; margin-bottom:8px'>"
            f"<span style='color:#3cddc7; opacity:0.5; font-family:monospace; "
            f"font-size:12px; white-space:nowrap'>{html_mod.escape(ts)}</span>"
            f"<span style='color:#c1c6d7; font-family:monospace; "
            f"font-size:12px'>{html_mod.escape(msg)}</span>"
            f"</div>"
        )
    return "<div>" + "".join(lines) + "</div>"


def _render_summary_html(result):
    """Render the processing summary box for the Review tab."""
    if not result:
        return ""
    lang = html_mod.escape(str(result.get("language", "—")).upper())
    segments = len(result.get("segments", []))
    speakers = result.get("speakers", "—")
    has_error = bool(result.get("diarize_error"))
    status_badge = (
        "<span style='background:rgba(60,221,199,0.1); border:1px solid rgba(60,221,199,0.2); "
        "color:#3cddc7; font-size:10px; padding:3px 9px; border-radius:9999px'>Success</span>"
        if not has_error else
        "<span style='background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.2); "
        "color:#fbbf24; font-size:10px; padding:3px 9px; border-radius:9999px'>Warning</span>"
    )
    return (
        f"<div style='padding:25px'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; "
        f"padding-bottom:16px'>"
        f"<span style='font-family:Inter,sans-serif; font-weight:600; font-size:12px; "
        f"color:#aac7ff; text-transform:uppercase; letter-spacing:1.2px'>Processing Summary</span>"
        f"{status_badge}"
        f"</div>"
        f"<div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px'>"
        f"<div>"
        f"<div style='font-family:Inter,sans-serif; font-weight:600; font-size:10px; "
        f"color:#c1c6d7; text-transform:uppercase; letter-spacing:-0.5px'>Language</div>"
        f"<div style='font-family:Manrope,sans-serif; font-weight:700; font-size:18px; "
        f"color:#dae2fd'>{lang}</div></div>"
        f"<div>"
        f"<div style='font-family:Inter,sans-serif; font-weight:600; font-size:10px; "
        f"color:#c1c6d7; text-transform:uppercase; letter-spacing:-0.5px'>Segments</div>"
        f"<div style='font-family:Manrope,sans-serif; font-weight:700; font-size:18px; "
        f"color:#dae2fd'>{segments}</div></div>"
        f"<div>"
        f"<div style='font-family:Inter,sans-serif; font-weight:600; font-size:10px; "
        f"color:#c1c6d7; text-transform:uppercase; letter-spacing:-0.5px'>Speakers</div>"
        f"<div style='font-family:Manrope,sans-serif; font-weight:700; font-size:18px; "
        f"color:#dae2fd'>{speakers}</div></div>"
        f"</div></div>"
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
            f"\n Speaker detection failed: {result['diarize_error']}"
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
        media_files = []
    if isinstance(media_files, str):
        media_files = [media_files]

    # Parse original paths supplied by the native file picker (if any).
    native_paths: list[str] = []
    if original_paths_json:
        try:
            parsed = json.loads(original_paths_json)
            if isinstance(parsed, list):
                native_paths = [str(p) for p in parsed if p]
        except (json.JSONDecodeError, TypeError):
            pass

    if not media_files and native_paths:
        media_files = native_paths

    if not media_files:
        raise gr.Error("Please upload at least one media file.")

    orig_paths: list[str | None] = [None] * len(media_files)
    for i in range(len(media_files)):
        if i < len(native_paths):
            orig_paths[i] = native_paths[i]
        else:
            mf = str(media_files[i])
            if "/tmp/" not in mf and "/gradio/" not in mf:
                orig_paths[i] = mf

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

    lang = language.strip().lower() if language else None
    total = len(media_files)
    filenames = [Path(mf).name for mf in media_files]
    statuses = ["pending"] * total
    errors: dict[int, str] = {}
    all_results: dict[int, dict] = {}
    all_txt_paths: list[str] = []

    cancel_event = threading.Event()
    if isinstance(cancel_state, dict):
        cancel_state.clear()
    cancel_state = {"event": cancel_event}

    _skip = gr.skip()

    # Outputs: transcript, speaker_transcript, txt_file, download_btn,
    #          info, progress_bar, cached_results, retry_btn, speaker_tab,
    #          batch_queue_html, file_selector, cancel_btn, cancel_state,
    #          summary_html, main_tabs

    for idx, media_file in enumerate(media_files):
        if cancel_event.is_set():
            for j in range(idx, total):
                statuses[j] = "cancelled"
            queue_html = _render_batch_queue(filenames, statuses, errors)
            yield (_skip, _skip, _skip, _skip, _skip, _skip, _skip, _skip,
                   _skip, queue_html, _skip, _skip, cancel_state, _skip, _skip, _skip)
            break

        statuses[idx] = "processing"
        queue_html = _render_batch_queue(filenames, statuses, errors)
        batch_msg = f"File {idx + 1} of {total}: {filenames[idx]}"

        # Switch to Process tab on first file
        tab_update = gr.Tabs(selected="process") if idx == 0 else _skip
        yield (_skip, _skip, _skip, _skip, _skip,
               _render_progress(0, batch_msg), _skip, _skip, _skip,
               queue_html, _skip, gr.Button(visible=True), cancel_state,
               _skip, _skip, tab_update)

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
            yield (_skip, _skip, _skip, _skip, _skip,
                   _render_progress(fraction, combined_msg), _skip, _skip, _skip,
                   _skip, _skip, _skip, _skip, _skip, _skip, _skip)

        thread.join()

        if error_holder:
            statuses[idx] = "error"
            errors[idx] = str(error_holder[0])
        else:
            statuses[idx] = "done"
            result = result_holder[0]
            all_results[idx] = result

            _, _, txt_path, _ = _format_result(result, media_file)
            all_txt_paths.append(txt_path)

            if save_alongside and orig_paths[idx]:
                try:
                    saved = save_txt_alongside(result, orig_paths[idx])
                    errors[idx] = f"Saved to {Path(saved).name}"
                except (PermissionError, OSError):
                    errors[idx] = "Could not save alongside source (permission denied)"

        queue_html = _render_batch_queue(filenames, statuses, errors)
        yield (_skip, _skip, _skip, _skip, _skip, _skip, _skip, _skip,
               _skip, queue_html, _skip, _skip, _skip, _skip, _skip, _skip)

    # --- Batch complete ---
    done_count = sum(1 for s in statuses if s == "done")
    err_count = sum(1 for s in statuses if s == "error")
    summary_parts = [f"{done_count} of {total} completed"]
    if err_count:
        summary_parts.append(f"{err_count} failed")

    first_result_idx = next((i for i in range(total) if i in all_results), None)

    if first_result_idx is not None:
        result = all_results[first_result_idx]
        plain_text, speaker_text, _, info = _format_result(
            result, media_files[first_result_idx]
        )
        show_retry = bool(result.get("diarize_error"))
        has_speakers = bool(speaker_text)
        summary_html = _render_summary_html(result)
    else:
        plain_text = ""
        speaker_text = ""
        info = "All files failed."
        show_retry = False
        has_speakers = False
        summary_html = ""

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

    selector_choices = [
        f"{i + 1}. {filenames[i]}" for i in sorted(all_results.keys())
    ]

    queue_html = _render_batch_queue(filenames, statuses, errors)

    # Switch to Review tab on completion
    yield (
        plain_text, speaker_text, download_path or "",
        gr.Button(visible=bool(download_path)),
        info,
        _render_progress(1.0, f"Done -- {' | '.join(summary_parts)}"),
        all_results,
        gr.Button(visible=show_retry),
        gr.Tab(visible=has_speakers),
        queue_html,
        gr.Dropdown(choices=selector_choices,
                    value=selector_choices[0] if selector_choices else None,
                    visible=len(selector_choices) > 1),
        gr.Button(visible=False),
        cancel_state,
        summary_html,
        gr.Column(visible=bool(summary_html)),
        gr.Tabs(selected="review"),
    )


def _on_file_select(filename, all_results, media_files):
    """Switch displayed transcript when user picks a file from the dropdown."""
    if not filename or not all_results or not media_files:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    if isinstance(media_files, str):
        media_files = [media_files]

    try:
        idx = int(filename.split(".", 1)[0]) - 1
    except (ValueError, IndexError):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    if isinstance(all_results, dict) and idx in all_results and idx < len(media_files):
        result = all_results[idx]
        plain_text, speaker_text, _, info = _format_result(result, media_files[idx])
        has_speakers = bool(speaker_text)
        summary_html_val = _render_summary_html(result)
        return (plain_text, speaker_text, info, gr.Tab(visible=has_speakers),
                summary_html_val, gr.Column(visible=bool(summary_html_val)))

    return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()


def _on_cancel_click(cancel_state):
    """Set the cancellation flag for batch processing."""
    if isinstance(cancel_state, dict) and "event" in cancel_state:
        cancel_state["event"].set()
    return gr.Button(interactive=False, value="Cancelling...")


def run_retry_diarize(media_files, hf_token, cached_results):
    """Retry only the speaker diarization step using the cached transcription."""
    if isinstance(media_files, list):
        media_file = media_files[0] if media_files else None
    else:
        media_file = media_files

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
    while True:
        try:
            item = progress_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            break
        fraction, message = item
        yield (_skip, _skip, _skip, _skip, _skip,
               _render_progress(fraction, message), _skip,
               _skip, _skip, _skip, _skip, _skip, _skip, _skip, _skip, _skip)

    thread.join()

    if error_holder:
        raise gr.Error(str(error_holder[0]))

    result = result_holder[0]
    plain_text, speaker_text, txt_path, info = _format_result(result, media_file)

    show_retry = bool(result.get("diarize_error"))
    has_speakers = bool(speaker_text)
    summary_html = _render_summary_html(result)

    yield (plain_text, speaker_text, str(txt_path),
           gr.Button(visible=True),
           info, "",
           {0: result}, gr.Button(visible=show_retry),
           gr.Tab(visible=has_speakers), _skip, _skip, _skip, _skip,
           summary_html, gr.Column(visible=bool(summary_html)), _skip)


def create_app(native_mode=False):
    from pywhispercpp.constants import MODELS_DIR as _models_dir

    _file_types = sorted(SUPPORTED_EXTENSIONS)
    _format_list = ", ".join(
        ext.lstrip(".").upper() for ext in sorted(SUPPORTED_EXTENSIONS)
    )
    _format_tags = " ".join(
        f"<span>{ext.lstrip('.').upper()}</span>"
        for ext in sorted(SUPPORTED_EXTENSIONS)[:5]
    )
    _more_count = max(0, len(SUPPORTED_EXTENSIONS) - 5)
    if _more_count:
        _format_tags += f" <span>+{_more_count} More</span>"

    with gr.Blocks(title="Media Transcriber") as app:

        # --- App Header ---
        gr.Markdown("# Media Transcriber", elem_classes=["app-header"])

        # Hidden state
        cached_results = gr.State(value=None)
        cancel_state = gr.State(value=None)
        original_paths = gr.Textbox(visible=False, elem_id="original_paths")

        # ===== Main 3-Tab Navigation =====
        with gr.Tabs(elem_classes=["dark-tabs"], elem_id="main-tabs") as main_tabs:

            # ==================== TAB 1: UPLOAD ====================
            with gr.Tab("Upload", id="upload"):
                with gr.Row(equal_height=False):

                    # --- Left: File Drop Zone ---
                    with gr.Column(scale=7):
                        media_input = gr.File(
                            label="Drop files here",
                            file_types=_file_types,
                            file_count="multiple",
                            elem_classes=["upload-zone"],
                        )

                        if native_mode:
                            native_browse_btn = gr.Button(
                                "Browse Files...", variant="secondary", size="sm"
                            )
                            native_browse_btn.click(
                                fn=None,
                                js="() => { window._nativeBrowse && window._nativeBrowse(); }",
                            )

                        gr.HTML(
                            f"<div class='format-tags'>{_format_tags}</div>",
                            elem_classes=["no-bg"],
                        )

                    # --- Right: Settings Panel ---
                    with gr.Column(scale=5, elem_classes=["settings-panel"]):
                        gr.HTML(
                            "<div style='display:flex; gap:12px; align-items:center; "
                            "margin-bottom:24px'>"
                            "<span style='font-size:18px; color:#c1c6d7'>&#9776;</span>"
                            "<span class='section-title'>Transcription Engine</span></div>",
                            elem_classes=["no-bg"],
                        )

                        model_dropdown = gr.Dropdown(
                            choices=MODEL_CHOICES,
                            value="large-v3",
                            label="Model Engine",
                        )
                        language_input = gr.Textbox(
                            label="Language (ISO Code)",
                            placeholder="e.g. 'en', 'fr' (empty for auto)",
                        )
                        diarize_checkbox = gr.Checkbox(
                            label="Detect Speakers",
                            value=False,
                            info="Identify unique voices in media",
                        )
                        hf_token_input = gr.Textbox(
                            label="HuggingFace Token",
                            placeholder="hf_****************",
                            type="password",
                            info="Required for speaker detection diarization",
                        )

                        save_alongside_checkbox = gr.Checkbox(
                            label="Save transcripts alongside source files",
                            value=native_mode,
                            visible=native_mode,
                            info="Write .txt next to each original media file",
                        )

                # --- Transcribe Button ---
                transcribe_btn = gr.Button(
                    "Transcribe",
                    variant="primary",
                    size="lg",
                    elem_classes=["transcribe-btn"],
                )
                gr.HTML(
                    "<div class='gpu-hint'>Engine will utilize GPU acceleration if available.</div>",
                    elem_classes=["no-bg"],
                )

            # ==================== TAB 2: PROCESS ====================
            with gr.Tab("Process", id="process"):
                with gr.Row(equal_height=False):

                    # --- Left: Processing Status ---
                    with gr.Column(scale=8):
                        with gr.Group(elem_classes=["process-card"]):
                            progress_bar = gr.HTML(
                                value=_render_progress(0, "Waiting to start..."),
                                elem_classes=["no-bg"],
                            )

                        with gr.Group(elem_classes=["log-section"]):
                            gr.HTML(
                                "<div style='padding:25px 25px 8px'>"
                                "<span style='font-family:Inter,sans-serif; font-size:12px; "
                                "color:#c1c6d7; text-transform:uppercase; letter-spacing:0.6px'>"
                                "Logs</span></div>",
                                elem_classes=["no-bg"],
                            )
                            log_html = gr.HTML(
                                value=_render_log_html(),
                                elem_classes=["no-bg"],
                            )

                    # --- Right: Job Queue ---
                    with gr.Column(scale=4):
                        with gr.Group(elem_classes=["job-queue"]):
                            gr.HTML(
                                "<div style='padding:24px 24px 0'>"
                                "<div class='job-queue-header'>Job Queue</div>"
                                "</div>",
                                elem_classes=["no-bg"],
                            )
                            batch_queue_html = gr.HTML(
                                value="<div style='padding:0 16px; color:#c1c6d7; "
                                "font-size:12px; opacity:0.5'>No files queued yet.</div>",
                                elem_classes=["no-bg"],
                            )
                            cancel_btn = gr.Button(
                                "Cancel Remaining",
                                variant="stop",
                                visible=False,
                                elem_classes=["cancel-btn"],
                            )

            # ==================== TAB 3: REVIEW ====================
            with gr.Tab("Review", id="review"):

                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        gr.Markdown("## Review Results", elem_classes=["review-title"])
                        gr.Markdown(
                            "Verification and export of your batch processing queue.",
                            elem_classes=["review-subtitle"],
                        )
                    with gr.Column(scale=1, visible=False) as summary_col:
                        summary_html = gr.HTML(
                            value="",
                            elem_classes=["summary-box"],
                        )

                # File selector
                file_selector = gr.Dropdown(
                    label="View results for",
                    choices=[],
                    visible=False,
                )

                # Info text
                info_text = gr.Textbox(
                    label="Info",
                    interactive=False,
                    lines=1,
                    max_lines=3,
                    elem_classes=["info-display"],
                    visible=False,
                )

                # Transcript content
                with gr.Group(elem_classes=["transcript-canvas"]):
                    with gr.Tabs(elem_classes=["file-tabs"]):
                        with gr.Tab("Transcript"):
                            transcript_output = gr.Textbox(
                                label="",
                                lines=20,
                                max_lines=40,
                                interactive=False,
                                buttons=["copy"],
                                elem_classes=["transcript-box"],
                                show_label=False,
                            )
                        with gr.Tab("Speakers", visible=False) as speaker_tab:
                            speaker_output = gr.Textbox(
                                label="",
                                lines=20,
                                max_lines=40,
                                interactive=False,
                                buttons=["copy"],
                                elem_classes=["transcript-box"],
                                show_label=False,
                            )

                txt_download = gr.Textbox(
                    visible=False,
                    elem_id="txt_download_path",
                )

                # Action footer
                with gr.Row():
                    download_btn = gr.Button(
                        "Download ALL (.zip)",
                        visible=False,
                        elem_classes=["download-btn"],
                    )
                    retry_btn = gr.Button(
                        "Retry Speaker Detection",
                        variant="secondary",
                        visible=False,
                        elem_classes=["retry-btn"],
                    )

        # --- Event wiring ---
        _outputs = [transcript_output, speaker_output, txt_download,
                     download_btn,
                     info_text, progress_bar, cached_results,
                     retry_btn, speaker_tab,
                     batch_queue_html, file_selector, cancel_btn,
                     cancel_state, summary_html, summary_col, main_tabs]

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

        download_btn.click(
            fn=None,
            inputs=[txt_download],
            outputs=[],
            js="""(path) => {
                if (window.pywebview && window.pywebview.api && path) {
                    window.pywebview.api.save_transcript(path);
                } else if (path) {
                    const a = document.createElement('a');
                    a.href = '/file=' + encodeURIComponent(path);
                    a.download = path.split('/').pop();
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                }
            }""",
        )

        cancel_btn.click(
            fn=_on_cancel_click,
            inputs=[cancel_state],
            outputs=[cancel_btn],
        )

        file_selector.change(
            fn=_on_file_select,
            inputs=[file_selector, cached_results, media_input],
            outputs=[transcript_output, speaker_output, info_text, speaker_tab,
                     summary_html, summary_col],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CUSTOM_CSS)
