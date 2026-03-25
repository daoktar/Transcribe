"""FastAPI backend for Transcribe — REST API + SSE progress streaming + static files."""

from __future__ import annotations

import asyncio
import atexit
import io
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

from transcribe.core import (
    SUPPORTED_EXTENSIONS,
    retry_diarize,
    save_txt,
    save_txt_alongside,
    transcribe_media,
)
from transcribe.paths import get_base_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]

# Maximum upload size per file: 2 GB
MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024

# Input validation limits
MAX_NUM_SPEAKERS = 20

# Job limits
MAX_JOBS = 100
JOB_TTL = 86400  # 24 hours

# Rate limiting
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX_UPLOADS = 10  # max uploads per minute per IP
_rate_limit_log: dict[str, list[float]] = defaultdict(list)

# Whisper supported language codes (ISO 639-1)
VALID_LANGUAGES = {
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
    "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
    "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
    "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
    "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru",
    "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
    "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
    "yi", "yo", "yue", "zh",
}

# Single reusable temp directory — cleaned up on process exit
_tmp_dir = tempfile.TemporaryDirectory(prefix="transcribe_")
atexit.register(_tmp_dir.cleanup)

STATIC_DIR = get_base_dir() / "static"


class _SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """In-memory state for one transcription job."""

    id: str
    files: list[Path] = field(default_factory=list)
    filenames: list[str] = field(default_factory=list)
    original_paths: list[str | None] = field(default_factory=list)
    status: str = "uploaded"  # uploaded | processing | done | cancelled
    statuses: list[str] = field(default_factory=list)  # per-file
    errors: dict[int, str] = field(default_factory=dict)
    results: dict[int, dict] = field(default_factory=dict)
    txt_paths: list[str] = field(default_factory=list)
    progress_fraction: float = 0.0
    progress_message: str = ""
    current_file_index: int = 0
    cancel_event: threading.Event = field(default_factory=threading.Event)
    progress_queue: asyncio.Queue | None = field(default=None, repr=False)
    settings: dict = field(default_factory=dict)
    _hf_token: str | None = field(default=None, repr=False)  # never in settings/logs
    created_at: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


_jobs: dict[str, Job] = {}


def _cleanup_old_jobs() -> None:
    """Evict expired jobs and their temp directories."""
    now = time.monotonic()
    expired = [
        jid for jid, job in _jobs.items()
        if now - job.created_at > JOB_TTL and job.status != "processing"
    ]
    for jid in expired:
        job_dir = Path(_tmp_dir.name) / jid
        if job_dir.is_dir():
            shutil.rmtree(job_dir, ignore_errors=True)
        del _jobs[jid]


def _check_rate_limit(client_ip: str) -> None:
    """Enforce upload rate limiting per IP."""
    now = time.monotonic()
    timestamps = _rate_limit_log[client_ip]
    # Prune old entries
    _rate_limit_log[client_ip] = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    if len(_rate_limit_log[client_ip]) >= _RATE_LIMIT_MAX_UPLOADS:
        raise HTTPException(status_code=429, detail="Too many uploads. Try again later.")
    _rate_limit_log[client_ip].append(now)

# ---------------------------------------------------------------------------
# Pydantic request bodies
# ---------------------------------------------------------------------------


class TranscribeRequest(BaseModel):
    model: str = "large-v3"
    language: str | None = None
    diarize: bool = False
    hf_token: str | None = None
    num_speakers: int | None = None
    save_alongside: bool = False
    original_paths: list[str | None] | None = None

    @field_validator("num_speakers")
    @classmethod
    def _validate_num_speakers(cls, v: int | None) -> int | None:
        if v is not None and (v < 1 or v > MAX_NUM_SPEAKERS):
            raise ValueError(f"num_speakers must be 1–{MAX_NUM_SPEAKERS}")
        return v


class RetryDiarizeRequest(BaseModel):
    hf_token: str | None = None
    num_speakers: int | None = None
    file_index: int = 0

    @field_validator("num_speakers")
    @classmethod
    def _validate_num_speakers(cls, v: int | None) -> int | None:
        if v is not None and (v < 1 or v > MAX_NUM_SPEAKERS):
            raise ValueError(f"num_speakers must be 1–{MAX_NUM_SPEAKERS}")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_env_hf_token() -> str | None:
    """Read HuggingFace token from environment (supports both common names)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def _sanitize_filename(filename: str) -> str:
    """Strip path components and dangerous characters from an upload filename."""
    # Take only the basename (prevent path traversal via ../../)
    filename = Path(filename).name
    # Remove any remaining path separators or null bytes
    filename = re.sub(r'[\x00/\\]', '', filename)
    return filename or "upload"


def _validate_save_alongside_path(user_path: str) -> Path:
    """Validate that a user-supplied path for save-alongside is safe.

    Only allows paths under the user's home directory to prevent
    arbitrary file writes.
    """
    resolved = Path(user_path).resolve()
    home = Path.home().resolve()
    if not resolved.is_relative_to(home):
        raise ValueError(f"Path not within home directory: {resolved}")
    return resolved


def _get_job(job_id: str) -> Job:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _enqueue(job: Job, event: dict[str, Any]) -> None:
    """Push an SSE event dict onto the job's asyncio queue from any thread."""
    q = job.progress_queue
    if q is None:
        return
    loop = job._loop  # type: ignore[attr-defined]
    loop.call_soon_threadsafe(q.put_nowait, event)


def _sanitize_error(msg: str) -> str:
    """Strip file paths and limit length for user-facing error messages."""
    # Remove absolute paths
    msg = re.sub(r'(/[^\s:]+)+', '<path>', msg)
    return msg[:500]


def _run_transcription(job: Job) -> None:
    """Run transcription for all files in the job (called in a background thread)."""
    with job.lock:
        job.status = "processing"
    done_count = 0
    error_count = 0
    first_result_index: int | None = None

    total_files = len(job.files)

    for i, fpath in enumerate(job.files):
        if job.cancel_event.is_set():
            job.status = "cancelled"
            for j in range(i, total_files):
                job.statuses[j] = "cancelled"
            _enqueue(job, {
                "type": "complete",
                "done_count": done_count,
                "error_count": error_count,
                "first_result_index": first_result_index,
                "cancelled": True,
            })
            return

        job.current_file_index = i
        job.statuses[i] = "processing"

        # Build per-file progress callback
        def _progress_cb(
            fraction: float,
            message: str,
            *,
            _file_index: int = i,
            _total: int = total_files,
        ) -> None:
            # Scale fraction across all files
            overall = (_file_index + fraction) / _total
            job.progress_fraction = overall
            job.progress_message = message
            _enqueue(job, {
                "type": "progress",
                "fraction": round(overall, 4),
                "message": message,
                "file_index": _file_index,
                "statuses": list(job.statuses),
                "errors": {str(k): v for k, v in job.errors.items()},
            })

        _progress_cb(0.0, f"Starting {job.filenames[i]}...")

        try:
            result = transcribe_media(
                media_path=str(fpath),
                model_size=job.settings.get("model", "large-v3"),
                language=job.settings.get("language"),
                progress_callback=_progress_cb,
                diarize=job.settings.get("diarize", False),
                num_speakers=job.settings.get("num_speakers"),
                hf_token=job._hf_token,
            )
            job.results[i] = result
            job.statuses[i] = "done"
            done_count += 1
            if first_result_index is None:
                first_result_index = i

            # Save transcript
            if job.settings.get("save_alongside") and job.original_paths[i]:
                try:
                    _validate_save_alongside_path(job.original_paths[i])
                    txt_path = save_txt_alongside(result, job.original_paths[i])
                except (ValueError, OSError):
                    logger.warning("save_alongside path rejected, using temp dir")
                    txt_path = save_txt(
                        result,
                        Path(_tmp_dir.name) / f"{fpath.stem}.txt",
                    )
            else:
                txt_path = save_txt(
                    result,
                    Path(_tmp_dir.name) / f"{fpath.stem}.txt",
                )
            job.txt_paths.append(str(txt_path))

        except Exception as exc:
            with job.lock:
                job.statuses[i] = "error"
                job.errors[i] = _sanitize_error(str(exc))
            error_count += 1

    with job.lock:
        job.status = "done"
    job.progress_fraction = 1.0
    job.progress_message = "Complete"
    _enqueue(job, {
        "type": "complete",
        "done_count": done_count,
        "error_count": error_count,
        "first_result_index": first_result_index if first_result_index is not None else 0,
    })


def _run_retry_diarize(job: Job, file_index: int, hf_token: str, num_speakers: int | None) -> None:
    """Retry diarization for a single file (called in a background thread)."""
    job.status = "processing"
    job.statuses[file_index] = "processing"

    def _progress_cb(fraction: float, message: str) -> None:
        job.progress_fraction = fraction
        job.progress_message = message
        _enqueue(job, {
            "type": "progress",
            "fraction": round(fraction, 4),
            "message": message,
            "file_index": file_index,
            "statuses": list(job.statuses),
            "errors": {str(k): v for k, v in job.errors.items()},
        })

    _progress_cb(0.0, f"Retrying speaker detection for {job.filenames[file_index]}...")

    try:
        cached = job.results.get(file_index)
        if cached is None:
            raise ValueError(f"No transcription result for file index {file_index}")

        result = retry_diarize(
            media_path=str(job.files[file_index]),
            result=cached,
            hf_token=hf_token,
            num_speakers=num_speakers,
            progress_callback=_progress_cb,
        )
        job.results[file_index] = result
        job.statuses[file_index] = "done"

        # Re-save transcript
        if job.settings.get("save_alongside") and job.original_paths[file_index]:
            try:
                _validate_save_alongside_path(job.original_paths[file_index])
                txt_path = save_txt_alongside(result, job.original_paths[file_index])
            except (ValueError, OSError):
                logger.warning("save_alongside path rejected, using temp dir")
                txt_path = save_txt(
                    result,
                    Path(_tmp_dir.name) / f"{job.files[file_index].stem}.txt",
                )
        else:
            txt_path = save_txt(
                result,
                Path(_tmp_dir.name) / f"{job.files[file_index].stem}.txt",
            )
        # Update the corresponding txt_path
        if file_index < len(job.txt_paths):
            job.txt_paths[file_index] = str(txt_path)
        else:
            job.txt_paths.append(str(txt_path))

        _enqueue(job, {
            "type": "complete",
            "done_count": 1,
            "error_count": 0,
            "first_result_index": file_index,
        })

    except Exception as exc:
        sanitized = _sanitize_error(str(exc))
        with job.lock:
            job.statuses[file_index] = "error"
            job.errors[file_index] = sanitized
        _enqueue(job, {
            "type": "error",
            "file_index": file_index,
            "error": sanitized,
        })

    with job.lock:
        job.status = "done"


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Transcribe", docs_url="/docs")
    app.add_middleware(_SecurityHeadersMiddleware)

    # --- Security headers middleware ---
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'"
        )
        return response

    # --- CORS — restrict to localhost origins ---
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://127\.0\.0\.1(:\d+)?$",
        allow_methods=["GET", "POST"],
        allow_credentials=False,
    )

    # Mount static files if the directory exists
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # GET / — serve the frontend
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            return HTMLResponse("<h1>Transcribe</h1><p>No frontend found.</p>")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # GET /api/config
    # ------------------------------------------------------------------
    @app.get("/api/config")
    async def get_config() -> JSONResponse:
        return JSONResponse({
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
            "model_choices": MODEL_CHOICES,
            "hf_token_set": bool(_get_env_hf_token()),
        })

    # ------------------------------------------------------------------
    # POST /api/upload
    # ------------------------------------------------------------------
    @app.post("/api/upload")
    async def upload_files(
        request: Request,
        files: list[UploadFile] | None = None,
        paths: list[str] | None = None,
    ) -> JSONResponse:
        """Upload media files or reference local paths (native mode)."""
        # Rate limiting
        client_ip = request.client.host if request.client else "unknown"
        _check_rate_limit(client_ip)

        # Cleanup expired jobs and enforce capacity
        _cleanup_old_jobs()
        if len(_jobs) >= MAX_JOBS:
            raise HTTPException(
                status_code=503,
                detail="Server at capacity. Try again later.",
            )

        has_files = files and len(files) > 0
        has_paths = paths and len(paths) > 0

        if not has_files and not has_paths:
            raise HTTPException(status_code=400, detail="No files provided")

        job_id = str(uuid.uuid4())
        job_dir = Path(_tmp_dir.name) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        job = Job(id=job_id)

        if has_paths:
            # Native mode: files are local paths from pywebview file picker
            for p in paths:  # type: ignore[union-attr]
                fp = Path(p)
                if not fp.is_file():
                    raise HTTPException(status_code=400, detail=f"File not found: {p}")
                suffix = fp.suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {suffix}",
                    )
                job.files.append(fp)
                job.filenames.append(fp.name)
                job.original_paths.append(str(fp))
                job.statuses.append("pending")
        else:
            # Browser mode: uploaded files
            for upload in files:  # type: ignore[union-attr]
                filename = _sanitize_filename(upload.filename or "unknown")
                suffix = Path(filename).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {suffix}",
                    )

                # Read with size limit
                content = await upload.read()
                if len(content) > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024**3)} GB)",
                    )

                dest = job_dir / filename
                counter = 1
                while dest.exists():
                    dest = job_dir / f"{Path(filename).stem}_{counter}{suffix}"
                    counter += 1

                # Verify dest is inside job_dir (defense in depth)
                if not dest.resolve().is_relative_to(job_dir.resolve()):
                    raise HTTPException(status_code=400, detail="Invalid filename")

                dest.write_bytes(content)

                job.files.append(dest)
                job.filenames.append(filename)
                job.original_paths.append(None)
                job.statuses.append("pending")

        _jobs[job_id] = job
        return JSONResponse({
            "job_id": job_id,
            "filenames": job.filenames,
            "file_count": len(job.files),
        })

    # ------------------------------------------------------------------
    # POST /api/transcribe/{job_id}
    # ------------------------------------------------------------------
    @app.post("/api/transcribe/{job_id}")
    async def start_transcription(job_id: str, body: TranscribeRequest) -> JSONResponse:
        job = _get_job(job_id)

        with job.lock:
            if job.status == "processing":
                raise HTTPException(status_code=409, detail="Job already processing")

            # Validate language code
            language = body.language.strip() if body.language else None
            if language and language not in VALID_LANGUAGES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid language code: {language}",
                )

            # Resolve HF token: UI > env var (kept separate from settings)
            job._hf_token = body.hf_token or _get_env_hf_token()

            # Store settings — don't persist the raw HF token
            job.settings = {
                "model": body.model if body.model in MODEL_CHOICES else "large-v3",
                "language": language,
                "diarize": body.diarize,
                "num_speakers": body.num_speakers,
                "save_alongside": body.save_alongside,
            }

            # Update original paths if provided (validate each path)
            if body.original_paths:
                for i, op in enumerate(body.original_paths):
                    if i < len(job.original_paths) and op:
                        try:
                            _validate_save_alongside_path(op)
                            job.original_paths[i] = op
                        except ValueError:
                            logger.warning("Rejected original_path: %s", op)
                            job.original_paths[i] = None

            # Reset state for re-runs
            job.status = "uploaded"
            job.statuses = ["pending"] * len(job.files)
            job.errors.clear()
            job.results.clear()
            job.txt_paths.clear()
            job.progress_fraction = 0.0
            job.progress_message = ""
            job.cancel_event.clear()

            # Create asyncio queue bound to the current event loop
            loop = asyncio.get_running_loop()
            job.progress_queue = asyncio.Queue()
            job._loop = loop  # type: ignore[attr-defined]

        # Launch background thread (outside lock)
        thread = threading.Thread(target=_run_transcription, args=(job,), daemon=True)
        thread.start()

        return JSONResponse({"status": "started", "job_id": job_id})

    # ------------------------------------------------------------------
    # GET /api/progress/{job_id} — Server-Sent Events
    # ------------------------------------------------------------------
    @app.get("/api/progress/{job_id}")
    async def progress_stream(job_id: str) -> StreamingResponse:
        job = _get_job(job_id)

        # Ensure there's a queue (may reconnect after start)
        if job.progress_queue is None:
            loop = asyncio.get_running_loop()
            job.progress_queue = asyncio.Queue()
            job._loop = loop  # type: ignore[attr-defined]

        async def event_generator():
            q = job.progress_queue
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
                    continue

                yield f"data: {json.dumps(event)}\n\n"

                if event.get("type") in ("complete", "error"):
                    # Also check for terminal "complete" to close stream
                    if event.get("type") == "complete":
                        break
                    # For single-file error during retry, also close
                    if event.get("type") == "error":
                        break

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # POST /api/cancel/{job_id}
    # ------------------------------------------------------------------
    @app.post("/api/cancel/{job_id}")
    async def cancel_job(job_id: str) -> JSONResponse:
        job = _get_job(job_id)
        job.cancel_event.set()
        return JSONResponse({"status": "cancel_requested"})

    # ------------------------------------------------------------------
    # GET /api/download/{job_id}
    # ------------------------------------------------------------------
    @app.get("/api/download/{job_id}", response_model=None)
    async def download_transcripts(job_id: str):
        job = _get_job(job_id)

        tmp_root = Path(_tmp_dir.name).resolve()
        valid_paths = []
        for p in job.txt_paths:
            pp = Path(p)
            if not pp.is_file() or pp.suffix != ".txt":
                continue
            # Only serve files inside the temp directory or alongside originals
            resolved = pp.resolve()
            if resolved.is_relative_to(tmp_root):
                valid_paths.append(p)
            elif any(
                op and resolved.parent == Path(op).resolve().parent
                for op in job.original_paths
            ):
                valid_paths.append(p)
            # else: skip — potential path traversal

        if not valid_paths:
            raise HTTPException(status_code=404, detail="No transcripts available")

        if len(valid_paths) == 1:
            p = Path(valid_paths[0])
            return FileResponse(
                path=str(p),
                filename=p.name,
                media_type="text/plain",
            )

        # Multiple files — return a zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in valid_paths:
                zf.write(p, Path(p).name)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=transcripts.zip",
            },
        )

    # ------------------------------------------------------------------
    # POST /api/retry-diarize/{job_id}
    # ------------------------------------------------------------------
    @app.post("/api/retry-diarize/{job_id}")
    async def retry_diarize_endpoint(job_id: str, body: RetryDiarizeRequest) -> JSONResponse:
        job = _get_job(job_id)

        if job.status == "processing":
            raise HTTPException(status_code=409, detail="Job already processing")

        file_index = body.file_index
        if file_index < 0 or file_index >= len(job.files):
            raise HTTPException(status_code=400, detail="Invalid file_index")

        if file_index not in job.results:
            raise HTTPException(
                status_code=400,
                detail="No transcription result for this file yet",
            )

        # Reset progress state
        job.progress_fraction = 0.0
        job.progress_message = ""
        job.cancel_event.clear()

        loop = asyncio.get_running_loop()
        job.progress_queue = asyncio.Queue()
        job._loop = loop  # type: ignore[attr-defined]

        hf_token = body.hf_token or job._hf_token or _get_env_hf_token()
        thread = threading.Thread(
            target=_run_retry_diarize,
            args=(job, file_index, hf_token, body.num_speakers),
            daemon=True,
        )
        thread.start()

        return JSONResponse({"status": "started", "job_id": job_id, "file_index": file_index})

    # ------------------------------------------------------------------
    # GET /api/result/{job_id}/{file_index}
    # ------------------------------------------------------------------
    @app.get("/api/result/{job_id}/{file_index}")
    async def get_result(job_id: str, file_index: int) -> JSONResponse:
        job = _get_job(job_id)

        if file_index < 0 or file_index >= len(job.files):
            raise HTTPException(status_code=400, detail="Invalid file_index")

        result = job.results.get(file_index)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="No result for this file yet",
            )

        # Build speaker text from segments if available
        segments = result.get("segments", [])
        speaker_text = ""
        if result.get("speakers") and segments:
            speaker_text = "\n".join(
                f"{seg.get('speaker', '?')}: {seg.get('text', '')}"
                for seg in segments
            )

        return JSONResponse({
            "filename": job.filenames[file_index],
            "status": job.statuses[file_index],
            "text": result.get("text", ""),
            "speaker_text": speaker_text,
            "segments_count": len(segments),
            "language": result.get("language"),
            "speakers": result.get("speakers"),
            "has_speakers": bool(speaker_text),
            "diarize_error": result.get("diarize_error"),
            "diarize_requested": job.settings.get("diarize", False),
        })

    return app


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=7860)
