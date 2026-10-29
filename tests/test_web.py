import asyncio
import io
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile

from transcribe import web


def _request(app, path, host="127.0.0.1:7860", method="GET", body=b""):
    """Drive one request through the ASGI app; returns (status, headers, body)."""
    messages = []
    pending = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if pending:
            return pending.pop()
        # Keep the client "connected": an early http.disconnect cancels streamed bodies.
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", host.encode()), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 7860),
    }
    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], headers, body


def _make_job(files, original_paths=None, settings=None):
    job = web.Job(id=str(uuid.uuid4()))
    job.files = [Path(f) for f in files]
    job.filenames = [Path(f).name for f in files]
    job.original_paths = original_paths or [None] * len(files)
    job.statuses = ["pending"] * len(files)
    job.settings = settings or {}
    (Path(web._tmp_dir.name) / job.id).mkdir(parents=True, exist_ok=True)
    return job


def _result(text):
    return {"text": text, "segments": [{"start": 0.0, "end": 1.0, "text": text}], "language": "en"}


@pytest.fixture
def registered():
    """Register jobs with the app's in-memory store and drop them afterwards."""
    added = []

    def _register(job):
        web._jobs[job.id] = job
        added.append(job.id)
        return job

    yield _register
    for jid in added:
        web._jobs.pop(jid, None)


class TestTrustedHost:
    def test_rejects_foreign_host(self):
        status, _, _ = _request(web.create_app(), "/api/config", host="attacker.example:7860")
        assert status == 400

    @pytest.mark.parametrize("host", ["127.0.0.1:7860", "localhost:7860", "127.0.0.1"])
    def test_allows_loopback_hosts(self, host):
        status, _, _ = _request(web.create_app(), "/api/config", host=host)
        assert status == 200


class TestTempTranscriptPaths:
    def test_path_is_inside_job_dir(self):
        job = _make_job(["/media/meeting.mp4"])
        assert web._temp_txt_path(job, 0) == Path(web._tmp_dir.name) / job.id / "meeting.txt"

    def test_shared_stem_gets_suffixed(self):
        job = _make_job(["/a/talk.mp3", "/b/talk.wav", "/c/other.mp3"])
        names = [web._temp_txt_path(job, i).name for i in range(3)]
        assert names == ["talk_1.txt", "talk_2.txt", "other.txt"]

    def test_jobs_with_same_filename_keep_their_own_transcript(self):
        job_a = _make_job(["/a/meeting.mp4"])
        job_b = _make_job(["/b/meeting.mp4"])

        with patch.object(web, "transcribe_media", return_value=_result("first")):
            web._run_transcription(job_a)
        with patch.object(web, "transcribe_media", return_value=_result("second")):
            web._run_transcription(job_b)

        assert Path(job_a.txt_paths[0]).read_text(encoding="utf-8") == "first"
        assert Path(job_b.txt_paths[0]).read_text(encoding="utf-8") == "second"


class TestTranscriptIndexing:
    def test_txt_paths_keyed_by_file_index_after_a_failure(self):
        job = _make_job(["/m/a.mp3", "/m/b.mp3", "/m/c.mp3"])

        def fake_transcribe(media_path, **_kwargs):
            if media_path.endswith("a.mp3"):
                raise RuntimeError("decode failed")
            return _result(Path(media_path).stem)

        with patch.object(web, "transcribe_media", side_effect=fake_transcribe):
            web._run_transcription(job)

        assert sorted(job.txt_paths) == [1, 2]
        assert Path(job.txt_paths[1]).read_text(encoding="utf-8") == "b"
        assert Path(job.txt_paths[2]).read_text(encoding="utf-8") == "c"

    def test_retry_diarize_does_not_clobber_other_transcripts(self):
        job = _make_job(["/m/a.mp3", "/m/b.mp3", "/m/c.mp3"])

        def fake_transcribe(media_path, **_kwargs):
            if media_path.endswith("a.mp3"):
                raise RuntimeError("decode failed")
            return _result(Path(media_path).stem)

        with patch.object(web, "transcribe_media", side_effect=fake_transcribe):
            web._run_transcription(job)

        labeled = {
            "text": "b",
            "segments": [{"start": 0.0, "end": 1.0, "text": "b", "speaker": "Speaker 1"}],
            "language": "en",
            "speakers": 1,
        }
        with patch.object(web, "retry_diarize", return_value=labeled):
            web._run_retry_diarize(job, 1, "hf_x", None)

        assert sorted(job.txt_paths) == [1, 2]
        assert Path(job.txt_paths[1]).read_text(encoding="utf-8") == "Speaker 1: b"
        assert Path(job.txt_paths[2]).read_text(encoding="utf-8") == "c"


class TestDownload:
    def test_zip_keeps_transcripts_that_share_a_name(self, tmp_path, registered):
        (tmp_path / "x").mkdir()
        (tmp_path / "y").mkdir()
        first = tmp_path / "x" / "meeting.txt"
        second = tmp_path / "y" / "meeting.txt"
        first.write_text("one", encoding="utf-8")
        second.write_text("two", encoding="utf-8")
        job = registered(_make_job(
            [tmp_path / "x" / "meeting.mp4", tmp_path / "y" / "meeting.mp4"],
            original_paths=[str(tmp_path / "x" / "meeting.mp4"), str(tmp_path / "y" / "meeting.mp4")],
        ))
        job.txt_paths = {0: str(first), 1: str(second)}

        status, headers, body = _request(web.create_app(), f"/api/download/{job.id}")

        assert status == 200
        assert headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            contents = {name: zf.read(name).decode() for name in zf.namelist()}
        assert contents == {"meeting.txt": "one", "meeting_1.txt": "two"}


class TestSaveUpload:
    def test_streams_file_to_disk(self, tmp_path):
        dest = tmp_path / "in.mp3"
        upload = UploadFile(file=io.BytesIO(b"x" * 2500), filename="in.mp3")
        with patch.object(web, "_UPLOAD_CHUNK_SIZE", 1000):
            asyncio.run(web._save_upload(upload, dest))
        assert dest.read_bytes() == b"x" * 2500

    def test_oversized_upload_is_rejected_and_removed(self, tmp_path):
        dest = tmp_path / "big.mp3"
        upload = UploadFile(file=io.BytesIO(b"x" * 2500), filename="big.mp3")
        with patch.object(web, "_UPLOAD_CHUNK_SIZE", 1000), patch.object(web, "MAX_UPLOAD_SIZE", 2000):
            with pytest.raises(HTTPException) as excinfo:
                asyncio.run(web._save_upload(upload, dest))
        assert excinfo.value.status_code == 413
        assert not dest.exists()


class TestDefaultEngine:
    @pytest.mark.parametrize("available, expected", [(True, "qwen"), (False, "whisper")])
    def test_config_reports_default_engine(self, available, expected):
        import json
        with patch("transcribe.qwen_engine.is_available", return_value=available):
            status, _, body = _request(web.create_app(), "/api/config")
        assert status == 200
        assert json.loads(body)["default_engine"] == expected

    @pytest.mark.parametrize("engine, expected", [(None, None), ("bogus", None), ("whisper", "whisper")])
    def test_validator_never_probes_the_backend(self, engine, expected):
        # Validation runs on the event loop; the default is resolved in start_transcription.
        with patch("transcribe.qwen_engine.backend", side_effect=AssertionError("probed")):
            body = web.TranscribeRequest(**({} if engine is None else {"engine": engine}))
        assert body.engine == expected


def test_lifespan_probes_the_backend_in_the_background():
    """Startup must not wait for the probe (the macOS app allows the server 30 s to bind)."""
    import asyncio
    import threading

    release = threading.Event()
    calls = []

    def slow_probe():
        calls.append(threading.current_thread().name)
        release.wait(5)
        return None

    app = web.create_app()

    async def run():
        with patch("transcribe.qwen_engine.backend", side_effect=slow_probe):
            async with app.router.lifespan_context(app):
                started = app.state.backend_warmup  # startup finished while the probe blocks
                assert started.is_alive()
                release.set()
                started.join(5)

    asyncio.run(run())
    assert calls == ["qwen-backend-probe"]


@pytest.mark.parametrize("available, expected", [(True, "qwen"), (False, "whisper")])
def test_start_without_engine_uses_the_default_probed_off_the_loop(registered, available, expected):
    import json
    import threading

    job = registered(_make_job(["/m/a.mp3"]))
    loop_thread = threading.main_thread()
    probed_in = []

    def probe():
        probed_in.append(threading.current_thread())
        return available

    with patch("transcribe.qwen_engine.is_available", side_effect=probe), \
         patch.object(web, "_run_transcription"):  # the job thread starts but does nothing
        status, _, _ = _request(web.create_app(), f"/api/transcribe/{job.id}",
                                method="POST", body=json.dumps({}).encode())
    assert status == 200
    assert job.settings["engine"] == expected
    assert probed_in and all(t is not loop_thread for t in probed_in)
