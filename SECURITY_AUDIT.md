# Security Audit Plan

Reusable checklist for auditing the Media Transcriber project. Run this audit before each release or after significant changes.

---

## 1. Network Isolation

The app must make NO internet connections except model downloads (whisper.cpp) and diarization library downloads (pyannote from HuggingFace).

### Checklist

- [ ] **External CDN links** — grep all HTML/CSS/JS for external URLs (`https://`, `http://`)
  ```bash
  grep -rn 'https\?://' transcribe/static/ --include='*.html' --include='*.css' --include='*.js' | grep -v '127.0.0.1' | grep -v 'localhost'
  ```
- [ ] **External fetch calls** — verify all `fetch()` in JS use relative paths only
  ```bash
  grep -n 'fetch(' transcribe/static/app.js | grep -v "fetch('/" | grep -v 'fetch(`/'
  ```
- [ ] **Python network imports** — check for `requests`, `urllib` (non-localhost), `httpx`, `aiohttp`, `socket` (non-localhost)
  ```bash
  grep -rn 'import requests\|import httpx\|import aiohttp' transcribe/
  grep -rn 'urllib.request' transcribe/ # should only be localhost polling in app.py
  ```
- [ ] **Server binding** — verify uvicorn binds to `127.0.0.1`, never `0.0.0.0`
  ```bash
  grep -rn '0\.0\.0\.0' transcribe/
  grep -rn 'host=' transcribe/web.py transcribe/app.py
  ```
- [ ] **Dependencies** — check no dependency phones home or collects telemetry
  ```bash
  pip-audit -r requirements.txt
  ```

---

## 2. Path Traversal & File Handling

### Upload Endpoint (`web.py:/api/upload`)

- [ ] Filenames sanitized — `Path(name).name` strips directory components
- [ ] Destination validated — `dest.resolve().is_relative_to(job_dir.resolve())`
- [ ] File size enforced — `MAX_UPLOAD_SIZE` checked before writing
- [ ] File extension validated — only `SUPPORTED_EXTENSIONS` allowed
- [ ] Test: upload file named `../../etc/passwd.mp3` — must be rejected or sanitized

### Download Endpoint (`web.py:/api/download`)

- [ ] All paths in `txt_paths` validated against temp dir root
- [ ] Alongside-original paths validated against original file's parent directory
- [ ] No symlink following for served files
- [ ] Test: manually inject a path like `/etc/passwd` into `job.txt_paths` — must be filtered out

### Native Mode (`app.py`)

- [ ] `pick_files()` uses native file dialog with extension filter
- [ ] `save_transcript()` validates source path exists before copying
- [ ] No arbitrary file read/write via JS API

---

## 3. Input Validation

- [ ] **Language codes** — validated against `VALID_LANGUAGES` whitelist
- [ ] **Model names** — validated against `MODEL_CHOICES`
- [ ] **Job IDs** — UUID format (generated server-side, not user-provided)
- [ ] **File index** — bounds-checked against `len(job.files)`
- [ ] **HF token** — never logged, never stored persistently, sanitized from error messages
- [ ] **num_speakers** — integer type enforced by Pydantic

### Test Commands
```bash
# Invalid language
curl -X POST http://127.0.0.1:7860/api/transcribe/TEST_JOB \
  -H 'Content-Type: application/json' \
  -d '{"language": "<script>alert(1)</script>"}'

# Invalid model
curl -X POST http://127.0.0.1:7860/api/transcribe/TEST_JOB \
  -H 'Content-Type: application/json' \
  -d '{"model": "nonexistent"}'
```

---

## 4. XSS & Frontend Security

- [ ] All `innerHTML` assignments use `escapeHtml()` for every interpolated variable
  ```bash
  grep -n 'innerHTML' transcribe/static/app.js
  ```
- [ ] `escapeHtml()` function exists and properly escapes `<>&"'`
- [ ] Error messages from backend are escaped before display
- [ ] No `eval()`, `Function()`, or `document.write()` usage
  ```bash
  grep -n 'eval\|Function(\|document\.write' transcribe/static/app.js
  ```
- [ ] CSP header blocks inline scripts: `Content-Security-Policy: default-src 'self'`

---

## 5. Security Headers

Verify all responses include:

```bash
curl -sI http://127.0.0.1:7860/ | grep -iE 'x-content|x-frame|content-security'
```

- [ ] `X-Content-Type-Options: nosniff`
- [ ] `X-Frame-Options: DENY`
- [ ] `Content-Security-Policy: default-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'`

---

## 6. CORS

- [ ] CORS middleware restricts origins to `127.0.0.1` only
- [ ] Only `GET` and `POST` methods allowed
- [ ] Credentials not allowed (`allow_credentials=False`)

### Test
```bash
# Should be rejected (wrong origin)
curl -sI -H "Origin: http://evil.com" http://127.0.0.1:7860/api/config | grep -i 'access-control'

# Should be allowed
curl -sI -H "Origin: http://127.0.0.1:7860" http://127.0.0.1:7860/api/config | grep -i 'access-control'
```

---

## 7. Resource Exhaustion & DoS

- [ ] **Upload size limit** — `MAX_UPLOAD_SIZE` (2 GB per file), returns HTTP 413
- [ ] **Job count limit** — `MAX_JOBS` (100), returns HTTP 503
- [ ] **Job TTL** — `JOB_TTL` (24h), expired jobs cleaned on next upload
- [ ] **Rate limiting** — 10 uploads/min per IP, returns HTTP 429
- [ ] **SSE timeout** — keepalive every 30s, stream closes on complete/error

### Test
```bash
# Rate limit test (should get 429 after 10 rapid uploads)
for i in $(seq 1 12); do
  echo "Request $i: $(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:7860/api/upload)"
done
```

---

## 8. Concurrency & Thread Safety

- [ ] `Job` dataclass has a `threading.Lock`
- [ ] All state mutations in `_run_transcription` wrapped with `with job.lock:`
- [ ] State reset in `start_transcription` wrapped with `with job.lock:`
- [ ] Processing status check under lock (prevents double-start race)
- [ ] Background thread launched outside lock

---

## 9. Credential Handling (HF Token)

- [ ] Token lookup order: UI input > `HF_TOKEN` env var > `.env` file
- [ ] Token never logged to console or file
- [ ] Token sanitized from error messages: `str(exc).replace(hf_token, "***")`
- [ ] Only SHA-256 hash cached for pipeline reuse (in `diarize.py`)
- [ ] `.env` in `.gitignore`
- [ ] `.env.example` provided without real tokens
- [ ] Token input uses `type="password"` in HTML

### Test
```bash
# Verify .env is gitignored
echo "HF_TOKEN=hf_test123" > .env
git status | grep '.env'  # should NOT appear as untracked
rm .env
```

---

## 10. Process & Temp File Security

- [ ] `sys.exit(0)` used instead of `os._exit(0)` — atexit handlers run
- [ ] Temp directory registered for cleanup via `atexit`
- [ ] Job subdirectories created inside shared temp dir
- [ ] No shell=True in subprocess calls
  ```bash
  grep -rn 'shell=True' transcribe/
  ```
- [ ] ffmpeg/ffprobe invoked via list args (no shell injection)
  ```bash
  grep -rn 'subprocess' transcribe/core.py
  ```

---

## 11. Dependency Audit

```bash
# Check for known vulnerabilities
pip-audit -r requirements.txt

# Check for outdated packages
pip list --outdated

# Scan for secrets in codebase
gitleaks detect --source .
```

- [ ] No known CVEs in dependencies
- [ ] No hardcoded secrets, tokens, or API keys in code
- [ ] No `.env`, `.pem`, `.key` files committed

---

## 12. Build Security (PyInstaller)

- [ ] `MediaTranscriber.spec` does not hardcode secrets or credentials
- [ ] Bundled ffmpeg/ffprobe sourced from trusted location (Homebrew)
- [ ] No world-writable paths in the .app bundle
- [ ] Code signing applied if distributing outside the App Store

---

## Audit Log

| Date | Auditor | Findings | Status |
|------|---------|----------|--------|
| 2026-03-26 | Claude Code | Initial comprehensive audit — 17 issues found and fixed (PR #13) | Resolved |
| | | | |

---

## How to Use This Document

1. Before each release, create a copy or branch of this file
2. Work through each section, checking boxes as you verify
3. Record findings in the Audit Log table
4. Fix any issues found before release
5. Keep this document updated as the codebase evolves
