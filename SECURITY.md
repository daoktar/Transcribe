# Security Policy

## Supported Versions

This project is currently maintained on the `main` branch.

## Reporting a Vulnerability

Please do not open public issues for security bugs.

Report privately by contacting the maintainer with:

- A clear description of the issue
- Reproduction steps or proof of concept
- Potential impact
- Suggested mitigation (if known)

You should receive an initial response within 7 days.

## Security Notes for Users

- The app is intended for local/offline use.
- The web UI binds to `127.0.0.1` by default.
- Keep `ffmpeg` and Python dependencies up to date.
- Treat media files as untrusted input; process them on a patched system.
- Do not commit transcripts, media, `.env` files, or private keys.

## Project Hardening Checklist

- [x] Local-only web default (`127.0.0.1`, no sharing)
- [x] Sensitive/data/cache files in `.gitignore`
- [x] Automated dependency audit in CI (`pip-audit`)
- [x] Automated secret scan in CI (`gitleaks`)
