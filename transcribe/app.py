"""Native macOS application wrapper for Media Transcriber.

Launches the FastAPI/uvicorn server in a background thread, displays it inside
a native WKWebView window via pywebview, and adds a menu-bar (system tray)
icon via PyObjC so the app can be minimised to the tray.

Usage::

    python -m transcribe.app
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
import threading
import time
import urllib.request

import webview

from transcribe.core import SUPPORTED_EXTENSIONS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to port 0 and let the OS pick a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 30.0):
    """Poll the local server until it responds or *timeout* expires."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.15)
    raise RuntimeError(f"Server did not start within {timeout}s")


def _run_uvicorn(app, host: str, port: int):
    """Run uvicorn in a background thread."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# JS API bridge — exposed to the webview for native file picking
# ---------------------------------------------------------------------------

# Build the file-type filter string for the native dialog
_ext_list = ";".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
_FILE_TYPES = (f"Media Files ({_ext_list})",)


class JsApi:
    """Python functions callable from JavaScript inside the webview."""

    def __init__(self, window: "webview.Window | None" = None, base_url: str = ""):
        self._window = window
        self._base_url = base_url.rstrip("/")

    def pick_files(self) -> list[str]:
        """Open a native macOS file dialog and return selected file paths."""
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=_FILE_TYPES,
        )
        return list(result) if result else []

    def save_transcript(self, url_or_path: str) -> str:
        """Save a transcript via native Save dialog.

        Accepts either a local filesystem path (legacy) or a URL (absolute or
        server-relative like ``/api/download/<job_id>``). For URLs, the file
        is fetched over loopback HTTP from the embedded uvicorn server, and
        the filename is taken from the ``Content-Disposition`` header.
        Returns the destination path on success, ``""`` on cancel, or a
        string starting with ``"error: "`` on failure.
        """
        from pathlib import Path

        # Legacy branch: existing local file → copy as before.
        if url_or_path and not url_or_path.startswith(("/", "http://", "https://")):
            src = Path(url_or_path)
            if src.exists():
                save_types = ("Text Files (*.txt;*.zip)",)
                result = self._window.create_file_dialog(
                    webview.SAVE_DIALOG,
                    save_filename=src.name,
                    file_types=save_types,
                )
                if result:
                    dest = result if isinstance(result, str) else result[0]
                    shutil.copy2(str(src), dest)
                    return dest
                return ""
            return f"error: file not found: {url_or_path}"

        # URL branch: fetch from the loopback server.
        if url_or_path.startswith(("http://", "https://")):
            url = url_or_path
        else:
            if not self._base_url:
                return "error: server base URL not configured"
            url = self._base_url + url_or_path

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
                content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                disposition = resp.headers.get("Content-Disposition") or ""
        except Exception as exc:  # noqa: BLE001
            print(f"save_transcript: fetch failed: {exc}", file=sys.stderr)
            return f"error: fetch failed: {exc}"

        # Parse filename from Content-Disposition via stdlib email parser.
        filename = ""
        if disposition:
            from email.message import Message
            msg = Message()
            msg["Content-Disposition"] = disposition
            filename = msg.get_filename() or ""
        if not filename:
            # Fallback: last URL path segment, then sensible default.
            tail = url.rsplit("/", 1)[-1].split("?", 1)[0]
            if tail and "." in tail:
                filename = tail
            else:
                filename = "transcripts.zip" if content_type == "application/zip" else "transcript.txt"

        if content_type == "application/zip":
            save_types = ("Zip Archive (*.zip)",)
        else:
            save_types = ("Text Files (*.txt)",)

        result = self._window.create_file_dialog(
            webview.SAVE_DIALOG,
            save_filename=filename,
            file_types=save_types,
        )
        if not result:
            return ""  # user cancelled

        dest = result if isinstance(result, str) else result[0]
        try:
            Path(dest).write_bytes(data)
        except Exception as exc:  # noqa: BLE001
            print(f"save_transcript: write failed: {exc}", file=sys.stderr)
            return f"error: write failed: {exc}"
        return dest


# ---------------------------------------------------------------------------
# Tray integration (dispatched to main thread after Cocoa event loop starts)
# ---------------------------------------------------------------------------

_tray = None  # keep a reference so it isn't garbage-collected
_closing_handler = None  # stored so we can detach before real quit
_quitting = False  # set True when the app should actually terminate


def _force_quit(window: webview.Window):
    """Detach the hide-on-close handler and destroy the window."""
    global _closing_handler, _quitting
    _quitting = True
    if _closing_handler is not None:
        window.events.closing -= _closing_handler
        _closing_handler = None
    window.destroy()


def _setup_tray(window: webview.Window):
    """Create the menu-bar status item.  Must run on the main Cocoa thread."""
    from PyObjCTools.AppHelper import callAfter
    from transcribe.tray import Tray

    def _show():
        window.show()

    def _quit():
        _force_quit(window)

    def _create():
        global _tray
        _tray = Tray(on_show=_show, on_quit=_quit)

    callAfter(_create)


# ---------------------------------------------------------------------------
# Window lifecycle
# ---------------------------------------------------------------------------


def _on_webview_started(window: webview.Window):
    """Called by pywebview in a background thread once the GUI is ready."""
    _setup_tray(window)

    # Hook Cmd+Q so it force-quits instead of being blocked by the
    # hide-on-close handler.  We wrap the existing NSApplication delegate's
    # applicationShouldTerminate: to set _quitting before the closing event.
    from PyObjCTools.AppHelper import callAfter
    from AppKit import NSApplication, NSTerminateNow
    from Foundation import NSObject
    import objc

    class TerminateInterceptor(NSObject):
        """Wraps the existing app delegate to intercept Cmd+Q."""

        def initWithOriginal_(self, original):
            self = objc.super(TerminateInterceptor, self).init()
            if self is None:
                return None
            self._original = original
            return self

        def applicationShouldTerminate_(self, sender):
            _force_quit(window)
            return NSTerminateNow

        def forwardingTargetForSelector_(self, sel):
            return self._original

        def respondsToSelector_(self, sel):
            if sel == b"applicationShouldTerminate:":
                return True
            if self._original is not None:
                return self._original.respondsToSelector_(sel)
            return False

    def _install():
        app = NSApplication.sharedApplication()
        original_delegate = app.delegate()
        interceptor = TerminateInterceptor.alloc().initWithOriginal_(original_delegate)
        app.setDelegate_(interceptor)
        # Keep a reference to prevent GC
        window._terminate_interceptor = interceptor

    callAfter(_install)

    # Bind the real window to the JsApi instance that was attached via
    # js_api= at create_window() time. We cannot pass window into JsApi at
    # construction because the window doesn't exist yet.
    api = getattr(window, "_js_api", None)
    if api is not None:
        api._window = window
        api._base_url = getattr(window, "_server_base_url", "").rstrip("/")
        # Also expose explicitly — belt-and-suspenders. In some pywebview
        # versions / configurations js_api= and expose() behave differently,
        # and we want every JsApi method reachable as window.pywebview.api.*
        try:
            window.expose(api.pick_files, api.save_transcript)
        except Exception as exc:  # noqa: BLE001
            print(f"_on_webview_started: expose failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    from transcribe.web import create_app

    port = _find_free_port()
    app = create_app()

    # Launch the FastAPI/uvicorn server in a background thread.
    server_thread = threading.Thread(
        target=_run_uvicorn,
        args=(app, "127.0.0.1", port),
        daemon=True,
    )
    server_thread.start()

    _wait_for_server(port)

    # Create the JS API up front so we can pass it to create_window via
    # js_api=. The window reference and base_url are filled in inside
    # _on_webview_started once the window exists.
    js_api = JsApi(window=None, base_url="")  # type: ignore[arg-type]

    window = webview.create_window(
        title="Media Transcriber",
        url=f"http://127.0.0.1:{port}",
        width=1100,
        height=900,
        min_size=(800, 600),
        background_color="#0b1326",
        js_api=js_api,
    )
    # Stash server URL and api instance so _on_webview_started can wire them up.
    window._server_base_url = f"http://127.0.0.1:{port}"
    window._js_api = js_api

    # Intercept window close-button → hide to tray instead of quitting.
    # Cmd+Q sets _quitting=True first, so the handler lets it through.
    def _hide_instead_of_close():
        if _quitting:
            return True  # allow the close
        window.hide()
        return False  # cancel the close — just hide to tray

    global _closing_handler
    _closing_handler = _hide_instead_of_close
    window.events.closing += _closing_handler

    # Start the native Cocoa event loop (blocks until window.destroy()).
    # Debug mode can be toggled via TRANSCRIBE_DEBUG=1 — enables the
    # WKWebView DevTools (right-click → Inspect Element) and verbose logs.
    debug = os.environ.get("TRANSCRIBE_DEBUG", "").strip() not in ("", "0", "false", "False")
    webview.start(
        func=_on_webview_started,
        args=(window,),
        gui="cocoa",
        debug=debug,
    )

    # Exit cleanly so atexit handlers run (e.g. temp dir cleanup).
    # Uvicorn runs in a daemon thread so it dies with us.
    sys.exit(0)


if __name__ == "__main__":
    main()
