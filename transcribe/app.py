"""Native macOS application wrapper for Media Transcriber.

Launches the Gradio server in a background thread, displays it inside a
native WKWebView window via pywebview, and adds a menu-bar (system tray)
icon via PyObjC so the app can be minimised to the tray.

Usage::

    python -m transcribe.app
"""

from __future__ import annotations

import json
import socket
import sys
import time
import urllib.request

import webview

from transcribe.core import SUPPORTED_EXTENSIONS
from transcribe.web import CUSTOM_CSS, THEME, create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Bind to port 0 and let the OS pick a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 30.0):
    """Poll the local Gradio server until it responds or *timeout* expires."""
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.15)
    raise RuntimeError(f"Gradio server did not start within {timeout}s")


# ---------------------------------------------------------------------------
# JS API bridge — exposed to the webview for native file picking
# ---------------------------------------------------------------------------

# Build the file-type filter string for the native dialog
_ext_list = ";".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
_FILE_TYPES = (f"Media Files ({_ext_list})",)


class JsApi:
    """Python functions callable from JavaScript inside the webview."""

    def __init__(self, window: webview.Window):
        self._window = window

    def pick_files(self) -> list[str]:
        """Open a native macOS file dialog and return selected file paths."""
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=True,
            file_types=_FILE_TYPES,
        )
        return list(result) if result else []


# ---------------------------------------------------------------------------
# Tray integration (dispatched to main thread after Cocoa event loop starts)
# ---------------------------------------------------------------------------

_tray = None  # keep a reference so it isn't garbage-collected
_closing_handler = None  # stored so we can detach before real quit


def _setup_tray(window: webview.Window):
    """Create the menu-bar status item.  Must run on the main Cocoa thread."""
    from PyObjCTools.AppHelper import callAfter
    from transcribe.tray import Tray

    def _show():
        window.show()

    def _quit():
        global _closing_handler
        # Detach the hide-on-close handler so destroy() actually terminates.
        if _closing_handler is not None:
            window.events.closing -= _closing_handler
            _closing_handler = None
        window.destroy()

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

    # Expose the JS API so Gradio pages can call window.pywebview.api.*
    api = JsApi(window)
    window.expose(api)

    # Inject a helper script that connects the native file picker to Gradio.
    # When the user picks files via the native dialog, we:
    #   1. Store original paths in the hidden #original_paths textarea so the
    #      backend knows the real on-disk locations (used for "save alongside").
    #   2. Update the visible #native_file_list element with the selected names.
    # run_transcription falls back to original_paths when no Gradio upload is
    # present, so native-mode users can transcribe without the upload widget.
    window.evaluate_js("""
    (function() {
        window._nativeBrowse = async function() {
            if (!window.pywebview || !window.pywebview.api) return;
            const paths = await window.pywebview.api.pick_files();
            if (!paths || paths.length === 0) return;

            // 1. Store original paths in the hidden textarea
            const pathsInput = document.querySelector('#original_paths textarea');
            if (pathsInput) {
                const setter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value').set;
                setter.call(pathsInput, JSON.stringify(paths));
                pathsInput.dispatchEvent(new Event('input', { bubbles: true }));
            }

            // 2. Update the visible file list display
            const display = document.querySelector('#native_file_list');
            if (display) {
                const names = paths.map(p => p.split('/').pop());
                const label = names.length === 1
                    ? names[0]
                    : names.length + ' files selected: ' + names.join(', ');
                display.innerHTML =
                    '<span style="color:#16a34a">&#10003; ' + label + '</span>';
            }
        };
    })();
    """)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    port = _find_free_port()

    # Launch the Gradio server without blocking the main thread.
    gradio_app = create_app(native_mode=True)
    gradio_app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        prevent_thread_lock=True,
        theme=THEME,
        css=CUSTOM_CSS,
    )

    _wait_for_server(port)

    window = webview.create_window(
        title="Media Transcriber",
        url=f"http://127.0.0.1:{port}",
        width=900,
        height=750,
        min_size=(600, 500),
        background_color="#f8f9fb",
    )

    # Intercept window close → hide to tray instead of quitting.
    def _hide_instead_of_close():
        window.hide()
        return False  # cancel the close

    global _closing_handler
    _closing_handler = _hide_instead_of_close
    window.events.closing += _closing_handler

    # Start the native Cocoa event loop (blocks until window.destroy()).
    webview.start(
        func=_on_webview_started,
        args=(window,),
        gui="cocoa",
    )

    # Cleanup: shut down the Gradio server.
    try:
        gradio_app.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
