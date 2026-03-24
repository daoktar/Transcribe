"""Native macOS application wrapper for Media Transcriber.

Launches the Gradio server in a background thread, displays it inside a
native WKWebView window via pywebview, and adds a menu-bar (system tray)
icon via PyObjC so the app can be minimised to the tray.

Usage::

    python -m transcribe.app
"""
from __future__ import annotations

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import json
import shutil
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

    def save_transcript(self, src_path: str) -> str:
        """Open a native Save dialog and copy the transcript file there."""
        from pathlib import Path
        src = Path(src_path)
        if not src.exists():
            return ""
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

    # Expose the JS API so Gradio pages can call window.pywebview.api.*
    api = JsApi(window)
    window.expose(api.pick_files, api.save_transcript)

    # Inject a helper script that connects the native file picker to Gradio.
    # When the user picks files via the native dialog, we populate the hidden
    # #original_paths textarea so the Gradio backend knows the real paths,
    # and programmatically add the files to the Gradio upload component.
    window.evaluate_js("""
    (function() {
        // Make the native browse function available globally
        window._nativeBrowse = async function() {
            if (!window.pywebview || !window.pywebview.api) return;
            const paths = await window.pywebview.api.pick_files();
            if (!paths || paths.length === 0) return;

            // Store original paths in the hidden textarea
            const pathsInput = document.querySelector('#original_paths textarea');
            if (pathsInput) {
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value').set;
                nativeInputValueSetter.call(pathsInput, JSON.stringify(paths));
                pathsInput.dispatchEvent(new Event('input', { bubbles: true }));
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
        width=1100,
        height=900,
        min_size=(800, 600),
        background_color="#0b1326",
    )

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
    webview.start(
        func=_on_webview_started,
        args=(window,),
        gui="cocoa",
    )

    # Cleanup: shut down the Gradio server and force-exit.
    # Gradio spawns non-daemon threads that prevent a clean exit.
    try:
        gradio_app.close()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
