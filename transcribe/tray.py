"""macOS menu bar (system tray) integration via PyObjC.

Creates an NSStatusItem with a dropdown menu for controlling the app window.
Must be initialised on the main thread (use ``PyObjCTools.AppHelper.callAfter``
when dispatching from a background thread).
"""

from __future__ import annotations

from typing import Callable

import objc
from AppKit import (
    NSImage,
    NSMenu,
    NSMenuItem,
    NSStatusBar,
    NSVariableStatusItemLength,
)
from Foundation import NSData, NSObject

from transcribe.paths import get_assets_dir


class TrayDelegate(NSObject):
    """Handles menu item actions via Objective-C target/action pattern."""

    def initWithCallbacks_(self, callbacks: dict[str, Callable]):
        self = objc.super(TrayDelegate, self).init()
        if self is None:
            return None
        self._callbacks = callbacks
        return self

    @objc.python_method
    def _dispatch(self, key):
        cb = self._callbacks.get(key)
        if cb:
            cb()

    def showWindow_(self, sender):
        self._dispatch("show")

    def quitApp_(self, sender):
        self._dispatch("quit")


class Tray:
    """Manages the macOS menu bar status item."""

    def __init__(self, on_show: Callable, on_quit: Callable):
        self._delegate = TrayDelegate.alloc().initWithCallbacks_({
            "show": on_show,
            "quit": on_quit,
        })

        self._status_item = (
            NSStatusBar.systemStatusBar()
            .statusItemWithLength_(NSVariableStatusItemLength)
        )

        # Try to set an icon; fall back to a text title.
        icon = self._load_icon()
        if icon:
            icon.setTemplate_(True)
            self._status_item.button().setImage_(icon)
        else:
            self._status_item.button().setTitle_("MT")

        self._menu = self._build_menu()
        self._status_item.setMenu_(self._menu)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_status(self, text: str):
        """Update the status display line in the menu (first item)."""
        item = self._menu.itemAtIndex_(0)
        if item:
            item.setTitle_(text)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_icon() -> NSImage | None:
        # Try @2x first for Retina, fall back to 1x
        assets = get_assets_dir()
        for name in ("tray_icon@2x.png", "tray_icon.png"):
            path = assets / name
            if path.exists():
                icon_bytes = path.read_bytes()
                data = NSData.dataWithBytes_length_(icon_bytes, len(icon_bytes))
                img = NSImage.alloc().initWithData_(data)
                if img:
                    img.setSize_((16, 16))
                    return img
        return None

    def _build_menu(self) -> NSMenu:
        menu = NSMenu.alloc().init()

        # Status line (disabled, informational)
        status = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Idle", None, ""
        )
        status.setEnabled_(False)
        menu.addItem_(status)

        menu.addItem_(NSMenuItem.separatorItem())

        show = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Show Window", "showWindow:", ""
        )
        show.setTarget_(self._delegate)
        menu.addItem_(show)

        menu.addItem_(NSMenuItem.separatorItem())

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Media Transcriber", "quitApp:", "q"
        )
        quit_item.setTarget_(self._delegate)
        menu.addItem_(quit_item)

        return menu
