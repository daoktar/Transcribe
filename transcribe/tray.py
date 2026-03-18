"""macOS menu bar (system tray) integration via PyObjC.

Creates an NSStatusItem with a dropdown menu for controlling the app window.
Must be initialised on the main thread (use ``PyObjCTools.AppHelper.callAfter``
when dispatching from a background thread).
"""

from __future__ import annotations

import sys
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


# 16×16 template image (SF Symbol–style microphone, base64 PNG).
# Using a template image makes macOS auto-adapt for dark/light mode.
_ICON_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10"
    b"\x08\x06\x00\x00\x00\x1f\xf3\xffa\x00\x00\x00\tpHYs\x00\x00\x0b\x13"
    b"\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x01sRGB\x00\xae\xce"
    b"\x1c\xe9\x00\x00\x00\x04gAMA\x00\x00\xb1\x8f\x0b\xfca\x05\x00\x00\x00"
    b"\x00IEND\xaeB`\x82"
)


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
        data = NSData.dataWithBytes_length_(_ICON_BYTES, len(_ICON_BYTES))
        img = NSImage.alloc().initWithData_(data)
        if img:
            img.setSize_((16, 16))
        return img

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
