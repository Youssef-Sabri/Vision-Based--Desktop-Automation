"""
automation.py — Desktop Automation Module

Controls the physical mouse and keyboard to interact with the OS.
Handles app launching, typing, saving, and window management.
"""

import os
import time

import pyautogui
import pygetwindow as gw

from typing import Dict, Any
from utils import logger, retry
from grounding import locate_icon

# Description sent to AI to locate the target application
NOTEPAD_TARGET = (
    "Notepad shortcut icon — a white document with horizontal lines "
    "and the label 'Notepad'"
)

# Cache to skip AI inference if icon hasn't moved
_CACHED_ICON_POS = None
_CACHED_ICON_CROP = None


class AutomationError(Exception):
    """Raised when an automation step fails after all retries."""


@retry(max_attempts=3, delay=1.0)
def launch_notepad() -> gw.Window:
    """
    Locates and launches Notepad.
    Uses pixel caching to bypass AI inference if the icon hasn't moved.
    """
    from PIL import ImageGrab, ImageChops, ImageStat

    global _CACHED_ICON_POS, _CACHED_ICON_CROP

    # Park cursor at top-left so it doesn't hover-highlight any icon
    pyautogui.moveTo(1, 1)
    time.sleep(0.1)

    # Capture a fresh desktop screenshot
    screenshot = ImageGrab.grab(all_screens=False)

    x, y = None, None

    # Check cache to bypass AI if possible
    if _CACHED_ICON_POS is not None and _CACHED_ICON_CROP is not None:
        cx, cy = _CACHED_ICON_POS
        box = (
            max(0, cx - 25),
            max(0, cy - 25),
            min(screenshot.width, cx + 25),
            min(screenshot.height, cy + 25),
        )
        current_crop = screenshot.crop(box)

        # Compare current screen region to cached crop
        diff = ImageChops.difference(current_crop, _CACHED_ICON_CROP)
        stat = ImageStat.Stat(diff)
        mean_diff = sum(stat.mean) / len(stat.mean)

        if mean_diff < 5.0:  # tolerance for minor rendering noise
            logger.info(f"Cache hit — icon unchanged at ({cx}, {cy}). Bypassing VLM.")
            x, y = cx, cy
        else:
            logger.info("Cache miss — icon moved or obscured. Running ScreenSeekeR.")
            _CACHED_ICON_POS = None
            _CACHED_ICON_CROP = None

    # Fallback to AI grounding if cache misses
    if x is None:
        try:
            x, y = locate_icon(NOTEPAD_TARGET, screenshot)
        except ValueError as exc:
            raise AutomationError(f"Icon not found: {exc}") from exc

        # Save crop to cache for next iteration
        _CACHED_ICON_POS = (x, y)
        box = (
            max(0, x - 25),
            max(0, y - 25),
            min(screenshot.width, x + 25),
            min(screenshot.height, y + 25),
        )
        _CACHED_ICON_CROP = screenshot.crop(box)

    logger.info(f"Notepad icon at ({x}, {y}) — launching…")
    pyautogui.FAILSAFE = True

    # Click an empty area first to ensure the desktop shell has focus
    pyautogui.click(5, 5)
    time.sleep(0.3)

    # Double-click the icon to launch Notepad
    pyautogui.moveTo(x, y, duration=0.2)
    time.sleep(0.05)
    pyautogui.doubleClick()

    # Wait for window to open
    deadline = time.time() + 4
    while time.time() < deadline:
        for win in gw.getWindowsWithTitle("Notepad"):
            if win.visible:
                logger.info(f"Notepad opened: '{win.title}'")
                win.activate()
                time.sleep(0.3)  # let the window finish its opening animation
                return win
        time.sleep(0.05)

    raise AutomationError("Notepad did not open within 4 s.")


def write_and_save(
    post: Dict[str, Any],
    target_dir: str,
    window: gw.Window,
) -> None:
    """Types content into active Notepad window and saves file."""
    post_id = post.get("id", "unknown")
    content = f"Title: {post.get('title', '')}\n\n{post.get('body', '')}\n"

    # Ensure target window is active before typing
    try:
        window.activate()
    except Exception as exc:
        logger.warning(f"activate() raised: {exc}")
    time.sleep(0.3)

    active = gw.getActiveWindow()
    if active is None or "Notepad" not in active.title:
        raise AutomationError(
            f"Expected Notepad to be active, "
            f"got '{active.title if active else 'None'}'."
        )

    # Type content
    logger.info(f"Typing content for post {post_id}…")
    pyautogui.write(content, interval=0.005)

    # Trigger Save As dialog
    pyautogui.hotkey("ctrl", "s")

    # Wait for dialog to appear
    deadline = time.time() + 5
    while time.time() < deadline:
        if any("save as" in str(t).lower() for t in gw.getAllTitles() if t):
            break
        time.sleep(0.05)
    time.sleep(0.2)  # tiny settle for the dialog to finish rendering

    filepath = os.path.abspath(os.path.join(target_dir, f"post_{post_id}.txt"))
    logger.info(f"Saving to {filepath}")
    pyautogui.write(filepath, interval=0.005)
    pyautogui.press("enter")

    # Handle potential overwrite confirmation
    deadline = time.time() + 1.0
    while time.time() < deadline:
        if any("confirm save as" in str(t).lower() for t in gw.getAllTitles() if t):
            logger.info("Confirming overwrite.")
            pyautogui.hotkey("alt", "y")
            break
        time.sleep(0.05)


def close_notepad(window: gw.Window) -> None:
    """Close the Notepad window gracefully, falling back to Alt+F4."""
    try:
        window.close()
    except Exception:
        try:
            window.activate()
        except Exception:
            pass
        pyautogui.hotkey("alt", "f4")
    time.sleep(0.3)


def process_post(post: Dict[str, Any], target_dir: str) -> None:
    """Executes full workflow for a single post."""
    window = launch_notepad()
    try:
        write_and_save(post, target_dir, window)
    finally:
        close_notepad(window)
