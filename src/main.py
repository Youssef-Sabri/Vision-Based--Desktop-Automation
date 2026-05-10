"""
main.py — Entry point for Vision-Based Desktop Automation.

Workflow:
1. Fetch the first 10 blog posts from JSONPlaceholder.
2. Minimize all windows to clear the desktop.
3. For each post:
    a. Locate the Notepad icon via AI or pixel cache.
    b. Launch Notepad, type content, save, and close.
4. Report final success rate.
"""

import os
import sys
import time

import keyboard
import pyautogui

from utils import logger
from api import fetch_posts, APIError
from automation import process_post

TEST_DELAY_SECONDS = 0


def main() -> None:
    logger.info("=== Vision-Based Desktop Automation ===")
    logger.info("Press Esc at any time to stop.")
    keyboard.add_hotkey(
        "esc", lambda: (logger.info("Stopped by user."), os._exit(0)),
    )

    # Fetch posts
    try:
        posts = fetch_posts(limit=10)
    except APIError as exc:
        logger.critical(f"Cannot fetch posts: {exc}")
        sys.exit(1)

    if not posts:
        logger.warning("No posts returned. Exiting.")
        sys.exit(0)

    # Prepare output directory
    target_dir = os.path.join(os.path.expanduser("~"), "Desktop", "tjm-project")
    os.makedirs(target_dir, exist_ok=True)
    logger.info(f"Output directory: {target_dir}")

    # Minimize all windows once
    logger.info("Minimising all windows…")
    pyautogui.hotkey("win", "d")
    time.sleep(1.5)

    # Process each post
    success = 0
    for i, post in enumerate(posts, 1):
        post_id = post.get("id", "?")
        logger.info(f"── Post {i}/{len(posts)} (id={post_id}) ──")
        try:
            process_post(post, target_dir)
            success += 1
            logger.info(f"Post {post_id} saved successfully.")
        except Exception as exc:
            logger.error(f"Post {post_id} failed: {exc}. Continuing…")

        if TEST_DELAY_SECONDS > 0 and i < len(posts):
            logger.info(f"⏳ Waiting {TEST_DELAY_SECONDS}s… (move the Notepad icon to test AI recovery)")
            time.sleep(TEST_DELAY_SECONDS)

    logger.info(f"Done. {success}/{len(posts)} posts saved to {target_dir}")


if __name__ == "__main__":
    main()
