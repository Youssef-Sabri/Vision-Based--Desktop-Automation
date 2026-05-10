import os
import sys
import time
from PIL import ImageGrab
import pyautogui

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import logger
from grounding import locate_icon
from automation import NOTEPAD_TARGET

# Change TARGET to test detection of a different icon
# TARGET = "Google Chrome icon — a colorful circular logo with red, yellow, and green sections around a blue center"
TARGET = NOTEPAD_TARGET

def capture_scenario(name: str):
    """Run a single screenshot capture scenario with automatic retries."""
    while True:
        logger.info(f"\n{'='*50}")
        logger.info(f"SCENARIO: {name.upper()}")
        logger.info(f"{'='*50}")
        input(f"👉 Please move the target icon to the {name} and press ENTER...")

        # Clear desktop for accurate AI inference
        pyautogui.hotkey("win", "d")
        time.sleep(1.5)

        logger.info("Taking screenshot and running ScreenSeekeR AI...")
        screenshot = ImageGrab.grab(all_screens=False)
        
        try:
            dest = os.path.join("debugging", f"annotated_{name.replace(' ', '_').lower()}.png")
            
            # Locate icon and output annotated image
            x, y = locate_icon(TARGET, screenshot, save_annotated_path=dest)
            
            logger.info(f"✅ Success! Found at ({x}, {y})")
            logger.info(f"📸 Saved deliverable to: {dest}")

            # Restore windows
            pyautogui.hotkey("win", "d")
            time.sleep(0.5)
            break
            
        except Exception as e:
            logger.error(f"❌ Failed to locate icon: {e}")

            # Restore windows on failure
            pyautogui.hotkey("win", "d")
            time.sleep(0.5)

            choice = input("⚠️  AI couldn't see the icon! Retry this scenario? (y/n): ")
            if choice.lower() != 'y':
                logger.info(f"⏭️ Skipping {name} scenario.")
                break

if __name__ == "__main__":
    logger.info("=== Deliverable Screenshot Generator ===")
    os.makedirs("debugging", exist_ok=True)
    
    capture_scenario("Top Left")
    capture_scenario("Center")
    capture_scenario("Bottom Right")
    
    logger.info("\n🎉 All done! Check the 'debugging/' folder for your 3 annotated screenshots.")
