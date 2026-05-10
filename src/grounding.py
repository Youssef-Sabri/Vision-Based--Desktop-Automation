"""
grounding.py — ScreenSeekeR visual grounding engine.

Implements the recursive search algorithm from arXiv:2504.07981 (ScreenSeekeR).

Pipeline:
1. PLANNER: VLM identifies up to 3 candidate screen regions.
2. GROUNDER: VLM predicts precise click-point coordinates for each region.
3. SCORING: Ranks candidates using Gaussian centrality.
4. NMS: Non-Maximum Suppression removes overlapping candidates.
5. RECURSE: Zooms into the top-ranked candidate and repeats until patch is small enough.
"""

import os
import re
import json
import math
import concurrent.futures
from typing import List, Optional, Tuple

from PIL import ImageGrab, Image, ImageDraw, ImageFont
from google import genai

from utils import logger, retry

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL     = "gemini-3.1-flash-lite"
MAX_DEPTH = 2
MIN_SIZE  = 1280                       # px — switch to direct grounding below this
NMS_IOU   = 0.5                        # IoU threshold for non-maximum suppression
SIGMA     = 0.3                        # Gaussian width
MAX_VLM   = 1024                       # downscale longest side before sending to API
DILATE_PX = 100                        # official dilation to prevent edge-cutoff
TEMP_DIR  = "temp"                     # folder for intermediate debug images
SAVE_DEBUG = True                     # set True to save intermediate images to temp/

# ─── Prompts ───────────────────────────────────────────────────────────────────
PLANNER_PROMPT = """\
You are a GUI understanding agent on Windows.
Identify up to 3 screen regions where "{target}" is most likely located.

Output ONLY XML area tags with coordinates in [0, 1000] space
(0=top-left corner, 1000=bottom-right corner):
<area x1="..." y1="..." x2="..." y2="...">description</area>

List regions from most likely to least likely. No other text.
"""

GROUNDER_PROMPT = """\
Find the exact centre of "{target}" in this image.
Output ONLY valid JSON — no markdown:
{{"x": <0-1000>, "y": <0-1000>, "confidence": <0.0-1.0>}}

Coordinates are in [0, 1000] normalised space within this image.
If the target is not visible output: {{"x": 0, "y": 0, "confidence": 0.0}}
"""


# ─── Helper functions ──────────────────────────────────────────────────────────

def _save_debug(img: Image.Image, name: str):
    """Save an intermediate image to the temp directory (if SAVE_DEBUG is on)."""
    if not SAVE_DEBUG:
        return
    os.makedirs(TEMP_DIR, exist_ok=True)
    path = os.path.join(TEMP_DIR, f"{name}.png")
    img.save(path)
    logger.debug(f"Saved debug image: {path}")

def _downscale(img: Image.Image) -> Tuple[Image.Image, float]:
    """Resize image so its longest side ≤ MAX_VLM pixels."""
    longest = max(img.size)
    if longest <= MAX_VLM:
        return img, 1.0
    scale = MAX_VLM / longest
    return img.resize(
        (int(img.width * scale), int(img.height * scale)),
        Image.Resampling.LANCZOS,
    ), scale


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from a model response string."""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    return json.loads(match.group())


def _gaussian_score(
    cx: float, cy: float,
    x1: float, y1: float, x2: float, y2: float,
) -> float:
    """
    Centrality-based Gaussian score (Equation 1 from ScreenSeekeR paper).

    Returns a value in (0, 1] — higher when the predicted click-point
    is closer to the centre of the candidate bounding box.
    """
    xn = (cx - x1) / max(x2 - x1, 1)
    yn = (cy - y1) / max(y2 - y1, 1)
    return math.exp(-((xn - 0.5) ** 2 + (yn - 0.5) ** 2) / (2 * SIGMA ** 2))


def _iou(a: Tuple, b: Tuple) -> float:
    """Compute Intersection-over-Union for two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(items: list) -> list:
    """
    Non-Maximum Suppression: keep highest-scored candidates and discard
    any lower-scored candidate whose IoU with a kept candidate exceeds
    the NMS_IOU threshold.
    """
    items = sorted(items, key=lambda x: x[0], reverse=True)
    kept = []
    for candidate in items:
        if all(_iou(candidate[1], k[1]) < NMS_IOU for k in kept):
            kept.append(candidate)
    return kept


# ─── VLM API calls ─────────────────────────────────────────────────────────────

@retry(max_attempts=3, delay=1.0)
def _call_planner(
    target: str, img: Image.Image, client: genai.Client,
) -> List[Tuple]:
    """
    Planner step: Identifies likely regions for the target.
    Returns up to 3 bounding boxes as (x1, y1, x2, y2) in [0, 1000] space.
    """
    small, _ = _downscale(img)
    prompt = PLANNER_PROMPT.format(target=target)
    response = client.models.generate_content(
        model=MODEL, contents=[prompt, small], config={"temperature": 0.0},
    )
    # Parse <area x1="..." y1="..." x2="..." y2="..."> tags
    pattern = r'<area\s+x1="(\d+)[",]\s*y1="(\d+)[",]\s*x2="(\d+)[",]\s*y2="(\d+)[",>]'
    regions = []
    for m in re.finditer(pattern, response.text):
        x1, y1, x2, y2 = map(int, m.groups())
        if x2 > x1 and y2 > y1:
            regions.append((x1, y1, x2, y2))
    return regions[:3]


@retry(max_attempts=3, delay=1.0)
def _call_grounder(
    target: str, img: Image.Image, client: genai.Client,
) -> dict:
    """
    Grounder step: Finds exact click-point inside the image crop.
    Returns {"x": int, "y": int, "confidence": float} in [0, 1000] space.
    """
    small, _ = _downscale(img)
    prompt = GROUNDER_PROMPT.format(target=target)
    response = client.models.generate_content(
        model=MODEL, contents=[prompt, small], config={"temperature": 0.0},
    )
    return _parse_json(response.text)


# ─── Core recursive search (Algorithm 1) ──────────────────────────────────────

def _visual_search(
    target: str,
    image: Image.Image,
    client: genai.Client,
    depth: int = 0,
    offset_x: int = 0,        # pixel offset of this crop within the full screenshot
    offset_y: int = 0,
) -> Optional[Tuple[int, int, float]]:
    """
    Recursive ScreenSeekeR search.

    Returns:
        Tuple of (global_x, global_y, confidence) if found, else None.
    """
    W, H = image.size

    # ── Base case: patch is small enough for direct grounding ──────────────
    if depth >= MAX_DEPTH or max(W, H) <= MIN_SIZE:
        _save_debug(image, f"seek_d{depth}_final")
        logger.info(f"[depth={depth}] Direct grounding on {W}×{H} patch…")
        result = _call_grounder(target, image, client)
        conf = result.get("confidence", 0.0)
        if conf > 0.0:
            gx = offset_x + int(result["x"] / 1000 * W)
            gy = offset_y + int(result["y"] / 1000 * H)
            logger.info(f"[depth={depth}] Found at ({gx}, {gy}), conf={conf:.2f}")
            return (gx, gy, conf)
        return None

    _save_debug(image, f"seek_d{depth}_full")
    regions = _call_planner(target, image, client)

    if not regions:
        logger.warning(f"[depth={depth}] Planner returned no regions → direct grounding.")
        return _visual_search(target, image, client, MAX_DEPTH, offset_x, offset_y)

    logger.info(f"[depth={depth}] {len(regions)} candidate region(s) found.")

    scored: list = []

    def _evaluate_region(i: int, rx1: int, ry1: int, rx2: int, ry2: int):
        """Score a single candidate region (runs inside a thread)."""
        # Convert normalised [0,1000] → pixel coordinates
        px1, py1 = int(rx1 / 1000 * W), int(ry1 / 1000 * H)
        px2, py2 = int(rx2 / 1000 * W), int(ry2 / 1000 * H)
        
        # Apply Box Dilation (Official ScreenSeekeR Step)
        px1, py1 = max(0, px1 - DILATE_PX), max(0, py1 - DILATE_PX)
        px2, py2 = min(W, px2 + DILATE_PX), min(H, py2 + DILATE_PX)
        px2, py2 = max(px2, px1 + 10), max(py2, py1 + 10)

        crop = image.crop((px1, py1, px2, py2))
        _save_debug(crop, f"seek_d{depth}_crop{i}")
        cW, cH = crop.size

        gr = _call_grounder(target, crop, client)
        conf = gr.get("confidence", 0.0)

        # Map the grounder's local point back into this image's pixel space
        gcx = px1 + int(gr["x"] / 1000 * cW)
        gcy = py1 + int(gr["y"] / 1000 * cH)

        g_score = _gaussian_score(gcx, gcy, px1, py1, px2, py2)
        total = g_score * conf
        return i, total, (px1, py1, px2, py2), conf, g_score

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(_evaluate_region, i, *r)
            for i, r in enumerate(regions)
        ]
        for future in concurrent.futures.as_completed(futures):
            i, total, box, conf, g_score = future.result()
            scored.append((total, box))
            logger.info(
                f"[depth={depth}] Region {i+1}: "
                f"score={total:.3f}  conf={conf:.2f}  gauss={g_score:.3f}"
            )

    ranked = _nms(scored)

    for _, (px1, py1, px2, py2) in ranked:
        crop = image.crop((px1, py1, px2, py2))

        result = _visual_search(
            target, crop, client,
            depth=depth + 1,
            offset_x=offset_x + px1,
            offset_y=offset_y + py1,
        )
        if result is not None:
            return result

    return None


# ─── Public API ────────────────────────────────────────────────────────────────

def locate_icon(
    target: str,
    screenshot: Optional[Image.Image] = None,
    save_annotated_path: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Locates an icon or UI element using the ScreenSeekeR algorithm.

    Args:
        target: Natural-language description of the target.
        screenshot: Pre-captured PIL Image. Takes a fresh screenshot if None.
        save_annotated_path: Optional path to save an annotated debug image.

    Returns:
        (x, y) absolute screen pixel coordinates.

    Raises:
        ValueError: If the element could not be found.
    """
    if screenshot is None:
        logger.info("Capturing desktop screenshot…")
        screenshot = ImageGrab.grab(all_screens=False)

    W, H = screenshot.size
    logger.info(f"ScreenSeekeR: locating '{target}' on {W}×{H} screenshot…")

    client = genai.Client()
    result = _visual_search(target, screenshot, client)

    if result is None:
        raise ValueError(f"ScreenSeekeR could not locate '{target}'.")

    x, y, conf = result
    logger.info(f"✓ Located '{target}' at ({x}, {y}) with confidence {conf:.2f}.")

    if save_annotated_path:
        # ── Annotate the final result for the deliverables ─────────────────────
        draw = ImageDraw.Draw(screenshot)
        r = 25
        # Draw red crosshair
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=4)
        draw.line((x - r - 15, y, x + r + 15, y), fill="red", width=3)
        draw.line((x, y - r - 15, x, y + r + 15), fill="red", width=3)

        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except Exception:
            font = ImageFont.load_default()

        # Draw text with a black outline for high visibility
        text = f"Conf: {conf:.2f}"
        
        # Calculate text size to perfectly center it above the crosshair
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = 120, 30  # Fallback for very old Pillow versions
            
        text_x = x - (text_w / 2)
        text_y = y - r - text_h - 15  # 15px padding above the crosshair

        for adj in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            draw.text((text_x + adj[0], text_y + adj[1]), text, fill="black", font=font)
        draw.text((text_x, text_y), text, fill="lime", font=font)

        os.makedirs(os.path.dirname(os.path.abspath(save_annotated_path)), exist_ok=True)
        screenshot.save(save_annotated_path)

    return (x, y)
