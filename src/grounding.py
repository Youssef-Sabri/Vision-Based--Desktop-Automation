"""
grounding.py — ScreenSeekeR Visual Grounding Engine

Implements the recursive Planner/Grounder search algorithm
from arXiv:2504.07981 with Box Dilation, Gaussian Centrality
Scoring, and Non-Maximum Suppression.
"""

import io
import os
import re
import json
import math
import concurrent.futures
from typing import List, Optional, Tuple

from PIL import ImageGrab, Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

from utils import logger, retry

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL = "gemini-3.1-flash-lite"
MAX_DEPTH = 2  # maximum recursive zoom levels
MIN_SIZE = 1280  # switch to direct grounding below this (px)
NMS_IOU = 0.5  # IoU threshold for non-maximum suppression
SIGMA = 0.3  # Gaussian centrality width (Eq. 1)
MAX_VLM = 768  # downscale longest side to reduce token usage
DILATE_PX = 100  # box dilation to prevent edge-cutoff
TEMP_DIR = "temp"  # debug image output folder
SAVE_DEBUG = True

# ─── Prompts (ScreenSeekeR Planner / Grounder) ────────────────────────────────
PLANNER_PROMPT = """\
You are a GUI understanding agent on Windows.
Identify up to 3 screen regions where "{target}" is most likely located.

CRITICAL: You must use Position Inference. Leverage common GUI knowledge to infer possible neighboring UI elements in proximity to the target (e.g., a "new" button typically appears near a "delete" button).

Return ONLY a JSON array of objects with coordinates in [0, 1000] space:
[{{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>, "label": "<description>", "reasoning": "<explain neighboring elements>"}}]

0 = top-left corner, 1000 = bottom-right corner.
Order regions from most likely to least likely. No other text.
"""

GROUNDER_PROMPT = """\
Find the exact centre of "{target}" in this image.
Return ONLY a JSON object with coordinates in [0, 1000] space:
{{"x": <int>, "y": <int>, "confidence": <float 0.0-1.0>}}

If the target is not visible: {{"x": 0, "y": 0, "confidence": 0.0}}
"""


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _save_debug(img: Image.Image, name: str):
    """Save debug image to temp/ folder."""
    if not SAVE_DEBUG:
        return
    os.makedirs(TEMP_DIR, exist_ok=True)
    path = os.path.join(TEMP_DIR, f"{name}.png")
    img.save(path)


def _to_jpeg(img: Image.Image) -> types.Part:
    """Compress and convert PIL image to JPEG bytes for faster API upload."""
    # Downscale if needed
    longest = max(img.size)
    if longest > MAX_VLM:
        scale = MAX_VLM / longest
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.Resampling.LANCZOS,
        )
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=75)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def _extract_json(text: str, default=None):
    """Extract the first JSON object or array from a model response."""
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if not match:
        if default is not None:
            return default
        raise ValueError(f"No JSON found in: {text[:200]}")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        if default is not None:
            return default
        raise ValueError(f"Failed to parse JSON: {e}")


def _gaussian_score(cx, cy, x1, y1, x2, y2) -> float:
    """Centrality-based Gaussian score (Equation 1 from ScreenSeekeR)."""
    xn = (cx - x1) / max(x2 - x1, 1)
    yn = (cy - y1) / max(y2 - y1, 1)
    return math.exp(-((xn - 0.5) ** 2 + (yn - 0.5) ** 2) / (2 * SIGMA**2))


def _iou(a: Tuple, b: Tuple) -> float:
    """Intersection-over-Union for two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(items: list) -> list:
    """Non-Maximum Suppression: discard overlapping lower-scored candidates."""
    items = sorted(items, key=lambda x: x[0], reverse=True)
    kept = []
    for candidate in items:
        if all(_iou(candidate[1], k[1]) < NMS_IOU for k in kept):
            kept.append(candidate)
    return kept


# ─── VLM API Calls ─────────────────────────────────────────────────────────────


@retry(max_attempts=3, delay=1.0)
def _call_planner(target: str, img: Image.Image, client: genai.Client) -> List[Tuple]:
    """Planner: returns up to 3 candidate regions as (x1, y1, x2, y2)."""
    image_part = _to_jpeg(img)
    prompt = PLANNER_PROMPT.format(target=target)
    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )

    items = _extract_json(response.text, default=[])
    if isinstance(items, dict):
        items = [items]
    regions = []
    for item in items:
        x1 = int(item.get("x1", 0))
        y1 = int(item.get("y1", 0))
        x2 = int(item.get("x2", 0))
        y2 = int(item.get("y2", 0))
        if x2 > x1 and y2 > y1:
            regions.append((x1, y1, x2, y2))

    if not regions:
        logger.warning("[Planner] No valid regions returned.")
    return regions[:3]


@retry(max_attempts=3, delay=1.0)
def _call_grounder(target: str, img: Image.Image, client: genai.Client) -> dict:
    """Grounder: returns {x, y, confidence} for exact click-point."""
    image_part = _to_jpeg(img)
    prompt = GROUNDER_PROMPT.format(target=target)
    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt, image_part],
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    return _extract_json(response.text)


# ─── Core Recursive Search (Algorithm 1) ──────────────────────────────────────


def _visual_search(
    target: str,
    image: Image.Image,
    client: genai.Client,
    depth: int = 0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Optional[Tuple[int, int, float]]:
    """Recursive ScreenSeekeR search. Returns (x, y, confidence) or None."""
    W, H = image.size

    # Base case: image is small enough or max depth reached
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

    # Planner step: identify candidate regions
    _save_debug(image, f"seek_d{depth}_full")
    regions = _call_planner(target, image, client)

    if not regions:
        logger.warning(f"[depth={depth}] No regions found → direct grounding.")
        return _visual_search(target, image, client, MAX_DEPTH, offset_x, offset_y)

    logger.info(f"[depth={depth}] {len(regions)} candidate region(s) found.")

    # Evaluate each region with Grounder + Gaussian scoring
    scored: list = []

    def _evaluate_region(i, rx1, ry1, rx2, ry2):
        """Score a single candidate region."""
        # Normalised [0,1000] → pixel coordinates
        px1, py1 = int(rx1 / 1000 * W), int(ry1 / 1000 * H)
        px2, py2 = int(rx2 / 1000 * W), int(ry2 / 1000 * H)

        # Box Dilation (ScreenSeekeR paper)
        px1, py1 = max(0, px1 - DILATE_PX), max(0, py1 - DILATE_PX)
        px2, py2 = min(W, px2 + DILATE_PX), min(H, py2 + DILATE_PX)
        px2, py2 = max(px2, px1 + 10), max(py2, py1 + 10)

        crop = image.crop((px1, py1, px2, py2))
        _save_debug(crop, f"seek_d{depth}_crop{i}")
        cW, cH = crop.size

        gr = _call_grounder(target, crop, client)
        conf = gr.get("confidence", 0.0)

        # Map local coordinates back to parent image space
        gcx = px1 + int(gr["x"] / 1000 * cW)
        gcy = py1 + int(gr["y"] / 1000 * cH)

        # Gaussian Centrality Score (Eq. 1)
        g_score = _gaussian_score(gcx, gcy, px1, py1, px2, py2)
        total = g_score * conf
        return i, total, (px1, py1, px2, py2), conf, g_score

    # Evaluate regions concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_evaluate_region, i, *r) for i, r in enumerate(regions)]
        for future in concurrent.futures.as_completed(futures):
            i, total, box, conf, g_score = future.result()
            scored.append((total, box))
            logger.info(
                f"[depth={depth}] Region {i + 1}: "
                f"score={total:.3f}  conf={conf:.2f}  gauss={g_score:.3f}"
            )

    # NMS: keep the best non-overlapping candidates
    ranked = _nms(scored)

    # Recurse into top-ranked regions
    for _, (px1, py1, px2, py2) in ranked:
        crop = image.crop((px1, py1, px2, py2))
        result = _visual_search(
            target,
            crop,
            client,
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
    Locate a UI element on screen using the ScreenSeekeR algorithm.

    Args:
        target: Natural-language description of the element.
        screenshot: Optional pre-captured image. Captures fresh if None.
        save_annotated_path: Optional path to save a debug image with crosshair.

    Returns:
        (x, y) absolute screen coordinates.

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

    # Draw annotated crosshair for diagnostics
    if save_annotated_path:
        draw = ImageDraw.Draw(screenshot)
        r = 25
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=4)
        draw.line((x - r - 15, y, x + r + 15, y), fill="red", width=3)
        draw.line((x, y - r - 15, x, y + r + 15), fill="red", width=3)

        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except Exception:
            font = ImageFont.load_default()

        text = f"Conf: {conf:.2f}"
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = 120, 30

        text_x = x - (text_w / 2)
        text_y = y - r - text_h - 15
        for adj in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            draw.text((text_x + adj[0], text_y + adj[1]), text, fill="black", font=font)
        draw.text((text_x, text_y), text, fill="lime", font=font)

        os.makedirs(
            os.path.dirname(os.path.abspath(save_annotated_path)), exist_ok=True
        )
        screenshot.save(save_annotated_path)

    return (x, y)
