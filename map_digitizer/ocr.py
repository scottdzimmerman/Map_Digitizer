"""
Stage 3 — OCR labeling.

Tiles the map image (full-page OCR misses small text), reconstructs
Tesseract's individual word boxes into full phrases, and associates each
phrase to the nearest shape by bounding-box gap (not centroid distance,
which systematically favors labels near the page center).

Uses pytesseract (local Tesseract binary) — no external API calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
import pytesseract

from .config import OCRConfig
from .geo_utils import bbox_gap


@dataclass
class OCRPhrase:
    text: str
    bbox_px: tuple  # (x0, y0, x1, y1) in full-image pixel coordinates


def _tile_bounds(width: int, height: int, tile_size: int, overlap: int) -> list:
    bounds = []
    y = 0
    while y < height:
        x = 0
        while x < width:
            x1 = min(x + tile_size, width)
            y1 = min(y + tile_size, height)
            bounds.append((x, y, x1, y1))
            if x1 >= width:
                break
            x += tile_size - overlap
        if y1 >= height:
            break
        y += tile_size - overlap
    return bounds


def ocr_tiles(image_path: str, config: OCRConfig) -> list:
    """Run tiled OCR over the full image and reconstruct word boxes into
    phrases. Returns a flat list of OCRPhrase in full-image pixel coords."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    tiles = _tile_bounds(width, height, config.tile_size_px, config.tile_overlap_px)

    phrases = []
    for (x0, y0, x1, y1) in tiles:
        tile_img = img.crop((x0, y0, x1, y1))
        if config.upscale_factor != 1.0:
            new_size = (int(tile_img.width * config.upscale_factor), int(tile_img.height * config.upscale_factor))
            tile_img = tile_img.resize(new_size, Image.LANCZOS)
        scale = config.upscale_factor

        data = pytesseract.image_to_data(
            tile_img, config=f"--psm {config.psm}", output_type=pytesseract.Output.DICT
        )
        phrases.extend(_reconstruct_phrases(data, x0, y0, scale))

    return _dedupe_overlap_phrases(phrases)


def _reconstruct_phrases(data: dict, tile_x0: float, tile_y0: float, scale: float) -> list:
    """Group Tesseract's per-word boxes by (block, par, line) and join them
    in x-order into full phrases — the single biggest label-quality lever
    per the procedure."""
    groups: dict = {}
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        key = (data.get("block_num", [0] * n)[i], data.get("par_num", [0] * n)[i], data.get("line_num", [0] * n)[i])
        groups.setdefault(key, []).append({
            "text": text,
            "left": data["left"][i], "top": data["top"][i],
            "width": data["width"][i], "height": data["height"][i],
        })

    phrases = []
    for words in groups.values():
        words.sort(key=lambda w: w["left"])
        text = " ".join(w["text"] for w in words)
        x0 = min(w["left"] for w in words) / scale + tile_x0
        y0 = min(w["top"] for w in words) / scale + tile_y0
        x1 = max(w["left"] + w["width"] for w in words) / scale + tile_x0
        y1 = max(w["top"] + w["height"] for w in words) / scale + tile_y0
        phrases.append(OCRPhrase(text=text, bbox_px=(x0, y0, x1, y1)))
    return phrases


def _dedupe_overlap_phrases(phrases: list, iou_threshold: float = 0.6) -> list:
    """Tile overlap means the same phrase can be OCR'd twice; drop near-
    duplicate boxes with the same text."""
    kept = []
    for p in phrases:
        is_dup = False
        for k in kept:
            if k.text == p.text and bbox_gap(k.bbox_px, p.bbox_px) < 5:
                is_dup = True
                break
        if not is_dup:
            kept.append(p)
    return kept


def associate_labels_to_shapes(shapes: list, phrases: list, max_gap_px: float) -> dict:
    """
    For each shape (needs a `.bbox_px` attribute), find the closest OCR
    phrase by rectangle gap and return {shape_index: (phrase_text, gap_px)}.
    Shapes with no phrase within max_gap_px are omitted (name stays
    "unlabeled" upstream).
    """
    result = {}
    for idx, shape in enumerate(shapes):
        best = None
        best_gap = float("inf")
        for phrase in phrases:
            gap = bbox_gap(shape.bbox_px, phrase.bbox_px)
            if gap < best_gap:
                best_gap = gap
                best = phrase
        if best is not None and best_gap <= max_gap_px:
            result[idx] = (best.text, best_gap)
    return result
