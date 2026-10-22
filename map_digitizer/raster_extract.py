"""
Stage 2B — Raster extraction.

For flattened/scanned map PDFs with no vector data: build a color mask per
feature class, restrict to the map panel, sever thin connecting lines from
area blobs with morphological opening, filter surviving connected components
by shape (aspect ratio / extent / solidity), and trace each one's outline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage as ndi
from skimage import morphology, measure

from .config import MapConfig, RGBColor, ShapeFilter
from .vector_extract import RawShape  # reuse the same output shape


def load_image_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def build_color_mask(img_rgb: np.ndarray, color: RGBColor) -> np.ndarray:
    """Boolean mask of pixels within Euclidean distance `tolerance` of the
    given RGB color. Sample the target color from a legend swatch, not a
    guess, per the procedure."""
    target = np.array(color.as_tuple(), dtype=np.float64)
    diff = img_rgb.astype(np.float64) - target
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return dist <= color.tolerance


def find_frame_edge(img_rgb: np.ndarray, border_color: RGBColor, axis: str,
                     from_start: bool, min_run: int = 50, search_limit: Optional[int] = None) -> int:
    """
    Scan rows (axis='y') or columns (axis='x') from one side of the image for
    the first long contiguous run of the frame's border color, to locate the
    map panel edge precisely (Stage 2B "restricting to the actual map panel").
    Returns the row/col index of that edge.
    """
    mask = build_color_mask(img_rgb, border_color)
    h, w = mask.shape[:2]
    n = h if axis == "y" else w
    limit = search_limit or n
    indices = range(n) if from_start else range(n - 1, -1, -1)

    checked = 0
    for i in indices:
        line = mask[i, :] if axis == "y" else mask[:, i]
        # longest run of True in this line
        if line.any():
            runs = np.diff(np.where(np.concatenate(([line[0]], line[:-1] != line[1:], [True])))[0])
            max_run = runs[::2].max() if len(runs) else 0
            if max_run >= min_run:
                return i
        checked += 1
        if checked >= limit:
            break
    return 0 if from_start else n - 1


def find_map_panel_bbox(img_rgb: np.ndarray, border_color: RGBColor, min_run: int = 50) -> tuple:
    """Locate all four edges of the map panel frame. Returns (x0, y0, x1, y1)."""
    h, w = img_rgb.shape[:2]
    y0 = find_frame_edge(img_rgb, border_color, axis="y", from_start=True, min_run=min_run, search_limit=h // 3)
    y1 = find_frame_edge(img_rgb, border_color, axis="y", from_start=False, min_run=min_run, search_limit=h // 3)
    x0 = find_frame_edge(img_rgb, border_color, axis="x", from_start=True, min_run=min_run, search_limit=w // 3)
    x1 = find_frame_edge(img_rgb, border_color, axis="x", from_start=False, min_run=min_run, search_limit=w // 3)
    return (x0, y0, x1, y1)


def restrict_to_panel(mask: np.ndarray, panel_bbox: tuple) -> np.ndarray:
    """Zero out everything outside the map panel bbox (legends, tables,
    title blocks, inset panels sharing the map's palette)."""
    x0, y0, x1, y1 = [int(v) for v in panel_bbox]
    restricted = np.zeros_like(mask)
    restricted[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return restricted


def sever_thin_lines(mask: np.ndarray, line_width_px: float) -> np.ndarray:
    """Morphological opening: erosion then dilation with a disk sized to the
    line's pixel width. Severs pipelines/roads touching a same-colored area
    feature while leaving compact blobs intact (Stage 2B step 2)."""
    radius = max(1, int(round(line_width_px)))
    selem = morphology.disk(radius)
    return morphology.opening(mask, selem)


def label_components(mask: np.ndarray) -> tuple:
    return ndi.label(mask, structure=np.ones((3, 3)))


def filter_regions_by_shape(labeled: np.ndarray, shape_filter: ShapeFilter) -> list:
    """Keep only connected components matching real-feature shape stats
    (Stage 2B step 3): aspect ratio, extent, solidity, and area bounds, and
    drop anything matching a configured fixed icon size."""
    kept = []
    for region in measure.regionprops(labeled):
        area = region.area
        if area < shape_filter.min_area_px:
            continue
        if shape_filter.max_area_px is not None and area > shape_filter.max_area_px:
            continue

        minr, minc, maxr, maxc = region.bbox
        h, w = maxr - minr, maxc - minc
        aspect = max(h, w) / max(1, min(h, w))
        if aspect > shape_filter.aspect_ratio_max:
            continue

        bbox_area = h * w
        extent = area / bbox_area if bbox_area else 0
        if extent < shape_filter.extent_min:
            continue

        solidity = region.solidity
        if solidity < shape_filter.solidity_min:
            continue

        is_icon = any(
            abs(area - fixed) <= fixed * shape_filter.icon_area_tolerance
            for fixed in shape_filter.fixed_icon_areas_px
        )
        if is_icon:
            continue

        kept.append(region)
    return kept


def trace_region_outline(labeled: np.ndarray, region) -> list:
    """Crop the region's bbox (with padding) and run cv2.findContours on the
    boolean sub-mask, keeping the largest contour (Stage 2B step 4)."""
    minr, minc, maxr, maxc = region.bbox
    pad = 2
    minr, minc = max(0, minr - pad), max(0, minc - pad)
    maxr, maxc = min(labeled.shape[0], maxr + pad), min(labeled.shape[1], maxc + pad)

    sub_mask = (labeled[minr:maxr, minc:maxc] == region.label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    # cv2 contour coords are (col, row) in the crop; shift back to full-image (x, y)
    points = [(float(pt[0][0] + minc), float(pt[0][1] + minr)) for pt in largest]
    if points and points[0] != points[-1]:
        points.append(points[0])
    return points


def extract_raster_shapes(config: MapConfig, image_path: str) -> list:
    """Run Stage 2B end to end for every feature class in config, and return
    a list of RawShape (same shape as vector_extract's output, so downstream
    code doesn't care which path produced them)."""
    img = load_image_rgb(image_path)

    if config.raster.map_panel_bbox_px is not None:
        panel_bbox = config.raster.map_panel_bbox_px
    elif config.raster.map_panel_border_color is not None:
        panel_bbox = find_map_panel_bbox(img, config.raster.map_panel_border_color)
    else:
        h, w = img.shape[:2]
        panel_bbox = (0, 0, w, h)

    all_shapes = []
    for fc in config.feature_classes:
        mask = build_color_mask(img, fc.fill_color)
        mask = restrict_to_panel(mask, panel_bbox)
        opened = sever_thin_lines(mask, config.raster.line_width_px)
        labeled, _ = label_components(opened)
        regions = filter_regions_by_shape(labeled, config.shape_filter)

        for region in regions:
            outline = trace_region_outline(labeled, region)
            if len(outline) < 3:
                continue
            minr, minc, maxr, maxc = region.bbox
            all_shapes.append(RawShape(
                feature_class=fc.label,
                pixel_points=outline,
                bbox_px=(minc, minr, maxc, maxr),
                area_px=float(region.area),
                source="raster",
            ))
    return all_shapes
