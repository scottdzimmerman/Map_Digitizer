"""
Stage 2A — Vector extraction.

Reads every filled/stroked path from a vector-authored PDF page
(`page.get_drawings()`), separates real features from decorative map
furniture (legend swatches, icons, inset-map duplicates, bleed artifacts)
using fill+stroke color and fixed-icon-size heuristics, and builds pixel-space
shapely geometries from the raw path segments.

Works for area features (Polygon), and can also emit LineString geometry
for line features drawn as vector strokes (feature_type="line").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

from .config import MapConfig, FeatureClass, FeatureType
from .geometry import flatten_cubic_bezier
from .geo_utils import point_in_bbox


@dataclass
class RawShape:
    feature_class: str
    pixel_points: list          # list of (x, y) pixel points, closed ring for polygons
    bbox_px: tuple               # (x0, y0, x1, y1)
    area_px: float
    source: str = "vector"


def _color_distance(c1: tuple, c2: tuple) -> float:
    # fitz colors are floats 0-1; normalize both sides to 0-255 for comparison
    def norm(c):
        return tuple(v * 255 if v <= 1.0 else v for v in c)
    a, b = norm(c1), norm(c2)
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _path_to_points(items: list, flatten_tolerance_px: float) -> list:
    """Flatten a fitz drawing path's raw items (lines, curves, rects, quads)
    into an ordered list of (x, y) pixel points."""
    n_segments = max(4, int(12 / max(0.25, flatten_tolerance_px)))
    points = []
    for item in items:
        kind = item[0]
        if kind == "l":  # line: (p1, p2)
            p1, p2 = item[1], item[2]
            if not points:
                points.append((p1.x, p1.y))
            points.append((p2.x, p2.y))
        elif kind == "c":  # cubic bezier: (p1, p2, p3, p4)
            p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
            if not points:
                points.append((p1.x, p1.y))
            seg = flatten_cubic_bezier((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y), n_segments)
            points.extend(seg[1:])
        elif kind == "re":  # rect
            rect = item[1]
            points.extend([
                (rect.x0, rect.y0), (rect.x1, rect.y0),
                (rect.x1, rect.y1), (rect.x0, rect.y1), (rect.x0, rect.y0),
            ])
        elif kind == "qu":  # quad
            quad = item[1]
            points.extend([(pt.x, pt.y) for pt in (quad.ul, quad.ur, quad.lr, quad.ll, quad.ul)])
    return points


def _is_excluded(bbox_px: tuple, exclude_regions: list, page_width: float, page_height: float) -> bool:
    x0, y0, x1, y1 = bbox_px
    # Bleed content: bounding box extends past the page's stated size.
    if x0 < -0.5 or y0 < -0.5 or x1 > page_width + 0.5 or y1 > page_height + 0.5:
        return True
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for region in exclude_regions:
        if point_in_bbox(cx, cy, (region.x0, region.y0, region.x1, region.y1)):
            return True
    return False


def _matches_fixed_icon_size(area_px: float, fixed_sizes: list, tolerance_frac: float) -> bool:
    for size in fixed_sizes:
        if abs(area_px - size) <= size * tolerance_frac:
            return True
    return False


def classify_feature(fill_rgb: Optional[tuple], stroke_rgb: Optional[tuple],
                      feature_classes: list, real_feature_stroke_colors: list) -> Optional[str]:
    """
    Returns the matching FeatureClass label, or None if this path doesn't
    look like a real feature.

    If real_feature_stroke_colors is non-empty, the path's stroke must match
    one of them first (this is the "combine fill *and* stroke color" trick
    from the procedure — decorative icons are stroked in a shade of their
    own fill, real features share one consistent near-black stroke).
    """
    if real_feature_stroke_colors and stroke_rgb is not None:
        if not any(_color_distance(stroke_rgb, sc.as_tuple()) <= sc.tolerance for sc in real_feature_stroke_colors):
            return None

    if fill_rgb is None:
        return None
    for fc in feature_classes:
        if _color_distance(fill_rgb, fc.fill_color.as_tuple()) <= fc.fill_color.tolerance:
            return fc.label
    return None


def extract_vector_shapes(config: MapConfig, page_index: int = 0) -> list:
    """Run Stage 2A end to end and return a list of RawShape."""
    doc = fitz.open(config.source_pdf)
    page = doc[page_index]
    drawings = page.get_drawings()

    raw_shapes = []
    icon_size_counter: dict = {}

    # First pass: build candidate shapes.
    candidates = []
    for d in drawings:
        fill = d.get("fill")
        stroke = d.get("color")
        rect = d.get("rect")
        if rect is None:
            continue
        bbox_px = (rect.x0, rect.y0, rect.x1, rect.y1)
        if _is_excluded(bbox_px, config.exclude_regions_px, page.rect.width, page.rect.height):
            continue

        label = classify_feature(fill, stroke, config.feature_classes, config.vector.real_feature_stroke_colors)
        if label is None:
            continue

        points = _path_to_points(d.get("items", []), config.vector.bezier_flatten_tolerance_px)
        if len(points) < 3:
            continue

        area_px = abs(rect.width * rect.height)
        rounded_area = round(area_px, -1)  # bucket to nearest 10 px^2 for icon-repeat detection
        icon_size_counter[rounded_area] = icon_size_counter.get(rounded_area, 0) + 1

        candidates.append((label, points, bbox_px, area_px))

    # Second pass: drop shapes matching repeated fixed icon sizes (stamped
    # icons cluster tightly on one or two exact sizes; real features vary
    # continuously) in addition to any explicitly configured fixed sizes.
    sf = config.shape_filter
    repeat_threshold = max(5, int(len(candidates) * 0.03))
    auto_icon_sizes = [size for size, count in icon_size_counter.items() if count >= repeat_threshold]

    for label, points, bbox_px, area_px in candidates:
        if area_px < sf.min_area_px:
            continue
        if sf.max_area_px is not None and area_px > sf.max_area_px:
            continue
        if _matches_fixed_icon_size(area_px, sf.fixed_icon_areas_px, sf.icon_area_tolerance):
            continue
        if _matches_fixed_icon_size(area_px, auto_icon_sizes, 0.02):
            continue
        raw_shapes.append(RawShape(feature_class=label, pixel_points=points, bbox_px=bbox_px, area_px=area_px))

    doc.close()
    return raw_shapes


def merge_touching_shapes(shapes: list, feature_type: FeatureType) -> list:
    """Optional cleanup: unary_union shapes of the same class whose polygons
    touch/overlap (e.g. a field drawn as several adjoining path fragments)."""
    if feature_type != FeatureType.AREA:
        return shapes
    by_class: dict = {}
    for s in shapes:
        by_class.setdefault(s.feature_class, []).append(s)

    merged = []
    for label, group in by_class.items():
        polys = [Polygon(s.pixel_points) for s in group if len(s.pixel_points) >= 3]
        polys = [p for p in polys if p.is_valid and not p.is_empty]
        if not polys:
            continue
        union = unary_union(polys)
        geoms = list(union.geoms) if hasattr(union, "geoms") else [union]
        for geom in geoms:
            if geom.is_empty:
                continue
            xs, ys = geom.exterior.coords.xy
            merged.append(RawShape(
                feature_class=label,
                pixel_points=list(zip(xs, ys)),
                bbox_px=geom.bounds,
                area_px=geom.area,
            ))
    return merged
