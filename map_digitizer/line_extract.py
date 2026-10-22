"""
Stage 9 — Line-feature extraction (pipelines, transmission lines, roads).

The two adaptations the procedure calls out for line features vs. area
features: skeletonize instead of erasing thin lines, and split the skeleton
graph at junction nodes so branching/merging networks come out as individual
segments rather than one tangled blob.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.morphology import skeletonize
from skan import Skeleton

from .config import MapConfig
from .raster_extract import load_image_rgb, build_color_mask, restrict_to_panel, find_map_panel_bbox
from .vector_extract import RawShape


@dataclass
class LineSegment:
    feature_class: str
    pixel_points: list   # ordered (x, y) pixel points along the segment
    length_px: float
    is_junction_endpoint: tuple  # (bool, bool) — whether each end sits at a network junction (degree >= 3)


def extract_line_shapes(config: MapConfig, image_path: str) -> list:
    """
    Run Stage 9 end to end: color mask -> skeletonize -> split into segments
    at junctions. Returns a list of LineSegment. If config.split_lines_at_junctions
    is False, segments are merged per-class into one flat list of all pixel
    paths without attempting a clean single MultiLineString join (callers can
    unary_union the resulting shapely LineStrings themselves if they just
    want overall network geometry).
    """
    img = load_image_rgb(image_path)

    if config.raster.map_panel_bbox_px is not None:
        panel_bbox = config.raster.map_panel_bbox_px
    elif config.raster.map_panel_border_color is not None:
        panel_bbox = find_map_panel_bbox(img, config.raster.map_panel_border_color)
    else:
        h, w = img.shape[:2]
        panel_bbox = (0, 0, w, h)

    all_segments = []
    for fc in config.feature_classes:
        mask = build_color_mask(img, fc.fill_color)
        mask = restrict_to_panel(mask, panel_bbox)
        if not mask.any():
            continue

        skeleton = skeletonize(mask)
        if not skeleton.any():
            continue

        try:
            skel_graph = Skeleton(skeleton)
        except ValueError:
            # No branches found (e.g. an isolated single pixel or empty mask)
            continue

        n_paths = skel_graph.n_paths
        for i in range(n_paths):
            coords_rc = skel_graph.path_coordinates(i)  # (row, col) order
            points_xy = [(float(c), float(r)) for r, c in coords_rc]
            if len(points_xy) < 2:
                continue
            length_px = float(skel_graph.path_lengths()[i])

            all_segments.append(LineSegment(
                feature_class=fc.label,
                pixel_points=points_xy,
                length_px=length_px,
                is_junction_endpoint=_endpoint_degrees(skel_graph, i),
            ))

    if not config.split_lines_at_junctions:
        all_segments = _merge_by_class(all_segments)

    return all_segments


def _endpoint_degrees(skel_graph: Skeleton, path_index: int) -> tuple:
    """Whether each end of this path sits at a skeleton-graph junction
    (degree >= 3) vs. a true network endpoint (degree 1)."""
    path_indices = skel_graph.path(path_index)
    degrees = skel_graph.degrees
    start_deg = degrees[path_indices[0]]
    end_deg = degrees[path_indices[-1]]
    return (start_deg >= 3, end_deg >= 3)


def _merge_by_class(segments: list) -> list:
    """Collapse all segments of the same class into single LineSegment
    entries per class (for callers who only want overall network geometry,
    not individually named segments)."""
    by_class: dict = {}
    for seg in segments:
        by_class.setdefault(seg.feature_class, []).append(seg)

    merged = []
    for label, group in by_class.items():
        merged.append(LineSegment(
            feature_class=label,
            pixel_points=[seg.pixel_points for seg in group],  # list of lists -> becomes MultiLineString downstream
            length_px=sum(seg.length_px for seg in group),
            is_junction_endpoint=(False, False),
        ))
    return merged
