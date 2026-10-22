"""
Stage 5 — Geographic sanity filtering.

The manual procedure did this as an eyeball scatter plot ("look for points
obviously outside the country/region"). Automated here as a real
point/geometry-vs-boundary containment check with a buffer to absorb
georeferencing error, which is both scriptable and more auditable than a
visual pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from shapely.geometry import shape, box, Point
from shapely.geometry.base import BaseGeometry


@dataclass
class SanityCheckResult:
    kept: list         # records that passed
    flagged: list       # records that failed, each with a "sanity_flag_reason"


def _load_boundary(boundary_config) -> BaseGeometry:
    if boundary_config.boundary_geojson_path:
        with open(boundary_config.boundary_geojson_path) as f:
            gj = json.load(f)
        if gj.get("type") == "FeatureCollection":
            geoms = [shape(feat["geometry"]) for feat in gj["features"]]
            from shapely.ops import unary_union
            return unary_union(geoms)
        return shape(gj.get("geometry", gj))
    if boundary_config.bbox:
        min_lon, min_lat, max_lon, max_lat = boundary_config.bbox
        return box(min_lon, min_lat, max_lon, max_lat)
    raise ValueError("BoundaryConfig needs either boundary_geojson_path or bbox")


def _buffer_deg_from_km(lat: float, km: float) -> float:
    """Rough degrees-per-km buffer at a given latitude (good enough for a
    sanity margin, not for precision work)."""
    import math
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * max(0.1, math.cos(math.radians(lat)))
    return km / min(km_per_deg_lat, km_per_deg_lon)


def filter_by_boundary(records: list, boundary_config, geometry_key: str = "geometry",
                        centroid_lat_key: str = "lat") -> SanityCheckResult:
    """
    records: list of dicts each with a shapely geometry (Point, Polygon, or
    LineString) under geometry_key. Flags any record whose geometry falls
    entirely outside boundary + buffer.
    """
    boundary = _load_boundary(boundary_config)
    kept, flagged = [], []

    for rec in records:
        geom = rec[geometry_key]
        centroid = geom.centroid
        buffer_deg = _buffer_deg_from_km(centroid.y, boundary_config.buffer_km)
        buffered_boundary = boundary.buffer(buffer_deg)

        if buffered_boundary.intersects(geom):
            kept.append(rec)
        else:
            dist_deg = buffered_boundary.distance(geom)
            rec = dict(rec)
            rec["sanity_flag_reason"] = (
                f"centroid ({centroid.x:.4f}, {centroid.y:.4f}) is "
                f"~{dist_deg * 111:.0f} km outside the expected region "
                f"(boundary + {boundary_config.buffer_km:.0f} km buffer)"
            )
            flagged.append(rec)

    return SanityCheckResult(kept=kept, flagged=flagged)
