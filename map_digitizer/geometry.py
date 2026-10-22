"""
Stage 6 — Converting pixel-space geometry to WGS84 WKT/GeoJSON.

Also holds the Bezier-flattening helper used by vector extraction (Stage 2A)
since it's pure geometry with no PDF-specific logic.
"""

from __future__ import annotations

from typing import Callable

from shapely.geometry import Polygon, LineString, Point, mapping
from shapely.geometry.base import BaseGeometry


def flatten_cubic_bezier(p0, p1, p2, p3, n_segments: int = 12) -> list:
    """Flatten a cubic Bezier curve (4 control points, each an (x, y) tuple)
    into a polyline of n_segments+1 points."""
    pts = []
    for i in range(n_segments + 1):
        t = i / n_segments
        mt = 1 - t
        x = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
        y = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
        pts.append((x, y))
    return pts


def transform_points(points_px: list, pixel_to_lonlat: Callable) -> list:
    """Apply a pixel->lonlat transform to a list of (x, y) pixel points."""
    return [pixel_to_lonlat(x, y) for x, y in points_px]


def polygon_from_pixel_points(points_px: list, pixel_to_lonlat: Callable,
                               simplify_tolerance_deg: float = 0.0008,
                               coordinate_decimals: int = 5) -> Polygon:
    """
    Build a simplified, rounded WGS84 Polygon from a traced pixel-space
    outline. buffer(0) repairs minor self-intersections from pixel tracing
    before simplification, matching the procedure's export step.
    """
    lonlat_pts = transform_points(points_px, pixel_to_lonlat)
    poly = Polygon(lonlat_pts).buffer(0)
    poly = poly.simplify(simplify_tolerance_deg, preserve_topology=True)
    return round_geometry(poly, coordinate_decimals)


def linestring_from_pixel_points(points_px: list, pixel_to_lonlat: Callable,
                                  simplify_tolerance_deg: float = 0.0008,
                                  coordinate_decimals: int = 5) -> LineString:
    """Same as polygon_from_pixel_points but for a traced line (pipeline
    skeleton edge, road centerline, etc.) — no buffer(0) since a line has no
    interior to repair."""
    lonlat_pts = transform_points(points_px, pixel_to_lonlat)
    line = LineString(lonlat_pts)
    line = line.simplify(simplify_tolerance_deg, preserve_topology=True)
    return round_geometry(line, coordinate_decimals)


def point_from_pixel(x: float, y: float, pixel_to_lonlat: Callable,
                      coordinate_decimals: int = 5) -> Point:
    lon, lat = pixel_to_lonlat(x, y)
    return round_geometry(Point(lon, lat), coordinate_decimals)


def round_geometry(geom: BaseGeometry, decimals: int) -> BaseGeometry:
    """Round every coordinate of a shapely geometry to `decimals` places
    (Stage 6: cuts multi-MB GeoJSON down to a fraction with no visible loss
    of shape fidelity at normal zoom)."""
    from shapely.ops import transform as shapely_transform

    def _round_coords(x, y, z=None):
        if z is not None:
            return (round(x, decimals), round(y, decimals), round(z, decimals))
        return (round(x, decimals), round(y, decimals))

    return shapely_transform(_round_coords, geom)


def to_wkt(geom: BaseGeometry) -> str:
    return geom.wkt


def features_to_geojson(records: list, geometry_key: str = "geometry") -> dict:
    """
    records: list of dicts, each containing a shapely geometry under
    geometry_key plus arbitrary properties. Returns a GeoJSON FeatureCollection.
    """
    features = []
    for rec in records:
        rec = dict(rec)
        geom = rec.pop(geometry_key)
        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": rec,
        })
    return {"type": "FeatureCollection", "features": features}
