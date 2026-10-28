"""
Config schema for a map-digitization run.

Every number that the original manual procedure required a human to read off
a specific map (a stroke color, a legend swatch, a control-point pixel
position, a match-score threshold) lives here, in a per-map config file, not
in code. The pipeline code is the same regardless of whether you're
digitizing oil fields, LNG terminals, or pipeline networks — only the config
changes.

Load with `load_config("my_map.yaml")` or `load_config("my_map.json")`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class FeatureType(str, Enum):
    AREA = "area"    # polygons: fields, concessions, protected areas...
    POINT = "point"  # point facilities: plants, terminals, stations...
    LINE = "line"    # linear features: pipelines, transmission lines, roads...


class SourceKind(str, Enum):
    VECTOR = "vector"    # real vector paths in the PDF (Stage 2A)
    RASTER = "raster"    # flattened / scanned image (Stage 2B)
    UNKNOWN = "unknown"  # let assess_pdf() decide


@dataclass
class RGBColor:
    """An RGB color with a per-channel or Euclidean match tolerance."""
    r: int
    g: int
    b: int
    tolerance: int = 20  # max Euclidean distance in 0-255 RGB space to count as a match

    def as_tuple(self) -> tuple:
        return (self.r, self.g, self.b)


@dataclass
class FeatureClass:
    """
    One category of feature on the map, e.g. 'Gas Field' (red) or
    'Oil Field' (green), or for a pipeline map, 'Gas Pipeline' vs
    'Oil Pipeline'. Vector extraction matches on stroke+fill; raster
    extraction matches on fill only (sampled from the legend swatch).
    """
    label: str
    fill_color: RGBColor
    stroke_color: Optional[RGBColor] = None  # vector extraction only


@dataclass
class ExcludeRegionPx:
    """A pixel-space bounding box to mask out before extraction (legend,
    inset/locator map, title block, bleed area, etc.)."""
    name: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class ShapeFilter:
    """Thresholds used to separate real features from icons / label callouts
    / pipeline fragments after color masking (Stage 2B step 3)."""
    min_area_px: float = 100
    max_area_px: Optional[float] = None
    aspect_ratio_max: float = 4.5      # bbox width/height (or vice versa)
    extent_min: float = 0.3            # area / bbox_area
    solidity_min: float = 0.6          # area / convex_hull_area
    fixed_icon_areas_px: list = field(default_factory=list)  # exact areas (+/- icon_area_tolerance) to drop as icons
    icon_area_tolerance: float = 0.1   # fractional tolerance around each fixed_icon_areas_px entry


@dataclass
class ControlPoint:
    """One (pixel, lon/lat) correspondence used to fit the georeferencing
    transform. label is just for readability in reports/QA."""
    label: str
    pixel_x: float
    pixel_y: float
    lon: float
    lat: float


@dataclass
class GeoreferenceConfig:
    control_points: list  # list[ControlPoint]
    degree: int = 1  # 1 = affine, 2 = quadratic (2nd-order polynomial)
    max_acceptable_residual_km: Optional[float] = None  # if set, drop_outliers() will iterate to this target
    region_bbox: Optional[tuple] = None  # (min_lon, min_lat, max_lon, max_lat); catches swapped lon/lat control points


@dataclass
class BoundaryConfig:
    """Geographic sanity filter (Stage 5): a region a valid feature must
    fall within (plus a buffer, since georeferencing has error)."""
    boundary_geojson_path: Optional[str] = None  # polygon/multipolygon covering the valid region
    bbox: Optional[tuple] = None  # fallback: (min_lon, min_lat, max_lon, max_lat)
    buffer_km: float = 50.0


@dataclass
class TrackerColumns:
    """Column-name mapping for the external tracker database CSV, so this
    isn't hardcoded to GOGET's schema."""
    id: str = "id"
    name: str = "name"
    lat: str = "latitude"
    lon: str = "longitude"
    existing_wkt: Optional[str] = "wkt"  # column holding existing geometry, if any; None if not tracked
    country: Optional[str] = None
    extra_label_cols: list = field(default_factory=list)  # extra columns to fold into the display label


@dataclass
class CrosscheckConfig:
    tracker_csv_path: str
    columns: TrackerColumns
    name_score_threshold: float = 70.0     # rapidfuzz token_sort_ratio, 0-100
    contained_distance_km: float = 15.0
    nearby_distance_km: float = 40.0
    name_match_search_cap_km: float = 200.0
    strip_generic_words: list = field(default_factory=lambda: [
        "oil field", "gas field", "oil and gas field", "project", "terminal",
        "plant", "station", "pipeline", "phase i", "phase ii", "phase 1", "phase 2",
    ])


@dataclass
class OCRConfig:
    enabled: bool = True
    tile_size_px: int = 1400
    tile_overlap_px: int = 100
    upscale_factor: float = 1.0  # set >1 (e.g. 2.0) if source is under ~200 DPI
    psm: int = 11  # tesseract page segmentation mode; 11 = sparse text
    max_label_shape_gap_px: float = 60.0  # ignore label candidates farther than this from a shape


@dataclass
class RasterExtractionConfig:
    line_width_px: float = 2.0  # measured pixel width of thin lines; drives morphology disk radius
    map_panel_border_color: Optional[RGBColor] = None
    map_panel_bbox_px: Optional[tuple] = None  # (x0, y0, x1, y1); overrides border-color detection if set


@dataclass
class VectorExtractionConfig:
    real_feature_stroke_colors: list = field(default_factory=list)  # list[RGBColor]; empty = don't filter on stroke
    bezier_flatten_tolerance_px: float = 1.0


@dataclass
class MapConfig:
    """Top-level config for one map-digitization run."""
    name: str
    source_pdf: str
    feature_type: FeatureType
    source_kind: SourceKind
    feature_classes: list          # list[FeatureClass]
    georeference: GeoreferenceConfig
    output_dir: str = "output"

    exclude_regions_px: list = field(default_factory=list)      # list[ExcludeRegionPx]
    shape_filter: ShapeFilter = field(default_factory=ShapeFilter)
    boundary: Optional[BoundaryConfig] = None
    crosscheck: Optional[CrosscheckConfig] = None
    ocr: OCRConfig = field(default_factory=OCRConfig)
    raster: RasterExtractionConfig = field(default_factory=RasterExtractionConfig)
    vector: VectorExtractionConfig = field(default_factory=VectorExtractionConfig)

    simplify_tolerance_deg: float = 0.0008
    coordinate_decimals: int = 5

    # Line-feature specific (Stage 9)
    split_lines_at_junctions: bool = True


def _dataclass_from_dict(cls, data):
    """Recursively build nested dataclasses from plain dicts (loaded from
    YAML/JSON). Handles the small set of nested types used in MapConfig."""
    if data is None:
        return None
    if cls is RGBColor:
        return RGBColor(**data)
    if cls is FeatureClass:
        d = dict(data)
        d["fill_color"] = RGBColor(**d["fill_color"])
        if d.get("stroke_color"):
            d["stroke_color"] = RGBColor(**d["stroke_color"])
        return FeatureClass(**d)
    if cls is ExcludeRegionPx:
        return ExcludeRegionPx(**data)
    if cls is ShapeFilter:
        return ShapeFilter(**data)
    if cls is ControlPoint:
        return ControlPoint(**data)
    if cls is GeoreferenceConfig:
        d = dict(data)
        d["control_points"] = [_dataclass_from_dict(ControlPoint, cp) for cp in d["control_points"]]
        if d.get("region_bbox"):
            d["region_bbox"] = tuple(d["region_bbox"])
        return GeoreferenceConfig(**d)
    if cls is BoundaryConfig:
        d = dict(data)
        if d.get("bbox"):
            d["bbox"] = tuple(d["bbox"])
        return BoundaryConfig(**d)
    if cls is TrackerColumns:
        return TrackerColumns(**data)
    if cls is CrosscheckConfig:
        d = dict(data)
        d["columns"] = _dataclass_from_dict(TrackerColumns, d["columns"])
        return CrosscheckConfig(**d)
    if cls is OCRConfig:
        return OCRConfig(**data)
    if cls is RasterExtractionConfig:
        d = dict(data)
        if d.get("map_panel_border_color"):
            d["map_panel_border_color"] = RGBColor(**d["map_panel_border_color"])
        if d.get("map_panel_bbox_px"):
            d["map_panel_bbox_px"] = tuple(d["map_panel_bbox_px"])
        return RasterExtractionConfig(**d)
    if cls is VectorExtractionConfig:
        d = dict(data)
        d["real_feature_stroke_colors"] = [RGBColor(**c) for c in d.get("real_feature_stroke_colors", [])]
        return VectorExtractionConfig(**d)
    raise TypeError(f"Unhandled dataclass type: {cls}")


def load_config(path: str) -> MapConfig:
    """Load a MapConfig from a YAML or JSON file."""
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    d = dict(data)
    d["feature_type"] = FeatureType(d["feature_type"])
    d["source_kind"] = SourceKind(d.get("source_kind", "unknown"))
    d["feature_classes"] = [_dataclass_from_dict(FeatureClass, fc) for fc in d["feature_classes"]]
    d["georeference"] = _dataclass_from_dict(GeoreferenceConfig, d["georeference"])
    d["exclude_regions_px"] = [_dataclass_from_dict(ExcludeRegionPx, r) for r in d.get("exclude_regions_px", [])]
    d["shape_filter"] = _dataclass_from_dict(ShapeFilter, d.get("shape_filter", {})) if "shape_filter" in d else ShapeFilter()
    d["boundary"] = _dataclass_from_dict(BoundaryConfig, d["boundary"]) if d.get("boundary") else None
    d["crosscheck"] = _dataclass_from_dict(CrosscheckConfig, d["crosscheck"]) if d.get("crosscheck") else None
    d["ocr"] = _dataclass_from_dict(OCRConfig, d.get("ocr", {})) if "ocr" in d else OCRConfig()
    d["raster"] = _dataclass_from_dict(RasterExtractionConfig, d.get("raster", {})) if "raster" in d else RasterExtractionConfig()
    d["vector"] = _dataclass_from_dict(VectorExtractionConfig, d.get("vector", {})) if "vector" in d else VectorExtractionConfig()

    return MapConfig(**d)


def _enums_to_values(obj):
    """Recursively convert Enum instances to their plain values so both the
    yaml and json writers can serialize the result of asdict()."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _enums_to_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_enums_to_values(v) for v in obj]
    return obj


def save_config(config: MapConfig, path: str) -> None:
    """Write a MapConfig back out (useful after programmatically building one,
    e.g. from an interactive control-point picker)."""
    p = Path(path)
    data = _enums_to_values(asdict(config))
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        p.write_text(yaml.safe_dump(data, sort_keys=False))
    else:
        p.write_text(json.dumps(data, indent=2))
