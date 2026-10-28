"""
End-to-end orchestration: wires the per-stage modules together according to
a MapConfig. This is the only place that needs to know the stage order; each
stage module stays independently testable and independently usable (e.g. you
can call georeference.fit_georeference() on its own control points without
running the rest of the pipeline).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon, LineString, Point

from .config import MapConfig, FeatureType, SourceKind
from .assess import assess_pdf, extract_native_raster
from .vector_extract import extract_vector_shapes, merge_touching_shapes
from .raster_extract import extract_raster_shapes
from .line_extract import extract_line_shapes
from .ocr import ocr_tiles, associate_labels_to_shapes
from .georeference import fit_georeference, choose_best_degree
from .geometry import polygon_from_pixel_points, linestring_from_pixel_points, point_from_pixel
from .sanity import filter_by_boundary
from .crosscheck import crosscheck_features, load_tracker_csv, find_missing_tracker_geometry
from .report_excel import build_extraction_workbook, build_crosscheck_workbook
from .report_html import build_leaflet_map


class PipelineResult:
    def __init__(self):
        self.assessment = None
        self.raw_shapes = None          # RawShape / LineSegment list, pixel space
        self.ocr_labels = None          # {shape_index: (text, gap_px)}
        self.georef = None
        self.extracted_records = None   # list of dicts with shapely geometry in WGS84
        self.sanity = None
        self.matches = None
        self.full_match_df = None
        self.missing_from_tracker_df = None
        self.missing_geometry_df = None
        self.excel_paths = []
        self.html_path = None


def run_pipeline(config: MapConfig, raster_image_path: str = None, page_index: int = 0) -> PipelineResult:
    """
    raster_image_path: required when config.source_kind == RASTER (or when
    UNKNOWN and assessment determines raster) — the native-resolution image
    pulled from the PDF (see assess.extract_native_raster), or a
    pre-processed image if you're not starting from a PDF embed at all.
    """
    result = PipelineResult()

    # Stage 0
    assessment = assess_pdf(config.source_pdf, page_index=page_index)
    result.assessment = assessment

    source_kind = config.source_kind
    if source_kind == SourceKind.UNKNOWN:
        source_kind = SourceKind.VECTOR if assessment.likely_kind == "vector" else SourceKind.RASTER

    if source_kind == SourceKind.RASTER and raster_image_path is None:
        if assessment.native_image_xref is None:
            raise ValueError("Raster source but no embedded image found and no raster_image_path given")
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        raster_image_path = str(out_dir / "native_image.png")
        extract_native_raster(config.source_pdf, page_index, assessment.native_image_xref, raster_image_path)

    # Stage 2A / 2B / 9
    if source_kind == SourceKind.VECTOR:
        raw_shapes = extract_vector_shapes(config, page_index=page_index)
        raw_shapes = merge_touching_shapes(raw_shapes, config.feature_type)
    elif config.feature_type == FeatureType.LINE:
        raw_shapes = extract_line_shapes(config, raster_image_path)
    else:
        raw_shapes = extract_raster_shapes(config, raster_image_path)
    result.raw_shapes = raw_shapes

    # Stage 3 (OCR) — only meaningful when we have a raster image to read text from.
    ocr_labels = {}
    if config.ocr.enabled and raster_image_path is not None:
        phrases = ocr_tiles(raster_image_path, config.ocr)
        ocr_labels = associate_labels_to_shapes(raw_shapes, phrases, config.ocr.max_label_shape_gap_px)
    result.ocr_labels = ocr_labels

    # Stage 4
    degree = config.georeference.degree
    if degree is None:
        degree = choose_best_degree(config.georeference.control_points)
    georef = fit_georeference(
        config.georeference.control_points,
        degree=degree,
        max_acceptable_residual_km=config.georeference.max_acceptable_residual_km,
        region_bbox=config.georeference.region_bbox,
    )
    result.georef = georef

    # Stage 6 — build WGS84 geometry per shape/segment
    records = []
    for idx, shape in enumerate(raw_shapes):
        name, gap_px = ocr_labels.get(idx, (None, None))
        base = {
            "shape_id": idx,
            "feature_class": shape.feature_class,
            "candidate_name": name,
            "name_source": "OCR (unverified)" if name else "unlabeled",
            "relative_size_px": getattr(shape, "area_px", getattr(shape, "length_px", None)),
            "label_match_distance_px": gap_px,
        }
        if config.feature_type == FeatureType.AREA:
            geom = polygon_from_pixel_points(
                shape.pixel_points, georef.pixel_to_lonlat,
                config.simplify_tolerance_deg, config.coordinate_decimals,
            )
        elif config.feature_type == FeatureType.LINE:
            pts = shape.pixel_points
            if pts and isinstance(pts[0], list):
                # merged multi-part segment (split_lines_at_junctions=False)
                from shapely.geometry import MultiLineString
                lines = [linestring_from_pixel_points(p, georef.pixel_to_lonlat,
                                                       config.simplify_tolerance_deg, config.coordinate_decimals)
                         for p in pts]
                geom = MultiLineString(lines)
            else:
                geom = linestring_from_pixel_points(
                    pts, georef.pixel_to_lonlat, config.simplify_tolerance_deg, config.coordinate_decimals,
                )
        else:  # POINT
            cx = (shape.bbox_px[0] + shape.bbox_px[2]) / 2
            cy = (shape.bbox_px[1] + shape.bbox_px[3]) / 2
            geom = point_from_pixel(cx, cy, georef.pixel_to_lonlat, config.coordinate_decimals)

        base["geometry"] = geom
        c = geom.centroid
        base["longitude"] = round(c.x, config.coordinate_decimals)
        base["latitude"] = round(c.y, config.coordinate_decimals)
        records.append(base)

    # Stage 5
    if config.boundary is not None:
        sanity = filter_by_boundary(records, config.boundary)
        result.sanity = sanity
        records = sanity.kept
    result.extracted_records = records

    # Deliverable: standalone extraction workbook (always produced)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "geometry"} | {"wkt": r["geometry"].wkt}
        for r in records
    ])
    georef_summary = {
        "degree": georef.degree,
        "n_used": len(georef.control_points_used),
        "n_dropped": len(georef.control_points_dropped),
        "mean_residual_km": georef.mean_residual_km,
        "max_residual_km": georef.max_residual_km,
    }
    method_notes = (
        f"Source PDF classified as {source_kind.value} ({assessment.n_vector_paths} vector paths, "
        f"{assessment.n_embedded_images} embedded images). Feature classes: "
        f"{', '.join(fc.label for fc in config.feature_classes)}."
    )
    extraction_xlsx = out_dir / f"{config.name.replace(' ', '_')}_extraction.xlsx"
    build_extraction_workbook(config, extracted_df, georef_summary, method_notes, str(extraction_xlsx))
    result.excel_paths.append(str(extraction_xlsx))

    # Stage 7 — crosscheck, if configured
    if config.crosscheck is not None:
        tracker_df = load_tracker_csv(config.crosscheck.tracker_csv_path, config.crosscheck.columns)
        matches = crosscheck_features(records, tracker_df, config.crosscheck)
        result.matches = matches

        cols = config.crosscheck.columns
        full_rows = []
        missing_rows = []
        for rec, m in zip(records, matches):
            row = {
                "shape_id": rec["shape_id"], "feature_class": rec["feature_class"],
                "candidate_name": rec["candidate_name"], "longitude": rec["longitude"], "latitude": rec["latitude"],
                "relative_size_px": rec["relative_size_px"], "match_type": m.match_type,
                "matched_tracker_name": m.closest_tracker_name if m.match_type != "no confident match" else None,
                "match_distance_km": m.closest_distance_km,
                "closest_tracker_name": m.closest_tracker_name, "closest_tracker_dist_km": m.closest_distance_km,
                "best_name_match": m.best_name_match_name, "name_score": m.best_name_score,
                "wkt": rec["geometry"].wkt,
            }
            full_rows.append(row)
            if m.match_type == "no confident match":
                missing_rows.append(row)

        result.full_match_df = pd.DataFrame(full_rows)
        result.missing_from_tracker_df = pd.DataFrame(missing_rows)
        result.missing_geometry_df = find_missing_tracker_geometry(tracker_df, config.crosscheck, records, matches)

        n_no_geom = result.missing_geometry_df.shape[0] if cols.existing_wkt not in tracker_df.columns else \
            tracker_df[cols.existing_wkt].isna().sum()
        headline_notes = (
            f"Tracker export: {len(tracker_df)} records. Map extraction: {len(records)} shapes. "
            f"{len(result.missing_from_tracker_df)} have no confident tracker match (candidate missing). "
            f"{result.missing_geometry_df['suggested_outline_wkt'].notna().sum()} tracker records without "
            f"existing geometry were given a candidate outline suggestion."
        )
        crosscheck_xlsx = out_dir / f"{config.name.replace(' ', '_')}_crosscheck.xlsx"
        build_crosscheck_workbook(
            config, extracted_df, result.full_match_df, result.missing_from_tracker_df,
            result.missing_geometry_df, headline_notes, str(crosscheck_xlsx),
        )
        result.excel_paths.append(str(crosscheck_xlsx))

        map_records = [dict(r, match_type=m.match_type) for r, m in zip(records, matches)]
    else:
        map_records = records

    # HTML QA map (always produced)
    html_path = out_dir / f"{config.name.replace(' ', '_')}_map.html"
    build_leaflet_map(map_records, str(html_path), title=f"{config.name} — digitization QA")
    result.html_path = str(html_path)

    return result
