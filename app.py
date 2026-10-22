#!/usr/bin/env python3
"""
GEM Map Digitizer — interactive front end for the map_digitizer pipeline.

Upload an infrastructure map (PDF or image), mark a handful of control
points and legend colors, optionally upload a GOGET (or any tracker) export
to crosscheck against, and get back the same deliverables the manual
procedure produces: a raw-extraction workbook, a tracker-crosscheck
workbook, and an interactive QA map — all georeferenced to WGS84.

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import os
import tempfile
import time
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from map_digitizer.config import (
    MapConfig, FeatureType, SourceKind, RGBColor, FeatureClass, ExcludeRegionPx,
    ShapeFilter, ControlPoint, GeoreferenceConfig, BoundaryConfig, TrackerColumns,
    CrosscheckConfig, OCRConfig, RasterExtractionConfig, VectorExtractionConfig,
    save_config,
)
from map_digitizer.assess import assess_pdf, extract_native_raster
from map_digitizer.vector_extract import extract_vector_shapes, merge_touching_shapes
from map_digitizer.raster_extract import extract_raster_shapes, load_image_rgb
from map_digitizer.line_extract import extract_line_shapes
from map_digitizer.ocr import ocr_tiles, associate_labels_to_shapes
from map_digitizer.georeference import fit_georeference, choose_best_degree
from map_digitizer.geometry import polygon_from_pixel_points, linestring_from_pixel_points, point_from_pixel
from map_digitizer.sanity import filter_by_boundary
from map_digitizer.crosscheck import crosscheck_features, load_tracker_csv, find_missing_tracker_geometry
from map_digitizer.report_excel import build_extraction_workbook, build_crosscheck_workbook
from map_digitizer.report_html import build_leaflet_map

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_CLICK_PICKER = True
except ImportError:
    HAS_CLICK_PICKER = False

st.set_page_config(page_title="GEM Map Digitizer", layout="wide")

# ---- Brand palette (matches GEM's production dashboard) ----
DARK = "#016B83"
LIGHT = "#F2F2EB"
HIGHLIGHT = "#9DF7E5"
CHART_COLORWAY = [
    "#016B83", "#FE4F2D", "#4A57A8", "#65BD8B", "#7F142A",
    "#099ED8", "#061F5F", "#CA4A50", "#004F61",
]
MATCH_COLORS = {
    "contained/very close": "#1a9850",
    "name match": "#66bd63",
    "nearby": "#fee08b",
    "no confident match": "#d73027",
}

FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")
WORKDIR = Path(tempfile.gettempdir()) / "map_digitizer_app"
WORKDIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def load_theme_css():
    def _b64(name):
        path = os.path.join(FONT_DIR, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")

    regular = _b64("PlusJakartaSans-VariableFont_wght.woff2")
    italic = _b64("PlusJakartaSans-Italic-VariableFont_wght.woff2")
    font_faces = ""
    if regular:
        font_faces += f"""
        @font-face {{
            font-family: 'Plus Jakarta Sans';
            src: url(data:font/woff2;base64,{regular}) format('woff2-variations'),
                 url(data:font/woff2;base64,{regular}) format('woff2');
            font-weight: 200 800; font-style: normal; font-display: swap;
        }}"""
    if italic:
        font_faces += f"""
        @font-face {{
            font-family: 'Plus Jakarta Sans';
            src: url(data:font/woff2;base64,{italic}) format('woff2-variations'),
                 url(data:font/woff2;base64,{italic}) format('woff2');
            font-weight: 200 800; font-style: italic; font-display: swap;
        }}"""

    return f"""
    <style>
    {font_faces}
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: {DARK}; }}
    [data-testid="stMetricValue"] {{ color: {DARK}; }}
    div[data-baseweb="tab-highlight"] {{ background-color: {HIGHLIGHT} !important; }}
    .stDownloadButton button, .stButton button {{ border-color: {DARK}; color: {DARK}; }}
    .stDownloadButton button:hover, .stButton button:hover {{
        border-color: {HIGHLIGHT}; color: {DARK}; background-color: {HIGHLIGHT};
    }}
    </style>
    """


st.markdown(load_theme_css(), unsafe_allow_html=True)

st.title("Map Digitizer")
st.caption(
    "Turns an infrastructure map (PDF or image) into georeferenced WGS84 features and, "
    "optionally, a crosscheck against a GOGET (or any) tracker export — following the same "
    "config-driven procedure as the manual worked examples (assess → extract → OCR → "
    "georeference → sanity-check → crosscheck → report). No AI extraction calls: every "
    "number below is something you read off the map once, same as the manual procedure."
)

if "control_points" not in st.session_state:
    st.session_state.control_points = pd.DataFrame(
        [{"label": "", "pixel_x": 0.0, "pixel_y": 0.0, "lon": 0.0, "lat": 0.0}]
    )
if "feature_classes" not in st.session_state:
    st.session_state.feature_classes = pd.DataFrame(
        [{"label": "Field", "r": 40, "g": 140, "b": 60, "tolerance": 25}]
    )
if "exclude_regions" not in st.session_state:
    st.session_state.exclude_regions = pd.DataFrame(
        columns=["name", "x0", "y0", "x1", "y1"]
    )
if "last_click_px" not in st.session_state:
    st.session_state.last_click_px = None
if "run_result" not in st.session_state:
    st.session_state.run_result = None

# ---------------------------------------------------------------------------
# 1. Upload the map
# ---------------------------------------------------------------------------
st.header("1. Upload your map")
map_file = st.file_uploader(
    "Map file (PDF, PNG, or JPG)", type=["pdf", "png", "jpg", "jpeg"], key="map_file"
)

map_name = st.text_input("Map / run name (used in output file names)", value="My_Map")

source_kind = None
working_image_path = None   # the exact pixel space used for extraction (native raster / plain image)
working_image_scale = 1.0   # display_px / native_px for the working image, when downscaled for preview
vector_source_pdf = None
vector_preview_scale = 1.0  # preview_px / PDF-point, for vector maps (click coords / this = point-space)
assessment = None
is_plain_image = False

if map_file is not None:
    suffix = Path(map_file.name).suffix.lower()
    upload_path = WORKDIR / f"upload{suffix}"
    upload_path.write_bytes(map_file.getvalue())

    if suffix == ".pdf":
        vector_source_pdf = str(upload_path)
        assessment = assess_pdf(vector_source_pdf)
        c1, c2, c3 = st.columns(3)
        c1.metric("Likely kind", assessment.likely_kind)
        c2.metric("Vector paths", assessment.n_vector_paths)
        c3.metric("Needs OCR", "Yes" if assessment.needs_ocr else "No")

        default_kind = "vector" if assessment.likely_kind == "vector" else "raster"
        kind_choice = st.radio(
            "Source kind", ["vector", "raster"],
            index=0 if default_kind == "vector" else 1, horizontal=True,
            help="Vector: the map was authored as real vector paths in the PDF (fills/strokes). "
                 "Raster: the PDF is a flattened scan/export — extraction works on pixel colors instead.",
        )
        source_kind = SourceKind.VECTOR if kind_choice == "vector" else SourceKind.RASTER

        if source_kind == SourceKind.VECTOR:
            import fitz
            doc = fitz.open(vector_source_pdf)
            page = doc[0]
            target_px = 1400
            vector_preview_scale = target_px / max(page.rect.width, page.rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(vector_preview_scale, vector_preview_scale))
            preview_path = WORKDIR / "vector_preview.png"
            pix.save(str(preview_path))
            doc.close()
            preview_img = Image.open(preview_path)
        else:
            if assessment.native_image_xref is None:
                st.error("No embedded raster image found in this PDF page.")
                st.stop()
            native_path = WORKDIR / "native_image.png"
            extract_native_raster(vector_source_pdf, 0, assessment.native_image_xref, str(native_path))
            working_image_path = str(native_path)
            preview_img = Image.open(working_image_path)
    else:
        is_plain_image = True
        source_kind = SourceKind.RASTER
        working_image_path = str(upload_path)
        preview_img = Image.open(working_image_path)

    # Downscale very large working images for a manageable preview; the click
    # picker's coordinates are converted back to native pixel space below.
    if not (suffix == ".pdf" and source_kind == SourceKind.VECTOR):
        max_dim = 1400
        w, h = preview_img.size
        if max(w, h) > max_dim:
            working_image_scale = max_dim / max(w, h)
            preview_img = preview_img.resize((int(w * working_image_scale), int(h * working_image_scale)))

    st.caption(
        "Click the preview to read off a pixel position (for control points or a legend swatch color). "
        if HAS_CLICK_PICKER else
        "Install `streamlit-image-coordinates` to click-pick pixel positions; for now, read coordinates "
        "off the preview manually (hover tools in your own image viewer work too)."
    )
    if HAS_CLICK_PICKER:
        click = streamlit_image_coordinates(preview_img, key="map_preview_click")
        if click is not None:
            disp_x, disp_y = click["x"], click["y"]
            if suffix == ".pdf" and source_kind == SourceKind.VECTOR:
                native_x, native_y = disp_x / vector_preview_scale, disp_y / vector_preview_scale
            else:
                native_x, native_y = disp_x / working_image_scale, disp_y / working_image_scale
            st.session_state.last_click_px = (native_x, native_y)
            swatch_rgb = None
            try:
                full_img = Image.open(working_image_path).convert("RGB") if working_image_path else None
                if full_img is not None:
                    swatch_rgb = full_img.getpixel((int(native_x), int(native_y)))
            except Exception:
                swatch_rgb = None
            msg = f"Last click → pixel ({native_x:.1f}, {native_y:.1f})"
            if swatch_rgb:
                msg += f" · RGB at that pixel: {swatch_rgb}"
            st.info(msg)
    else:
        st.image(preview_img)

# ---------------------------------------------------------------------------
# 2. Feature type & classes
# ---------------------------------------------------------------------------
st.header("2. Feature type & classes")
feature_type_label = st.selectbox(
    "Feature type", ["Area (polygons — fields, concessions...)",
                      "Point (facilities — plants, terminals...)",
                      "Line (pipelines, transmission lines...)"],
)
feature_type = {
    "Area (polygons — fields, concessions...)": FeatureType.AREA,
    "Point (facilities — plants, terminals...)": FeatureType.POINT,
    "Line (pipelines, transmission lines...)": FeatureType.LINE,
}[feature_type_label]

st.caption(
    "One row per legend category (e.g. 'Gas Field' vs 'Oil Field'). Sample the fill color from a "
    "legend swatch on the map — click it above, then copy the reported RGB in here."
)
st.session_state.feature_classes = st.data_editor(
    st.session_state.feature_classes, num_rows="dynamic", use_container_width=True,
    column_config={
        "label": st.column_config.TextColumn("Label", required=True),
        "r": st.column_config.NumberColumn("R", min_value=0, max_value=255),
        "g": st.column_config.NumberColumn("G", min_value=0, max_value=255),
        "b": st.column_config.NumberColumn("B", min_value=0, max_value=255),
        "tolerance": st.column_config.NumberColumn("Tolerance", min_value=1, max_value=120, help="Euclidean RGB match tolerance"),
    },
    key="feature_classes_editor",
)

# ---------------------------------------------------------------------------
# 3. Control points (georeferencing)
# ---------------------------------------------------------------------------
st.header("3. Control points (georeferencing)")
st.caption(
    "At least 3 (pixel, lon/lat) pairs for known locations on the map — named places, or (best) "
    "fields/facilities whose coordinates are already in your tracker export. Click the map above to "
    "read off a pixel position, then type its known real-world longitude/latitude here."
)
if st.session_state.last_click_px and HAS_CLICK_PICKER:
    cpx, cpy = st.session_state.last_click_px
    st.caption(f"Last click was at pixel ({cpx:.1f}, {cpy:.1f}) — paste those into a new row's pixel_x / pixel_y.")

st.session_state.control_points = st.data_editor(
    st.session_state.control_points, num_rows="dynamic", use_container_width=True,
    column_config={
        "label": st.column_config.TextColumn("Label"),
        "pixel_x": st.column_config.NumberColumn("Pixel X", format="%.1f"),
        "pixel_y": st.column_config.NumberColumn("Pixel Y", format="%.1f"),
        "lon": st.column_config.NumberColumn("Longitude", format="%.5f"),
        "lat": st.column_config.NumberColumn("Latitude", format="%.5f"),
    },
    key="control_points_editor",
)

gc1, gc2 = st.columns(2)
degree_choice = gc1.selectbox("Georeference fit", ["Auto (cross-validated)", "Affine (degree 1)", "Quadratic (degree 2)"])
max_residual = gc2.number_input(
    "Max acceptable mean residual (km) — drop worst outlier control points to reach this",
    min_value=0.0, value=15.0, step=1.0,
    help="Set to 0 to disable outlier-dropping and use all control points as given.",
)

# ---------------------------------------------------------------------------
# 4. Exclude regions (optional)
# ---------------------------------------------------------------------------
with st.expander("4. Exclude regions (optional) — legend box, inset map, title block, etc."):
    st.session_state.exclude_regions = st.data_editor(
        st.session_state.exclude_regions, num_rows="dynamic", use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Name"),
            "x0": st.column_config.NumberColumn("x0"), "y0": st.column_config.NumberColumn("y0"),
            "x1": st.column_config.NumberColumn("x1"), "y1": st.column_config.NumberColumn("y1"),
        },
        key="exclude_regions_editor",
    )

# ---------------------------------------------------------------------------
# 5. Advanced extraction settings
# ---------------------------------------------------------------------------
with st.expander("5. Advanced extraction settings"):
    st.subheader("Shape filter (separates real features from icons / fragments)")
    sf1, sf2, sf3, sf4 = st.columns(4)
    min_area_px = sf1.number_input("Min area (px²)", min_value=1.0, value=100.0)
    aspect_ratio_max = sf2.number_input("Max aspect ratio", min_value=1.0, value=4.5)
    extent_min = sf3.number_input("Min extent", min_value=0.0, max_value=1.0, value=0.3)
    solidity_min = sf4.number_input("Min solidity", min_value=0.0, max_value=1.0, value=0.6)

    st.subheader("Raster-specific")
    r1, r2 = st.columns(2)
    line_width_px = r1.number_input("Thin-line width to sever (px)", min_value=0.5, value=2.0)
    use_panel_bbox = r2.checkbox("Restrict to a map-panel bounding box", value=False)
    panel_bbox = None
    if use_panel_bbox:
        pb1, pb2, pb3, pb4 = st.columns(4)
        panel_bbox = (
            pb1.number_input("panel x0", value=0.0), pb2.number_input("panel y0", value=0.0),
            pb3.number_input("panel x1", value=1000.0), pb4.number_input("panel y1", value=1000.0),
        )

    st.subheader("Vector-specific")
    real_stroke_rgb = st.text_input(
        "Real-feature stroke color as 'r,g,b' (optional — separates real features from decorative icons "
        "sharing a fill color)", value="",
    )
    bezier_tol = st.number_input("Bezier flatten tolerance (px)", min_value=0.1, value=1.0)

# ---------------------------------------------------------------------------
# 6. OCR
# ---------------------------------------------------------------------------
with st.expander("6. OCR label reading"):
    ocr_available = working_image_path is not None
    if not ocr_available:
        st.caption("OCR needs a raster image — not used for a vector-source PDF unless you also supply one.")
    ocr_enabled = st.checkbox("Read feature labels with OCR (Tesseract)", value=ocr_available, disabled=not ocr_available)
    o1, o2, o3 = st.columns(3)
    tile_size_px = o1.number_input("Tile size (px)", min_value=200, value=1400)
    upscale_factor = o2.number_input("Upscale factor", min_value=1.0, value=1.0, step=0.5)
    max_label_gap_px = o3.number_input("Max label-to-shape gap (px)", min_value=1.0, value=60.0)

# ---------------------------------------------------------------------------
# 7. Geographic sanity boundary (optional)
# ---------------------------------------------------------------------------
with st.expander("7. Geographic sanity boundary (optional)"):
    use_boundary = st.checkbox("Flag features outside an expected bounding box", value=False)
    boundary_bbox, buffer_km = None, 50.0
    if use_boundary:
        b1, b2, b3, b4, b5 = st.columns(5)
        boundary_bbox = (
            b1.number_input("min lon", value=-10.0), b2.number_input("min lat", value=-10.0),
            b3.number_input("max lon", value=10.0), b4.number_input("max lat", value=10.0),
        )
        buffer_km = b5.number_input("Buffer (km)", min_value=0.0, value=50.0)

# ---------------------------------------------------------------------------
# 8. Crosscheck against a tracker export (e.g. GOGET)
# ---------------------------------------------------------------------------
st.header("8. Crosscheck against a tracker export (e.g. GOGET)")
tracker_file = st.file_uploader(
    "Tracker export (CSV or XLSX) — optional. Skip this to produce extraction-only output.",
    type=["csv", "xlsx"], key="tracker_file",
)
crosscheck_cfg = None
tracker_csv_path = None
if tracker_file is not None:
    if tracker_file.name.lower().endswith(".xlsx"):
        tracker_df_preview = pd.read_excel(tracker_file)
    else:
        tracker_df_preview = pd.read_csv(tracker_file)
    tracker_csv_path = str(WORKDIR / "tracker.csv")
    tracker_df_preview.to_csv(tracker_csv_path, index=False)
    st.dataframe(tracker_df_preview.head(5), use_container_width=True)

    cols = list(tracker_df_preview.columns)

    def _guess(options, *names):
        for n in names:
            if n in options:
                return options.index(n)
        return 0

    t1, t2, t3, t4, t5 = st.columns(5)
    id_col = t1.selectbox("ID column", cols, index=_guess(cols, "GEM Field ID", "id", "ID"))
    name_col = t2.selectbox("Name column", cols, index=_guess(cols, "Field Name", "name", "Name"))
    lat_col = t3.selectbox("Latitude column", cols, index=_guess(cols, "Latitude", "latitude", "lat"))
    lon_col = t4.selectbox("Longitude column", cols, index=_guess(cols, "Longitude", "longitude", "lon"))
    wkt_options = ["(none)"] + cols
    wkt_col = t5.selectbox("Existing geometry (WKT) column", wkt_options, index=_guess(wkt_options, "Wkt Area", "wkt", "Wkt"))

    x1, x2, x3, x4 = st.columns(4)
    name_score_threshold = x1.number_input("Name match threshold (0-100)", min_value=0.0, max_value=100.0, value=70.0)
    contained_distance_km = x2.number_input("Contained/very-close distance (km)", min_value=0.0, value=15.0)
    nearby_distance_km = x3.number_input("Nearby distance (km)", min_value=0.0, value=40.0)
    name_match_search_cap_km = x4.number_input("Name-match search cap (km)", min_value=0.0, value=200.0)

    crosscheck_cfg = CrosscheckConfig(
        tracker_csv_path=tracker_csv_path,
        columns=TrackerColumns(
            id=id_col, name=name_col, lat=lat_col, lon=lon_col,
            existing_wkt=None if wkt_col == "(none)" else wkt_col,
        ),
        name_score_threshold=name_score_threshold,
        contained_distance_km=contained_distance_km,
        nearby_distance_km=nearby_distance_km,
        name_match_search_cap_km=name_match_search_cap_km,
    )

st.divider()


# ---------------------------------------------------------------------------
# Build the MapConfig from everything above
# ---------------------------------------------------------------------------
def build_config() -> MapConfig:
    fc_rows = st.session_state.feature_classes.dropna(subset=["label"])
    feature_classes = [
        FeatureClass(label=str(r["label"]), fill_color=RGBColor(int(r["r"]), int(r["g"]), int(r["b"]), int(r["tolerance"])))
        for _, r in fc_rows.iterrows()
    ]

    cp_rows = st.session_state.control_points.dropna(subset=["lon", "lat"])
    cp_rows = cp_rows[(cp_rows["pixel_x"] != 0) | (cp_rows["pixel_y"] != 0)]
    control_points = [
        ControlPoint(label=str(r.get("label") or f"cp{i}"), pixel_x=float(r["pixel_x"]), pixel_y=float(r["pixel_y"]),
                     lon=float(r["lon"]), lat=float(r["lat"]))
        for i, r in cp_rows.iterrows()
    ]

    degree = None if degree_choice.startswith("Auto") else (1 if "1" in degree_choice else 2)

    exclude_regions = [
        ExcludeRegionPx(name=str(r["name"]), x0=float(r["x0"]), y0=float(r["y0"]), x1=float(r["x1"]), y1=float(r["y1"]))
        for _, r in st.session_state.exclude_regions.dropna().iterrows()
    ] if len(st.session_state.exclude_regions) else []

    vector_cfg = VectorExtractionConfig(bezier_flatten_tolerance_px=bezier_tol)
    if real_stroke_rgb.strip():
        try:
            r, g, b = [int(v.strip()) for v in real_stroke_rgb.split(",")]
            vector_cfg.real_feature_stroke_colors = [RGBColor(r, g, b, tolerance=15)]
        except ValueError:
            pass

    raster_cfg = RasterExtractionConfig(line_width_px=line_width_px)
    if use_panel_bbox and panel_bbox is not None:
        raster_cfg.map_panel_bbox_px = panel_bbox

    boundary_cfg = BoundaryConfig(bbox=boundary_bbox, buffer_km=buffer_km) if (use_boundary and boundary_bbox) else None

    cfg = MapConfig(
        name=map_name or "Map",
        source_pdf=vector_source_pdf or (working_image_path or "map"),
        feature_type=feature_type,
        source_kind=source_kind or SourceKind.UNKNOWN,
        feature_classes=feature_classes,
        georeference=GeoreferenceConfig(control_points=control_points, degree=degree,
                                         max_acceptable_residual_km=(max_residual or None)),
        output_dir=str(WORKDIR / "output"),
        exclude_regions_px=exclude_regions,
        shape_filter=ShapeFilter(min_area_px=min_area_px, aspect_ratio_max=aspect_ratio_max,
                                  extent_min=extent_min, solidity_min=solidity_min),
        boundary=boundary_cfg,
        crosscheck=crosscheck_cfg,
        ocr=OCRConfig(enabled=ocr_enabled if working_image_path else False, tile_size_px=int(tile_size_px),
                      upscale_factor=upscale_factor, max_label_shape_gap_px=max_label_gap_px),
        raster=raster_cfg,
        vector=vector_cfg,
    )
    return cfg


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
st.header("9. Run")
ready = map_file is not None and len(st.session_state.feature_classes.dropna(subset=["label"])) > 0
run_clicked = st.button("Run digitization", type="primary", disabled=not ready, use_container_width=True)

if not ready:
    st.caption("Upload a map and define at least one feature class to enable this.")

if run_clicked:
    progress = st.progress(0, text="Starting…")
    log = st.status("Running digitization pipeline…", expanded=True)
    try:
        cfg = build_config()
        n_cps = len(cfg.georeference.control_points)
        if n_cps < 3:
            log.update(label="Need at least 3 control points", state="error")
            st.error(f"Only {n_cps} control point(s) given — at least 3 are required to georeference the map.")
            st.stop()

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        steps_total = 8
        step_counter = {"n": 0}

        def advance(label):
            step_counter["n"] += 1
            progress.progress(step_counter["n"] / steps_total, text=label)
            log.write(f"✓ {label}")

        # Stage 2: extract raw shapes
        log.write("Extracting shapes…")
        if cfg.source_kind == SourceKind.VECTOR:
            raw_shapes = extract_vector_shapes(cfg, page_index=0)
            raw_shapes = merge_touching_shapes(raw_shapes, cfg.feature_type)
        elif cfg.feature_type == FeatureType.LINE:
            raw_shapes = extract_line_shapes(cfg, working_image_path)
        else:
            raw_shapes = extract_raster_shapes(cfg, working_image_path)
        advance(f"Extracted {len(raw_shapes)} raw shapes")

        # Stage 3: OCR
        ocr_labels = {}
        if cfg.ocr.enabled and working_image_path is not None:
            phrases = ocr_tiles(working_image_path, cfg.ocr)
            ocr_labels = associate_labels_to_shapes(raw_shapes, phrases, cfg.ocr.max_label_shape_gap_px)
        advance(f"OCR: labeled {len(ocr_labels)} of {len(raw_shapes)} shapes" if cfg.ocr.enabled else "OCR skipped")

        # Stage 4: georeference
        degree = cfg.georeference.degree
        if degree is None:
            degree = choose_best_degree(cfg.georeference.control_points)
        georef = fit_georeference(cfg.georeference.control_points, degree=degree,
                                   max_acceptable_residual_km=cfg.georeference.max_acceptable_residual_km)
        advance(f"Georeferenced (degree {georef.degree}, mean residual {georef.mean_residual_km:.1f} km, "
                f"max {georef.max_residual_km:.1f} km)")

        # Stage 6: build WGS84 geometry
        records = []
        for idx, shape in enumerate(raw_shapes):
            name, gap_px = ocr_labels.get(idx, (None, None))
            base = {
                "shape_id": idx, "feature_class": shape.feature_class, "candidate_name": name,
                "name_source": "OCR (unverified)" if name else "unlabeled",
                "relative_size_px": getattr(shape, "area_px", getattr(shape, "length_px", None)),
                "label_match_distance_px": gap_px,
            }
            if cfg.feature_type == FeatureType.AREA:
                geom = polygon_from_pixel_points(shape.pixel_points, georef.pixel_to_lonlat,
                                                  cfg.simplify_tolerance_deg, cfg.coordinate_decimals)
            elif cfg.feature_type == FeatureType.LINE:
                pts = shape.pixel_points
                if pts and isinstance(pts[0], list):
                    from shapely.geometry import MultiLineString
                    lines = [linestring_from_pixel_points(p, georef.pixel_to_lonlat,
                                                           cfg.simplify_tolerance_deg, cfg.coordinate_decimals)
                             for p in pts]
                    geom = MultiLineString(lines)
                else:
                    geom = linestring_from_pixel_points(pts, georef.pixel_to_lonlat,
                                                         cfg.simplify_tolerance_deg, cfg.coordinate_decimals)
            else:
                cx = (shape.bbox_px[0] + shape.bbox_px[2]) / 2
                cy = (shape.bbox_px[1] + shape.bbox_px[3]) / 2
                geom = point_from_pixel(cx, cy, georef.pixel_to_lonlat, cfg.coordinate_decimals)
            base["geometry"] = geom
            c = geom.centroid
            base["longitude"] = round(c.x, cfg.coordinate_decimals)
            base["latitude"] = round(c.y, cfg.coordinate_decimals)
            records.append(base)
        advance(f"Built {len(records)} WGS84 geometries")

        # Stage 5: sanity filter
        sanity = None
        if cfg.boundary is not None:
            sanity = filter_by_boundary(records, cfg.boundary)
            records = sanity.kept
        advance(f"Sanity filter flagged {len(sanity.flagged)} feature(s)" if sanity else "Sanity filter skipped")

        # Extraction workbook (always)
        extracted_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "geometry"} | {"wkt": r["geometry"].wkt} for r in records
        ])
        georef_summary = {
            "degree": georef.degree, "n_used": len(georef.control_points_used),
            "n_dropped": len(georef.control_points_dropped),
            "mean_residual_km": georef.mean_residual_km, "max_residual_km": georef.max_residual_km,
        }
        method_notes = (
            f"Source classified as {cfg.source_kind.value}. Feature classes: "
            f"{', '.join(fc.label for fc in cfg.feature_classes)}."
        )
        extraction_xlsx = out_dir / f"{cfg.name.replace(' ', '_')}_extraction.xlsx"
        build_extraction_workbook(cfg, extracted_df, georef_summary, method_notes, str(extraction_xlsx))
        advance("Built raw-extraction workbook")

        full_match_df = missing_from_tracker_df = missing_geometry_df = None
        crosscheck_xlsx = None
        if cfg.crosscheck is not None:
            tracker_df = load_tracker_csv(cfg.crosscheck.tracker_csv_path, cfg.crosscheck.columns)
            matches = crosscheck_features(records, tracker_df, cfg.crosscheck)
            cols_ = cfg.crosscheck.columns
            full_rows, missing_rows = [], []
            for rec, m in zip(records, matches):
                row = {
                    "shape_id": rec["shape_id"], "feature_class": rec["feature_class"],
                    "candidate_name": rec["candidate_name"], "longitude": rec["longitude"], "latitude": rec["latitude"],
                    "relative_size_px": rec["relative_size_px"], "match_type": m.match_type,
                    "matched_tracker_name": m.closest_tracker_name if m.match_type != "no confident match" else None,
                    "match_distance_km": m.closest_distance_km, "closest_tracker_name": m.closest_tracker_name,
                    "closest_tracker_dist_km": m.closest_distance_km, "best_name_match": m.best_name_match_name,
                    "name_score": m.best_name_score, "wkt": rec["geometry"].wkt,
                }
                full_rows.append(row)
                if m.match_type == "no confident match":
                    missing_rows.append(row)
            full_match_df = pd.DataFrame(full_rows)
            missing_from_tracker_df = pd.DataFrame(missing_rows)
            missing_geometry_df = find_missing_tracker_geometry(tracker_df, cfg.crosscheck, records, matches)
            headline_notes = (
                f"Tracker export: {len(tracker_df)} records. Map extraction: {len(records)} shapes. "
                f"{len(missing_from_tracker_df)} have no confident tracker match (candidate missing). "
                f"{missing_geometry_df['suggested_outline_wkt'].notna().sum()} tracker records without "
                f"existing geometry were given a candidate outline suggestion."
            )
            crosscheck_xlsx = out_dir / f"{cfg.name.replace(' ', '_')}_crosscheck.xlsx"
            build_crosscheck_workbook(cfg, extracted_df, full_match_df, missing_from_tracker_df,
                                       missing_geometry_df, headline_notes, str(crosscheck_xlsx))
            map_records = [dict(r, match_type=m.match_type) for r, m in zip(records, matches)]
            advance(f"Crosschecked vs {len(tracker_df)} tracker records")
        else:
            map_records = records
            advance("Crosscheck skipped (no tracker file uploaded)")

        html_path = out_dir / f"{cfg.name.replace(' ', '_')}_map.html"
        build_leaflet_map(map_records, str(html_path), title=f"{cfg.name} — digitization QA")
        config_yaml_path = out_dir / f"{cfg.name.replace(' ', '_')}_config.yaml"
        try:
            save_config(cfg, str(config_yaml_path))
        except Exception:
            config_yaml_path = None
        advance("Built interactive QA map")

        progress.progress(1.0, text="Done")
        log.update(label="Pipeline complete", state="complete")

        st.session_state.run_result = {
            "records": records, "extracted_df": extracted_df, "georef_summary": georef_summary,
            "extraction_xlsx": str(extraction_xlsx),
            "crosscheck_xlsx": str(crosscheck_xlsx) if crosscheck_xlsx else None,
            "full_match_df": full_match_df, "html_path": str(html_path),
            "config_yaml_path": str(config_yaml_path) if config_yaml_path else None,
        }
    except Exception as e:
        log.update(label="Pipeline failed", state="error")
        st.error(f"{type(e).__name__}: {e}")
        st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.run_result
if result:
    st.header("Results")
    gsum = result["georef_summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Features extracted", len(result["records"]))
    m2.metric("Georef degree", gsum["degree"])
    m3.metric("Mean residual (km)", f"{gsum['mean_residual_km']:.1f}")
    m4.metric("Max residual (km)", f"{gsum['max_residual_km']:.1f}")

    if result["full_match_df"] is not None:
        import plotly.express as px
        counts = result["full_match_df"]["match_type"].value_counts().reset_index()
        counts.columns = ["match_type", "count"]
        fig = px.bar(counts, x="match_type", y="count", color="match_type",
                     color_discrete_map=MATCH_COLORS, title="Crosscheck match breakdown")
        fig.update_layout(showlegend=False, paper_bgcolor=LIGHT, plot_bgcolor=LIGHT,
                           font=dict(color=DARK))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Interactive QA map")
    html_content = Path(result["html_path"]).read_text()
    st.components.v1.html(html_content, height=650, scrolling=False)

    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)
    with open(result["extraction_xlsx"], "rb") as f:
        d1.download_button("Raw extraction (.xlsx)", f, file_name=Path(result["extraction_xlsx"]).name)
    if result["crosscheck_xlsx"]:
        with open(result["crosscheck_xlsx"], "rb") as f:
            d2.download_button("Tracker crosscheck (.xlsx)", f, file_name=Path(result["crosscheck_xlsx"]).name)
    with open(result["html_path"], "rb") as f:
        d3.download_button("QA map (.html)", f, file_name=Path(result["html_path"]).name)
    if result["config_yaml_path"]:
        with open(result["config_yaml_path"], "rb") as f:
            d4.download_button("Run config (.yaml)", f, file_name=Path(result["config_yaml_path"]).name)

    with st.expander("Extracted features (table)"):
        st.dataframe(result["extracted_df"], use_container_width=True)
