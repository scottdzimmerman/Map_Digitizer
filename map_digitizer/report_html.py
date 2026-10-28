"""
Stage 10 — Self-contained interactive HTML map (Leaflet).

The original map_digitizer package vendors Leaflet's JS/CSS/images locally
and inlines them as base64 data URIs for a fully offline deliverable. This
Streamlit-app build doesn't carry that vendor bundle, so it loads Leaflet
from a CDN (unpkg) instead -- the output HTML is still a single self-
contained file, it just needs an internet connection to render (same as the
live OpenStreetMap tile layer underneath it already requires).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_LEAFLET_VERSION = "1.9.4"
_LEAFLET_CSS_CDN = f"https://unpkg.com/leaflet@{_LEAFLET_VERSION}/dist/leaflet.css"
_LEAFLET_JS_CDN = f"https://unpkg.com/leaflet@{_LEAFLET_VERSION}/dist/leaflet.js"

_CLASS_COLORS = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d", "#666666",
]

_MATCH_STYLE = {
    "contained/very close": {"color": "#1a9850", "label": "Matched (contained/very close)"},
    "name match": {"color": "#66bd63", "label": "Matched (name match)"},
    "nearby": {"color": "#fee08b", "label": "Weak match (nearby)"},
    "no confident match": {"color": "#d73027", "label": "No confident match"},
    None: {"color": "#4575b4", "label": "Unmatched / not crosschecked"},
}


def _class_color(label: str, class_labels: list) -> str:
    idx = class_labels.index(label) if label in class_labels else 0
    return _CLASS_COLORS[idx % len(_CLASS_COLORS)]


def _feature_to_geojson(rec: dict, geometry_field: str, class_field: str, name_field: str,
                         match_field: str, extra_fields: list) -> dict:
    geom = rec[geometry_field]
    from shapely.geometry import mapping
    props = {
        "feature_class": rec.get(class_field),
        "name": rec.get(name_field),
        "match_type": rec.get(match_field),
    }
    for f in extra_fields:
        if f in rec:
            props[f] = rec[f]
    return {"type": "Feature", "geometry": mapping(geom), "properties": props}


def build_leaflet_map(records: list, out_path: str, title: str = "Map digitization QA",
                       geometry_field: str = "geometry", class_field: str = "feature_class",
                       name_field: str = "candidate_name", match_field: str = "match_type",
                       extra_fields: list = None, center: tuple = None, zoom: int = 6) -> str:
    """
    records: list of dicts, each with a shapely geometry under geometry_field
    plus feature_class / candidate name / optional match_type (from
    crosscheck.py — omit or leave None for a pure extraction QA map with no
    crosscheck yet).

    Layers are split by feature_class, each independently toggleable, and
    features are styled by match_type when present (green/yellow/red) or by
    class color otherwise.
    """
    extra_fields = extra_fields or []
    class_labels = sorted({r.get(class_field) for r in records if r.get(class_field)})

    geojson_by_class = {}
    for rec in records:
        cls = rec.get(class_field, "unclassified")
        geojson_by_class.setdefault(cls, []).append(
            _feature_to_geojson(rec, geometry_field, class_field, name_field, match_field, extra_fields)
        )

    if center is None:
        # Rough centroid of all features for the initial view.
        lats, lons = [], []
        for rec in records:
            c = rec[geometry_field].centroid
            lons.append(c.x)
            lats.append(c.y)
        center = (sum(lats) / len(lats), sum(lons) / len(lons)) if lats else (0, 0)

    layers_js = []
    controls_js = []
    for cls in geojson_by_class:
        fc = {"type": "FeatureCollection", "features": geojson_by_class[cls]}
        var_name = "layer_" + re.sub(r"\W+", "_", cls)
        color = _class_color(cls, class_labels)
        layers_js.append(f"""
        var {var_name} = L.geoJSON({json.dumps(fc)}, {{
            style: function(feature) {{
                var mt = feature.properties.match_type;
                var styleInfo = MATCH_STYLE[mt] || MATCH_STYLE["__default__"];
                return {{color: styleInfo.color, weight: 2, fillColor: styleInfo.color, fillOpacity: 0.35}};
            }},
            pointToLayer: function(feature, latlng) {{
                var mt = feature.properties.match_type;
                var styleInfo = MATCH_STYLE[mt] || MATCH_STYLE["__default__"];
                return L.circleMarker(latlng, {{radius: 6, color: styleInfo.color, fillColor: styleInfo.color, fillOpacity: 0.8}});
            }},
            onEachFeature: function(feature, layer) {{
                var p = feature.properties;
                var html = "<b>" + (p.name || "(unnamed)") + "</b><br>Class: " + (p.feature_class || "") +
                    (p.match_type ? "<br>Match: " + p.match_type : "");
                {"".join(f'html += "<br>{f}: " + (p["{f}"] != null ? p["{f}"] : "");' for f in extra_fields)}
                layer.bindPopup(html);
            }}
        }}).addTo(map);
        """)
        controls_js.append(f'"{cls}": {var_name}')

    match_style_js = "var MATCH_STYLE = " + json.dumps({
        (k if k is not None else "__default__"): v for k, v in _MATCH_STYLE.items()
    }) + ";"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="{_LEAFLET_CSS_CDN}" />
<style>
html, body, #map {{ height: 100%; margin: 0; }}
.legend {{ background: white; padding: 8px 10px; font: 13px sans-serif; line-height: 1.5; }}
.legend div span {{ display:inline-block; width:12px; height:12px; margin-right:6px; border-radius:2px; }}
</style>
</head>
<body>
<div id="map"></div>
<script src="{_LEAFLET_JS_CDN}"></script>
<script>
{match_style_js}
var map = L.map('map').setView([{center[0]}, {center[1]}], {zoom});
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 18
}}).addTo(map);

// Click-to-mark: drop a visible dot wherever the map is clicked, showing the
// exact lat/lon under the cursor -- handy for sanity-checking a feature's
// real-world position or reading off a coordinate for a new control point.
var _clickMarker = null;
map.on('click', function(e) {{
    if (_clickMarker) {{ map.removeLayer(_clickMarker); }}
    _clickMarker = L.circleMarker(e.latlng, {{
        radius: 7, color: '#111111', weight: 2, fillColor: '#ffeb3b', fillOpacity: 0.95
    }}).addTo(map);
    _clickMarker.bindPopup(
        'Lat: ' + e.latlng.lat.toFixed(5) + '<br>Lon: ' + e.latlng.lng.toFixed(5)
    ).openPopup();
}});

{"".join(layers_js)}

var overlays = {{ {", ".join(controls_js)} }};
L.control.layers(null, overlays, {{collapsed: false}}).addTo(map);

var legend = L.control({{position: 'bottomright'}});
legend.onAdd = function() {{
    var div = L.DomUtil.create('div', 'legend');
    var rows = "";
    for (var key in MATCH_STYLE) {{
        rows += "<div><span style='background:" + MATCH_STYLE[key].color + "'></span>" + MATCH_STYLE[key].label + "</div>";
    }}
    div.innerHTML = "<b>Match status</b>" + rows;
    return div;
}};
legend.addTo(map);
</script>
</body>
</html>
"""
    Path(out_path).write_text(html)
    return out_path
