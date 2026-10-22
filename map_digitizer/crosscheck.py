"""
Stage 7 — Crosschecking extracted features against a tracker database.

Generic over the tracker's schema (column names come from config, not
hardcoded to GOGET) and over feature type (area/point/line — line matching
uses endpoint/sampled-point distance instead of centroid distance, per
Stage 9's guidance).

rapidfuzz is a classic string-similarity library (no LLM involved).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz
from shapely.geometry import Point

from .config import CrosscheckConfig, FeatureType
from .geo_utils import haversine_km


MATCH_CONTAINED = "contained/very close"
MATCH_NAME = "name match"
MATCH_NEARBY = "nearby"
MATCH_NONE = "no confident match"


def normalize_name(name: Optional[str], strip_words: list) -> str:
    """Lowercase, strip trailing '(location, country)' parentheticals and
    configured generic words, collapse whitespace."""
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"\([^)]*\)\s*$", "", name).strip()  # trailing parenthetical
    for w in strip_words:
        name = re.sub(rf"\b{re.escape(w.lower())}\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def name_score(a: Optional[str], b: Optional[str], strip_words: list) -> float:
    na, nb = normalize_name(a, strip_words), normalize_name(b, strip_words)
    if not na or not nb:
        return 0.0
    return float(fuzz.token_sort_ratio(na, nb))


@dataclass
class MatchResult:
    extracted_index: int
    match_type: str
    closest_tracker_id: Optional[str]
    closest_tracker_name: Optional[str]
    closest_distance_km: Optional[float]
    best_name_match_id: Optional[str]
    best_name_match_name: Optional[str]
    best_name_score: Optional[float]


def _centroid_lonlat(geom) -> tuple:
    c = geom.centroid
    return (c.x, c.y)


def load_tracker_csv(path: str, columns) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [columns.id, columns.name, columns.lat, columns.lon]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Tracker CSV is missing expected columns: {missing}")
    return df


def crosscheck_features(extracted: list, tracker_df: pd.DataFrame, config: CrosscheckConfig,
                         name_field: str = "candidate_name", geometry_field: str = "geometry") -> list:
    """
    extracted: list of dicts, each with at least `geometry_field` (a shapely
    geometry) and `name_field` (OCR'd/candidate name, may be None/garbled).

    For line features, pass geometries whose centroid is a reasonable stand-
    in only if you don't have per-line tracker geometry; for endpoint-based
    matching on sparse tracker data, see `match_line_by_endpoints` instead.

    Returns a list of MatchResult, one per extracted feature (same order,
    same length).
    """
    cols = config.columns
    tracker_lats = tracker_df[cols.lat].to_numpy(dtype=float)
    tracker_lons = tracker_df[cols.lon].to_numpy(dtype=float)
    tracker_names = tracker_df[cols.name].tolist()
    tracker_ids = tracker_df[cols.id].tolist()

    results = []
    for i, rec in enumerate(extracted):
        geom = rec[geometry_field]
        lon, lat = _centroid_lonlat(geom)
        cand_name = rec.get(name_field)

        distances = [
            haversine_km(lon, lat, t_lon, t_lat) if not (pd.isna(t_lat) or pd.isna(t_lon)) else float("inf")
            for t_lon, t_lat in zip(tracker_lons, tracker_lats)
        ]
        closest_idx = min(range(len(distances)), key=lambda j: distances[j]) if distances else None
        closest_dist = distances[closest_idx] if closest_idx is not None else None

        # Name matching only against tracker records within the search cap.
        best_name_idx, best_score = None, 0.0
        for j, (t_name, dist) in enumerate(zip(tracker_names, distances)):
            if dist > config.name_match_search_cap_km:
                continue
            score = name_score(cand_name, t_name, config.strip_generic_words)
            if score > best_score:
                best_score, best_name_idx = score, j

        if closest_dist is not None and closest_dist <= config.contained_distance_km:
            match_type = MATCH_CONTAINED
        elif best_name_idx is not None and best_score >= config.name_score_threshold:
            match_type = MATCH_NAME
        elif closest_dist is not None and closest_dist <= config.nearby_distance_km:
            match_type = MATCH_NEARBY
        else:
            match_type = MATCH_NONE

        results.append(MatchResult(
            extracted_index=i,
            match_type=match_type,
            closest_tracker_id=str(tracker_ids[closest_idx]) if closest_idx is not None else None,
            closest_tracker_name=tracker_names[closest_idx] if closest_idx is not None else None,
            closest_distance_km=closest_dist,
            best_name_match_id=str(tracker_ids[best_name_idx]) if best_name_idx is not None else None,
            best_name_match_name=tracker_names[best_name_idx] if best_name_idx is not None else None,
            best_name_score=best_score if best_name_idx is not None else None,
        ))

    return results


def find_missing_tracker_geometry(tracker_df: pd.DataFrame, config: CrosscheckConfig,
                                   extracted: list, matches: list,
                                   name_field: str = "candidate_name", geometry_field: str = "geometry") -> pd.DataFrame:
    """
    The reverse direction: for every tracker record with no existing geometry
    (config.columns.existing_wkt column empty/absent), find whichever
    extracted feature matched it best (by distance and by name) and suggest
    that as a candidate outline, with confidence noted.

    Returns a DataFrame — one row per tracker record lacking geometry.
    """
    cols = config.columns
    has_wkt_col = cols.existing_wkt and cols.existing_wkt in tracker_df.columns

    # Build reverse index: tracker_id -> list of (extracted_index, distance, name_score)
    reverse: dict = {}
    for m in matches:
        if m.closest_tracker_id is not None:
            reverse.setdefault(m.closest_tracker_id, []).append(("distance", m.extracted_index, m.closest_distance_km))
        if m.best_name_match_id is not None:
            reverse.setdefault(m.best_name_match_id, []).append(("name", m.extracted_index, m.best_name_score))

    rows = []
    for _, trow in tracker_df.iterrows():
        tid = str(trow[cols.id])
        already_has_geom = has_wkt_col and pd.notna(trow.get(cols.existing_wkt)) and str(trow.get(cols.existing_wkt)).strip()
        if already_has_geom:
            continue

        candidates = reverse.get(tid, [])
        best_by_distance = min((c for c in candidates if c[0] == "distance"), key=lambda c: c[2], default=None)
        best_by_name = max((c for c in candidates if c[0] == "name"), key=lambda c: c[2], default=None)

        suggested_idx, confidence = None, "Low / none"
        if best_by_distance and best_by_distance[2] <= config.contained_distance_km:
            suggested_idx, confidence = best_by_distance[1], "High (spatially very close)"
        elif best_by_name and best_by_name[2] >= config.name_score_threshold:
            suggested_idx, confidence = best_by_name[1], "Higher (name match)"
        elif best_by_distance and best_by_distance[2] <= config.nearby_distance_km:
            suggested_idx, confidence = best_by_distance[1], "Higher (spatially close)"

        suggested_wkt = extracted[suggested_idx][geometry_field].wkt if suggested_idx is not None else None

        rows.append({
            cols.name: trow[cols.name],
            cols.lat: trow[cols.lat],
            cols.lon: trow[cols.lon],
            "closest_extracted_index": best_by_distance[1] if best_by_distance else None,
            "closest_extracted_dist_km": best_by_distance[2] if best_by_distance else None,
            "best_name_extracted_index": best_by_name[1] if best_by_name else None,
            "best_name_score": best_by_name[2] if best_by_name else None,
            "confidence": confidence,
            "suggested_outline_wkt": suggested_wkt,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Line-feature-specific matching (Stage 9).
#
# A line has no single centroid worth matching on, so crosscheck_features()
# (centroid distance) is a poor fit for pipelines/roads/transmission lines.
# Two alternatives, in order of how much the tracker database gives you:
# ---------------------------------------------------------------------------

def sampled_point_distance(line_a, line_b, n_samples: int = 20) -> float:
    """
    Average nearest-point distance (km) between two LineStrings, sampled at
    n_samples regular intervals along line_a. Use when the tracker DOES carry
    reference-line geometry for the feature (e.g. an existing pipeline route)
    and you want a full-line similarity measure rather than just endpoints.
    """
    if line_a.length == 0:
        pts = [line_a.coords[0]]
    else:
        pts = [line_a.interpolate(i / n_samples, normalized=True) for i in range(n_samples + 1)]
        pts = [(p.x, p.y) for p in pts]

    dists = []
    for lon, lat in pts:
        nearest = line_b.interpolate(line_b.project(Point(lon, lat)))
        dists.append(haversine_km(lon, lat, nearest.x, nearest.y))
    return sum(dists) / len(dists)


def match_line_by_endpoints(line_geom, tracker_endpoint_a: tuple, tracker_endpoint_b: tuple) -> float:
    """
    Distance (km) between an extracted line's two endpoints and a tracker
    record's two named endpoint facilities (each an (lon, lat) tuple),
    trying both pairings and keeping the smaller total. This is the
    procedure's recommended fallback "often more robust than any full-line
    distance metric when your database is sparse on middle-of-line geometry"
    — e.g. a tracker that only records "Hassi R'Mel -> Arzew" as two
    facility points, not a route geometry.

    Returns the average of the two endpoint distances under the best pairing.
    """
    coords = list(line_geom.coords)
    line_start, line_end = coords[0], coords[-1]

    def _pair_cost(p1, t1, p2, t2):
        return (haversine_km(p1[0], p1[1], t1[0], t1[1]) + haversine_km(p2[0], p2[1], t2[0], t2[1])) / 2

    direct = _pair_cost(line_start, tracker_endpoint_a, line_end, tracker_endpoint_b)
    swapped = _pair_cost(line_start, tracker_endpoint_b, line_end, tracker_endpoint_a)
    return min(direct, swapped)


def crosscheck_lines_by_endpoints(extracted: list, tracker_df: pd.DataFrame, config: CrosscheckConfig,
                                   endpoint_a_lat_col: str, endpoint_a_lon_col: str,
                                   endpoint_b_lat_col: str, endpoint_b_lon_col: str,
                                   name_field: str = "candidate_name", geometry_field: str = "geometry") -> list:
    """
    Line-feature counterpart to crosscheck_features(): for each extracted
    LineString, find the tracker record whose two named endpoint facilities
    are closest to the line's two endpoints, combined with the usual name
    fuzzy-match. Tracker records missing either endpoint coordinate are
    skipped for distance purposes (name matching still applies).
    """
    cols = config.columns
    tracker_names = tracker_df[cols.name].tolist()
    tracker_ids = tracker_df[cols.id].tolist()

    results = []
    for i, rec in enumerate(extracted):
        geom = rec[geometry_field]
        cand_name = rec.get(name_field)

        distances = []
        for _, trow in tracker_df.iterrows():
            a = (trow.get(endpoint_a_lon_col), trow.get(endpoint_a_lat_col))
            b = (trow.get(endpoint_b_lon_col), trow.get(endpoint_b_lat_col))
            if any(pd.isna(v) for v in (*a, *b)):
                distances.append(float("inf"))
                continue
            distances.append(match_line_by_endpoints(geom, a, b))

        closest_idx = min(range(len(distances)), key=lambda j: distances[j]) if distances else None
        closest_dist = distances[closest_idx] if closest_idx is not None else None

        best_name_idx, best_score = None, 0.0
        for j, (t_name, dist) in enumerate(zip(tracker_names, distances)):
            if dist > config.name_match_search_cap_km:
                continue
            score = name_score(cand_name, t_name, config.strip_generic_words)
            if score > best_score:
                best_score, best_name_idx = score, j

        if closest_dist is not None and closest_dist <= config.contained_distance_km:
            match_type = MATCH_CONTAINED
        elif best_name_idx is not None and best_score >= config.name_score_threshold:
            match_type = MATCH_NAME
        elif closest_dist is not None and closest_dist <= config.nearby_distance_km:
            match_type = MATCH_NEARBY
        else:
            match_type = MATCH_NONE

        results.append(MatchResult(
            extracted_index=i,
            match_type=match_type,
            closest_tracker_id=str(tracker_ids[closest_idx]) if closest_idx is not None and closest_dist != float("inf") else None,
            closest_tracker_name=tracker_names[closest_idx] if closest_idx is not None and closest_dist != float("inf") else None,
            closest_distance_km=closest_dist if closest_dist != float("inf") else None,
            best_name_match_id=str(tracker_ids[best_name_idx]) if best_name_idx is not None else None,
            best_name_match_name=tracker_names[best_name_idx] if best_name_idx is not None else None,
            best_name_score=best_score if best_name_idx is not None else None,
        ))

    return results
