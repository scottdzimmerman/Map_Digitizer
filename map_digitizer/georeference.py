"""
Stage 4 — Georeferencing (pixel -> WGS84).

Fits a least-squares polynomial transform from a set of (pixel, lon/lat)
control points, in the *pixel -> lon/lat* direction (the direction actually
needed to place extracted shapes), validates it by cross-checking predicted
vs. true control-point positions with haversine distance, and can iteratively
drop outlier control points (mislabeled places, inset-map duplicates, etc. —
per the procedure's warnings) and refit.

Supports both an affine (degree=1) and quadratic (degree=2) fit, to capture
gentle projection curvature the way the worked examples did for a printed
graticule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .geo_utils import haversine_km, likely_swapped_lonlat


@dataclass
class GeoreferenceResult:
    pixel_to_lonlat: Callable[[float, float], tuple]  # (px, py) -> (lon, lat)
    lonlat_to_pixel: Callable[[float, float], tuple]  # (lon, lat) -> (px, py)
    degree: int
    control_points_used: list      # the ControlPoints actually kept
    control_points_dropped: list   # ControlPoints dropped as outliers
    residuals_km: list             # residual per kept control point, same order
    mean_residual_km: float
    max_residual_km: float


def _design_matrix(xs: np.ndarray, ys: np.ndarray, degree: int) -> np.ndarray:
    """Build the polynomial design matrix for fitting. degree=1 -> affine
    [x, y, 1]; degree=2 -> quadratic [x, y, x^2, y^2, x*y, 1]."""
    ones = np.ones_like(xs)
    if degree == 1:
        return np.column_stack([xs, ys, ones])
    if degree == 2:
        return np.column_stack([xs, ys, xs ** 2, ys ** 2, xs * ys, ones])
    raise ValueError(f"Unsupported degree: {degree} (use 1 or 2)")


def _fit_direction(src_a: np.ndarray, src_b: np.ndarray, dst: np.ndarray, degree: int) -> np.ndarray:
    """Fit dst ~= f(src_a, src_b) via least squares; returns coefficients."""
    A = _design_matrix(src_a, src_b, degree)
    coef, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return coef


def _apply(coef: np.ndarray, a: float, b: float, degree: int) -> float:
    A = _design_matrix(np.array([a]), np.array([b]), degree)
    return float((A @ coef)[0])


def fit_georeference(control_points: list, degree: int = 1,
                      max_acceptable_residual_km: Optional[float] = None,
                      max_drop_fraction: float = 0.25,
                      region_bbox: Optional[tuple] = None) -> GeoreferenceResult:
    """
    Fit a pixel<->lon/lat transform from a list of config.ControlPoint.

    If max_acceptable_residual_km is set, iteratively drops the single
    worst-residual control point and refits until either the mean residual
    is at/under the target or max_drop_fraction of the original points have
    been dropped (whichever comes first) — mirroring the procedure's
    "drop the outsized-residual point, then refit" step.

    region_bbox, if given, is an expected (min_lon, min_lat, max_lon,
    max_lat) for every control point's real-world location — e.g. roughly
    the country/region the map covers. Any control point whose (lon, lat)
    falls outside this box but whose (lat, lon) would fall inside it is very
    likely a swapped-column typo, and is reported as a loud ValueError up
    front rather than being silently fit into a garbage transform (this is
    the single most common manual-entry bug in this stage — it produces no
    error on its own, just a map that lands nowhere near the real place).
    """
    points = list(control_points)
    dropped = []
    n_original = len(points)
    max_drops = max(0, int(n_original * max_drop_fraction))

    if n_original < 3:
        raise ValueError("Need at least 3 control points to fit a transform")

    if region_bbox is not None:
        suspects = [p for p in points if likely_swapped_lonlat(p.lon, p.lat, region_bbox)]
        if suspects:
            labels = ", ".join(f"'{p.label}' (lon={p.lon}, lat={p.lat})" for p in suspects)
            raise ValueError(
                f"{len(suspects)} control point(s) look like their lon/lat columns are swapped "
                f"relative to region_bbox={region_bbox}: {labels}. Swap lon and lat for these points "
                f"(or pass region_bbox=None to skip this check if the region really is correct)."
            )

    while True:
        pxs = np.array([p.pixel_x for p in points], dtype=float)
        pys = np.array([p.pixel_y for p in points], dtype=float)
        lons = np.array([p.lon for p in points], dtype=float)
        lats = np.array([p.lat for p in points], dtype=float)

        coef_lon = _fit_direction(pxs, pys, lons, degree)
        coef_lat = _fit_direction(pxs, pys, lats, degree)

        residuals = []
        for p in points:
            pred_lon = _apply(coef_lon, p.pixel_x, p.pixel_y, degree)
            pred_lat = _apply(coef_lat, p.pixel_x, p.pixel_y, degree)
            residuals.append(haversine_km(p.lon, p.lat, pred_lon, pred_lat))

        mean_res = float(np.mean(residuals))
        max_res = float(np.max(residuals))

        if (max_acceptable_residual_km is None
                or mean_res <= max_acceptable_residual_km
                or len(dropped) >= max_drops
                or len(points) <= 3):
            break

        worst_idx = int(np.argmax(residuals))
        dropped.append(points.pop(worst_idx))

    # Fit the inverse direction too (lon/lat -> pixel), needed to place
    # QA control points back on the map image, and useful for reprojecting.
    coef_px = _fit_direction(lons, lats, pxs, degree)
    coef_py = _fit_direction(lons, lats, pys, degree)

    def pixel_to_lonlat(px: float, py: float) -> tuple:
        return (_apply(coef_lon, px, py, degree), _apply(coef_lat, px, py, degree))

    def lonlat_to_pixel(lon: float, lat: float) -> tuple:
        return (_apply(coef_px, lon, lat, degree), _apply(coef_py, lon, lat, degree))

    return GeoreferenceResult(
        pixel_to_lonlat=pixel_to_lonlat,
        lonlat_to_pixel=lonlat_to_pixel,
        degree=degree,
        control_points_used=points,
        control_points_dropped=dropped,
        residuals_km=residuals,
        mean_residual_km=mean_res,
        max_residual_km=max_res,
    )


def choose_best_degree(control_points: list, degrees=(1, 2), holdout_fraction: float = 0.25,
                        n_trials: int = 20, seed: int = 0) -> int:
    """
    Cross-validate degree 1 (affine) vs degree 2 (quadratic) by repeated
    random holdout, and return whichever has the lower mean held-out
    residual — the procedure's "try both, keep whichever gives lower
    cross-validated residuals" instruction, made concrete and reproducible.
    """
    rng = np.random.RandomState(seed)
    n = len(control_points)
    n_holdout = max(1, int(n * holdout_fraction))
    if n - n_holdout < 3:
        # Not enough points to hold any out meaningfully; just use degree 1.
        return degrees[0]

    scores = {d: [] for d in degrees}
    idx = np.arange(n)
    for _ in range(n_trials):
        rng.shuffle(idx)
        holdout_idx = set(idx[:n_holdout].tolist())
        train = [cp for i, cp in enumerate(control_points) if i not in holdout_idx]
        test = [cp for i, cp in enumerate(control_points) if i in holdout_idx]
        for d in degrees:
            try:
                fit = fit_georeference(train, degree=d)
            except ValueError:
                continue
            errs = []
            for cp in test:
                pred_lon, pred_lat = fit.pixel_to_lonlat(cp.pixel_x, cp.pixel_y)
                errs.append(haversine_km(cp.lon, cp.lat, pred_lon, pred_lat))
            scores[d].append(np.mean(errs))

    means = {d: (np.mean(v) if v else float("inf")) for d, v in scores.items()}
    return min(means, key=means.get)
