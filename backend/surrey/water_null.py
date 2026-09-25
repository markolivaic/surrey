"""A physical null for water: below the measured sea level plus a fixed margin.

M0prime has no water output, so "water called water" against M0prime is zero by
construction and says nothing. M1's water has to beat something that knows where the
sea surface was when the aircraft flew. Written before M1 or M0prime was trained and
before any lidar height was compared with any gauge reading.

Gauge
    NOAA CO-OPS station 9414290, San Francisco, CA (37.806305, -122.46589), the
    verified six minute water level, product water_level, datum NAVD, metric, GMT.
    Public domain; NOS asks for attribution, given in THIRD_PARTY_NOTICES.md.

Time
    Each point's own GPS time, read from the EPT (dimension GpsTime, adjusted standard
    GPS time, seconds since the GPS epoch minus 1e9) and matched to the cached point by
    position. The gauge level is interpolated linearly to that time. Flight lines over
    one area can be hours apart, so one level per area is not used.

Heights
    The source files declare NAVD88 height, Geoid18, in metres (EPSG:5703). The EPT is
    published in EPSG:3857 by a horizontal reprojection with no vertical step, checked
    in scripts/pipeline/water_null.py by comparing each source file's z range with its EPT
    bounds. The gauge reading is requested in NAVD88. Two realisations of the same
    datum remain: the lidar reaches NAVD88 through GEOID18 from ellipsoid heights, the
    gauge through levelling to bench marks. Their difference is not measured here.

Null
    A clean point (not noise 7 or 18) is water when its height is below the level at its
    time plus MARGIN_M.

MARGIN_M = 0.5, fixed before looking, with its reasons:
    - 3DEP accuracy: quality levels 1 and 2 both allow 10 cm RMSEz in non-vegetated
      terrain (USGS Lidar Base Specification), about 0.2 m at 95 percent (1.96 times
      RMSEz), and water returns are no better than open ground.
    - the gauge is at the Golden Gate and the areas are inside the bay, several
      kilometres east: an allowance for the tide being higher or later there, not
      measured.
    - wind chop on the bay surface: an allowance, not measured.
    - half a metre is already the band this project calls "at the surface" for hard
      water errors (score_variant.HARD_WITHIN_M), so the null and the error analysis
      use the same idea of the surface.
    The allowances are guesses and are written as guesses. The margin is not tuned
    after the fact; a sensitivity at 0.25 and 1.0 m is reported beside it and never
    replaces it.

Scoring, per area (Embarcadero and Bay Bridge only, the areas with delivery water):
reference is delivery class 9. For the null and for each model: precision, recall and
IoU of the water call. Lift over the null is (IoU_model - IoU_null) / (1 - IoU_null),
the share of the room above the null the model takes, the same shape as the building
lift. IoU is the one figure that neither calling few points (precision) nor calling
everything low (recall) can game.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

STATION_ID = "9414290"
STATION_NAME = "San Francisco, CA"
STATION_LATLON = (37.806305, -122.46589)
PRODUCT = "water_level"
DATUM = "NAVD"
MARGIN_M = 0.5
SENSITIVITY_MARGINS_M = (0.25, 1.0)
AREAS = ("embarcadero", "baybridge")
ASPRS_WATER = 9
ASPRS_NOISE = (7, 18)

GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.UTC)
# GPS time runs ahead of UTC by the leap seconds since 1980; 18 since 2017-01-01 and
# unchanged through 2023.
GPS_MINUS_UTC_S = 18


def gps_adjusted_to_utc(t: np.ndarray) -> np.ndarray:
    """Adjusted standard GPS seconds to POSIX UTC seconds."""
    return np.asarray(t, np.float64) + 1e9 - GPS_MINUS_UTC_S + GPS_EPOCH.timestamp()


def level_at(times_utc: np.ndarray, gauge_utc: np.ndarray, gauge_level: np.ndarray) -> np.ndarray:
    """The gauge level linearly interpolated to each time. A time outside the record is
    an error, never an extrapolation."""
    times_utc = np.asarray(times_utc, np.float64)
    order = np.argsort(gauge_utc)
    gt, gl = np.asarray(gauge_utc, np.float64)[order], np.asarray(gauge_level, np.float64)[order]
    if times_utc.size and (times_utc.min() < gt[0] or times_utc.max() > gt[-1]):
        raise ValueError("point times fall outside the gauge record")
    return np.interp(times_utc, gt, gl)


def null_water(z: np.ndarray, level: np.ndarray, margin_m: float = MARGIN_M) -> np.ndarray:
    return np.asarray(z) < np.asarray(level) + margin_m


def water_scores(called: np.ndarray, truth: np.ndarray, population: np.ndarray) -> dict:
    """Precision, recall and IoU of a water call over a population of points."""
    c, t = called & population, truth & population
    tp = int((c & t).sum())
    fp = int((c & ~t).sum())
    fn = int((~c & t).sum())
    return {
        "called": int(c.sum()),
        "water_points": int(t.sum()),
        "true_positive": tp,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else None,
    }


def as_null(scores: dict) -> dict:
    """The same scores named as the null's own, so no precision stands anywhere without the
    null it is read against (tests/integrity/test_artifacts.py)."""
    return {
        (f"null_{k}" if k in ("precision", "recall", "iou") else k): v for k, v in scores.items()
    }


def lift_over_null(model_iou: float | None, null_iou: float | None) -> float | None:
    if model_iou is None or null_iou is None or null_iou >= 1.0:
        return None
    return round((model_iou - null_iou) / (1.0 - null_iou), 4)
