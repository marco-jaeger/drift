"""Outline-based waterline metrics and outline extraction.

Adapted from Ethan Welty's shape.py prototype (glimpse). The apparent
waterline width of an iceberg is the distance between the two x-extrema of
its outline in roll-corrected normalized camera coordinates, projected onto
the sea-surface plane; the tracer point is the midpoint of those endpoints.

Two stages:
    extract-outlines command (Stage A, once per tracking run):
        Segment every MOT detection into an outline polygon with SAM or Otsu
        thresholding and store them packed in outlines.npz, keyed
        "{frame}_{id}", each with a quality record and a trusted flag.
    Library (Stage B, used by georeference and circulation):
        per_detection_waterlines computes widths and tracer centers aligned
        with a tracks dict; aggregate_by_iceberg summarises widths per track.

Every detection gets an outline (bbox rectangle as last resort), so velocity
tracers have full coverage; only trusted outlines feed the size split.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from tqdm import tqdm

from helpers import (
    MODELS_DIR, build_frame_to_image, get_glimpse, get_sequences,
    load_mot_tracks, log_config, log_section, resolve_device,
    save_config_snapshot,
)

logger = logging.getLogger(__name__)

_GLIMPSE = None


def _glimpse():
    """Lazily imported, module-cached glimpse (see helpers.get_glimpse)."""
    global _GLIMPSE
    if _GLIMPSE is None:
        _GLIMPSE = get_glimpse()
    return _GLIMPSE


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OutlineExtractionConfig:
    """Parameters for the extract-outlines command."""

    dataset: str = "???"
    run_name: Optional[str] = None     # Read tracking/<run_name>/track.txt

    method: str = "sam"                # 'sam' or 'otsu'
    sam_model: str = "mobile_sam"      # 'mobile_sam' or 'sam_vit_b'

    pad_px: int = 8                    # Crop margin around each box
    approx_epsilon_px: float = 0.5     # Douglas-Peucker simplification; keep
                                       # small so waterline extrema are not
                                       # clipped (1 px here is ~0.2-0.35 m)
    device: Optional[str] = None       # None = auto (cuda > mps > cpu)


# SAM variants: name -> (registry type, checkpoint filename, download URL).
# Checkpoints live in MODELS_DIR and are downloaded on first use.
SAM_MODELS = {
    "mobile_sam": ("vit_t", "mobile_sam.pt",
                   "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/"
                   "master/weights/mobile_sam.pt"),
    "sam_vit_b": ("vit_b", "sam_vit_b_01ec64.pth",
                  "https://dl.fbaipublicfiles.com/segment_anything/"
                  "sam_vit_b_01ec64.pth"),
}


@dataclass
class _QualityGate:
    """Fixed acceptance gate for a candidate outline (crop coordinates).

    An outline fails the gate (emitted but not trusted) when the segmentation
    clearly failed: implausible area relative to the detection box, leaking to
    the crop border, too few vertices, or -- for Otsu -- no genuine
    two-population intensity split (illumination-invariant separability).
    """
    min_area_frac: float = 0.08
    max_area_frac: float = 3.00    # Tight detector boxes make correct outlines
                                   # routinely exceed the box; >3x is over-grab
    max_border_frac: float = 0.25
    border_tol_px: float = 2.0
    min_vertices: int = 6
    min_separability: float = 0.65  # Otsu eta gate; SAM skips it


# ============================================================================
# PROJECTION (shared with georeference.py)
# ============================================================================

def project_to_sea(camera, uv, z_sea):
    """Project image points onto the horizontal sea plane at height z_sea.

    The glimpse camera model supplies world-frame ray directions from the
    surveyed camera position; each ray is intersected with the plane. z_sea
    may be a scalar or a per-point array.

    Returns:
        (N, 2) UTM easting/northing [m]; NaN where the ray misses the plane.
    """
    glimpse = _glimpse()
    uv = np.atleast_2d(np.asarray(uv, dtype=float))
    z_sea = np.broadcast_to(np.asarray(z_sea, dtype=float), (len(uv),))

    directions = camera.uv_to_xyz(uv=uv, directions=True)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.repeat(np.atleast_2d(camera.xyz), len(uv), axis=0)
    rays = np.column_stack((origins, directions))

    world = np.full((len(uv), 3), np.nan)
    for z in np.unique(z_sea):
        mask = z_sea == z
        plane = [0, 0, float(z), 1, 0, 0, 0, 1, 0]
        world[mask] = glimpse.helpers.intersect_rays_plane(rays=rays[mask],
                                                           plane=plane)
    return world[:, :2]


# ============================================================================
# WATERLINE GEOMETRY (Ethan's algorithm)
# ============================================================================

@dataclass
class WaterlineResult:
    """Waterline metrics for one detection."""
    width_m: float                 # NaN if missing outline or degenerate
    center_xy: np.ndarray          # (2,) UTM easting/northing, NaN if invalid
    flagged_vertical: bool         # Near-vertical left/right boundary
    endpoints_uv: Optional[np.ndarray] = None   # (2, 2) image coords, debug
    waterline_uv: Optional[np.ndarray] = None   # (K, 2) waterline arc, image
    waterline_xy: Optional[np.ndarray] = None   # (K, 2) waterline arc, UTM
    outline_xy: Optional[np.ndarray] = None     # (M, 2) full outline, UTM


def _roll_matrix(roll_deg):
    """2x2 rotation matrix correcting for camera roll."""
    theta = np.deg2rad(roll_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])


def _open_polygon(uv):
    """Return the polygon without a repeated closing vertex."""
    uv = np.asarray(uv, dtype=float)
    if len(uv) > 1 and np.allclose(uv[0], uv[-1]):
        uv = uv[:-1]
    return uv


def _walk_arc(nxy, index, direction, min_arc):
    """Walk the closed polygon from `index` in `direction` (+1/-1) until the
    cumulative arc length reaches `min_arc`; return the end vertex. Robust to
    sparse simplified polygons where a vertex-count window would span wildly
    varying arc lengths."""
    n = len(nxy)
    pos, travelled = index, 0.0
    for _ in range(n - 1):
        nxt = (pos + direction) % n
        travelled += float(np.linalg.norm(nxy[nxt] - nxy[pos]))
        pos = nxt
        if travelled >= min_arc:
            break
    return nxy[pos]


def _endpoint_is_near_vertical(nxy, index, threshold_deg, arc_fraction=0.15):
    """True if the outline adjacent to vertex `index` is near-vertical.

    Walks an arc of `arc_fraction` of the outline's larger bounding dimension
    in both directions and takes the chord back to the endpoint; a chord
    within `threshold_deg` of vertical means the x-extremum is poorly
    constrained (Ethan's flagged edge case).
    """
    n = len(nxy)
    if n < 3:
        return True
    extent = nxy.max(axis=0) - nxy.min(axis=0)
    min_arc = arc_fraction * float(extent.max())
    for direction in (+1, -1):
        other = _walk_arc(nxy, index, direction, min_arc)
        d = other - nxy[index]
        if np.allclose(d, 0):
            continue
        angle_from_vertical = np.degrees(np.arctan2(abs(d[0]), abs(d[1])))
        if angle_from_vertical < threshold_deg:
            return True
    return False


def _waterline_arc_indices(nxy, index_min):
    """Vertex indices tracing the waterline: the arc between the two x-extrema
    with the larger mean y (lowest in the image, origin upper-left)."""
    order = np.roll(np.arange(len(nxy)), -index_min)
    nxy_r = nxy[order]
    index_max_r = int(np.argmax(nxy_r[:, 0]))
    a, b = slice(None, index_max_r + 1), slice(index_max_r, None)
    is_waterline = a if nxy_r[a, 1].mean() > nxy_r[b, 1].mean() else b
    return order[is_waterline]


def waterline_from_outline(camera, uv, z_sea, vertical_threshold_deg=10.0,
                           densify=True, project_shape=False,
                           keep_debug=False):
    """Exact waterline metric for a single iceberg outline (reference path).

    `uv` is the outline polygon in image coordinates (open or closed);
    `z_sea` the sea-surface height in the camera vertical datum.

    `densify=True` interpolates the polygon to ~1 px vertex spacing before the
    x-extrema are picked, so the endpoint lies on the boundary rather than the
    nearest stored vertex. `project_shape=True` additionally projects the full
    outline and waterline arc onto the sea plane (map-view figures).

    The batch path (_waterlines_fast) is used by the pipeline; this function
    is the validated reference and the map-view/debug tool.
    """
    nan = WaterlineResult(np.nan, np.full(2, np.nan), flagged_vertical=False)
    uv = _open_polygon(uv)
    if len(uv) < 3:
        return nan

    if densify:
        closed = np.vstack([uv, uv[:1]])
        uv = _open_polygon(_glimpse().helpers.interpolate_line(vertices=closed,
                                                               dx=1))

    # Normalized (distortion-corrected) camera coords, roll-corrected so the
    # x-axis is parallel to the sea-surface plane.
    nxy = camera._uv_to_xy(uv) @ _roll_matrix(camera.viewdir[2]).T

    index_min = int(np.argmin(nxy[:, 0]))
    index_max = int(np.argmax(nxy[:, 0]))
    if index_min == index_max:
        return nan

    # Near-vertical outline at an endpoint makes the x-extremum less stable.
    # FLAGGED (annotation), not rejected: the width is still usable and the
    # per-iceberg median absorbs unstable single frames.
    flagged = (
        _endpoint_is_near_vertical(nxy, index_min, vertical_threshold_deg)
        or _endpoint_is_near_vertical(nxy, index_max, vertical_threshold_deg)
    )

    endpoints_uv = uv[[index_min, index_max]]
    xy = project_to_sea(camera, endpoints_uv, z_sea)
    if not np.all(np.isfinite(xy)):
        return nan

    width = float(np.linalg.norm(xy[1] - xy[0]))
    center = xy.mean(axis=0)
    result = WaterlineResult(width, center, flagged_vertical=bool(flagged))

    if project_shape or keep_debug:
        arc = _waterline_arc_indices(nxy, index_min)
        result.endpoints_uv = endpoints_uv
        result.waterline_uv = uv[arc]
    if project_shape:
        result.outline_xy = project_to_sea(camera, uv, z_sea)
        result.waterline_xy = project_to_sea(camera, uv[arc], z_sea)
    return result


# ============================================================================
# OUTLINE STORE
# ============================================================================

# Columns of the per-outline quality table stored in the .npz.
QUALITY_COLUMNS = ("eta", "contrast", "area_frac", "border_frac",
                   "n_vertices", "box_area_px", "method_code", "trusted")
# Codes are part of the on-disk format; 'grabcut' is retained for reading
# existing files even though the backend was removed.
METHOD_CODES = {"classical": 0, "grabcut": 1, "sam": 2, "bbox_fallback": 3}
METHOD_NAMES = {v: k for k, v in METHOD_CODES.items()}


def outline_key(frame, iceberg_id):
    """Key format of outlines.npz: raw ints, '123_45'.

    Deliberately DIFFERENT from helpers.emb_key (zero-padded '000123_45'):
    existing outline files use this format, so do not unify them.
    """
    return f"{int(frame)}_{int(iceberg_id)}"


def load_outlines(path):
    """Load outline polygons from an .npz written by extract-outlines.

    Supports both layouts transparently: PACKED (all polygons in one
    __packed_xy__ array plus index arrays -- loads in seconds at any scale)
    and legacy PER-KEY (one array per outline; slow for large runs).
    """
    with np.load(path, allow_pickle=True) as data:
        if "__packed_xy__" in set(data.files):
            xy = data["__packed_xy__"]
            keys = [str(k) for k in data["__packed_keys__"]]
            split = data["__packed_split__"]  # cumulative vertex counts
            pieces = np.split(xy, split[:-1]) if len(split) > 1 else [xy]
            outlines = {k: p for k, p in zip(keys, pieces)}
        else:
            outlines = {k: data[k] for k in data.files if not k.startswith("__")}
    logger.info(f"Loaded {len(outlines)} outlines from {path}")
    return outlines


def load_outline_quality(path):
    """Load the per-outline quality table: {'frame_id': {column: value}} with
    a boolean 'trusted' flag. Empty dict if the file has no table."""
    with np.load(path, allow_pickle=True) as data:
        if "__quality_keys__" not in data.files:
            return {}
        keys = [str(k) for k in data["__quality_keys__"]]
        cols = [str(c) for c in data["__quality_cols__"]]
        table = data["__quality__"]
        out = {}
        for i, key in enumerate(keys):
            row = {c: float(table[i, j]) for j, c in enumerate(cols)}
            row["trusted"] = bool(row.get("trusted", 0.0))
            out[key] = row
    logger.info(f"Loaded quality records for {len(out)} outlines from {path}")
    return out


# ============================================================================
# PER-DETECTION WATERLINES (Stage B)
# ============================================================================

@dataclass
class WaterlineArrays:
    """Per-detection waterline metrics aligned with a tracks dict."""
    widths: np.ndarray      # (N,) metres, NaN if no outline / degenerate
    centers: np.ndarray     # (N, 2) UTM, NaN if invalid
    flagged: np.ndarray     # (N,) bool, near-vertical endpoint (annotation)
    missing: np.ndarray     # (N,) bool, no outline available
    trusted: np.ndarray     # (N,) bool, passed the extraction quality gate;
                            # the size split uses only trusted widths


def _near_vertical_at(nxy, a, b, i, threshold_deg):
    """Cheap near-vertical test at vertex i (in [a, b)) using its immediate
    neighbours in the corrected outline (no arc walk)."""
    if threshold_deg <= 0:
        return False
    prev_i = i - 1 if i - 1 >= a else b - 1
    next_i = i + 1 if i + 1 < b else a
    for nb in (prev_i, next_i):
        d = nxy[nb] - nxy[i]
        ang = np.degrees(np.arctan2(abs(d[0]), abs(d[1]) + 1e-12))
        if ang <= threshold_deg:
            return True
    return False


def _waterlines_fast(tracks, camera, outlines, z_sea, vertical_threshold_deg):
    """Vectorised waterline widths and centres for all detections at once.

    Equivalent to waterline_from_outline per detection, but skips densify
    (moves endpoints by <~0.05 m on the ground; verified against the exact
    path) and batches _uv_to_xy over all vertices and project_to_sea over all
    endpoints -- ~50-100x faster at 10^5-10^6 detections.
    """
    n = len(tracks["frame"])
    widths = np.full(n, np.nan)
    centers = np.full((n, 2), np.nan)
    flagged = np.zeros(n, dtype=bool)

    idx_present, polys = [], []
    for k in range(n):
        uv = outlines.get(outline_key(tracks["frame"][k], tracks["id"][k]))
        if uv is None or len(uv) < 3:
            continue
        idx_present.append(k)
        polys.append(np.asarray(uv, dtype=float))
    if not polys:
        return widths, centers, flagged

    counts = np.array([len(p) for p in polys])
    offsets = np.concatenate([[0], np.cumsum(counts)])
    flat_uv = np.concatenate(polys, axis=0)

    nxy = camera._uv_to_xy(flat_uv) @ _roll_matrix(camera.viewdir[2]).T
    x = nxy[:, 0]

    ep_uv = np.empty((2 * len(polys), 2))
    valid = np.ones(len(polys), dtype=bool)
    for j in range(len(polys)):
        a, b = int(offsets[j]), int(offsets[j + 1])
        xs = x[a:b]
        i_min = a + int(np.argmin(xs))
        i_max = a + int(np.argmax(xs))
        if i_min == i_max:
            valid[j] = False
            ep_uv[2 * j] = ep_uv[2 * j + 1] = flat_uv[a]
            continue
        ep_uv[2 * j] = flat_uv[i_min]
        ep_uv[2 * j + 1] = flat_uv[i_max]
        if (_near_vertical_at(nxy, a, b, i_min, vertical_threshold_deg)
                or _near_vertical_at(nxy, a, b, i_max, vertical_threshold_deg)):
            flagged[idx_present[j]] = True

    xy = project_to_sea(camera, ep_uv, z_sea)
    fin = np.isfinite(xy).all(axis=1)
    left, right = xy[0::2], xy[1::2]
    finite_pair = fin[0::2] & fin[1::2] & valid
    w = np.linalg.norm(right - left, axis=1)
    c = (left + right) / 2.0
    for j, k in enumerate(idx_present):
        if finite_pair[j]:
            widths[k] = w[j]
            centers[k] = c[j]
    return widths, centers, flagged


def per_detection_waterlines(tracks, camera, outlines, z_sea_mean=42.0,
                             vertical_threshold_deg=10.0,
                             quality=None) -> WaterlineArrays:
    """Outline-based widths and tracer centers, in tracks order.

    With the always-emit extractor, (almost) every detection has an outline,
    so .centers is a full-coverage velocity tracer. .widths is computed for
    all, but the size split should use only .trusted (the quality gate stored
    at extraction time; pass load_outline_quality's dict). Without a quality
    table, trusted falls back to "width is finite". The near-vertical flag is
    an annotation only -- the per-iceberg median absorbs unstable frames.
    """
    n = len(tracks["frame"])
    widths, centers, flagged = _waterlines_fast(
        tracks, camera, outlines, z_sea_mean, vertical_threshold_deg)

    missing = np.zeros(n, dtype=bool)
    trusted = np.zeros(n, dtype=bool)
    for k in range(n):
        key = outline_key(tracks["frame"][k], tracks["id"][k])
        uv = outlines.get(key)
        if uv is None or len(uv) < 3:
            missing[k] = True
            continue
        finite = bool(np.isfinite(widths[k]))
        if quality and key in quality:
            trusted[k] = quality[key]["trusted"] and finite
        else:
            trusted[k] = finite

    logger.info(
        f"per_detection_waterlines: {n} detections -> "
        f"{int(np.isfinite(widths).sum())} with width, {int(trusted.sum())} "
        f"trusted ({int(missing.sum())} no outline, {int(flagged.sum())} "
        f"flagged near-vertical); median trusted width "
        f"{np.nanmedian(widths[trusted]) if trusted.any() else float('nan'):.2f} m")
    return WaterlineArrays(widths, centers, flagged, missing, trusted)


# ============================================================================
# PER-ICEBERG ROBUST SIZE SUMMARY
# ============================================================================

@dataclass
class IcebergSize:
    """Robust width summary for one iceberg over its track.

    The apparent width varies as the iceberg rotates, so a track yields many
    samples of one berg: the median is the representative size, the IQR the
    reportable rotation/noise spread, and a robust z-screen drops rollover and
    fragmentation jumps before the statistics are taken.
    """
    iceberg_id: int
    n: int            # samples used (after screening)
    n_raw: int        # finite trusted samples before screening
    median: float
    mean: float
    q25: float
    q75: float
    iqr: float        # q75 - q25, the rotation spread
    mad: float        # median absolute deviation
    width_min: float
    width_max: float


def aggregate_by_iceberg(tracks, arrays: WaterlineArrays, min_samples=3,
                         robust_z=3.5) -> Dict[int, IcebergSize]:
    """Robust per-iceberg width summary from trusted per-detection widths.

    Sits alongside the per-detection widths (velocity segments keep using the
    instantaneous width); use this for reporting one size per iceberg and for
    the per-iceberg size split. `robust_z` screens outliers with an
    Iglewicz-Hoaglin median/MAD z-score (np.inf disables); icebergs with fewer
    than `min_samples` trusted widths are skipped.
    """
    ids = np.asarray(tracks["id"])
    widths = arrays.widths
    usable = np.isfinite(widths) & arrays.trusted
    out: Dict[int, IcebergSize] = {}

    for iceberg_id in np.unique(ids):
        w = widths[(ids == iceberg_id) & usable]
        n_raw = len(w)
        if n_raw < min_samples:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        if np.isfinite(robust_z) and mad > 0:
            z = 0.6745 * np.abs(w - med) / mad
            w = w[z <= robust_z]
        if len(w) < min_samples:
            continue
        q25, q75 = np.percentile(w, [25, 75])
        out[int(iceberg_id)] = IcebergSize(
            iceberg_id=int(iceberg_id), n=len(w), n_raw=n_raw,
            median=float(np.median(w)), mean=float(np.mean(w)),
            q25=float(q25), q75=float(q75), iqr=float(q75 - q25),
            mad=float(np.median(np.abs(w - np.median(w)))),
            width_min=float(w.min()), width_max=float(w.max()),
        )

    if out:
        meds = np.array([s.median for s in out.values()])
        iqrs = np.array([s.iqr for s in out.values()])
        logger.info(f"aggregate_by_iceberg: {len(out)} icebergs summarised; "
                    f"median-of-medians {np.median(meds):.2f} m, "
                    f"typical IQR {np.median(iqrs):.2f} m")
    return out


def iceberg_median_per_detection(tracks, arrays: WaterlineArrays, sizes=None,
                                 **kwargs) -> np.ndarray:
    """Broadcast each iceberg's median width back to a per-detection array
    (NaN where the iceberg was not summarised) -- the per-iceberg size split."""
    if sizes is None:
        sizes = aggregate_by_iceberg(tracks, arrays, **kwargs)
    ids = np.asarray(tracks["id"])
    out = np.full(len(ids), np.nan)
    for iceberg_id, size in sizes.items():
        out[ids == iceberg_id] = size.median
    return out


def write_iceberg_sizes_csv(sizes: Dict[int, IcebergSize], path):
    """Write the per-iceberg size summary to CSV, sorted by iceberg id."""
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "n", "n_raw", "median_m", "mean_m", "q25_m", "q75_m",
              "iqr_m", "mad_m", "min_m", "max_m"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for iceberg_id in sorted(sizes):
            s = sizes[iceberg_id]
            writer.writerow([
                s.iceberg_id, s.n, s.n_raw,
                f"{s.median:.3f}", f"{s.mean:.3f}",
                f"{s.q25:.3f}", f"{s.q75:.3f}", f"{s.iqr:.3f}", f"{s.mad:.3f}",
                f"{s.width_min:.3f}", f"{s.width_max:.3f}",
            ])
    logger.info(f"Wrote {len(sizes)} iceberg sizes to {path}")


# ============================================================================
# SEGMENTATION BACKENDS (Stage A)
# ============================================================================
# Backends return (contour_crop, crop_shape, offset, metrics): the contour in
# crop coordinates plus the crop origin, so the quality record can inspect
# border contact before the contour is offset to full-image coordinates.

def _crop_box(image, box, pad):
    """Crop image to box (xyxy) with pad px margin; return (crop, origin)."""
    h, w = image.shape[:2]
    x0 = max(int(round(box[0])) - pad, 0)
    y0 = max(int(round(box[1])) - pad, 0)
    x1 = min(int(round(box[2])) + pad, w)
    y1 = min(int(round(box[3])) + pad, h)
    if x1 <= x0 or y1 <= y0:
        return None, (0, 0)
    return image[y0:y1, x0:x1], (x0, y0)


def _pick_contour(mask, box_in_crop):
    """Largest external contour whose centroid lies inside the detection box;
    falls back to the largest overall. Returns (K, 2) crop coords or None."""
    import cv2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    bx0, by0, bx1, by1 = box_in_crop
    best, best_area = None, -1.0
    best_in_box, best_in_box_area = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        if bx0 <= cx <= bx1 and by0 <= cy <= by1 and area > best_in_box_area:
            best_in_box, best_in_box_area = c, area
        if area > best_area:
            best, best_area = c, area
    chosen = best_in_box if best_in_box is not None else best
    return None if chosen is None else chosen.reshape(-1, 2).astype(float)


def _otsu_separability(gray):
    """Otsu separability eta in [0, 1]: illumination-invariant bimodality.
    High when the crop has two intensity populations (berg vs water, day or
    night), ~0.64 for a uniform/noise crop."""
    import cv2
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    levels = np.arange(256)
    omega = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    var_total = float(((levels - mu_t) ** 2 * p).sum())
    if var_total <= 0 or not np.isfinite(np.nanmax(sigma_b)):
        return 0.0
    return float(np.nanmax(sigma_b) / var_total)


def _fg_bg_contrast(gray, mask):
    """Mean foreground minus mean background intensity (signed); None if
    either region is empty."""
    fg = gray[mask > 0]
    bg = gray[mask == 0]
    if fg.size == 0 or bg.size == 0:
        return None
    return float(fg.mean() - bg.mean())


def _denoise_auto(gray):
    """Non-Local Means denoising, applied only to dark or low-separability
    crops so daytime crops stay fast. On night crops this lifts the fraction
    of bergs clearing the separability gate from ~0.5 to ~0.94."""
    import cv2
    if gray.mean() < 40.0 or _otsu_separability(gray) < 0.70:
        return cv2.fastNlMeansDenoising(gray, None, h=10,
                                        templateWindowSize=7,
                                        searchWindowSize=21)
    return gray


def _clahe(gray):
    """Contrast-limited adaptive histogram equalisation so one Otsu threshold
    works day and night (low light, compressed range, brightness gradients)."""
    import cv2
    h, w = gray.shape[:2]
    tiles = (max(2, min(8, w // 16)), max(2, min(8, h // 16)))
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=tiles).apply(gray)


def _segment_otsu(image, box, pad):
    """Otsu threshold in the (denoised, CLAHE-enhanced) crop -> outline.

    Tries BOTH a bright-foreground and a dark-foreground hypothesis and keeps
    the centred component with the least crop-border contact; this recovers
    shadowed bergs darker than bright sky-reflecting water, where a fixed
    'keep the bright region' rule fails.

    Returns (contour_crop, crop_shape, offset, metrics) with the
    illumination-invariant eta and signed contrast in metrics.
    """
    import cv2
    crop, (x0, y0) = _crop_box(image, box, pad)
    if crop is None or crop.size == 0:
        return None, None, None, None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    gray = _denoise_auto(gray)
    eta = _otsu_separability(gray)  # invariant, post-denoise
    blur = cv2.GaussianBlur(_clahe(gray), (3, 3), 0)
    thresh, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    box_in_crop = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
    h, w = gray.shape[:2]

    best = None  # ((border_frac, -area), contour, mask)
    for polarity in ("bright", "dark"):
        raw = (blur >= thresh) if polarity == "bright" else (blur < thresh)
        mask = raw.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contour = _pick_contour(mask, box_in_crop)
        if contour is None:
            continue
        x, y = contour[:, 0], contour[:, 1]
        border = ((x <= 1) | (x >= w - 2) | (y <= 1) | (y >= h - 2)).mean()
        area = cv2.contourArea(contour.astype(np.float32))
        # The berg is the centred blob with the LEAST border contact; the
        # wrong polarity (background) hugs the crop edges.
        key = (border, -area)
        if best is None or key < best[0]:
            best = (key, contour, mask)
    if best is None:
        return None, None, None, {"eta": eta, "contrast": None}
    contour, mask = best[1], best[2]
    return contour, crop.shape, (x0, y0), {"eta": eta,
                                           "contrast": _fg_bg_contrast(gray, mask)}


class _SamRunner:
    """Lazily loaded box-promptable SAM, segmenting ONE box per crop.

    Per-crop (not per-frame batch) inference: on a 24 MP frame with hundreds
    of detections, decoding all boxes at once upscales every mask to the full
    image and exhausts memory. Cropping to box + margin keeps peak memory
    bounded by one small crop, and small bergs get full encoder resolution.
    """

    def __init__(self, sam_model, device=None):
        if sam_model not in SAM_MODELS:
            raise ValueError(f"sam_model must be one of {list(SAM_MODELS)}, "
                             f"got '{sam_model}'")
        self.model_type, self.checkpoint_name, self.url = SAM_MODELS[sam_model]
        self.device = device
        self._predictor = None
        self._torch = None

    def _ensure(self):
        if self._predictor is not None:
            return self._predictor
        import urllib.request
        import torch
        self._torch = torch

        checkpoint = MODELS_DIR / self.checkpoint_name
        if not checkpoint.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading SAM checkpoint to {checkpoint}")
            urllib.request.urlretrieve(self.url, checkpoint)

        if self.model_type == "vit_t":
            try:
                from mobile_sam import SamPredictor, sam_model_registry
            except ImportError as exc:
                raise SystemExit(
                    "sam_model 'mobile_sam' needs the mobile_sam package:\n"
                    "    pip install git+https://github.com/ChaoningZhang/MobileSAM.git"
                ) from exc
        else:
            from segment_anything import SamPredictor, sam_model_registry

        device = str(resolve_device(self.device))
        sam = sam_model_registry[self.model_type](checkpoint=str(checkpoint))
        sam.to(device=device).eval()
        self._predictor = SamPredictor(sam)
        logger.info(f"Loaded SAM '{self.model_type}' on {device} "
                    f"(per-crop inference)")
        return self._predictor

    def segment_box(self, image, box, pad):
        """Segment one box on a crop around it. Returns the backend tuple
        (contour_crop, crop_shape, offset, metrics); eta is None because the
        separability gate is meaningless for SAM masks."""
        predictor = self._ensure()
        torch = self._torch
        # Generous margin so the mask does not hug the crop border
        bw, bh = box[2] - box[0], box[3] - box[1]
        margin = int(max(pad, 0.6 * min(bw, bh)))
        crop, (x0, y0) = _crop_box(image, box, margin)
        if crop is None or crop.size == 0:
            return None, None, None, {"eta": None, "contrast": None}
        box_in_crop = np.array([box[0] - x0, box[1] - y0,
                                box[2] - x0, box[3] - y0], dtype=float)
        with torch.inference_mode():
            predictor.set_image(crop)
            masks, _, _ = predictor.predict(box=box_in_crop,
                                            multimask_output=False)
        mask = masks[0].astype(np.uint8) * 255
        contour = _pick_contour(mask, tuple(box_in_crop))
        predictor.reset_image()  # free the cached image embedding promptly
        return contour, crop.shape, (x0, y0), {"eta": None, "contrast": None}


# ============================================================================
# EXTRACTION ORCHESTRATION
# ============================================================================

def _box_xyxy(tracks, k):
    return np.array([
        tracks["left"][k], tracks["top"][k],
        tracks["left"][k] + tracks["width"][k],
        tracks["top"][k] + tracks["height"][k],
    ], dtype=float)


def _bbox_polygon(box):
    """Bounding-box rectangle as a closed (5, 2) polygon -- the last-resort
    outline so every detection has one."""
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float)


def _quality_record(contour_crop, crop_shape, box, gate, metrics, method):
    """Per-outline quality record including the trusted flag.

    trusted is the accept/reject gate; the outline is emitted either way so
    downstream has full coverage and can inspect these fields.
    """
    import cv2
    metrics = metrics or {}
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
    rec = {
        "eta": float(metrics.get("eta") or 0.0),
        "contrast": float(metrics.get("contrast") or 0.0),
        "area_frac": 0.0,
        "border_frac": 1.0,
        "n_vertices": 0.0,
        "box_area_px": float(box_area),
        "method_code": float(METHOD_CODES.get(method, 3)),
        "trusted": 0.0,
    }
    if contour_crop is None or len(contour_crop) < 3:
        return rec

    rec["n_vertices"] = float(len(contour_crop))
    area = cv2.contourArea(contour_crop.astype(np.float32))
    rec["area_frac"] = float(area / box_area)
    h, w = crop_shape[:2]
    x, y = contour_crop[:, 0], contour_crop[:, 1]
    rec["border_frac"] = float((
        (x <= gate.border_tol_px) | (x >= w - 1 - gate.border_tol_px)
        | (y <= gate.border_tol_px) | (y >= h - 1 - gate.border_tol_px)
    ).mean())

    eta = metrics.get("eta")
    trusted = (
        len(contour_crop) >= gate.min_vertices
        and (eta is None or eta >= gate.min_separability)  # SAM skips eta
        and gate.min_area_frac <= rec["area_frac"] <= gate.max_area_frac
        and rec["border_frac"] <= gate.max_border_frac
    )
    rec["trusted"] = 1.0 if trusted else 0.0
    return rec


def _simplify(contour, epsilon_px):
    """Light Douglas-Peucker simplification in full-image coordinates."""
    import cv2
    if epsilon_px <= 0:
        return contour.astype(float)
    return cv2.approxPolyDP(contour.astype(np.float32), epsilon_px,
                            closed=True).reshape(-1, 2).astype(float)


def _write_packed_npz(out_path, outlines, qkeys, qrows):
    """Write outlines in the PACKED layout (a few members, loads in seconds
    at any scale) together with the quality table."""
    from pathlib import Path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(outlines.keys())
    polys = [np.asarray(outlines[k], dtype=float) for k in keys]
    counts = np.array([len(p) for p in polys], dtype=np.int64)
    arrays = {
        "__packed_xy__": (np.concatenate(polys, axis=0) if polys
                          else np.zeros((0, 2), dtype=float)),
        "__packed_keys__": np.array(keys),
        "__packed_split__": np.cumsum(counts),
        "__quality_keys__": np.array(qkeys),
        "__quality_cols__": np.array(QUALITY_COLUMNS),
        "__quality__": (np.array(qrows, dtype=float)
                        if qrows else np.zeros((0, len(QUALITY_COLUMNS)))),
    }
    np.savez_compressed(out_path, **arrays)


def _extract_sequence(config, tracks, image_dir, out_path):
    """Segment every detection of one sequence and write outlines.npz."""
    import cv2

    gate = _QualityGate()
    method_name = "classical" if config.method == "otsu" else "sam"
    sam = _SamRunner(config.sam_model, config.device) \
        if config.method == "sam" else None

    frame_to_image = build_frame_to_image(image_dir)
    by_frame: Dict[int, list] = {}
    for k in range(len(tracks["frame"])):
        by_frame.setdefault(int(tracks["frame"][k]), []).append(k)

    outlines: Dict[str, np.ndarray] = {}
    qkeys, qrows = [], []
    stats = {"emitted": 0, "trusted": 0, "bbox_fallback": 0, "missing_frame": 0}

    def emit(frame, k, contour, rec):
        outlines[outline_key(frame, tracks["id"][k])] = contour
        qkeys.append(outline_key(frame, tracks["id"][k]))
        qrows.append([rec[c] for c in QUALITY_COLUMNS])
        stats["emitted"] += 1
        stats["trusted"] += int(rec["trusted"] >= 1.0)
        stats["bbox_fallback"] += int(
            int(rec["method_code"]) == METHOD_CODES["bbox_fallback"])

    progress = tqdm(sorted(by_frame.items()), desc="Extracting outlines",
                    unit="frame")
    for frame, indices in progress:
        image_name = frame_to_image.get(frame)
        if image_name is None:
            stats["missing_frame"] += 1
            for k in indices:  # emit bbox fallbacks so detections are not lost
                box = _box_xyxy(tracks, k)
                emit(frame, k, _bbox_polygon(box),
                     _quality_record(None, (0, 0), box, gate, None,
                                     "bbox_fallback"))
            continue

        image = cv2.cvtColor(cv2.imread(str(image_dir / image_name)),
                             cv2.COLOR_BGR2RGB)
        for k in indices:
            box = _box_xyxy(tracks, k)
            if sam is not None:
                c, shape, offset, metrics = sam.segment_box(image, box,
                                                            config.pad_px)
            else:
                c, shape, offset, metrics = _segment_otsu(image, box,
                                                          config.pad_px)
            if c is not None and len(c) >= 3:
                rec = _quality_record(c, shape, box, gate, metrics, method_name)
                contour = _simplify(c + np.array(offset, dtype=float),
                                    config.approx_epsilon_px)
            else:
                rec = _quality_record(None, (0, 0), box, gate, None,
                                      "bbox_fallback")
                contour = _bbox_polygon(box)
            emit(frame, k, contour, rec)
        progress.set_postfix({"outlines": stats["emitted"],
                              "trusted": stats["trusted"]})

    _write_packed_npz(out_path, outlines, qkeys, qrows)
    logger.info(f"Extracted {stats['emitted']} outlines "
                f"({stats['trusted']} trusted, "
                f"{stats['bbox_fallback']} bbox fallbacks, "
                f"{stats['missing_frame']} frames without image)")
    logger.info(f"Saved to: {out_path}")
    return stats


def extract_dataset_outlines(config: OutlineExtractionConfig):
    """Extract outlines for every sequence of a dataset (extract-outlines)."""
    if config.method not in ("sam", "otsu"):
        raise ValueError(f"method must be 'sam' or 'otsu', got '{config.method}'")
    log_config(config, title="Outline Extraction Configuration")

    sequences = get_sequences(config.dataset, run_name=config.run_name)
    for sequence_name, paths in sequences.items():
        if not paths["tracking"].exists():
            logger.warning(f"No track.txt found for {sequence_name}, skipping")
            continue
        log_section(f"Extracting outlines: {sequence_name}")
        tracks = load_mot_tracks(paths["tracking"])
        logger.info(f"{len(tracks['id'])} detections, "
                    f"{len(np.unique(tracks['id']))} icebergs")
        _extract_sequence(config, tracks, paths["images"], paths["outlines"])
        save_config_snapshot(config, paths["outlines_config"])

    log_section("Outline extraction complete")
