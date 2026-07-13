"""Fjord circulation fields from georeferenced iceberg tracks.

Aggregates the UTM trajectories from georeference.py into gridded velocity
fields and circulation streamlines: velocities are
displacement over physical elapsed time (frame gap x frame interval, correct
across detection gaps), interpolated onto an analysis grid and smoothed with
a NaN-aware Gaussian filter. Panels are rendered by visualize.py.

Optional size split: each velocity segment is classified small/large by the
apparent waterline width (outline-based when outlines.npz exists, bbox
lower-corner distance otherwise). With size_basis='per-iceberg' (default)
every segment of an iceberg inherits that iceberg's robust track-median
width; 'per-detection' uses the instantaneous width (noisier). The 'all'
field always contains every segment with a finite position, so distant bergs
that cannot be size-classified still populate the flow field.

Known limitations (carried into the manuscript): flat-sea assumption (valid
for relative patterns), tracer-point jitter from rotation (absorbed by
gridding + smoothing), and size as an *apparent* single-azimuth width.
"""

import copy
import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

from helpers import (
    get_sequences, load_mot_tracks, log_config, log_section,
    save_config_snapshot,
)
from georeference import (
    UTM_EPSG, assemble_tracks, bounding_box_points, load_camera,
    load_waterlines, project_detections, project_to_sea, sea_level_for_frames,
    window_tracks,
)
import outlines

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CirculationConfig:
    """Parameters for the circulation command."""

    dataset: str = "???"
    run_name: Optional[str] = None    # Read tracking/<run_name>/track.txt

    # Tracer / size metric (outline-based when outlines.npz exists)
    use_outlines: bool = True
    vertical_threshold_deg: float = 10.0

    # Sea surface and temporal sampling
    z_sea_mean: float = 42.0
    frame_interval_min: float = 2.0
    max_gap_frames: int = 3           # Drop segments spanning larger gaps

    # Time windowing: single window (frame_start/length_hours) or batch mode
    # (window_hours: consecutive non-overlapping windows, one subdir each)
    frame_start: Optional[int] = None
    length_hours: Optional[float] = None
    window_hours: Optional[float] = None

    # Analysis grid. Bounds default to the data extent (padded by two grid
    # cells, snapped to the resolution); set explicitly for identical extents
    # across runs.
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    grid_resolution_m: float = 100.0
    smoothing_sigma: float = 1.25
    max_distance_from_data_m: float = 100.0  # No extrapolation beyond this

    # Size split: None -> one combined field; 'p50'/'p25'/... percentile of
    # segment widths, or '12m'/'16m' fixed width in metres.
    size_split: Optional[str] = None
    size_basis: str = "per-iceberg"   # or 'per-detection'
    min_samples_per_iceberg: int = 5  # Trusted widths needed for a median

    # Rendering
    vmax_percentile: float = 98.0     # Robust shared colour-scale maximum
    streamline_density: float = 3.0
    streamline_boundary_cells: int = 0  # Extend field so streamlines reach
                                        # the data boundary (0 disables)
    quiver_length_power: float = 0.5  # <1 compresses arrow lengths (visual)
    basemap: bool = True
    dpi: int = 300
    utm_epsg: str = UTM_EPSG


# ============================================================================
# PER-DETECTION SIZE (bbox fallback) AND SPLIT PARSING
# ============================================================================

def per_detection_widths(config, tracks, camera):
    """Bbox fallback width [m]: distance between the projected lower corners.

    Systematically biased (the lowest pixel row rarely coincides with the
    widest columns); prefer the outline-based metric when outlines exist.
    """
    _, bottom_left, bottom_right = bounding_box_points(tracks)
    z_sea = sea_level_for_frames(tracks["frame"], config.z_sea_mean)
    bl_world = project_to_sea(camera, bottom_left, z_sea)
    br_world = project_to_sea(camera, bottom_right, z_sea)
    return np.linalg.norm(br_world - bl_world, axis=1)


def parse_size_split(spec, segment_widths):
    """Resolve a size-split spec to a threshold in metres.

    Returns:
        (threshold_m, description): description records the derivation for
        the metadata sidecar.
    """
    spec = spec.strip().lower()
    m_pct = re.fullmatch(r"p(\d+(?:\.\d+)?)", spec)
    m_metres = re.fullmatch(r"(\d+(?:\.\d+)?)m?", spec)
    if m_pct:
        pct = float(m_pct.group(1))
        if not (0 < pct < 100):
            raise ValueError(f"percentile must be in (0, 100), got {pct}")
        return (float(np.nanpercentile(segment_widths, pct)),
                f"p{pct:g} of segment widths")
    if m_metres:
        threshold = float(m_metres.group(1))
        return threshold, f"fixed {threshold:g} m"
    raise ValueError(f"size_split must be a percentile ('p50', ...) or a "
                     f"width in metres ('12m'), got {spec!r}")


# ============================================================================
# VELOCITY SEGMENTS
# ============================================================================

def velocity_segments(assembled, frame_interval_min, max_gap_frames):
    """Per-segment velocities [m/min] for the whole population (no split).

    Returns:
        (points, vectors): segment midpoints and velocity vectors, (M, 2).
    """
    points, vectors = [], []
    for track in assembled.values():
        frames = track[:, 0]
        xy = track[:, 1:3]
        for k in range(len(frames) - 1):
            gap = int(frames[k + 1] - frames[k])
            if gap < 1 or gap > max_gap_frames:
                continue
            p0, p1 = xy[k], xy[k + 1]
            if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
                continue
            points.append((p0 + p1) / 2.0)
            vectors.append((p1 - p0) / (gap * frame_interval_min))
    return np.array(points), np.array(vectors)


def velocity_segments_with_widths(detections, widths, frame_interval_min,
                                  max_gap_frames):
    """All velocity segments plus each segment's mean endpoint width.

    A segment is kept whenever both POSITIONS are finite; the width may be
    NaN (untrusted / distant berg), in which case the segment still feeds the
    velocity field but cannot be size-classified.

    Args:
        detections: (N, 4) [id, frame, easting, northing] in tracks order.
        widths: Aligned per-detection widths (NaN allowed).

    Returns:
        (points, vectors, segment_widths); segment_widths may contain NaN.
    """
    points, vectors, segment_widths = [], [], []
    by_id = {}
    for k in range(len(detections)):
        by_id.setdefault(int(detections[k, 0]), []).append(k)

    for indices in by_id.values():
        indices = sorted(indices, key=lambda i: detections[i, 1])
        rows = detections[indices]
        w = widths[indices]
        frames = rows[:, 1]
        xy = rows[:, 2:4]
        for k in range(len(frames) - 1):
            gap = int(frames[k + 1] - frames[k])
            if gap < 1 or gap > max_gap_frames:
                continue
            p0, p1 = xy[k], xy[k + 1]
            if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
                continue
            points.append((p0 + p1) / 2.0)
            vectors.append((p1 - p0) / (gap * frame_interval_min))
            segment_widths.append((w[k] + w[k + 1]) / 2.0)

    return np.array(points), np.array(vectors), np.array(segment_widths)


# ============================================================================
# GRIDDING AND SMOOTHING
# ============================================================================

def resolve_grid_bounds(config, detections):
    """Grid bounds: explicit config values, or the finite data extent padded
    by two grid cells and snapped to the resolution."""
    res = config.grid_resolution_m
    x, y = detections[:, 2], detections[:, 3]
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise RuntimeError("No finite projected positions to derive grid bounds.")
    pad = 2 * res

    def snap(v, up):
        return (np.ceil(v / res) if up else np.floor(v / res)) * res

    bounds = (
        config.x_min if config.x_min is not None else snap(x[finite].min() - pad, False),
        config.x_max if config.x_max is not None else snap(x[finite].max() + pad, True),
        config.y_min if config.y_min is not None else snap(y[finite].min() - pad, False),
        config.y_max if config.y_max is not None else snap(y[finite].max() + pad, True),
    )
    logger.info(f"Grid bounds: x [{bounds[0]:.0f}, {bounds[1]:.0f}], "
                f"y [{bounds[2]:.0f}, {bounds[3]:.0f}] "
                f"({res:.0f} m resolution)")
    return bounds


def build_grid(bounds, resolution_m):
    """Regular analysis grid over (x_min, x_max, y_min, y_max)."""
    x_min, x_max, y_min, y_max = bounds
    xi = np.arange(x_min, x_max, resolution_m)
    yi = np.arange(y_min, y_max, resolution_m)
    return np.meshgrid(xi, yi)


def nan_gaussian_filter(field, sigma):
    """Gaussian smoothing that ignores NaNs (normalised convolution)."""
    valid = np.isfinite(field).astype(float)
    filled = np.where(np.isfinite(field), field, 0.0)
    numerator = gaussian_filter(filled, sigma=sigma)
    denominator = gaussian_filter(valid, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = numerator / denominator
    out[denominator < 1e-6] = np.nan
    return out


def grid_velocity_field(config, X, Y, points, vectors):
    """Interpolate scattered velocities onto the grid, smooth, and mask cells
    farther than max_distance_from_data_m from any observation.

    Returns:
        (U, V, speed) with invalid cells NaN.
    """
    if len(points) == 0:
        empty = np.full(X.shape, np.nan)
        return empty, empty.copy(), empty.copy()

    U = griddata(points, vectors[:, 0], (X, Y), method="linear")
    V = griddata(points, vectors[:, 1], (X, Y), method="linear")
    U = nan_gaussian_filter(U, config.smoothing_sigma)
    V = nan_gaussian_filter(V, config.smoothing_sigma)

    tree = cKDTree(points)
    distance, _ = tree.query(np.column_stack((X.ravel(), Y.ravel())), k=1)
    near_data = distance.reshape(X.shape) < config.max_distance_from_data_m

    valid = near_data & np.isfinite(U) & np.isfinite(V)
    U = np.where(valid, U, np.nan)
    V = np.where(valid, V, np.nan)
    return U, V, np.sqrt(U ** 2 + V ** 2)


def extend_field_for_streamlines(U, V, iterations=1):
    """Prepare the field so streamlines reach the data boundary.

    streamplot terminates integration at NaN neighbours, so streamlines stop
    one cell short of the boundary. Every NaN is filled by nearest-neighbour
    extrapolation and the result restricted to the valid region dilated by
    `iterations` cells -- enough margin to reach the true boundary without
    flooding into unsampled water. iterations=0 returns the field unchanged.
    """
    valid = np.isfinite(U) & np.isfinite(V)
    if iterations <= 0 or not valid.any():
        return U, V, np.hypot(U, V)
    idx = distance_transform_edt(~valid, return_distances=False,
                                 return_indices=True)
    U_fill = U[tuple(idx)]
    V_fill = V[tuple(idx)]
    draw = binary_dilation(valid, iterations=iterations)
    U_ext = np.where(draw, U_fill, np.nan)
    V_ext = np.where(draw, V_fill, np.nan)
    return U_ext, V_ext, np.hypot(U_ext, V_ext)


# ============================================================================
# ORCHESTRATION
# ============================================================================

def _resolve_vmax(config, speed_fields):
    """Shared colour-scale maximum: robust percentile over all fields."""
    finite = np.concatenate([s[np.isfinite(s)].ravel() for s in speed_fields])
    if finite.size == 0:
        return 1.0
    return round(float(np.nanpercentile(finite, config.vmax_percentile)), 1)


def _run_sequence(config, paths, output_dir):
    """Circulation analysis for one sequence; writes panels, legend, and metadata."""
    from visualize import (
        render_streamline_panel, render_vector_panel, save_speed_legend,
        setup_publication_style,
    )
    setup_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_mot_tracks(paths["tracking"])
    if len(tracks["id"]) == 0:
        raise RuntimeError("Tracks file is empty.")
    tracks = window_tracks(tracks, config.frame_start, config.length_hours,
                           config.frame_interval_min)
    if len(tracks["id"]) == 0:
        raise RuntimeError("No detections in the selected window.")

    camera = load_camera(paths["camera"])
    use_outlines = config.use_outlines and paths["outlines"].exists()
    if config.use_outlines and not use_outlines:
        logger.warning(f"No outlines at {paths['outlines']} "
                       f"(run extract-outlines); falling back to bbox metrics")

    # --- Georeferencing --- #
    waterline_arrays = None
    if use_outlines:
        waterline_arrays = load_waterlines(paths, tracks, camera,
                                           config.z_sea_mean,
                                           config.vertical_threshold_deg)
        detections = project_detections(camera, tracks, config.z_sea_mean,
                                        centers=waterline_arrays.centers)
        # Full flow coverage: where the outline centre did not project (e.g.
        # near-horizon endpoint), fall back to the bbox tracer; the width
        # stays NaN so those bergs remain unclassified but not dropped.
        nan_pos = ~np.isfinite(detections[:, 2:4]).all(axis=1)
        if nan_pos.any():
            bbox_det = project_detections(camera, tracks, config.z_sea_mean)
            detections[nan_pos, 2:4] = bbox_det[nan_pos, 2:4]
            logger.info(f"Filled {int(nan_pos.sum())} detections with the "
                        f"bbox tracer where the outline centre did not project")
    else:
        detections = project_detections(camera, tracks, config.z_sea_mean)

    assembled = assemble_tracks(detections)
    logger.info(f"Projected {len(detections)} detections into "
                f"{len(assembled)} tracks")

    if waterline_arrays is not None:
        sizes = outlines.aggregate_by_iceberg(tracks, waterline_arrays)
        if sizes:
            outlines.write_iceberg_sizes_csv(sizes,
                                              output_dir / "iceberg_sizes.csv")

    bounds = resolve_grid_bounds(config, detections)
    X, Y = build_grid(bounds, config.grid_resolution_m)

    # --- Velocity segments, optionally size-classified --- #
    size_meta = None
    if config.size_split is None:
        points, vectors = velocity_segments(assembled,
                                            config.frame_interval_min,
                                            config.max_gap_frames)
        class_segments = [("all", points, vectors)]
        logger.info(f"No size split: one combined field over "
                    f"{len(assembled)} icebergs")
    else:
        if waterline_arrays is not None:
            # Only trusted outlines feed the size metric
            widths = np.where(waterline_arrays.trusted,
                              waterline_arrays.widths, np.nan)
            width_metric = ("georeferenced apparent waterline width (m), "
                            "outline-based (trusted outlines only)")
        else:
            widths = per_detection_widths(config, tracks, camera)
            width_metric = ("georeferenced apparent waterline width (m), "
                            "bounding-box lower-corner distance")

        if config.size_basis == "per-iceberg":
            if waterline_arrays is None:
                # Minimal arrays object so bbox widths can be aggregated
                wa = outlines.WaterlineArrays(
                    widths=widths,
                    centers=np.full((len(widths), 2), np.nan),
                    flagged=np.zeros(len(widths), bool),
                    missing=~np.isfinite(widths),
                    trusted=np.isfinite(widths))
            else:
                wa = waterline_arrays
            iceberg_sizes = outlines.aggregate_by_iceberg(
                tracks, wa, min_samples=config.min_samples_per_iceberg)
            # Each detection inherits its iceberg's median width (NaN when
            # the iceberg lacked enough trusted samples)
            class_widths = outlines.iceberg_median_per_detection(
                tracks, wa, sizes=iceberg_sizes)
            classification = (f"per-iceberg (track median width, "
                              f">={config.min_samples_per_iceberg} "
                              f"trusted samples)")
        else:
            class_widths = widths
            classification = "per-detection (instantaneous segment width)"

        all_points, all_vectors, segment_widths = velocity_segments_with_widths(
            detections, class_widths, config.frame_interval_min,
            config.max_gap_frames)
        if len(all_points) == 0:
            raise RuntimeError("No valid velocity segments.")

        finite = np.isfinite(segment_widths)
        if finite.sum() == 0:
            raise RuntimeError("No segments have a trustworthy width to split "
                               "on (check outlines / min_samples_per_iceberg).")
        threshold, description = parse_size_split(config.size_split,
                                                  segment_widths[finite])
        small_mask = finite & (segment_widths < threshold)
        large_mask = finite & (segment_widths >= threshold)
        unclassified_mask = ~finite  # Finite position, untrusted width

        class_segments = [
            ("all", all_points, all_vectors),
            ("small", all_points[small_mask], all_vectors[small_mask]),
            ("large", all_points[large_mask], all_vectors[large_mask]),
        ]
        if unclassified_mask.any():
            class_segments.append(("unclassified",
                                   all_points[unclassified_mask],
                                   all_vectors[unclassified_mask]))

        size_meta = {
            "size_metric": width_metric,
            "classification": classification,
            "size_basis": config.size_basis,
            "size_split": config.size_split,
            "threshold_m": round(threshold, 2),
            "threshold_definition": description + " (trustworthy widths only)",
            "n_segments_all": int(len(all_points)),
            "n_segments_small": int(small_mask.sum()),
            "n_segments_large": int(large_mask.sum()),
            "n_segments_unclassified": int(unclassified_mask.sum()),
            "note": ("'all' includes every segment with a finite position so "
                     "distant bergs are not dropped; small/large use only "
                     "trustworthy widths."),
        }
        logger.info(f"Size split {config.size_split} ({config.size_basis}) -> "
                    f"threshold {threshold:.1f} m ({description}); "
                    f"{len(all_points)} all / {int(small_mask.sum())} small / "
                    f"{int(large_mask.sum())} large / "
                    f"{int(unclassified_mask.sum())} unclassified segments")

    # --- Grid, render --- #
    fields = {}
    for label, points, vectors in class_segments:
        U, V, speed = grid_velocity_field(config, X, Y, points, vectors)
        fields[label] = dict(points=points, U=U, V=V, speed=speed)
        logger.info(f"{label} class: {len(points)} velocity segments")

    vmax = _resolve_vmax(config, [f["speed"] for f in fields.values()])
    logger.info(f"Shared colour-scale maximum: {vmax:.1f} m/min")

    outputs = {}
    for label, f in fields.items():
        vec_path = output_dir / f"circulation_vectors_{label}.png"
        stream_path = output_dir / f"circulation_streamlines_{label}.png"
        render_vector_panel(config, X, Y, f["U"], f["V"], f["speed"], vmax,
                            bounds, vec_path)
        U_ext, V_ext, speed_ext = extend_field_for_streamlines(
            f["U"], f["V"], iterations=config.streamline_boundary_cells)
        render_streamline_panel(config, X, Y, U_ext, V_ext, speed_ext, vmax,
                                bounds, stream_path)
        outputs[f"vectors_{label}"] = str(vec_path)
        outputs[f"streamlines_{label}"] = str(stream_path)

    legend_path = output_dir / "circulation_legend.png"
    save_speed_legend(legend_path, vmin=0, vmax=vmax)
    outputs["legend"] = str(legend_path)

    # --- Metadata sidecar (figure caption material) --- #
    frames = tracks["frame"]
    span_frames = int(frames.max() - frames.min())
    metadata = {
        "input_tracks": str(paths["tracking"]),
        "n_velocity_segments": {label: int(len(f["points"]))
                                for label, f in fields.items()},
        "velocity_units": "m/min",
        "observation_window": {
            "first_frame": int(frames.min()),
            "last_frame": int(frames.max()),
            "frame_interval_min": config.frame_interval_min,
            "duration_hours": round(span_frames * config.frame_interval_min / 60.0, 2),
        },
        "grid_bounds": {"x_min": bounds[0], "x_max": bounds[1],
                        "y_min": bounds[2], "y_max": bounds[3]},
        "grid_resolution_m": config.grid_resolution_m,
        "smoothing_sigma": config.smoothing_sigma,
        "sea_surface": f"constant mean sea level (z = {config.z_sea_mean} m)",
        "vmax_m_per_min": vmax,
        "config": asdict(config),
    }
    if size_meta is not None:
        metadata["size"] = size_meta
    meta_path = output_dir / "circulation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote {meta_path}")
    outputs["metadata"] = str(meta_path)

    save_config_snapshot(config, output_dir / "config.yaml")
    return {"outputs": outputs, "metadata": metadata}


def _run_windowed(config, paths):
    """Batch mode: consecutive non-overlapping windows of window_hours each,
    one numbered subdirectory per window plus a manifest. Window failures are
    recorded, not fatal."""
    window_hours = config.window_hours
    if not window_hours or window_hours <= 0:
        raise ValueError("window_hours must be a positive number.")

    tracks_full = load_mot_tracks(paths["tracking"])
    if len(tracks_full["frame"]) == 0:
        raise RuntimeError("Tracks file is empty.")

    first_frame = int(tracks_full["frame"].min())
    last_frame = int(tracks_full["frame"].max())
    window_frames = int(round(window_hours * 60.0 / config.frame_interval_min))
    if window_frames < 2:
        raise ValueError(f"window_hours={window_hours} gives only "
                         f"{window_frames} frames per window.")

    starts = list(range(first_frame, last_frame + 1, window_frames))
    logger.info(f"Windowed run: "
                f"{(last_frame - first_frame) * config.frame_interval_min / 60.0:.1f} h "
                f"sequence, {len(starts)} windows of {window_hours:.1f} h "
                f"({window_frames} frames each)")

    base_dir = paths["circulation"]
    base_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    n_ok = 0
    for i, window_start in enumerate(starts):
        t_start_h = (window_start - first_frame) * config.frame_interval_min / 60.0
        t_end_h = t_start_h + window_hours
        subdir = base_dir / f"window_{i + 1:02d}_h{t_start_h:.1f}-h{t_end_h:.1f}"
        logger.info(f"Window {i + 1}/{len(starts)}: frame_start "
                    f"{window_start} ({t_start_h:.1f} h - {t_end_h:.1f} h)")

        window_config = copy.copy(config)
        window_config.window_hours = None  # Avoid recursion
        window_config.frame_start = window_start
        window_config.length_hours = window_hours

        entry = {"index": i + 1, "frame_start": window_start,
                 "t_start_h": round(t_start_h, 2), "t_end_h": round(t_end_h, 2),
                 "output_dir": str(subdir)}
        try:
            _run_sequence(window_config, paths, subdir)
            entry["status"] = "ok"
            n_ok += 1
        except Exception as exc:
            logger.warning(f"Window {i + 1} skipped: {exc}")
            entry["status"] = f"failed: {exc}"
        manifests.append(entry)

    manifest = {
        "input_tracks": str(paths["tracking"]),
        "window_hours": window_hours,
        "frame_interval_min": config.frame_interval_min,
        "size_split": config.size_split,
        "n_windows_total": len(starts),
        "n_windows_ok": n_ok,
        "n_windows_failed": len(starts) - n_ok,
        "windows": manifests,
    }
    manifest_path = base_dir / "windows_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Windowed run complete: {n_ok}/{len(starts)} windows OK, "
                f"manifest at {manifest_path}")
    return manifest


def run_circulation(config: CirculationConfig):
    """Circulation fields for every sequence of a dataset (circulation)."""
    log_config(config, title="Fjord Circulation Configuration")
    sequences = get_sequences(config.dataset, run_name=config.run_name)

    for sequence_name, paths in sequences.items():
        if not paths["tracking"].exists():
            logger.warning(f"No track.txt found for {sequence_name}, skipping")
            continue
        log_section(f"Circulation analysis: {sequence_name}")
        if config.window_hours is not None:
            _run_windowed(config, paths)
        else:
            _run_sequence(config, paths, paths["circulation"])

    log_section("Circulation analysis complete")
