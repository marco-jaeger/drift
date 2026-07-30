"""Georeferencing: project iceberg detections from pixel space to UTM.

Uses a calibrated glimpse camera (data/<site>/camera.json: surveyed position,
laboratory intrinsics, skyline-refined orientation) to cast a world-frame ray
per image point and intersect it with the sea surface, modelled as a
horizontal plane at the constant mean height z_sea_mean. The flat-sea
assumption is valid for relative flow patterns; a constant offset largely
cancels in velocities.

Chain: this module projects detections and assembles per-iceberg UTM
trajectories; circulation.py aggregates them into velocity fields and
streamlines. Trajectory figures live in visualize.py.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from helpers import (
    get_glimpse, get_sequences, load_mot_tracks, log_config, log_section,
    save_config_snapshot,
)
import outlines
from outlines import project_to_sea

logger = logging.getLogger(__name__)

UTM_EPSG = "EPSG:32623"  # UTM zone 23N (study site)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GeoreferenceConfig:
    """Parameters for the georeference command."""

    dataset: str = "???"
    run_name: Optional[str] = None    # Read tracking/<run_name>/track.txt

    # Tracer point: outline-based waterline centre (requires outlines.npz from
    # extract-outlines) or the bbox bottom-edge midpoint as fallback.
    use_outlines: bool = True
    vertical_threshold_deg: float = 10.0  # Near-vertical endpoint annotation

    # Sea surface and timing
    z_sea_mean: float = 42.0          # Camera/DEM vertical datum [m]
    frame_interval_min: float = 2.0   # Minutes between consecutive frames

    # Track filtering (same fields as CirculationConfig)
    min_track_length: int = 1
    max_tracks: Optional[int] = None  # Keep only the N longest tracks
    frame_start: Optional[int] = None
    length_hours: Optional[float] = None

    # Figures (rendered by visualize.py)
    make_figures: bool = True
    basemap: bool = True


# ============================================================================
# CAMERA AND TRACK FILTERS
# ============================================================================

def load_camera(path):
    """Load a calibrated glimpse camera from JSON."""
    camera = get_glimpse().Camera.from_json(path=str(path))
    logger.info(f"Loaded camera from {path}")
    return camera


def filter_tracks(tracks, min_confidence=0.0, min_track_length=1):
    """Drop low-confidence detections and short track fragments.

    Short, low-score fragments let detector noise leak into the circulation
    field and the size distribution. Returns a new tracks dict.
    """
    keep = tracks["conf"] >= min_confidence
    subset = {key: value[keep] for key, value in tracks.items()}

    ids, counts = np.unique(subset["id"], return_counts=True)
    long_enough = set(ids[counts >= min_track_length].tolist())
    keep_long = np.array([berg in long_enough for berg in subset["id"]],
                         dtype=bool)
    subset = {key: value[keep_long] for key, value in subset.items()}

    logger.info(f"filter_tracks: {len(tracks['id'])} -> {len(subset['id'])} "
                f"detections, {len(np.unique(tracks['id']))} -> "
                f"{len(np.unique(subset['id']))} icebergs "
                f"(min_confidence={min_confidence:.2f}, "
                f"min_track_length={min_track_length})")
    return subset


def limit_to_longest_tracks(tracks, max_tracks):
    """Keep only the max_tracks icebergs with the most detections (None or a
    large value returns the tracks unchanged). Useful for readable plots."""
    if max_tracks is None:
        return tracks
    ids, counts = np.unique(tracks["id"], return_counts=True)
    if max_tracks >= len(ids):
        return tracks
    keep_ids = set(ids[np.argsort(counts)[::-1][:max_tracks]].tolist())
    keep = np.array([berg in keep_ids for berg in tracks["id"]], dtype=bool)
    subset = {key: value[keep] for key, value in tracks.items()}
    logger.info(f"limit_to_longest_tracks: kept longest {len(keep_ids)} of "
                f"{len(ids)} icebergs ({len(subset['id'])} detections)")
    return subset


def window_tracks(tracks, frame_start, length_hours, frame_interval_min):
    """Restrict detections to one time window.

    frame_start defaults to the first frame in the file; length_hours
    (converted to frames via the frame interval) defaults to the rest of the
    sequence. No-op when both are None. Returns a new tracks dict.
    """
    if frame_start is None and length_hours is None:
        return tracks
    fmin = frame_start if frame_start is not None else int(tracks["frame"].min())
    if length_hours is not None:
        fmax = fmin + int(round(length_hours * 60.0 / frame_interval_min))
    else:
        fmax = int(tracks["frame"].max())
    keep = (tracks["frame"] >= fmin) & (tracks["frame"] <= fmax)
    subset = {k: v[keep] for k, v in tracks.items()}
    logger.info(f"window_tracks: frames [{fmin}, {fmax}] -> "
                f"{len(subset['id'])} of {len(tracks['id'])} detections")
    return subset


# ============================================================================
# PROJECTION AND TRACK ASSEMBLY
# ============================================================================

def bounding_box_points(tracks):
    """Image-space (u, v) arrays for the bbox centre and lower corners.

    The image v-axis increases downward, so the bottom edge (top + height) is
    the part of the box closest to the waterline.
    """
    left, top = tracks["left"], tracks["top"]
    w, h = tracks["width"], tracks["height"]
    centre = np.column_stack([left + w / 2.0, top + h / 2.0])
    bottom_left = np.column_stack([left, top + h])
    bottom_right = np.column_stack([left + w, top + h])
    return centre, bottom_left, bottom_right


def sea_level_for_frames(frames, z_sea_mean=42.0):
    """Constant sea-surface height per detection (flat-sea model)."""
    return np.full(np.asarray(frames).shape, float(z_sea_mean))


def project_detections(camera, tracks, z_sea_mean=42.0, point="waterline",
                       centers=None):
    """Project every detection from pixel space to UTM.

    With `centers` (an (N, 2) UTM array aligned with tracks, e.g.
    per_detection_waterlines(...).centers), those are used directly as the
    tracer points -- the outline-based waterline centre is a better flow
    tracer than any bounding-box point. Otherwise a bbox point is projected:
    'waterline' (bottom-edge midpoint) or 'center'.

    Returns:
        (N, 4) array [iceberg_id, frame, easting, northing].
    """
    if centers is not None:
        xy = np.asarray(centers, dtype=float)
        if xy.shape != (len(tracks["id"]), 2):
            raise ValueError(f"centers must be (N, 2) aligned with tracks; "
                             f"got {xy.shape} for N={len(tracks['id'])}.")
        return np.column_stack([tracks["id"], tracks["frame"], xy])

    centre, bottom_left, bottom_right = bounding_box_points(tracks)
    if point == "center":
        uv = centre
    elif point == "waterline":
        uv = (bottom_left + bottom_right) / 2.0
    else:
        raise ValueError("point must be 'center' or 'waterline'.")

    z_sea = sea_level_for_frames(tracks["frame"], z_sea_mean)
    xy = project_to_sea(camera, uv, z_sea)
    return np.column_stack([tracks["id"], tracks["frame"], xy])


def assemble_tracks(detections, out_csv=None):
    """Assemble projected detections into per-iceberg UTM trajectories.

    Args:
        detections: (N, 4) [id, frame, easting, northing] array.
        out_csv: Optional CSV output path.

    Returns:
        {iceberg_id: (M, 3) [frame, easting, northing]} ordered by frame.
    """
    tracks = {}
    for berg in np.unique(detections[:, 0]):
        rows = detections[detections[:, 0] == berg]
        rows = rows[np.argsort(rows[:, 1])]
        tracks[int(berg)] = rows[:, 1:4]

    if out_csv:
        from pathlib import Path
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_csv, detections, delimiter=",",
                   header="id,frame,easting,northing", comments="",
                   fmt=["%d", "%d", "%.3f", "%.3f"])
        logger.info(f"Wrote {out_csv}")
    return tracks


# ============================================================================
# DRIFT SPEEDS (physical, used to colour the trajectory figures)
# ============================================================================

def segment_speeds(frames, xy, frame_interval_min):
    """Per-segment drift speed [m/min] for one frame-ordered track.

    Speed of segment i->i+1 is UTM distance / (frame gap x frame interval).
    Segments with non-positive elapsed time or a non-finite endpoint are NaN.
    """
    frames = np.asarray(frames, dtype=float)
    xy = np.asarray(xy, dtype=float)
    dt = np.diff(frames) * float(frame_interval_min)     # minutes
    dist = np.linalg.norm(np.diff(xy, axis=0), axis=1)   # metres
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dt > 0, dist / dt, np.nan)


def compute_track_speeds(tracks_utm, frame_interval_min=2.0):
    """Per-segment speeds for every assembled UTM track.

    Returns:
        (speeds_by_id, all_speeds): speeds_by_id[id] maps each segment's
        starting frame to its speed (so pixel-space tracks can look speeds up
        regardless of ordering); all_speeds concatenates all finite speeds
        for a shared colour scale.
    """
    speeds_by_id = {}
    all_speeds = []
    for berg, arr in tracks_utm.items():
        order = np.argsort(arr[:, 0])
        frames = arr[order, 0]
        xy = arr[order, 1:3]
        speed = segment_speeds(frames, xy, frame_interval_min)
        speeds_by_id[int(berg)] = {int(frames[i]): speed[i]
                                   for i in range(len(speed))}
        all_speeds.append(speed[np.isfinite(speed)])
    all_speeds = (np.concatenate(all_speeds) if all_speeds
                  else np.array([], dtype=float))
    return speeds_by_id, all_speeds


# ============================================================================
# SHARED LOADING (used by georeference and circulation commands)
# ============================================================================

def load_waterlines(paths, tracks, camera, z_sea_mean, vertical_threshold_deg):
    """Load outlines + quality for a sequence and compute waterline arrays."""
    berg_outlines = outlines.load_outlines(paths["outlines"])
    quality = outlines.load_outline_quality(paths["outlines"])
    return outlines.per_detection_waterlines(
        tracks, camera, berg_outlines, z_sea_mean=z_sea_mean,
        vertical_threshold_deg=vertical_threshold_deg, quality=quality)


# ============================================================================
# COMMAND
# ============================================================================

def run_georeference(config: GeoreferenceConfig):
    """Project, assemble, and (optionally) plot trajectories per sequence.

    Outputs per sequence (under tracking[/run_name]/georeference/):
    tracks_utm.csv, iceberg_sizes.csv (outline mode), trajectory figures,
    and the resolved config snapshot.
    """
    log_config(config, title="Georeference Configuration")
    sequences = get_sequences(config.dataset, run_name=config.run_name)

    for sequence_name, paths in sequences.items():
        if not paths["tracking"].exists():
            logger.warning(f"No track.txt found for {sequence_name}, skipping")
            continue
        log_section(f"Georeferencing sequence: {sequence_name}")

        tracks = load_mot_tracks(paths["tracking"])
        tracks = window_tracks(tracks, config.frame_start, config.length_hours,
                               config.frame_interval_min)
        tracks = filter_tracks(tracks, min_track_length=config.min_track_length)
        tracks = limit_to_longest_tracks(tracks, config.max_tracks)
        if len(tracks["id"]) == 0:
            logger.warning("No detections left after filtering, skipping")
            continue

        camera = load_camera(paths["camera"])
        out_dir = paths["georeference"]
        out_dir.mkdir(parents=True, exist_ok=True)

        use_outlines = config.use_outlines and paths["outlines"].exists()
        if config.use_outlines and not use_outlines:
            logger.warning(f"No outlines at {paths['outlines']} "
                           f"(run extract-outlines); falling back to bbox tracer")

        if use_outlines:
            wl = load_waterlines(paths, tracks, camera, config.z_sea_mean,
                                 config.vertical_threshold_deg)
            detections = project_detections(camera, tracks, config.z_sea_mean,
                                            centers=wl.centers)
            finite = np.isfinite(detections[:, 2:4]).all(axis=1)
            if not finite.all():
                logger.info(f"Dropping {int((~finite).sum())} detections "
                            f"without a valid waterline centre")
                detections = detections[finite]
            sizes = outlines.aggregate_by_iceberg(tracks, wl)
            if sizes:
                outlines.write_iceberg_sizes_csv(
                    sizes, out_dir / "iceberg_sizes.csv")
        else:
            detections = project_detections(camera, tracks, config.z_sea_mean)

        assembled = assemble_tracks(detections,
                                    out_csv=out_dir / "tracks_utm.csv")
        logger.info(f"Projected {len(detections)} detections into "
                    f"{len(assembled)} iceberg tracks")

        if config.make_figures:
            from visualize import render_track_figures
            render_track_figures(config, tracks, assembled, paths, out_dir)

        save_config_snapshot(config, out_dir / "config.yaml")

    log_section("Georeferencing complete")
