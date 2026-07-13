"""Iceberg multi-object tracking with Kalman filtering and spatial indexing.

Combines appearance embeddings (DINOv2, see embedding.py), spatial distance
(Euclidean and Kalman-predicted), and size consistency into a weighted
similarity score, then matches tracks to detections per frame using either
bidirectional (mutual best) or pure greedy assignment.

Per-frame pipeline: predict -> index -> match -> update -> delete -> create.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from filterpy.kalman import KalmanFilter
from tqdm import tqdm

from helpers import (
    bbox_center, emb_key, extract_candidates, extract_matches, get_sequences,
    load_icebergs_by_frame, log_config, log_section, save_config_snapshot,
    write_mot_file,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IcebergTrackingConfig:
    """All hyperparameters for the tracking pipeline (single source of defaults)."""

    # Data
    dataset: str = "???"
    seq_start_index: int = 0
    seq_length_limit: Optional[int] = None
    run_name: Optional[str] = None  # Subfolder under tracking/ for ablation runs

    # Algorithm
    use_kalman: bool = True
    use_spatial_index: bool = True
    bidirectional_matching: bool = True  # False = pure greedy

    # Thresholds (minimum appearance/size similarity, maximum distance in px)
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "appearance": 0.8102,
        "distance": 99.18,
        "size": 0.3143,
    })
    threshold_tolerance: float = 0.3  # Relaxation factor applied to thresholds
    use_appearance_threshold: bool = True
    use_size_threshold: bool = True
    use_distance_threshold: bool = True
    get_gt_thresholds: bool = False   # Derive thresholds from GT statistics
    gt_thresholds: str = "ekas-hill-train"  # Dataset used for threshold derivation

    # Feature weights
    weight_appearance: float = 0.2
    weight_euclidean_distance: float = 0.2
    weight_kalman_distance: float = 0.5
    weight_size: float = 0.1

    # Track management
    max_age: int = 3                 # Frames a track survives without a match
    min_iceberg_id_count: int = 1    # Minimum track length kept in output
    min_iceberg_size: float = 100.0  # Minimum bbox area for new tracks

    # Kalman filter
    process_noise: float = 10.0
    measurement_noise: float = 18.0


# ============================================================================
# SIMILARITY FEATURES
# ============================================================================

def get_distance(iceberg_a, iceberg_b):
    """Euclidean distance in pixels between two iceberg bbox centers."""
    return float(np.linalg.norm(
        np.subtract(bbox_center(iceberg_a["bbox"]), bbox_center(iceberg_b["bbox"]))
    ))


def get_appearance_similarity(features_a, features_b, device="cpu"):
    """Cosine similarity of two embeddings, rescaled from [-1, 1] to [0, 1]."""
    features_a = features_a.to(device).unsqueeze(0)
    features_b = features_b.to(device).unsqueeze(0)
    cosine_sim = F.cosine_similarity(features_a, features_b, dim=1)
    return ((cosine_sim + 1) / 2).item()


def get_size_similarity(iceberg_a, iceberg_b):
    """Ratio of smaller to larger bbox area in [0, 1] (1 = identical size)."""
    _, _, a_w, a_h = iceberg_a["bbox"]
    _, _, b_w, b_h = iceberg_b["bbox"]
    size_a, size_b = a_w * a_h, b_w * b_h
    if size_a == 0 or size_b == 0:
        return 0.0
    return min(size_a, size_b) / max(size_a, size_b)


def get_score(appearance_similarity, eucl_distance_similarity,
              kalman_distance_similarity, size_similarity,
              appearance_weight=0.2, eucl_distance_weight=0.2,
              kalman_distance_weight=0.5, size_weight=0.1):
    """Weighted average of the four similarity components, in [0, 1]."""
    total_weight = (appearance_weight + eucl_distance_weight
                    + kalman_distance_weight + size_weight)
    return (appearance_similarity * appearance_weight
            + eucl_distance_similarity * eucl_distance_weight
            + kalman_distance_similarity * kalman_distance_weight
            + size_similarity * size_weight) / total_weight


def min_max_normalize(v, v_min, v_max):
    """Linearly map v from [v_min, v_max] to [0, 1]."""
    return (v - v_min) / (v_max - v_min)


# ============================================================================
# GROUND-TRUTH THRESHOLD DERIVATION
# ============================================================================

def get_gt_thresholds(dataset, print_stats=True):
    """Derive matching thresholds from ground-truth match statistics.

    Computes appearance, distance, and size similarity for every GT pair of
    the same iceberg in consecutive frames, then takes conservative bounds
    (min appearance, max distance, min size) as thresholds. Requires
    ground-truth embeddings (run `embed embed_source=ground_truth` first).

    Returns:
        dict: {"appearance": float, "distance": float, "size": float}
    """
    log_section("EXTRACT GROUND TRUTH SIMILARITY FEATURES")
    logger.info(f"Dataset: {dataset}")

    sequences = get_sequences(dataset)
    logger.info(f"Found {len(sequences)} sequences: {list(sequences.keys())}")

    total = {"appearance": [], "distance": [], "size": []}
    for sequence_name, paths in sequences.items():
        logger.info(f"\nProcessing sequence: {sequence_name}")
        iceberg_embeddings = torch.load(paths["gt_embeddings"])
        icebergs_by_frame = load_icebergs_by_frame(paths["ground_truth"])
        features = _get_similarity_features(
            icebergs_by_frame, iceberg_embeddings, paths["ground_truth"]
        )
        for key in total:
            total[key].extend(features[key])

    similarity_stats = {}
    for feature, values in total.items():
        arr = np.array(values)
        similarity_stats[feature] = {
            "Mean": np.mean(arr), "Median": np.median(arr),
            "Std Dev": np.std(arr), "Min": np.min(arr), "Max": np.max(arr),
        }

    thresholds = {
        "appearance": float(similarity_stats["appearance"]["Min"]),
        "distance": float(similarity_stats["distance"]["Max"]),
        "size": float(similarity_stats["size"]["Min"]),
    }

    if print_stats:
        df = pd.DataFrame(similarity_stats)
        logger.info("\nSimilarities between matched icebergs:")
        logger.info("\n" + df.to_string(float_format="%.4f"))
        logger.info("\nDerived thresholds:")
        for key, value in thresholds.items():
            logger.info(f"  {key}: {value:.4f}")

    return thresholds


def _get_similarity_features(icebergs_by_frame, iceberg_embeddings, ground_truth_file):
    """Compute appearance, distance, and size features for all GT matches."""
    candidates = extract_candidates(ground_truth_file)
    matches = extract_matches(candidates)
    logger.info(f"Processing {len(matches)} ground truth matches...")

    features = {"appearance": [], "distance": [], "size": []}
    for match in matches:
        iceberg_id = match["id"]
        frame, next_frame = match["frame"], match["next_frame"]

        iceberg_a = icebergs_by_frame[f"{int(frame):06d}"][iceberg_id]
        iceberg_b = icebergs_by_frame[f"{int(next_frame):06d}"][iceberg_id]
        features_a = iceberg_embeddings.get(emb_key(frame, iceberg_id))
        features_b = iceberg_embeddings.get(emb_key(next_frame, iceberg_id))
        if features_a is None or features_b is None:
            continue

        features["appearance"].append(
            get_appearance_similarity(features_a, features_b))
        features["distance"].append(get_distance(iceberg_a, iceberg_b))
        features["size"].append(get_size_similarity(iceberg_a, iceberg_b))

    logger.info(f"Computed features for {len(features['appearance'])} pairs")
    return features


# ============================================================================
# TRACK REPRESENTATION
# ============================================================================

class IcebergTrack:
    """One iceberg track with optional Kalman state estimation.

    The Kalman filter uses a constant-velocity model with state
    [x, y, vx, vy, w, h] and measurements [x, y, w, h].
    """

    def __init__(self, initial_detection, track_id, frame_id, config, distance_threshold):
        self.track_id = track_id
        self.config = config
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.distance_threshold = distance_threshold

        # History of (frame_id, detection_id, bbox, confidence) tuples
        self.history = [(frame_id, initial_detection["id"],
                         initial_detection["bbox"], initial_detection["conf"])]
        self.last_bbox = initial_detection["bbox"]
        self.last_detection = initial_detection

        if config.use_kalman:
            self.kf = self._init_kalman_filter(initial_detection["bbox"], config)
        else:
            self.kf = None
            self.predicted_bbox = initial_detection["bbox"]

    def _init_kalman_filter(self, bbox, config):
        kf = KalmanFilter(dim_x=6, dim_z=4)
        # Constant-velocity transition: x += vx, y += vy; w, h constant
        kf.F = np.array([
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        # Measure position and size
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])
        q = config.process_noise
        kf.Q = np.diag([q, q, q, q, q, q])
        r = config.measurement_noise
        kf.R = np.diag([r, r, r, r])
        kf.P = np.eye(6) * 100  # High initial uncertainty (velocity unknown)

        x, y, w, h = bbox
        kf.x = np.array([x, y, 0, 0, w, h])
        return kf

    def predict(self):
        """Advance the motion model one step and return the predicted bbox."""
        if self.kf is not None:
            self.kf.predict()
            state = self.kf.x
            self.predicted_bbox = [state[0], state[1], state[4], state[5]]
        else:
            self.predicted_bbox = self.last_bbox

        self.age += 1
        self.time_since_update += 1
        return self.predicted_bbox

    def update(self, detection, frame_id):
        """Correct the filter with a matched detection and extend history."""
        bbox = detection["bbox"]
        if self.kf is not None:
            self.kf.update(np.array(bbox))

        self.last_bbox = bbox
        self.last_detection = detection
        self.hits += 1
        self.time_since_update = 0
        self.history.append((frame_id, detection["id"], bbox, detection["conf"]))

    def get_state(self):
        """Current estimated bbox [x, y, w, h]."""
        if self.kf is not None:
            state = self.kf.x
            return [state[0], state[1], state[4], state[5]]
        return self.last_bbox

    def get_velocity(self):
        """Estimated (vx, vy) in pixels/frame; (0, 0) without Kalman."""
        if self.kf is not None:
            return (self.kf.x[2], self.kf.x[3])
        return (0, 0)

    def get_uncertainty(self):
        """2-sigma position uncertainty radius in pixels.

        Used as an adaptive search radius: grows when the track has been
        predicted without updates. Falls back to the configured distance
        threshold when Kalman filtering is disabled.
        """
        if self.kf is not None:
            return 2 * np.sqrt(self.kf.P[0, 0] + self.kf.P[1, 1])
        return self.distance_threshold


# ============================================================================
# SPATIAL INDEXING
# ============================================================================

class SpatialIndex:
    """Grid-based spatial hash for fast radius queries over detections.

    Detections are binned by bbox center into cells of `cell_size` pixels;
    a radius query only inspects nearby cells instead of all detections.
    """

    def __init__(self, cell_size=100):
        self.cell_size = cell_size
        self.index = defaultdict(list)  # (cell_x, cell_y) -> [(det_id, det_data)]

    def build(self, detections):
        """Populate the index from a {det_id: det_data} dictionary."""
        self.index.clear()
        for det_id, det_data in detections.items():
            center_x, center_y = bbox_center(det_data["bbox"])
            cell = (int(center_x // self.cell_size), int(center_y // self.cell_size))
            self.index[cell].append((det_id, det_data))

    def query_radius(self, position, radius):
        """Return all (det_id, det_data) within `radius` of `position`."""
        x, y = position
        r_cells = int(np.ceil(radius / self.cell_size))
        center_cell_x = int(x // self.cell_size)
        center_cell_y = int(y // self.cell_size)

        candidates = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                candidates.extend(
                    self.index.get((center_cell_x + dx, center_cell_y + dy), [])
                )

        filtered = []
        for det_id, det_data in candidates:
            det_x, det_y = bbox_center(det_data["bbox"])
            if np.hypot(det_x - x, det_y - y) <= radius:
                filtered.append((det_id, det_data))
        return filtered


# ============================================================================
# TRACKER
# ============================================================================

class IcebergTracker:
    """Orchestrates tracking across all sequences of a dataset."""

    def __init__(self, config: IcebergTrackingConfig):
        self.config = config
        self.dataset = config.dataset
        self.sequences = get_sequences(self.dataset, run_name=config.run_name)

        self.thresholds = dict(config.thresholds)
        if config.get_gt_thresholds:
            try:
                self.thresholds = get_gt_thresholds(config.gt_thresholds,
                                                    print_stats=False)
            except (FileNotFoundError, KeyError, OSError) as e:
                logger.warning(
                    f"Could not derive thresholds from '{config.gt_thresholds}' "
                    f"({e}); using configured thresholds instead."
                )

        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0

        log_config(config, title="Iceberg Tracking Configuration")
        logger.info(f"Active thresholds: "
                    f"appearance={self.thresholds['appearance']:.4f}, "
                    f"distance={self.thresholds['distance']:.0f}px, "
                    f"size={self.thresholds['size']:.4f}")

    def track(self):
        """Process all sequences: load detections and embeddings, track, save."""
        for sequence_name, paths in self.sequences.items():
            log_section(f"Processing sequence: {sequence_name}")

            if not paths["detections"].exists():
                logger.warning(f"No det.txt found for {sequence_name}, skipping")
                continue
            if not paths["det_embeddings"].exists():
                logger.warning(f"No embeddings.pt found for {sequence_name}, skipping")
                continue

            # Reset per-sequence state
            self.tracks = []
            self.next_track_id = 1
            self.frame_count = 0

            icebergs_by_frame = load_icebergs_by_frame(paths["detections"])
            logger.info(f"Loading embeddings from {paths['det_embeddings']}")
            iceberg_embeddings = torch.load(paths["det_embeddings"])
            logger.info(f"Loaded {len(iceberg_embeddings)} embeddings")

            all_results = self._process_sequence(icebergs_by_frame, iceberg_embeddings)
            self._save_tracking_results(all_results, paths["tracking"])
            save_config_snapshot(self.config, paths["track_config"])

        log_section("Tracking complete")

    def _process_sequence(self, icebergs_by_frame, embeddings):
        """Track all frames of one sequence with a progress bar."""
        frames = sorted(icebergs_by_frame.keys())
        start_frame = self.config.seq_start_index
        if self.config.seq_length_limit is None:
            end_frame = len(frames)
        else:
            end_frame = min(self.config.seq_length_limit, len(frames))

        logger.info(f"Processing {end_frame - start_frame} frames")
        all_results = []
        progress_bar = tqdm(range(start_frame, end_frame),
                            desc="Tracking frames", unit="frame")

        for frame_idx in progress_bar:
            frame_id = frames[frame_idx]
            frame_results = self._track_frame(
                frame_id, icebergs_by_frame[frame_id], embeddings
            )
            all_results.extend(frame_results)
            progress_bar.set_postfix({
                "tracks": len(self.tracks),
                "matches": len(frame_results),
            })
        return all_results

    def _track_frame(self, frame_id, detections, embeddings):
        """Run the per-frame pipeline; return results for matched tracks."""
        self.frame_count += 1

        # 1. Predict new positions for all active tracks
        for track in self.tracks:
            track.predict()

        # 2. Build spatial index for fast candidate lookup
        if self.config.use_spatial_index:
            spatial_index = SpatialIndex(cell_size=100)
            spatial_index.build(detections)
        else:
            spatial_index = None

        # 3. Match tracks to detections
        if self.config.bidirectional_matching:
            matches, unmatched_tracks, unmatched_dets = self._matching_bidirectional(
                frame_id, detections, embeddings, spatial_index)
        else:
            matches, unmatched_tracks, unmatched_dets = self._matching_pure_greedy(
                frame_id, detections, embeddings, spatial_index)

        # 4. Update matched tracks
        for track, detection in matches:
            track.update(detection, frame_id)

        # 5. Delete tracks unmatched for longer than max_age
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.config.max_age]

        # 6. Create new tracks for unmatched detections above the size filter
        for detection in unmatched_dets:
            _, _, w, h = detection["bbox"]
            if w * h >= self.config.min_iceberg_size:
                self.tracks.append(IcebergTrack(
                    detection, self.next_track_id, frame_id, self.config,
                    distance_threshold=self.thresholds["distance"],
                ))
                self.next_track_id += 1

        # 7. Output results for tracks matched in this frame
        return [
            {
                "frame_id": frame_id,
                "track_id": track.track_id,
                "bbox": track.last_bbox,
                "confidence": track.last_detection["conf"],
            }
            for track in self.tracks if track.time_since_update == 0
        ]

    def _compute_similarity(self, track, detection, features_a, features_b):
        """Gated, weighted similarity for one track-detection pair.

        Size and appearance act as gates (with tolerance relaxation); pairs
        passing both gates get a weighted score over appearance, Euclidean
        distance, Kalman-predicted distance, and size. Returns None if gated out.
        """
        last_iceberg = {"bbox": track.last_bbox}
        detection_iceberg = {"bbox": detection["bbox"]}
        tolerance = 1 - self.config.threshold_tolerance

        size_similarity = get_size_similarity(last_iceberg, detection_iceberg)
        if (self.config.use_size_threshold
                and size_similarity < self.thresholds["size"] * tolerance):
            return None

        appearance_similarity = get_appearance_similarity(features_a, features_b)
        if (self.config.use_appearance_threshold
                and appearance_similarity < self.thresholds["appearance"] * tolerance):
            return None

        distance_eucl = get_distance(last_iceberg, detection_iceberg)
        distance_kalman = get_distance({"bbox": track.predicted_bbox}, detection_iceberg)

        dist_range = (self.thresholds["distance"]
                      if self.config.use_distance_threshold else float("inf"))
        kalman_distance_norm = 1 - min_max_normalize(distance_kalman, 0, dist_range)
        eucl_distance_norm = 1 - min_max_normalize(distance_eucl, 0, dist_range)

        # Re-normalize gated features to [0, 1] above their threshold
        size_similarity = min_max_normalize(
            size_similarity,
            self.thresholds["size"] if self.config.use_size_threshold else 0.0, 1.0)
        appearance_similarity = min_max_normalize(
            appearance_similarity,
            self.thresholds["appearance"] if self.config.use_appearance_threshold else 0.0, 1.0)

        return get_score(
            appearance_similarity, eucl_distance_norm, kalman_distance_norm,
            size_similarity,
            self.config.weight_appearance, self.config.weight_euclidean_distance,
            self.config.weight_kalman_distance, self.config.weight_size,
        )

    def _iter_candidates(self, frame_id, detections, embeddings, spatial_index):
        """Yield (similarity, track, detection) for all plausible pairs.

        Shared candidate generation for both matching algorithms: spatial
        query around each track's predicted position (adaptive radius when
        Kalman is enabled), embedding lookup, and gated similarity.
        """
        all_detections = list(detections.values())
        if self.config.use_distance_threshold:
            base_radius = self.thresholds["distance"] * (1 + self.config.threshold_tolerance)
        else:
            base_radius = float("inf")

        for track in self.tracks:
            radius = (max(track.get_uncertainty(), base_radius)
                      if self.config.use_kalman else base_radius)

            if spatial_index is None or radius == float("inf"):
                candidates = [(d["id"], d) for d in all_detections]
            else:
                candidates = spatial_index.query_radius(
                    bbox_center(track.predicted_bbox), radius)

            track_embedding = embeddings.get(
                emb_key(track.history[-1][0], track.history[-1][1]))
            if track_embedding is None:
                continue

            for det_id, detection in candidates:
                det_embedding = embeddings.get(emb_key(frame_id, det_id))
                if det_embedding is None:
                    continue
                similarity = self._compute_similarity(
                    track, detection, track_embedding, det_embedding)
                if similarity is not None:
                    yield similarity, track, detection

    def _matching_bidirectional(self, frame_id, detections, embeddings, spatial_index=None):
        """Mutual-best matching: track and detection must prefer each other.

        Conservative (fewer false matches, potentially more fragmentation).

        Returns:
            (matches, unmatched_tracks, unmatched_dets)
        """
        track_best = {}  # track -> (detection, similarity)
        for similarity, track, detection in self._iter_candidates(
                frame_id, detections, embeddings, spatial_index):
            if track not in track_best or similarity > track_best[track][1]:
                track_best[track] = (detection, similarity)

        det_best = {}  # det_id -> (track, similarity)
        for track, (detection, similarity) in track_best.items():
            det_id = detection["id"]
            if det_id not in det_best or similarity > det_best[det_id][1]:
                det_best[det_id] = (track, similarity)

        matches, matched_tracks, matched_det_ids = [], set(), set()
        for track, (detection, _) in track_best.items():
            det_id = detection["id"]
            if det_best[det_id][0] is track:
                matches.append((track, detection))
                matched_tracks.add(track)
                matched_det_ids.add(det_id)

        unmatched_tracks = [t for t in self.tracks if t not in matched_tracks]
        unmatched_dets = [d for d in detections.values()
                          if d["id"] not in matched_det_ids]
        return matches, unmatched_tracks, unmatched_dets

    def _matching_pure_greedy(self, frame_id, detections, embeddings, spatial_index=None):
        """Global greedy matching: assign highest-similarity pairs first.

        More permissive than bidirectional (better continuity in dense scenes).

        Returns:
            (matches, unmatched_tracks, unmatched_dets)
        """
        all_candidates = list(self._iter_candidates(
            frame_id, detections, embeddings, spatial_index))
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        matches, matched_tracks, matched_det_ids = [], set(), set()
        for _, track, detection in all_candidates:
            det_id = detection["id"]
            if track in matched_tracks or det_id in matched_det_ids:
                continue
            matches.append((track, detection))
            matched_tracks.add(track)
            matched_det_ids.add(det_id)

        unmatched_tracks = [t for t in self.tracks if t not in matched_tracks]
        unmatched_dets = [d for d in detections.values()
                          if d["id"] not in matched_det_ids]
        return matches, unmatched_tracks, unmatched_dets

    def _save_tracking_results(self, all_results, output_path):
        """Filter out short tracks and write MOT-format track.txt."""
        logger.info("\nSaving tracking results...")

        track_lengths = defaultdict(int)
        for result in all_results:
            track_lengths[result["track_id"]] += 1
        valid_track_ids = {tid for tid, length in track_lengths.items()
                           if length >= self.config.min_iceberg_id_count}

        rows = [
            (r["frame_id"], r["track_id"], *r["bbox"], r["confidence"])
            for r in all_results if r["track_id"] in valid_track_ids
        ]
        write_mot_file(rows, output_path)

        logger.info(f"Total tracking entries: {len(all_results)}")
        logger.info(f"Filtered out (too short): {len(all_results) - len(rows)}")
        logger.info(f"Valid tracks: {len(valid_track_ids)}")
        logger.info(f"Saved to: {output_path}")
