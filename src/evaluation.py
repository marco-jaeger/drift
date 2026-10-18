"""Tracking evaluation against ground truth with the official TrackEval metrics.

Two steps: (1) match_tracking_to_gt keeps, per frame, the tracking box assigned
to each GT box (one-to-one Hungarian assignment on IoU) and writes eval.txt.
The GT annotates only a subset of the icebergs, so boxes of unannotated
icebergs are ignored instead of counted as false positives; a second box on
the same annotated iceberg (e.g. split by shading) is ignored as well.
(2) calc_metrics passes the full GT and eval.txt to TrackEval's HOTA, CLEAR,
Identity, and Count metrics per sequence, combines sequences with TrackEval,
prints the tables, and writes metrics.json next to the tracking outputs for
scriptable comparison across runs.
"""

import contextlib
import io
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# TrackEval (pinned commit) still uses np.int/np.float/np.bool, which numpy
# 1.24 removed; restore them as the builtins they were aliases of.
for _name, _type in (("int", int), ("float", float), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _type)

# Importing trackeval prints a warning about its optional BURST dataset
# (needs pycocotools), which is not used here.
with contextlib.redirect_stdout(io.StringIO()):
    from trackeval.datasets._base_dataset import _BaseDataset  # noqa: E402
    from trackeval.metrics import CLEAR, HOTA, Count, Identity  # noqa: E402

from helpers import (  # noqa: E402
    get_sequences, load_icebergs_by_frame, log_config, log_section,
    normalize_frame, save_metrics_json, write_mot_file,
)

logger = logging.getLogger(__name__)

# Key under which compute_sequence_metrics keeps TrackEval's raw results,
# needed by combine_metrics; not written to metrics.json
RAW_KEY = "_trackeval"


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """Configuration for tracking evaluation."""

    dataset: str = "???"
    iou_threshold: float = 0.5       # Minimum IoU for a GT-track match
    run_name: Optional[str] = None   # Evaluate tracking/<run_name>/ outputs


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def eval_tracking(config: EvalConfig):
    """Run the full evaluation: GT matching, then metric computation."""
    log_config(config, title="Evaluation Configuration")
    match_tracking_to_gt(config)
    return calc_metrics(config)


def match_tracking_to_gt(config: EvalConfig):
    """Write eval.txt with the tracking boxes assigned to GT boxes.

    See select_gt_matched_tracks. GT boxes without an assigned tracking box
    count as false negatives during metric computation.
    """
    log_section("FILTER TRACKING RESULTS TO GROUND TRUTH")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"IoU threshold: {config.iou_threshold}")

    sequences = get_sequences(config.dataset, run_name=config.run_name)
    logger.info(f"Sequences ({len(sequences)}): {', '.join(sequences.keys())}\n")

    for sequence_name, paths in sequences.items():
        logger.info(f"Processing sequence: {sequence_name}")

        gt_by_frame = _load_unique_ids(paths["ground_truth"])
        track_by_frame = _load_unique_ids(paths["tracking"])
        selected = select_gt_matched_tracks(gt_by_frame, track_by_frame,
                                            config.iou_threshold)

        rows = [(int(frame_id), track["id"], *track["bbox"], track["conf"])
                for frame_id in sorted(selected)
                for track in selected[frame_id].values()]
        write_mot_file(rows, paths["eval"])
        logger.info(f"Filtered tracking saved to: {paths['eval']}")


def _load_unique_ids(path):
    """load_icebergs_by_frame, after checking that no ID occurs twice in a frame.

    load_icebergs_by_frame keys boxes by ID and would silently keep only the
    last of two boxes with the same ID; TrackEval rejects such files.
    """
    counts = Counter()
    with open(path) as f:
        for line in f:
            if line.strip():
                frame, obj_id = line.split(",", 2)[:2]
                counts[normalize_frame(frame), int(obj_id)] += 1
    duplicates = sorted(key for key, n in counts.items() if n > 1)
    if duplicates:
        raise ValueError(f"{path}: same ID more than once in a frame, "
                         f"(frame, id): {duplicates[:10]}")
    return load_icebergs_by_frame(path)


def select_gt_matched_tracks(gt_by_frame, track_by_frame, iou_threshold=0.5):
    """Per frame, the tracking boxes assigned one-to-one to GT boxes.

    GT and tracking boxes are assigned by maximising IoU (Hungarian), with
    pairs below the IoU threshold excluded as in TrackEval's CLEAR matching,
    so each GT box keeps the tracking box with the highest IoU unless another
    GT box needs it. All other tracking boxes are ignored: boxes of icebergs
    that are not annotated, and second boxes on an annotated iceberg.

    Returns:
        dict: {frame_id: {track_id: track}} for frames that contain GT boxes.
    """
    selected = {}
    for frame_id, gts in gt_by_frame.items():
        tracks = track_by_frame.get(frame_id, {})
        selected[frame_id] = {}
        if not gts or not tracks:
            continue

        track_ids = list(tracks)
        ious = _box_ious(gts.values(), tracks.values())
        ious[ious < iou_threshold - np.finfo("float").eps] = 0
        rows, cols = linear_sum_assignment(-ious)
        for r, c in zip(rows, cols):
            if ious[r, c] > np.finfo("float").eps:
                selected[frame_id][track_ids[c]] = tracks[track_ids[c]]
    return selected


def calc_metrics(config: EvalConfig):
    """Compute all tracking metrics, print tables, and write metrics.json.

    Returns:
        dict: {sequence_name: metrics} plus a "COMBINED" entry for
        multi-sequence datasets.
    """
    log_section("COMPUTING TRACKING METRICS", width=80)

    sequences = get_sequences(config.dataset, run_name=config.run_name)
    all_metrics = {}

    for sequence_name, paths in sequences.items():
        logger.info(f"Processing sequence: {sequence_name}")

        gt_by_frame = _load_unique_ids(paths["ground_truth"])
        track_by_frame = _load_unique_ids(paths["eval"])

        metrics = compute_sequence_metrics(gt_by_frame, track_by_frame,
                                           config.iou_threshold)
        all_metrics[sequence_name] = metrics
        save_metrics_json({k: v for k, v in metrics.items() if k != RAW_KEY},
                          paths["metrics"])

    if len(all_metrics) > 1:
        all_metrics["COMBINED"] = combine_metrics(all_metrics)

    print_all_metrics(all_metrics)
    return all_metrics


# ============================================================================
# METRIC COMPUTATION (TrackEval)
# ============================================================================

def compute_sequence_metrics(gt_by_frame, track_by_frame, iou_threshold=0.5):
    """TrackEval HOTA, CLEAR, Identity, and Count metrics for one sequence.

    Returns:
        dict: TrackEval's summary fields (e.g. IDF1, IDSW, MOTA, HOTA), with
        HOTA's fields averaged over its localisation thresholds as TrackEval
        reports them, plus the raw results under RAW_KEY for combine_metrics.
    """
    data = _to_trackeval_data(gt_by_frame, track_by_frame)
    raw = {metric.get_name(): metric.eval_sequence(data)
           for metric in _metrics(iou_threshold)}
    return _summarize(raw)


def combine_metrics(all_metrics):
    """Combine sequences (or datasets) with TrackEval's combine_sequences.

    Takes {name: metrics} as returned by compute_sequence_metrics or by this
    function. Combining combined results again is exact, as TrackEval
    combines by sums and by averages weighted with summed counts.
    """
    raw = {}
    for metric in _metrics():
        name = metric.get_name()
        raw[name] = metric.combine_sequences(
            {seq: m[RAW_KEY][name] for seq, m in all_metrics.items()})
    return _summarize(raw)


def _metrics(iou_threshold=0.5):
    """TrackEval metric objects; CLEAR and Identity match at iou_threshold.

    HOTA evaluates its own range of IoU thresholds (0.05 to 0.95).
    """
    config = {"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}
    return [HOTA(), CLEAR(config), Identity(config), Count()]


def _summarize(raw):
    """Flat dict of TrackEval's summary fields, plus raw under RAW_KEY."""
    summary = {}
    for metric in _metrics():
        res = raw[metric.get_name()]
        integer_fields = set(getattr(metric, "integer_fields", []))
        for field in metric.summary_fields:
            value = res[field]
            summary[field] = (int(value) if field in integer_fields
                              else float(np.mean(value)))
    summary[RAW_KEY] = raw
    return summary


def _to_trackeval_data(gt_by_frame, track_by_frame):
    """One sequence in the form TrackEval's dataset classes pass to metrics.

    IDs are renumbered to 0..N-1, every timestep gets an entry (also frames
    without boxes, as CLEAR's IDSW and Frag depend on temporal adjacency),
    and similarities are TrackEval's own box IoUs.
    """
    timesteps = _timesteps(gt_by_frame, track_by_frame)
    gt_ids = sorted({i for gts in gt_by_frame.values() for i in gts})
    track_ids = sorted({i for tracks in track_by_frame.values() for i in tracks})
    gt_idx = {gt_id: k for k, gt_id in enumerate(gt_ids)}
    track_idx = {track_id: k for k, track_id in enumerate(track_ids)}

    data = {"num_timesteps": len(timesteps),
            "num_gt_ids": len(gt_ids), "num_tracker_ids": len(track_ids),
            "num_gt_dets": 0, "num_tracker_dets": 0,
            "gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    for frame_id in timesteps:
        gts = gt_by_frame.get(frame_id, {})
        tracks = track_by_frame.get(frame_id, {})
        data["gt_ids"].append(np.array([gt_idx[i] for i in gts], dtype=int))
        data["tracker_ids"].append(
            np.array([track_idx[i] for i in tracks], dtype=int))
        data["similarity_scores"].append(_box_ious(gts.values(), tracks.values()))
        data["num_gt_dets"] += len(gts)
        data["num_tracker_dets"] += len(tracks)
    return data


def _timesteps(gt_by_frame, track_by_frame):
    """All frame ids of a sequence in temporal order.

    Numeric frame ids are filled in from the first to the last frame, so that
    frames without any box remain timesteps; other ids (e.g. filenames) are
    taken as they occur.
    """
    frames = set(gt_by_frame) | set(track_by_frame)
    if not frames:
        return []
    try:
        numbers = sorted(int(frame) for frame in frames)
    except ValueError:
        return sorted(frames)
    return [normalize_frame(n) for n in range(numbers[0], numbers[-1] + 1)]


def _box_ious(gt_boxes, track_boxes):
    """IoU matrix (GT x tracking) of (x, y, w, h) boxes, computed by TrackEval."""
    return _BaseDataset._calculate_box_ious(
        np.array([b["bbox"] for b in gt_boxes], dtype=float).reshape(-1, 4),
        np.array([b["bbox"] for b in track_boxes], dtype=float).reshape(-1, 4),
        box_format="xywh")


# ============================================================================
# OUTPUT
# ============================================================================

def print_all_metrics(all_metrics):
    """Print Count, CLEAR, Identity, HOTA, and derived tables."""
    _log_table("Count", all_metrics, [], ["Dets", "GT_Dets", "IDs", "GT_IDs"])
    _log_table("CLEAR", all_metrics,
               ["MOTA", "MOTP", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR"],
               ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "Frag", "MT", "PT", "ML"])
    _log_table("Identity", all_metrics, ["IDF1", "IDR", "IDP"],
               ["IDTP", "IDFN", "IDFP"])
    _log_table("HOTA", all_metrics,
               ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr",
                "LocA"], [])

    width = 66
    logger.info(f"\n{'=' * width}")
    logger.info("Derived Metrics:")
    logger.info(f"{'=' * width}")
    logger.info(f"{'Sequence':<30}{'ID_Ratio':<12}{'IDSW/track':<12}{'Frag/track':<12}")
    logger.info(f"{'-' * width}")
    for seq_name, m in all_metrics.items():
        gt_ids = m["GT_IDs"]
        id_ratio = m["IDs"] / gt_ids if gt_ids > 0 else 0.0
        idsw_per_track = m["IDSW"] / gt_ids if gt_ids > 0 else 0.0
        frag_per_track = m["Frag"] / gt_ids if gt_ids > 0 else 0.0
        logger.info(f"{seq_name:<30}{id_ratio:<12.2f}"
                    f"{idsw_per_track:<12.2f}{frag_per_track:<12.2f}")
    logger.info(f"\n{'=' * width}\n")


def _log_table(title, all_metrics, float_fields, int_fields):
    """Log one metric table: a row per sequence, a column per field."""
    width = 30 + 10 * (len(float_fields) + len(int_fields))
    logger.info(f"\n{'=' * width}")
    logger.info(f"{title}:")
    logger.info(f"{'=' * width}")
    logger.info(f"{'Sequence':<30}"
                + "".join(f"{field:<10}" for field in float_fields + int_fields))
    logger.info(f"{'-' * width}")
    for seq_name, m in all_metrics.items():
        logger.info(f"{seq_name:<30}"
                    + "".join(f"{m[field]:<10.3f}" for field in float_fields)
                    + "".join(f"{m[field]:<10}" for field in int_fields))
