"""Tracking evaluation against ground truth (MOTChallenge protocol).

Two steps: (1) match_tracking_to_gt filters tracking results to GT-matched
detections via frame-by-frame greedy IoU matching and writes eval.txt;
(2) calc_metrics computes Count, CLEAR, and Identity metrics per sequence,
prints TrackEval-style tables, and writes metrics.json next to the tracking
outputs for scriptable comparison across runs.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from collections import defaultdict

from helpers import (
    calculate_iou, get_sequences, load_icebergs_by_frame, log_config,
    log_section, save_metrics_json, write_mot_file,
)

logger = logging.getLogger(__name__)


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
    """Filter tracking results to GT-matched detections and write eval.txt.

    For each GT object in each frame, the tracking detection with the highest
    IoU above the threshold is kept; everything else is discarded. Unmatched
    GT objects count as false negatives during metric computation.
    """
    log_section("FILTER TRACKING RESULTS TO GROUND TRUTH")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"IoU threshold: {config.iou_threshold}")

    sequences = get_sequences(config.dataset, run_name=config.run_name)
    logger.info(f"Sequences ({len(sequences)}): {', '.join(sequences.keys())}\n")

    for sequence_name, paths in sequences.items():
        logger.info(f"Processing sequence: {sequence_name}")

        gt_by_frame = load_icebergs_by_frame(paths["ground_truth"])
        track_by_frame = load_icebergs_by_frame(paths["tracking"])

        rows = []
        for frame_id in sorted(gt_by_frame.keys()):
            gts = gt_by_frame.get(frame_id, {})
            tracks = track_by_frame.get(frame_id, {})
            frame_matches = []

            for gt in gts.values():
                gt_bb = gt["bbox"]
                best_match = 0.0
                candidate = None

                for track in tracks.values():
                    track_bb = track["bbox"]
                    box1 = [gt_bb[0], gt_bb[1], gt_bb[0] + gt_bb[2], gt_bb[1] + gt_bb[3]]
                    box2 = [track_bb[0], track_bb[1],
                            track_bb[0] + track_bb[2], track_bb[1] + track_bb[3]]
                    iou = calculate_iou(box1, box2)
                    if iou > best_match and iou > config.iou_threshold:
                        best_match = iou
                        candidate = track

                if candidate is not None and candidate not in frame_matches:
                    frame_matches.append(candidate)
                    x, y, w, h = candidate["bbox"]
                    rows.append((int(frame_id), candidate["id"],
                                 x, y, w, h, candidate["conf"]))

        write_mot_file(rows, paths["eval"])
        logger.info(f"Filtered tracking saved to: {paths['eval']}")


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

        gt_by_frame = load_icebergs_by_frame(paths["ground_truth"])
        track_by_frame = load_icebergs_by_frame(paths["eval"])

        metrics = compute_sequence_metrics(gt_by_frame, track_by_frame,
                                           config.iou_threshold)
        all_metrics[sequence_name] = metrics
        save_metrics_json(metrics, paths["metrics"])

    if len(all_metrics) > 1:
        all_metrics["COMBINED"] = combine_metrics(all_metrics)

    print_all_metrics(all_metrics)
    return all_metrics


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def compute_sequence_metrics(gt_by_frame, track_by_frame, iou_threshold=0.5):
    """Compute Count, CLEAR, and Identity metrics for one sequence.

    Frame-by-frame greedy IoU matching (each GT matched to at most one track
    per frame and vice versa), followed by CLEAR statistics (recall, LocA,
    IDSW, Frag, MT/PT/ML) and Identity metrics (IDF1/IDR/IDP via longest
    continuous GT-track associations).
    """
    frames = sorted(gt_by_frame.keys())

    # ---- 1. Frame-by-frame matching ---- #
    gt_to_track = {}  # frame_id -> {gt_id: track_id}
    track_to_gt = {}  # frame_id -> {track_id: gt_id}
    CLR_TP = 0
    CLR_FN = 0
    all_ious = []

    for frame_id in frames:
        gts = gt_by_frame.get(frame_id, {})
        tracks = track_by_frame.get(frame_id, {})
        gt_to_track[frame_id] = {}
        track_to_gt[frame_id] = {}

        for gt_id, gt in gts.items():
            gt_bb = gt["bbox"]
            best_iou = 0.0
            best_track_id = None

            for track_id, track in tracks.items():
                track_bb = track["bbox"]
                box1 = [gt_bb[0], gt_bb[1], gt_bb[0] + gt_bb[2], gt_bb[1] + gt_bb[3]]
                box2 = [track_bb[0], track_bb[1],
                        track_bb[0] + track_bb[2], track_bb[1] + track_bb[3]]
                iou = calculate_iou(box1, box2)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_iou >= iou_threshold and best_track_id is not None:
                gt_to_track[frame_id][gt_id] = best_track_id
                track_to_gt[frame_id][best_track_id] = gt_id
                all_ious.append(best_iou)
                CLR_TP += 1
            else:
                CLR_FN += 1

    # ---- 2. Count metrics ---- #
    Dets = sum(len(tracks) for tracks in track_by_frame.values())
    GT_Dets = sum(len(gts) for gts in gt_by_frame.values())

    all_track_ids = set()
    for tracks in track_by_frame.values():
        all_track_ids.update(tracks.keys())
    IDs = len(all_track_ids)

    all_gt_ids = set()
    for gts in gt_by_frame.values():
        all_gt_ids.update(gts.keys())
    GT_IDs = len(all_gt_ids)

    # ---- 3. Recall and localization accuracy ---- #
    CLR_Re = CLR_TP / GT_Dets if GT_Dets > 0 else 0.0
    LocA = np.mean(all_ious) if len(all_ious) > 0 else 0.0

    # ---- 4. ID switches and fragmentations ---- #
    IDSW = 0
    Frag = 0
    gt_last_track = {}
    gt_last_frame = {}

    for frame_id in frames:
        for gt_id, track_id in gt_to_track.get(frame_id, {}).items():
            if gt_id in gt_last_track:
                if gt_last_track[gt_id] != track_id:
                    IDSW += 1
                    gt_last_track[gt_id] = track_id
                if int(frame_id) > int(gt_last_frame[gt_id]) + 1:
                    Frag += 1  # Gap in tracking of this GT
            else:
                gt_last_track[gt_id] = track_id
            gt_last_frame[gt_id] = int(frame_id)

    # ---- 5. Track coverage (MT >= 80%, PT 20-80%, ML < 20%) ---- #
    MT = PT = ML = 0
    for gt_id in all_gt_ids:
        gt_frames = [f for f in frames if gt_id in gt_by_frame.get(f, {})]
        if not gt_frames:
            continue
        matched = sum(1 for f in gt_frames if gt_id in gt_to_track.get(f, {}))
        coverage = matched / len(gt_frames)
        if coverage >= 0.8:
            MT += 1
        elif coverage >= 0.2:
            PT += 1
        else:
            ML += 1

    MTR = MT / GT_IDs if GT_IDs > 0 else 0.0
    PTR = PT / GT_IDs if GT_IDs > 0 else 0.0
    MLR = ML / GT_IDs if GT_IDs > 0 else 0.0

    # ---- 6. Identity metrics ---- #
    gt_trajectories = defaultdict(list)
    track_trajectories = defaultdict(list)
    for frame_id in frames:
        for gt_id, track_id in gt_to_track.get(frame_id, {}).items():
            gt_trajectories[gt_id].append((frame_id, track_id))
        for track_id, gt_id in track_to_gt.get(frame_id, {}).items():
            track_trajectories[track_id].append((frame_id, gt_id))

    IDTP = sum(_longest_segment(traj) for traj in gt_trajectories.values())
    IDTP_from_tracks = sum(_longest_segment(traj)
                           for traj in track_trajectories.values())
    IDFN = CLR_TP - IDTP
    IDFP = CLR_TP - IDTP_from_tracks

    IDR = IDTP / (IDTP + IDFN) if (IDTP + IDFN) > 0 else 0.0
    IDP = IDTP / (IDTP + IDFP) if (IDTP + IDFP) > 0 else 0.0
    IDF1 = (2 * IDTP / (2 * IDTP + IDFN + IDFP)
            if (2 * IDTP + IDFN + IDFP) > 0 else 0.0)

    return {
        # Count
        "Dets": Dets, "GT_Dets": GT_Dets, "IDs": IDs, "GT_IDs": GT_IDs,
        # CLEAR
        "CLR_Re": CLR_Re, "LocA": LocA, "CLR_TP": CLR_TP, "CLR_FN": CLR_FN,
        "IDSW": IDSW, "Frag": Frag, "MT": MT, "PT": PT, "ML": ML,
        "MTR": MTR, "PTR": PTR, "MLR": MLR,
        # Identity
        "IDF1": IDF1, "IDR": IDR, "IDP": IDP,
        "IDTP": IDTP, "IDFN": IDFN, "IDFP": IDFP,
    }


def _longest_segment(trajectory):
    """Longest continuous run of the same partner ID in a (frame, id) list.

    Used for IDTP: from the GT side the partner is a track ID, from the
    track side it is a GT ID.
    """
    segments = defaultdict(int)
    current_partner = None
    current_length = 0

    for _, partner in trajectory:
        if partner == current_partner:
            current_length += 1
        else:
            if current_partner is not None:
                segments[current_partner] = max(segments[current_partner],
                                                current_length)
            current_partner = partner
            current_length = 1

    if current_partner is not None:
        segments[current_partner] = max(segments[current_partner], current_length)

    return max(segments.values()) if segments else 0


def combine_metrics(all_metrics):
    """Aggregate per-sequence metrics: sum counts, weight-average LocA,
    recompute derived ratios from the summed counts."""
    combined = {}

    count_metrics = ["Dets", "GT_Dets", "IDs", "GT_IDs", "CLR_TP", "CLR_FN",
                     "IDSW", "Frag", "MT", "PT", "ML", "IDTP", "IDFN", "IDFP"]
    for metric in count_metrics:
        combined[metric] = sum(m[metric] for m in all_metrics.values())

    combined["CLR_Re"] = (combined["CLR_TP"] / combined["GT_Dets"]
                          if combined["GT_Dets"] > 0 else 0.0)
    combined["MTR"] = combined["MT"] / combined["GT_IDs"] if combined["GT_IDs"] > 0 else 0.0
    combined["PTR"] = combined["PT"] / combined["GT_IDs"] if combined["GT_IDs"] > 0 else 0.0
    combined["MLR"] = combined["ML"] / combined["GT_IDs"] if combined["GT_IDs"] > 0 else 0.0

    total_matches = sum(m["CLR_TP"] for m in all_metrics.values())
    if total_matches > 0:
        combined["LocA"] = sum(m["LocA"] * m["CLR_TP"]
                               for m in all_metrics.values()) / total_matches
    else:
        combined["LocA"] = 0.0

    combined["IDR"] = (combined["IDTP"] / (combined["IDTP"] + combined["IDFN"])
                       if (combined["IDTP"] + combined["IDFN"]) > 0 else 0.0)
    combined["IDP"] = (combined["IDTP"] / (combined["IDTP"] + combined["IDFP"])
                       if (combined["IDTP"] + combined["IDFP"]) > 0 else 0.0)
    combined["IDF1"] = (2 * combined["IDTP"]
                        / (2 * combined["IDTP"] + combined["IDFN"] + combined["IDFP"])
                        if (2 * combined["IDTP"] + combined["IDFN"] + combined["IDFP"]) > 0
                        else 0.0)
    return combined


def print_all_metrics(all_metrics):
    """Print Count, CLEAR, Identity, and Derived tables in TrackEval style."""

    logger.info(f"\n{'=' * 80}")
    logger.info("Count:")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Sequence':<30} {'Dets':<12} {'GT_Dets':<12} {'IDs':<12} {'GT_IDs':<12}")
    logger.info(f"{'-' * 80}")
    for seq_name, m in all_metrics.items():
        logger.info(f"{seq_name:<30} {m['Dets']:<12} {m['GT_Dets']:<12} "
                    f"{m['IDs']:<12} {m['GT_IDs']:<12}")

    logger.info(f"\n{'=' * 80}")
    logger.info("CLEAR:")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"{'Sequence':<30} "
        f"{'CLR_Re':<10} {'LocA':<10} {'MTR':<10} {'PTR':<10} {'MLR':<10} "
        f"{'CLR_TP':<10} {'CLR_FN':<10} {'IDSW':<10} {'Frag':<10} "
        f"{'MT':<8} {'PT':<8} {'ML':<8}")
    logger.info(f"{'-' * 150}")
    for seq_name, m in all_metrics.items():
        logger.info(
            f"{seq_name:<30} "
            f"{m['CLR_Re']:<10.3f} {m['LocA']:<10.3f} {m['MTR']:<10.3f} "
            f"{m['PTR']:<10.3f} {m['MLR']:<10.3f} "
            f"{m['CLR_TP']:<10} {m['CLR_FN']:<10} {m['IDSW']:<10} {m['Frag']:<10} "
            f"{m['MT']:<8} {m['PT']:<8} {m['ML']:<8}")

    logger.info(f"\n{'=' * 80}")
    logger.info("Identity:")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Sequence':<30} {'IDF1':<10} {'IDR':<10} {'IDP':<10} "
                f"{'IDTP':<10} {'IDFN':<10} {'IDFP':<10}")
    logger.info(f"{'-' * 100}")
    for seq_name, m in all_metrics.items():
        logger.info(f"{seq_name:<30} {m['IDF1']:<10.3f} {m['IDR']:<10.3f} "
                    f"{m['IDP']:<10.3f} {m['IDTP']:<10} {m['IDFN']:<10} {m['IDFP']:<10}")

    logger.info(f"\n{'=' * 80}")
    logger.info("Derived Metrics:")
    logger.info(f"{'=' * 80}")
    logger.info(f"{'Sequence':<30} {'ID_Ratio':<12} {'IDSW/track':<12} {'Frag/track':<12}")
    logger.info(f"{'-' * 80}")
    for seq_name, m in all_metrics.items():
        gt_ids = m["GT_IDs"]
        id_ratio = m["IDs"] / gt_ids if gt_ids > 0 else 0.0
        idsw_per_track = m["IDSW"] / gt_ids if gt_ids > 0 else 0.0
        frag_per_track = m["Frag"] / gt_ids if gt_ids > 0 else 0.0
        logger.info(f"{seq_name:<30} {id_ratio:<12.2f} "
                    f"{idsw_per_track:<12.2f} {frag_per_track:<12.2f}")

    logger.info(f"\n{'=' * 80}\n")
