"""Shared utilities: paths, logging, configuration, and MOT-format file I/O.

Directory layout (single source of truth for the whole pipeline):
    DATA_DIR     Immutable inputs: images, ground-truth annotations, masks.
    RESULTS_DIR  Derived, regenerable outputs (detections, embeddings, tracks,
                 eval files, visualizations) in a tree mirroring DATA_DIR.
    MODELS_DIR   Trained checkpoints. One detection and one embedding model
                 are shared across all datasets.
"""

import argparse
import json
import logging
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(level=logging.INFO):
    """Configure root logging once. Called from run.py, never from modules."""
    logging.basicConfig(level=level, format="%(message)s", force=True)


def log_section(title, width=60):
    """Log a uniform section banner."""
    logger.info("\n" + "=" * width)
    logger.info(title)
    logger.info("=" * width)


def log_config(config, title="Configuration"):
    """Pretty-print any config dataclass in a uniform format."""
    log_section(title)
    for key, value in vars(config).items():
        logger.info(f"  {key:<28} {value}")
    logger.info("=" * 60)


# ============================================================================
# CLI AND CONFIGURATION
# ============================================================================

COMMANDS = (
    "train-detection", "detect", "train-embedding", "embed",
    "track", "eval", "visualize",
    "extract-outlines", "georeference", "circulation",
)

# command -> (module, config class). Imported lazily so e.g. an eval run
# never loads torch/timm.
CONFIG_CLASSES = {
    "train-detection": ("detection", "IcebergDetectionConfig"),
    "detect": ("detection", "IcebergDetectionConfig"),
    "train-embedding": ("embedding", "IcebergEmbeddingsConfig"),
    "embed": ("embedding", "IcebergEmbeddingsConfig"),
    "track": ("tracking", "IcebergTrackingConfig"),
    "eval": ("evaluation", "EvalConfig"),
    "visualize": ("visualize", "VisualizationConfig"),
    "extract-outlines": ("outlines", "OutlineExtractionConfig"),
    "georeference": ("georeference", "GeoreferenceConfig"),
    "circulation": ("circulation", "CirculationConfig"),
}


def parse_cli_args():
    """Parse `<command> [cfg=file.yaml] [key=value ...]`.

    Returns:
        (command, cfg_file, dotlist): command name, YAML path (or None), and
        the remaining key=value overrides as a list of strings.
    """
    parser = argparse.ArgumentParser(description="Iceberg tracking pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in COMMANDS:
        sub.add_parser(cmd)
    args, unknown = parser.parse_known_args()

    dotlist = [a for a in unknown if "=" in a]
    cfg_file = None
    for arg in list(dotlist):
        if arg.startswith("cfg="):
            cfg_file = arg.split("=", 1)[1]
            dotlist.remove(arg)
    return args.cmd, cfg_file, dotlist


def load_config(command, cfg_file=None, overrides=None):
    """Build a validated config dataclass for a pipeline command.

    Merge priority: dataclass defaults < YAML file < overrides. The dataclass
    is the single source of truth for defaults and types; unknown keys and
    type mismatches raise immediately.

    Args:
        command: One of COMMANDS (selects the config dataclass).
        cfg_file: Optional YAML path (relative to PROJECT_ROOT) passed via
            `cfg=...` for override sets you don't want to retype; raises if it
            doesn't exist. Usually omitted -- dataclass defaults plus CLI
            key=value overrides are enough.
        overrides: dict (Python/notebook use) or list of "key=value" strings
            (CLI use).

    Returns:
        Config dataclass instance for the given command.
    """
    import importlib

    if command not in CONFIG_CLASSES:
        raise ValueError(
            f"Unknown command '{command}'. Choose from: {', '.join(COMMANDS)}"
        )
    module_name, class_name = CONFIG_CLASSES[command]
    config_class = getattr(importlib.import_module(module_name), class_name)

    cfg = OmegaConf.structured(config_class)

    if cfg_file:
        cfg_path = PROJECT_ROOT / cfg_file
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg_path))

    if overrides:
        if isinstance(overrides, dict):
            cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
        else:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    if OmegaConf.is_missing(cfg, "dataset") or not cfg.dataset:
        raise ValueError(
            "Dataset not specified. Provide it via the command line "
            "(dataset=ekas-hill), the YAML config, or in Python "
            "(load_config(..., overrides={'dataset': 'ekas-hill'}))."
        )
    return OmegaConf.to_object(cfg)


def save_config_snapshot(config, out_path):
    """Save the fully resolved config that produced an output, for provenance.

    Written next to the output files it describes (e.g. tracking/config.yaml),
    including a timestamp and the current git commit if available.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = OmegaConf.to_container(OmegaConf.structured(config))
    snapshot["_meta"] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    OmegaConf.save(OmegaConf.create(snapshot), out_path)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def resolve_device(device=None):
    """Return the configured torch device, auto-detecting when None."""
    import torch
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# PATH MANAGEMENT
# ============================================================================

def get_sequences(dataset, run_name=None):
    """Map every sequence in a dataset to its input and output file paths.

    Inputs (images, gt.txt) live under DATA_DIR; all derived files live under
    RESULTS_DIR in a mirrored tree. When `run_name` is set, the tracking,
    georeference, and circulation outputs each go to a <stage>/<run_name>/
    subfolder so variants can coexist for ablations (these stages all derive
    from one tracking run's track IDs; outlines.npz stays inside the tracking
    directory for the same reason). Visualizations are written next to the
    annotations they show: <stage_dir>/images/ and <stage_dir>/videos/.

    Handles both single-sequence datasets (images/ at the dataset root) and
    multi-sequence datasets (one subdirectory per sequence).

    Args:
        dataset: Dataset name (directory under DATA_DIR), e.g. "ekas-hill".
        run_name: Optional tracking-run identifier.

    Returns:
        dict: {sequence_name: {path_key: Path}} with keys: images,
        ground_truth, camera, gt_dir, gt_embeddings, detections, det_config,
        det_embeddings, tracking, track_config, eval, metrics, outlines,
        outlines_config, georeference, circulation.
    """
    base_path = DATA_DIR / dataset
    camera_path = DATA_DIR / Path(dataset).parts[0] / "camera.json"
    if (base_path / "images").exists():
        sequence_dirs = [base_path]
    else:
        sequence_dirs = sorted(p for p in base_path.iterdir() if p.is_dir())

    sequences = {}
    for sequence_dir in sequence_dirs:
        images_dir = sequence_dir / "images"
        if not images_dir.exists():
            logger.debug(f"No images/ directory in {sequence_dir}, skipping")
            continue

        results_dir = RESULTS_DIR / sequence_dir.relative_to(DATA_DIR)
        run = run_name if run_name else ""
        tracking_dir = results_dir / "tracking" / run

        sequences[sequence_dir.name] = {
            "images": images_dir,
            "ground_truth": sequence_dir / "ground_truth" / "gt.txt",
            "gt_dir": results_dir / "ground_truth",
            "gt_embeddings": results_dir / "ground_truth" / "embeddings.pt",
            "detections": results_dir / "detections" / "det.txt",
            "det_config": results_dir / "detections" / "config.yaml",
            "det_embeddings": results_dir / "detections" / "embeddings.pt",
            "tracking": tracking_dir / "track.txt",
            "track_config": tracking_dir / "config.yaml",
            "eval": tracking_dir / "eval.txt",
            "metrics": tracking_dir / "metrics.json",
            "camera": camera_path,
            "outlines": tracking_dir / "outlines.npz",
            "outlines_config": tracking_dir / "outlines_config.yaml",
            "georeference": results_dir / "georeference" / run,
            "circulation": results_dir / "circulation" / run,
        }
    return sequences


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def get_image_ext(image_dir):
    """Return the extension (without dot) of the first image in a directory.

    Skips non-image files such as .DS_Store or hidden files.
    """
    for path in sorted(Path(image_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path.suffix.lstrip(".")
    raise FileNotFoundError(f"No image files found in {image_dir}")


def first_image(image_dir):
    """Return the path of the first image in a directory (non-images skipped)."""
    ext = get_image_ext(image_dir)
    for path in sorted(Path(image_dir).iterdir()):
        if path.is_file() and path.suffix.lstrip(".") == ext:
            return path
    raise FileNotFoundError(f"No image files found in {image_dir}")


# ============================================================================
# MOT-FORMAT FILE I/O
# ============================================================================

def normalize_frame(frame):
    """Zero-pad numeric frame identifiers to six digits; pass others through."""
    try:
        return f"{int(frame):06d}"
    except (ValueError, TypeError):
        return frame


def emb_key(frame, iceberg_id):
    """Key format used in embeddings.pt dictionaries: '{frame}_{id}'."""
    return f"{normalize_frame(frame)}_{iceberg_id}"


def write_mot_file(rows, path):
    """Write rows in MOTChallenge format and sort the file.

    Args:
        rows: Iterable of (frame, id, x, y, w, h, conf) tuples.
        path: Output file path; parent directories are created.

    Line format: frame,id,x,y,w,h,conf,1,-1,-1
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for frame, obj_id, x, y, w, h, conf in rows:
            f.write(f"{frame},{obj_id},{x},{y},{w},{h},{conf},1,-1,-1\n")
    sort_file(path)


def sort_file(txt_file):
    """Sort a MOT annotation file in-place by object ID, then by frame.

    ID-major ordering keeps all entries of one track together, which is the
    layout downstream consumers of these files expect.
    """
    with open(txt_file, "r") as f:
        lines = f.readlines()

    parsed = []
    for line in lines:
        img_name, line_rest = line.split(",", 1)
        img_name = int(img_name) if img_name.isdigit() else img_name
        iceberg_id, line_rest = line_rest.split(",", 1)
        new_line = f"{img_name},{iceberg_id},{line_rest}"
        parsed.append((img_name, iceberg_id, new_line))

    parsed.sort(key=lambda x: (int(x[1]), x[0]))

    with open(txt_file, "w") as f:
        for _, _, line in parsed:
            f.write(line)


def parse_annotations(txt_file):
    """Parse a MOT annotation file into flat detection dicts.

    Malformed lines are logged and skipped.

    Returns:
        list[dict]: One dict per line with keys frame (zero-padded str),
        id (int), bb_left, bb_top, bb_width, bb_height (float).
    """
    detections = []
    with open(txt_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.strip().split(",")
            try:
                detections.append({
                    "frame": normalize_frame(parts[0]),
                    "id": int(parts[1]),
                    "bb_left": float(parts[2]),
                    "bb_top": float(parts[3]),
                    "bb_width": float(parts[4]),
                    "bb_height": float(parts[5]),
                })
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"Could not parse line {line_num} in {txt_file}, skipping "
                    f"('{line.strip()}': {e})"
                )
    return detections


def load_icebergs_by_frame(det_file):
    """Load a MOT file into a nested {frame: {iceberg_id: data}} dictionary.

    Each iceberg entry contains id, bbox (x, y, w, h), conf, and the three
    trailing MOT fields (x, y, z). Frames are sorted for deterministic
    iteration.
    """
    icebergs_by_frame = defaultdict(dict)
    with open(det_file, "r") as f:
        for line in f:
            frame, id_, left, top, width, height, conf, x, y, z = line.strip().split(",")
            frame = normalize_frame(frame)
            icebergs_by_frame[frame][int(id_)] = {
                "id": int(id_),
                "bbox": (float(left), float(top), float(width), float(height)),
                "conf": float(conf),
                "x": int(x),
                "y": int(y),
                "z": int(z),
            }
    return dict(sorted(icebergs_by_frame.items()))


# ============================================================================
# GEOMETRY AND MATCH EXTRACTION
# ============================================================================

def bbox_center(bbox):
    """Return the (x, y) center of an (x, y, w, h) bounding box."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def calculate_iou(box1, box2):
    """Intersection over Union of two [xmin, ymin, xmax, ymax] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def extract_candidates(txt_path):
    """Map each frame number (int) to the list of iceberg IDs it contains."""
    candidates = defaultdict(list)
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            candidates[int(parts[0])].append(int(parts[1]))
    return dict(sorted(candidates.items()))


def extract_matches(candidates):
    """Find iceberg IDs that appear in consecutive frames.

    Args:
        candidates: {frame_num: [iceberg_ids]} from extract_candidates().

    Returns:
        list[dict]: Matches with keys id, frame, next_frame.
    """
    matches = []
    frame_list = sorted(candidates.keys())
    for current_frame, next_frame in zip(frame_list[:-1], frame_list[1:]):
        next_ids = set(candidates[next_frame])
        for iceberg_id in candidates[current_frame]:
            if iceberg_id in next_ids:
                matches.append({
                    "id": iceberg_id,
                    "frame": current_frame,
                    "next_frame": next_frame,
                })
    return matches


def save_metrics_json(metrics, path):
    """Dump a metrics dict as JSON (numpy scalars converted to floats)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)


def load_mot_tracks(path):
    """Load a MOT file into per-detection numpy arrays (for vectorised analysis).

    Counterpart to load_icebergs_by_frame for code that operates on 10^5-10^6
    detections at once. The frame column may be an integer or a filename such
    as '_MG_6953'; in the latter case the trailing digits become the frame ID.

    Returns:
        dict: frame, id (int arrays), left, top, width, height, conf.
    """
    import re

    try:
        data = np.loadtxt(path, delimiter=",")
    except ValueError:
        # First column contains strings; extract the trailing digits.
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                match = re.search(r"(\d+)$", parts[0].strip())
                if not match:
                    raise ValueError(
                        f"Cannot extract a frame number from '{parts[0]}' in {path}.")
                parts[0] = match.group(1)
                rows.append([float(x) for x in parts])
        data = np.array(rows)
        logger.info(f"Non-standard frame IDs in {path}; using trailing digits.")

    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 6:
        raise ValueError(f"{path}: expected >=6 MOT columns, found {data.shape[1]}.")
    conf = data[:, 6] if data.shape[1] >= 7 else np.ones(len(data))
    return {
        "frame": data[:, 0].astype(int),
        "id": data[:, 1].astype(int),
        "left": data[:, 2],
        "top": data[:, 3],
        "width": data[:, 4],
        "height": data[:, 5],
        "conf": conf,
    }


def build_frame_to_image(image_dir, pattern="*"):
    """Map integer frame number -> image filename via trailing filename digits.

    Inverse of the digit extraction in load_mot_tracks; works both for
    zero-padded frames (000123.jpg) and camera names (_MG_6953.JPG). Files
    without trailing digits are skipped.
    """
    import re

    mapping = {}
    for path in sorted(Path(image_dir).glob(pattern)):
        match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
        if match:
            mapping[int(match.group(1))] = path.name
    logger.info(f"build_frame_to_image: {len(mapping)} images in {image_dir}")
    return mapping


# ============================================================================
# GLIMPSE (photogrammetry) IMPORT SHIM
# ============================================================================

def get_glimpse():
    """Import glimpse with stand-in modules for its heavy dependencies.

    glimpse eagerly imports GDAL, OpenCV, sharedmem, lmfit, and piexif at load
    time; the projection helpers used here (Camera.from_json, uv_to_xyz,
    intersect_rays_plane) never call into them. Registering stub modules first
    lets unmodified glimpse import without those packages installed. Uses
    sys.modules.setdefault, so genuinely installed libraries are left alone.
    Safe ONLY while no code path calls tracking/calibration/raster I/O.
    """
    import sys
    import types

    class _AnyMeta(type):  # attribute access yields another stub type
        def __getattr__(cls, name):
            return _AnyMeta(f"{cls.__name__}.{name}", (), {})

    def _stub_module(fullname):
        module = types.ModuleType(fullname)
        module.__path__ = []  # mark as package so submodule imports resolve
        module.__getattr__ = lambda attr, _n=fullname: _AnyMeta(f"{_n}.{attr}", (), {})
        return module

    for name in ("osgeo", "osgeo.gdal", "osgeo.gdal_array", "osgeo.ogr",
                 "osgeo.osr", "sharedmem", "lmfit", "lmfit.parameter",
                 "piexif", "cv2"):
        sys.modules.setdefault(name, _stub_module(name))

    import glimpse
    return glimpse
