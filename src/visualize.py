"""Visualization: annotated frames/videos/GIFs and publication figures.

Frame annotation (visualize command): draws IDs, boxes, and the stored
outlines from extract-outlines onto the frames of any annotation source
(ground_truth, detections, tracking, eval) and renders a video and/or an
animated GIF. Outlines are read from outlines.npz rather than re-segmented,
so the figures show exactly the geometry the analysis used. Outputs under
results/<seq>/visualization/.

Publication figures (called by the georeference and circulation commands):
UTM/pixel trajectory maps (plain and drift-speed-coloured), the standalone
speed legend, and the circulation vector/streamline panels, all on a shared
serif/bold style so the paper figures match.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PIL import Image
from tqdm import tqdm

from helpers import (
    first_image, get_image_ext, get_sequences, load_icebergs_by_frame,
    log_config, log_section, save_config_snapshot,
)
from outlines import load_outlines, outline_key
from georeference import (
    UTM_EPSG, bounding_box_points, compute_track_speeds, segment_speeds,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class VisualizationConfig:
    """Parameters for the visualize command."""

    dataset: str = "???"
    run_name: Optional[str] = None      # Visualize tracking/<run_name>/ results

    # Annotation source: ground_truth | detections | tracking | eval
    annotation_source: str = "tracking"

    # Frame range
    start_index: int = 0
    seq_length_limit: Optional[int] = None
    show_images: bool = False

    # Size of the images, video and GIF as a percentage of the source
    # resolution; None = 100, lowered by make_gif (see Visualizer.GIF_QUALITY)
    quality: Optional[int] = None

    # Annotation elements. Outlines/masks come from outlines.npz
    # (extract-outlines) and are keyed by track ID, so they are available for
    # the tracking and eval sources.
    draw_ids: bool = True
    draw_boxes: bool = True
    draw_outlines: bool = False
    draw_masks: bool = False            # Filled outlines, alpha-blended
    outline_thickness: int = 2

    # Single colour for all icebergs as hex "#RRGGBB"; None = per-ID colours
    uniform_color: Optional[str] = None

    # Video and GIF, both compiled from the annotated images at fps
    make_video: bool = False
    make_gif: bool = False
    fps: int = 7


def _parse_color(color):
    """Hex string "#RRGGBB" -> OpenCV BGR tuple; None passes through."""
    if color is None:
        return None
    s = str(color).lstrip("#")
    if len(s) != 6:
        raise ValueError(f"uniform_color must be '#RRGGBB', got '{color}'.")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    return (b, g, r)


# ============================================================================
# FRAME ANNOTATION AND VIDEO
# ============================================================================

class Visualizer:
    """Annotates iceberg frames and renders videos/GIFs for one dataset."""

    VALID_SOURCES = ("ground_truth", "detections", "tracking", "eval")
    OUTLINE_SOURCES = ("tracking", "eval")  # outlines are keyed by track ID
    MASK_ALPHA = 0.5

    # Default quality when make_gif is set: GIF is a palette format, and
    # full-resolution frames run to gigabytes
    GIF_QUALITY = 20

    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.annotation_source = config.annotation_source
        self._uniform_color = _parse_color(config.uniform_color)
        self._colormap = {}  # Consistent per-ID colours across frames
        self._annotated = {}  # sequence -> frames written, for video/GIF

        if self.annotation_source not in self.VALID_SOURCES:
            raise ValueError(
                f"Invalid annotation source '{self.annotation_source}'. "
                f"Must be one of: {', '.join(self.VALID_SOURCES)}")

        quality = config.quality
        if quality is None:
            quality = self.GIF_QUALITY if config.make_gif else 100
        if not 0 < quality <= 100:
            raise ValueError(f"quality must be in (0, 100], got {quality}.")
        self._scale = quality / 100

        log_config(config, title="Visualization Configuration")

    def run(self):
        """Annotate all sequences and (optionally) render videos and GIFs."""
        self.annotate_icebergs()
        if self.config.make_video:
            self.render_video()
        if self.config.make_gif:
            self.render_gif()

    # ------------------------------------------------------------------ #
    # Annotation
    # ------------------------------------------------------------------ #

    def annotate_icebergs(self):
        """Annotate the selected frames of every sequence."""
        sequences = get_sequences(self.config.dataset,
                                  run_name=self.config.run_name)
        logger.info(f"Rendering frames at {self._scale:.0%} of the source "
                    "resolution")

        for sequence_name, paths in sequences.items():
            log_section(f"Annotating sequence: {sequence_name}")

            if not paths[self.annotation_source].exists():
                logger.warning(f"No {self.annotation_source} annotations at "
                               f"{paths[self.annotation_source]}, skipping")
                continue

            icebergs_by_frame, image_ext = self._select_frames(paths)
            outlines = self._load_sequence_outlines(paths)

            out_dir, _ = self._source_dirs(paths)
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving annotated images to: {out_dir}")

            n_missing_outlines = 0
            total = sum(len(icebergs) for icebergs in icebergs_by_frame.values())
            progress = tqdm(total=total, desc="Annotating icebergs",
                            unit="iceberg")

            for frame_name, icebergs in icebergs_by_frame.items():
                image_path = paths["images"] / f"{frame_name}.{image_ext}"
                img = cv2.imread(str(image_path))
                if img is None:
                    raise FileNotFoundError(f"Image not found: {image_path}")
                img = self._scale_frame(img)

                for iceberg_id, iceberg_data in icebergs.items():
                    progress.update(1)
                    outline = None
                    if outlines is not None:
                        outline = outlines.get(outline_key(frame_name,
                                                           iceberg_id))
                        if outline is None:
                            n_missing_outlines += 1
                    self._draw_iceberg(img, iceberg_id, iceberg_data, outline)

                cv2.imwrite(str(out_dir / f"{frame_name}.jpg"), img)
                self._annotated.setdefault(sequence_name, []).append(
                    f"{frame_name}.jpg")
                if self.config.show_images:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title(frame_name)
                    plt.axis("off")
                    plt.show()

            progress.close()
            if n_missing_outlines:
                logger.warning(f"{n_missing_outlines} annotations had no "
                               f"stored outline")
            save_config_snapshot(self.config, out_dir / "config.yaml")

    def _scale_frame(self, img):
        """Resize a frame to the configured quality. Annotations are drawn
        afterwards, at the matching scale."""
        if self._scale == 1:
            return img
        size = (max(1, round(img.shape[1] * self._scale)),
                max(1, round(img.shape[0] * self._scale)))
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def _draw_iceberg(self, img, iceberg_id, iceberg_data, outline):
        """Draw the configured annotation elements for one iceberg.

        Coordinates and line/font weights are scaled together, so a reduced
        quality looks like the full-resolution figure shrunk -- the icebergs
        keep their share of the frame instead of disappearing under labels.
        Weights are floored so no stroke is lost entirely.
        """
        color = self._get_object_color(iceberg_id)
        x1, y1, w, h = np.asarray(iceberg_data["bbox"], dtype=float) * self._scale

        if outline is not None:
            pts = np.round(np.asarray(outline, dtype=float) * self._scale
                           ).astype(np.int32).reshape(-1, 1, 2)
            if self.config.draw_masks:
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                region = mask > 0
                img[region] = ((1 - self.MASK_ALPHA) * img[region]
                               + self.MASK_ALPHA * np.array(color)
                               ).astype(np.uint8)
            if self.config.draw_outlines:
                cv2.polylines(img, [pts], isClosed=True, color=color,
                              thickness=self._scaled(
                                  self.config.outline_thickness),
                              lineType=cv2.LINE_AA)

        if self.config.draw_boxes:
            cv2.rectangle(img, (int(x1), int(y1)),
                          (int(x1 + w), int(y1 + h)), color, self._scaled(3))
        if self.config.draw_ids:
            text_y = max(int(y1) - self._scaled(10), self._scaled(20))
            cv2.putText(img, str(iceberg_id), (int(x1), text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.3, 0.7 * self._scale),  # 0.3: cv2's legible floor
                        color, self._scaled(2))

    def _scaled(self, pixels, minimum=1):
        """A pixel weight or offset at the output scale, never below 1 px."""
        return max(minimum, round(pixels * self._scale))

    def _load_sequence_outlines(self, paths):
        """Load stored outlines when outline/mask drawing is enabled.

        Returns None (with a warning) when drawing is disabled, the source's
        IDs cannot match outline keys, or no outlines.npz exists yet.
        """
        if not (self.config.draw_outlines or self.config.draw_masks):
            return None
        if self.annotation_source not in self.OUTLINE_SOURCES:
            logger.warning(
                f"Outlines are keyed by track ID and cannot be matched to "
                f"'{self.annotation_source}' annotations; drawing without them")
            return None
        if not paths["outlines"].exists():
            logger.warning(f"No outlines at {paths['outlines']} "
                           f"(run extract-outlines); drawing without them")
            return None
        return load_outlines(paths["outlines"])

    def _select_frames(self, paths):
        """Apply the frame range and load the matching annotations.

        Returns:
            ({frame_name: icebergs}, image_ext) restricted to frames that
            exist and have annotations.
        """
        image_ext = get_image_ext(paths["images"])
        images = sorted(f.name for f in paths["images"].iterdir()
                        if f.name.endswith(image_ext))

        if self.config.seq_length_limit is None:
            end_frame = len(images)
        else:
            end_frame = min(self.config.start_index
                            + self.config.seq_length_limit, len(images))
        images = images[self.config.start_index:end_frame]
        logger.info(f"Selected {len(images)} frames "
                    f"(from index {self.config.start_index})")

        icebergs_by_frame = load_icebergs_by_frame(
            paths[self.annotation_source])
        selection = {}
        missing = 0
        for image in images:
            frame_name = image.split("." + image_ext)[0]
            if frame_name in icebergs_by_frame:
                selection[frame_name] = icebergs_by_frame[frame_name]
            else:
                missing += 1
        if missing:
            logger.warning(f"Skipped {missing} frames without "
                           f"{self.annotation_source} annotations")
        return selection, image_ext

    def _get_object_color(self, object_id):
        """Uniform colour if configured, else a cached per-ID seeded colour."""
        if self._uniform_color is not None:
            return self._uniform_color
        if object_id not in self._colormap:
            np.random.seed(object_id)
            self._colormap[object_id] = tuple(
                map(int, np.random.randint(0, 255, 3)))
        return self._colormap[object_id]

    def _source_dirs(self, paths, media="videos"):
        """(images_dir, media_dir) inside the directory that owns the
        annotations being visualized; run_name scoping comes from the paths
        themselves. eval shares the tracking directory with the tracking
        source, so its frames go to images_eval/."""
        owner = {
            "detections": paths["detections"].parent,
            "ground_truth": paths["gt_dir"],
            "tracking": paths["tracking"].parent,
            "eval": paths["tracking"].parent,
        }[self.annotation_source]
        images_name = "images_eval" if self.annotation_source == "eval" else "images"
        return owner / images_name, owner / media

    # ------------------------------------------------------------------ #
    # Video and GIF
    # ------------------------------------------------------------------ #

    def _annotated_images(self, sequence_name, paths, media):
        """(image_dir, media_dir, frame names) for one sequence, or None when
        it has no annotated frames to compile.

        Only the frames this run annotated are compiled. The directory can
        also hold frames from earlier runs -- outside the configured frame
        range, or at a different quality, which a video or GIF cannot mix.
        Everything in it is used only when the render is called on its own,
        without annotate_icebergs.
        """
        image_dir, out_dir = self._source_dirs(paths, media)
        if not image_dir.exists():
            logger.warning(f"No annotated images at {image_dir}, skipping")
            return None

        images = self._annotated.get(sequence_name)
        on_disk = sorted(f.name for f in image_dir.iterdir()
                         if f.name.lower().endswith("jpg"))
        if images is None:
            images = on_disk
        elif len(on_disk) > len(images):
            logger.warning(f"{image_dir} also holds {len(on_disk) - len(images)}"
                           f" frame(s) from earlier runs; skipping those")
        if not images:
            logger.warning(f"No annotated images in {image_dir}, skipping")
            return None
        return image_dir, out_dir, images

    def render_video(self):
        """Compile the annotated images of every sequence into an MP4."""
        logger.info(f"\nRendering videos at {self.config.fps} fps")
        sequences = get_sequences(self.config.dataset,
                                  run_name=self.config.run_name)

        for sequence_name, paths in sequences.items():
            logger.info(f"\nProcessing sequence: {sequence_name}")

            selected = self._annotated_images(sequence_name, paths, "videos")
            if selected is None:
                continue
            image_dir, video_dir, images = selected

            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{self.annotation_source}.mp4"

            first_image = cv2.imread(str(image_dir / images[0]))
            if first_image is None:
                raise FileNotFoundError(f"Could not read {image_dir / images[0]}")
            height, width = first_image.shape[:2]
            logger.info(f"Video: {width}x{height}, {len(images)} frames")

            # Codec fallback chain (H.264 -> MPEG-4 -> Xvid)
            video_writer = None
            for codec, ext in (("avc1", ".mp4"), ("mp4v", ".mp4"),
                               ("XVID", ".avi")):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_path = str(video_path).replace(".mp4", ext)
                    writer = cv2.VideoWriter(test_path, fourcc,
                                             self.config.fps, (width, height))
                    if writer.isOpened():
                        video_writer = writer
                        video_path = Path(test_path)
                        logger.info(f"Using codec: {codec}")
                        break
                    writer.release()
                except Exception as e:
                    logger.warning(f"Codec {codec} failed: {e}")
            if video_writer is None:
                raise RuntimeError("Failed to initialize any video codec")

            failed_frames = 0
            for frame_name in tqdm(images, desc="Writing video", unit="frame"):
                img = cv2.imread(str(image_dir / frame_name))
                if img is None:
                    failed_frames += 1
                    continue
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                video_writer.write(img)
            video_writer.release()

            if failed_frames:
                logger.warning(f"Failed to write {failed_frames} frames")
            logger.info(f"Video saved to: {video_path}")

    def render_gif(self):
        """Compile the annotated images of every sequence into an animated GIF.

        The frames were already written at the configured quality, so the GIF
        only adds palette reduction: each frame is opened one at a time and
        becomes an 8-bit palette image (one byte per pixel instead of three),
        and consecutive frames are stored as transparent deltas
        (optimize=True), which is nearly free on the static parts of a
        time-lapse. Pillow holds every frame until the file is written, so
        quality is what keeps both the file and the peak memory small.
        """
        # 20 ms: below that most viewers substitute a default frame duration
        duration_ms = max(20, round(1000.0 / self.config.fps))
        logger.info(f"\nRendering GIFs at {self.config.fps} fps")
        sequences = get_sequences(self.config.dataset,
                                  run_name=self.config.run_name)

        for sequence_name, paths in sequences.items():
            logger.info(f"\nProcessing sequence: {sequence_name}")

            selected = self._annotated_images(sequence_name, paths, "gifs")
            if selected is None:
                continue
            image_dir, gif_dir, images = selected

            gif_dir.mkdir(parents=True, exist_ok=True)
            gif_path = gif_dir / f"{self.annotation_source}.gif"

            with Image.open(image_dir / images[0]) as probe:
                width, height = probe.size
            logger.info(f"GIF: {width}x{height}, {len(images)} frames")

            frames = (self._load_gif_frame(image_dir / name)
                      for name in tqdm(images, desc="Writing GIF",
                                       unit="frame"))
            first = next(frames)
            first.save(gif_path, save_all=True, append_images=frames,
                       duration=duration_ms, loop=0, optimize=True)

            size_mb = gif_path.stat().st_size / 1e6
            logger.info(f"GIF saved to: {gif_path} ({size_mb:.1f} MB)")

    @staticmethod
    def _load_gif_frame(image_path):
        """One annotated frame as an 8-bit palette image. 128 adaptive colours
        are indistinguishable from GIF's maximum 256 on these scenes."""
        with Image.open(image_path) as src:
            img = src.convert("RGB")
        return img.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


# ============================================================================
# PUBLICATION STYLE AND MAP ELEMENTS
# ============================================================================

SPEED_LABEL = r"Drift speed (m min$^{-1}$)"
AXIS_LABEL_SIZE = 22  # Publication-scale label size


def setup_publication_style():
    """Serif, bold rcParams shared by all trajectory/circulation figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelweight": "bold",
        "font.size": 16,
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def draw_km_scale_bar(ax, length_m=1000.0, label="1 km"):
    """Metric scale bar in the lower-left (UTM axes only, where one axis unit
    is one metre). Positioned relative to the current axis limits."""
    x0, x1 = ax.get_xlim()
    y0, y1 = sorted(ax.get_ylim())
    span_x, span_y = x1 - x0, y1 - y0
    bx = x0 + 0.05 * span_x
    by = y0 + 0.06 * span_y
    height_m = 0.012 * span_y
    ax.add_patch(Rectangle((bx, by), length_m, height_m,
                           facecolor="white", edgecolor="black",
                           linewidth=1.5, zorder=100))
    for x, text in ((bx, "0"), (bx + length_m, label)):
        ax.text(x, by + height_m + 0.012 * span_y, text,
                ha="center", va="bottom", fontsize=AXIS_LABEL_SIZE,
                fontweight="bold", color="black", zorder=101)


def add_basemap(ax, utm_epsg=UTM_EPSG, alpha=0.7, zoom=None,
                attribution_size=8):
    """OpenStreetMap basemap with consistent attribution styling; tile or
    network failures are downgraded to a warning."""
    import contextily as ctx
    kwargs = dict(crs=utm_epsg, source=ctx.providers.OpenStreetMap.Mapnik,
                  alpha=alpha, attribution_size=attribution_size)
    if zoom is not None:
        kwargs["zoom"] = zoom
    try:
        ctx.add_basemap(ax, **kwargs)
    except Exception as exc:
        logger.warning(f"Basemap unavailable ({exc}); plotting without it.")


def _speed_norm(all_speeds, clip=(2.0, 98.0), vmin=None, vmax=None):
    """Colour normalisation shared across the velocity figures: robust
    percentiles of the observed speeds unless vmin/vmax are given."""
    if all_speeds.size:
        lo, hi = np.percentile(all_speeds, clip)
    else:
        lo, hi = 0.0, 1.0
    vmin = lo if vmin is None else vmin
    vmax = hi if vmax is None else vmax
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _coloured_segments(ax, xy, speeds, cmap, norm, linewidth):
    """Add one track as a speed-coloured LineCollection."""
    points = np.asarray(xy, dtype=float).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.asarray(speeds, dtype=float))
    lc.set_linewidth(linewidth)
    lc.set_capstyle("round")
    ax.add_collection(lc)
    return lc


def _prepare_background(image_path, fade=0.5, fade_to="white", saturation=0.0):
    """Load a frame and mute it so coloured tracks stand out: `saturation`
    interpolates grayscale <-> original colour, then `fade` blends toward
    white or black."""
    img = np.asarray(plt.imread(image_path), dtype=float)
    if img.max() > 1.0:
        img = img / 255.0
    rgb = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img[..., :3]
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])  # Rec. 709 luminance
    gray = np.stack([lum] * 3, axis=-1)
    saturation = float(np.clip(saturation, 0.0, 1.0))
    desaturated = gray * (1.0 - saturation) + rgb * saturation
    fade = float(np.clip(fade, 0.0, 1.0))
    target = 1.0 if fade_to == "white" else 0.0
    return np.clip(desaturated * (1.0 - fade) + target * fade, 0.0, 1.0)


def _save_figure(fig, out_path, dpi=150):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# TRAJECTORY FIGURES (called by the georeference command)
# ============================================================================

def plot_track_map(tracks_utm, out_path, basemap=True, utm_epsg=UTM_EPSG):
    """Plain UTM trajectory map (one line per iceberg)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    for xy in tracks_utm.values():
        ax.plot(xy[:, 1], xy[:, 2], "-", linewidth=1)
    if basemap:
        add_basemap(ax, utm_epsg=utm_epsg)
    _save_figure(fig, out_path)
    logger.info(f"Wrote {out_path}")


def plot_tracks_pixel(tracks, image_path, out_path, linewidth=1.0, alpha=0.8):
    """Trajectories in IMAGE space over a time-lapse frame.

    Sanity-check companion to the UTM map: a track that is smooth in pixels
    but distorts in UTM points to a georeferencing issue, not a tracking one.
    """
    image = plt.imread(image_path)
    centre, _, _ = bounding_box_points(tracks)

    fig, ax = plt.subplots(
        figsize=(16, 16.0 * image.shape[0] / image.shape[1]))
    ax.imshow(image)

    n_tracks = 0
    for berg in np.unique(tracks["id"]):
        mask = tracks["id"] == berg
        order = np.argsort(tracks["frame"][mask])
        uv = centre[mask][order]
        if len(uv) < 2:
            continue
        ax.plot(uv[:, 0], uv[:, 1], "-", linewidth=linewidth, alpha=alpha)
        n_tracks += 1

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Image frame: origin top-left, y down
    _save_figure(fig, out_path)
    logger.info(f"Wrote {out_path} ({n_tracks} trajectories)")


def plot_track_map_velocity(tracks_utm, out_path, frame_interval_min=2.0,
                            cmap="plasma", basemap=True, utm_epsg=UTM_EPSG,
                            linewidth=1.0, vmin=None, vmax=None,
                            colorbar=True, scalebar_length_m=1000.0,
                            scalebar_label="1 km", dpi=300):
    """UTM trajectory map with each segment coloured by drift speed."""
    setup_publication_style()
    _, all_speeds = compute_track_speeds(tracks_utm, frame_interval_min)
    norm = _speed_norm(all_speeds, vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    n_tracks = 0
    for berg, arr in tracks_utm.items():
        order = np.argsort(arr[:, 0])
        frames = arr[order, 0]
        xy = arr[order, 1:3]
        if len(xy) < 2:
            continue
        speeds = segment_speeds(frames, xy, frame_interval_min)
        _coloured_segments(ax, xy, speeds, cmap_obj, norm, linewidth)
        n_tracks += 1

    ax.autoscale()  # LineCollections do not autoscale the view
    if basemap:
        add_basemap(ax, utm_epsg=utm_epsg)
    ax.set_xticks([])
    ax.set_yticks([])
    if scalebar_length_m and scalebar_length_m > 0:
        draw_km_scale_bar(ax, length_m=scalebar_length_m,
                          label=scalebar_label)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, extend="max")
        cbar.set_label(SPEED_LABEL, fontsize=AXIS_LABEL_SIZE,
                       fontweight="bold")
    _save_figure(fig, out_path, dpi=dpi)
    logger.info(f"Wrote {out_path} ({n_tracks} trajectories)")


def plot_tracks_pixel_velocity(tracks, tracks_utm, image_path, out_path,
                               frame_interval_min=2.0, cmap="plasma",
                               linewidth=1.0, image_fade=0.5,
                               image_fade_to="white", image_saturation=0.0,
                               vmin=None, vmax=None, colorbar=True, dpi=300):
    """Image-space trajectories coloured by the GEOREFERENCED drift speed.

    Each pixel segment reuses its UTM segment's m/min speed (matched per
    iceberg by starting frame), so the colours mean the same physical speed
    as the UTM figure. No scale bar: the oblique view has no single
    pixel-to-metre ratio. The background frame is muted for legibility.
    """
    setup_publication_style()
    image = _prepare_background(image_path, fade=image_fade,
                                fade_to=image_fade_to,
                                saturation=image_saturation)
    centre, _, _ = bounding_box_points(tracks)

    speeds_by_id, all_speeds = compute_track_speeds(tracks_utm,
                                                    frame_interval_min)
    norm = _speed_norm(all_speeds, vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(
        figsize=(16, 16.0 * image.shape[0] / image.shape[1]))
    ax.imshow(image)

    n_tracks = 0
    for berg in np.unique(tracks["id"]):
        mask = tracks["id"] == berg
        order = np.argsort(tracks["frame"][mask])
        frames = tracks["frame"][mask][order]
        uv = centre[mask][order]
        if len(uv) < 2:
            continue
        seg_map = speeds_by_id.get(int(berg), {})
        speeds = np.array([seg_map.get(int(frames[i]), np.nan)
                           for i in range(len(frames) - 1)], dtype=float)
        _coloured_segments(ax, uv, speeds, cmap_obj, norm, linewidth)
        n_tracks += 1

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_xticks([])
    ax.set_yticks([])

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, extend="max")
        cbar.set_label(SPEED_LABEL, fontsize=AXIS_LABEL_SIZE,
                       fontweight="bold")
    _save_figure(fig, out_path, dpi=dpi)
    logger.info(f"Wrote {out_path} ({n_tracks} trajectories)")


def save_speed_legend(out_path, vmin, vmax, cmap="plasma",
                      label=r"Speed in m/min", dpi=300,
                      tick_fontsize=22, label_fontsize=24):
    """Standalone drift-speed colour scale shared by the velocity figures.

    Bold weight and tick size are applied via rc_context: a colorbar
    regenerates its tick labels at draw time, which on several matplotlib
    versions discards weights set on the tick Text objects directly.
    """
    setup_publication_style()
    with plt.rc_context({"font.weight": "bold",
                         "axes.labelweight": "bold",
                         "ytick.labelsize": tick_fontsize}):
        fig, ax = plt.subplots(figsize=(1.6, 8))
        ax.axis("off")
        mappable = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax),
                                         cmap=plt.get_cmap(cmap))
        cax = fig.add_axes([0.50, 0.08, 0.32, 0.84])
        cbar = fig.colorbar(mappable, cax=cax, extend="max")
        cbar.set_label(label, fontsize=label_fontsize, fontweight="bold",
                       labelpad=18)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight("bold")
            tick.set_fontsize(tick_fontsize)
        _save_figure(fig, out_path, dpi=dpi)
    logger.info(f"Wrote {out_path} (legend: {cmap}, "
                f"{vmin:.2f}-{vmax:.2f} m/min)")


def render_track_figures(config, tracks, assembled, paths, out_dir):
    """All trajectory figures for one georeferenced sequence, on one shared
    drift-speed colour scale (0 to a rounded robust 98th percentile)."""
    setup_publication_style()
    out_dir = Path(out_dir)

    _, all_speeds = compute_track_speeds(assembled, config.frame_interval_min)
    finite = all_speeds[np.isfinite(all_speeds)]
    vmin = 0.0
    vmax = round(float(np.percentile(finite, 98.0)), 1) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    logger.info(f"Shared drift-speed colour scale: {vmin:.2f}-{vmax:.2f} m/min")

    plot_track_map(assembled, out_dir / "tracks_map.png",
                   basemap=config.basemap)
    plot_track_map_velocity(assembled, out_dir / "tracks_map_velocity.png",
                            frame_interval_min=config.frame_interval_min,
                            basemap=config.basemap, vmin=vmin, vmax=vmax,
                            colorbar=False)

    background = first_image(paths["images"])
    plot_tracks_pixel(tracks, background, out_dir / "tracks_pixel.png")
    plot_tracks_pixel_velocity(tracks, assembled, background,
                               out_dir / "tracks_pixel_velocity.png",
                               frame_interval_min=config.frame_interval_min,
                               vmin=vmin, vmax=vmax, colorbar=False)
    save_speed_legend(out_dir / "speed_legend.png", vmin=vmin, vmax=vmax)


# ============================================================================
# CIRCULATION PANELS (called by the circulation command)
# ============================================================================

def _map_figsize(bounds, width=10.0):
    """Figure size matching the UTM extent's aspect ratio at the same width
    as the trajectory maps, so fixed-size elements stay comparable in print."""
    x_min, x_max, y_min, y_max = bounds
    return (width, width * (y_max - y_min) / (x_max - x_min))


def _prepare_map_axes(ax, bounds):
    x_min, x_max, y_min, y_max = bounds
    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])


def render_vector_panel(config, X, Y, U, V, speed, vmax, bounds, out_path):
    """Velocity-vector (quiver) panel: colour encodes the quantitative speed;
    arrow length is a non-linear (quiver_length_power) visual aid only."""
    fig, ax = plt.subplots(figsize=_map_figsize(bounds))
    _prepare_map_axes(ax, bounds)
    if config.basemap:
        add_basemap(ax, utm_epsg=config.utm_epsg)

    smax = np.nanmax(speed) if np.any(np.isfinite(speed)) else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_u = np.where(speed > 0, U / speed, 0.0)
        unit_v = np.where(speed > 0, V / speed, 0.0)
        normalised = speed / smax if smax > 0 else np.zeros_like(speed)
    length = (np.power(normalised, config.quiver_length_power)
              * (config.grid_resolution_m * 0.9))
    arrow_u = unit_u * length
    arrow_v = unit_v * length

    drawn = np.isfinite(arrow_u) & np.isfinite(arrow_v)
    if np.any(drawn):
        ax.quiver(X[drawn], Y[drawn], arrow_u[drawn], arrow_v[drawn],
                  speed[drawn], cmap="plasma", clim=(0, vmax),
                  pivot="mid", units="xy", angles="xy",
                  scale=1.0, scale_units="xy",
                  width=config.grid_resolution_m * 0.12,
                  headwidth=3.0, headlength=3.5, headaxislength=3.0,
                  edgecolor="black", linewidth=0.5)
    draw_km_scale_bar(ax)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=config.dpi)
    logger.info(f"Wrote {out_path}")


def render_streamline_panel(config, X, Y, U, V, speed, vmax, bounds, out_path):
    """Streamline panel showing the continuous circulation topology.

    Expects the field already extended by
    circulation.extend_field_for_streamlines when
    streamline_boundary_cells > 0.
    """
    fig, ax = plt.subplots(figsize=_map_figsize(bounds))
    _prepare_map_axes(ax, bounds)
    if config.basemap:
        add_basemap(ax, utm_epsg=config.utm_epsg)

    if np.any(np.isfinite(U)):
        ax.streamplot(X, Y, U, V, color=speed, cmap="plasma",
                      norm=plt.Normalize(0, vmax),
                      linewidth=1.5, arrowsize=1.5,
                      density=config.streamline_density)
    draw_km_scale_bar(ax)
    fig.tight_layout()
    _save_figure(fig, out_path, dpi=config.dpi)
    logger.info(f"Wrote {out_path}")
