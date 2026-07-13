"""Iceberg detection with Faster R-CNN.

Training: k-fold cross-validation over all sequences of the training dataset,
keeping the best model by validation loss. One shared model (models/detection.pt)
is used for all datasets.

Inference: multi-scale detection with sliding windows for large images, inline
filtering (confidence, size, edge, mask), NMS, and nested-detection removal.
Optionally generates appearance embeddings for the detections inline.
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF
from tqdm import tqdm

from helpers import (
    DATA_DIR, MODELS_DIR, calculate_iou, get_image_ext, get_sequences,
    log_config, log_section, resolve_device, save_config_snapshot,
    write_mot_file,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IcebergDetectionConfig:
    """All hyperparameters for detection training and inference."""

    # Data
    dataset: str = "???"
    num_workers: int = 4

    # Model architecture (anchors tuned for wide, small-to-medium icebergs)
    num_classes: int = 2  # Background + iceberg
    anchor_sizes: List[List[int]] = field(
        default_factory=lambda: [[16], [32], [64], [128], [256]])
    anchor_aspect_ratios: List[List[float]] = field(
        default_factory=lambda: [[0.4, 0.6, 0.8]] * 5)

    # Training
    k_folds: int = 5
    num_epochs: int = 4
    batch_size: int = 2
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    save_checkpoints: bool = False

    # Faster R-CNN inference internals
    box_detections_per_img: int = 5000
    box_nms_thresh: float = 0.5
    rpn_nms_thresh: float = 0.7

    # Multi-scale / sliding-window inference
    confidence_threshold: float = 0.1
    scales: List[float] = field(default_factory=lambda: [0.5, 1.0])
    window_size: List[int] = field(default_factory=lambda: [1536, 1536])
    overlap: float = 0.35

    # Inline filtering during inference
    edge_tolerance: int = 0
    mask_ratio_threshold: float = 0.1
    filter_masked_regions: bool = True
    min_iceberg_size: float = 100.0

    # Embedding generation after detection
    generate_embeddings: bool = True

    # Hardware (None = auto-detect)
    device: Optional[str] = None


# ============================================================================
# DATASET
# ============================================================================

class IcebergSequenceDataset(Dataset):
    """One timelapse sequence: images plus MOT-format ground-truth boxes."""

    def __init__(self, sequence_name, images_dir, gt_file, image_ext, transforms=None):
        self.sequence_name = sequence_name
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.image_ext = image_ext

        column_names = ["image", "iceberg_id", "bb_left", "bb_top", "bb_width",
                        "bb_height", "conf", "unused_1", "unused_2", "unused_3"]
        self.detections = pd.read_csv(gt_file, names=column_names)
        self.unique_images = self.detections["image"].unique()

    def __len__(self):
        return len(self.unique_images)

    def __getitem__(self, idx):
        """Return (image_tensor, target) with boxes in (x1, y1, x2, y2) format."""
        img_name = self.unique_images[idx]
        img_detections = self.detections[self.detections["image"] == img_name]

        # Resolve image path (plain or zero-padded frame naming)
        img_path = self.images_dir / f"{img_name}.{self.image_ext}"
        if not img_path.exists():
            img_path = self.images_dir / f"{img_name:06d}.{self.image_ext}"
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_name} in {self.images_dir}")

        img = Image.open(img_path).convert("RGB")

        boxes, labels = [], []
        for _, row in img_detections.iterrows():
            x1, y1 = row["bb_left"], row["bb_top"]
            boxes.append([x1, y1, x1 + row["bb_width"], y1 + row["bb_height"]])
            labels.append(1)

        # Globally unique image id across sequences (names may repeat)
        global_image_id = hash(f"{self.sequence_name}_{img_name}") % (2 ** 31)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([global_image_id], dtype=torch.int64),
        }

        img = self.transforms(img) if self.transforms else TF.to_tensor(img)
        return img, target

    def get_sequence_info(self):
        """Return counts of images, annotations, and unique iceberg IDs."""
        return {
            "sequence_name": self.sequence_name,
            "num_images": len(self.unique_images),
            "num_annotations": len(self.detections),
            "unique_icebergs": len(self.detections["iceberg_id"].unique()),
        }


# ============================================================================
# DETECTOR
# ============================================================================

class IcebergDetector:
    """Coordinates detection training and inference for a dataset."""

    def __init__(self, config: IcebergDetectionConfig):
        self.config = config
        self.dataset = config.dataset
        self.device = resolve_device(config.device)
        self.model = None
        self.transform = self._get_transforms()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model_file = MODELS_DIR / "detection.pt"
        self.checkpoint_dir = MODELS_DIR / "checkpoints"
        if self.config.save_checkpoints:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_config(config, title="Iceberg Detection Configuration")
        logger.info(f"Device: {self.device}")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self):
        """Train with k-fold cross-validation; save the best model overall."""
        log_section("TRAINING PHASE")
        logger.info(f"{self.config.k_folds}-fold cross-validation, "
                    f"{self.config.num_epochs} epochs per fold")

        dataset = self._get_multi_seq_dataset()
        total_start_time = time.time()
        kf = KFold(n_splits=self.config.k_folds, shuffle=True, random_state=42)
        best_val_loss_overall = float("inf")
        best_model_state_overall = None

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            logger.info(f"\nFold {fold + 1}/{self.config.k_folds}")

            collate = lambda batch: tuple(zip(*batch))
            train_loader = DataLoader(
                Subset(dataset, train_idx), batch_size=self.config.batch_size,
                shuffle=True, num_workers=self.config.num_workers, collate_fn=collate)
            val_loader = DataLoader(
                Subset(dataset, val_idx), batch_size=self.config.batch_size,
                shuffle=False, num_workers=self.config.num_workers, collate_fn=collate)

            model = self._build_model()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.config.learning_rate,
                momentum=self.config.momentum, weight_decay=self.config.weight_decay)

            best_val_loss = float("inf")
            best_model_state = None

            for epoch in range(self.config.num_epochs):
                train_loss = self._train_one_epoch(model, optimizer, train_loader)
                val_loss = self._evaluate_model(model, val_loader)

                elapsed, remaining, per_epoch = self._calculate_time_estimates(
                    total_start_time, self.config.k_folds, fold,
                    self.config.num_epochs, epoch)
                logger.info(
                    f"Epoch [{epoch + 1}/{self.config.num_epochs}] "
                    f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} | "
                    f"Time: {timedelta(seconds=int(elapsed))}<"
                    f"{timedelta(seconds=int(remaining))}, "
                    f"{timedelta(seconds=int(per_epoch))}/Epoch")

                if self.config.save_checkpoints:
                    checkpoint_path = self.checkpoint_dir / (
                        f"detection_fold{fold + 1}_epoch{epoch + 1}"
                        f"_valloss{val_loss:.4f}.pt")
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info(f"  Checkpoint saved: {checkpoint_path.name}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())

            if best_val_loss < best_val_loss_overall:
                best_val_loss_overall = best_val_loss
                best_model_state_overall = best_model_state
                logger.info(f"New best model (val loss {best_val_loss_overall:.4f})")

        if best_model_state_overall:
            torch.save(best_model_state_overall, self.model_file)
            save_config_snapshot(self.config, MODELS_DIR / "detection_config.yaml")
            logger.info(f"\nBest model saved to {self.model_file} "
                        f"(val loss {best_val_loss_overall:.4f})")
            logger.info(f"Training completed in "
                        f"{timedelta(seconds=int(time.time() - total_start_time))}")

    def _get_multi_seq_dataset(self):
        """Concatenate IcebergSequenceDatasets from all sequences with GT."""
        sequence_datasets = []
        sequences = get_sequences(self.dataset)

        for sequence_name, paths in sequences.items():
            if not paths["ground_truth"].exists():
                logger.warning(f"No gt.txt found for {sequence_name}, skipping")
                continue
            seq_dataset = IcebergSequenceDataset(
                sequence_name=sequence_name,
                images_dir=paths["images"],
                gt_file=paths["ground_truth"],
                image_ext=get_image_ext(paths["images"]),
                transforms=self.transform,
            )
            sequence_datasets.append(seq_dataset)
            info = seq_dataset.get_sequence_info()
            logger.info(f"Loaded '{sequence_name}': {info['num_images']} images, "
                        f"{info['num_annotations']} annotations, "
                        f"{info['unique_icebergs']} unique icebergs")

        if not sequence_datasets:
            raise ValueError(f"No valid sequences found in {self.dataset}")

        combined = ConcatDataset(sequence_datasets)
        logger.info(f"\n{self.dataset}: {len(sequence_datasets)} sequences, "
                    f"{len(combined)} images total")
        return combined

    def _build_model(self):
        """Faster R-CNN (ResNet-50 FPN) with custom anchors and 2-class head."""
        anchor_generator = AnchorGenerator(
            sizes=tuple(tuple(s) for s in self.config.anchor_sizes),
            aspect_ratios=tuple(tuple(r) for r in self.config.anchor_aspect_ratios),
        )
        model = fasterrcnn_resnet50_fpn(
            weights=None,  # Compatible pretrained weights loaded selectively below
            rpn_anchor_generator=anchor_generator,
            box_detections_per_img=self.config.box_detections_per_img,
            box_nms_thresh=self.config.box_nms_thresh,
            rpn_nms_thresh=self.config.rpn_nms_thresh,
        )
        self._load_pretrained_weights(model)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.config.num_classes)
        return model.to(self.device)

    def _load_pretrained_weights(self, model):
        """Load pretrained weights, skipping layers with shape mismatches
        (custom anchors and class head)."""
        pretrained_dict = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.get_state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def _load_model(self, checkpoint_path=None):
        """Load the shared trained model (or a specific checkpoint)."""
        if self.model is None:
            self.model = self._build_model()

        file_path = Path(checkpoint_path) if checkpoint_path else self.model_file
        if not file_path.exists():
            raise FileNotFoundError(f"No trained model found at {file_path}")
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {file_path}")
        return self.model

    def _get_transforms(self):
        """Convert PIL images or numpy arrays to normalized tensors."""
        def transform(image):
            if isinstance(image, Image.Image):
                return TF.to_tensor(image)
            if isinstance(image, np.ndarray):
                return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            return image
        return transform

    def _train_one_epoch(self, model, optimizer, data_loader):
        """One training pass; returns average loss."""
        model.train()
        running_loss = 0.0
        for images, targets in data_loader:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        return running_loss / len(data_loader)

    def _evaluate_model(self, model, data_loader):
        """Validation loss. Model stays in train mode because Faster R-CNN
        only returns losses in train mode; weights are not updated."""
        model.train()
        running_val_loss = 0.0
        with torch.set_grad_enabled(True):
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                running_val_loss += sum(loss for loss in loss_dict.values()).item()
        return running_val_loss / len(data_loader)

    def _calculate_time_estimates(self, start_time, k_folds, fold, num_epochs, epoch):
        """Return (elapsed, estimated_remaining, avg_seconds_per_epoch)."""
        elapsed = time.time() - start_time
        total_epochs = k_folds * num_epochs
        current_epoch_global = fold * num_epochs + (epoch + 1)
        avg_per_epoch = elapsed / current_epoch_global
        remaining = (total_epochs - current_epoch_global) * avg_per_epoch
        return elapsed, remaining, avg_per_epoch

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(self, checkpoint_path=None):
        """Detect icebergs in all sequences and save MOT-format det.txt files.

        Args:
            checkpoint_path: Optional specific checkpoint instead of the
                shared best model.
        """
        log_section("INFERENCE PHASE")
        self._load_model(checkpoint_path)
        self.model.eval()
        mask, img_width, img_height = self._load_filter()

        sequences = get_sequences(self.dataset)
        logger.info(f"Sequences ({len(sequences)}): {', '.join(sequences.keys())}")

        for sequence_name, paths in sequences.items():
            image_ext = get_image_ext(paths["images"])
            image_files = sorted(f.name for f in paths["images"].iterdir()
                                 if f.name.endswith(image_ext))
            all_detections = []
            start_time = time.time()

            logger.info(f"\nProcessing {len(image_files)} images in {sequence_name}")
            progress_bar = tqdm(image_files, desc=f"Detecting ({sequence_name})",
                                unit="image")
            for img_file in progress_bar:
                img_name = img_file.rsplit(".", 1)[0]
                detections = self._run_multi_scale_sliding_window_prediction(
                    paths["images"] / img_file, mask, img_width, img_height)
                for i, det in enumerate(detections):
                    det["image"] = img_name
                    det["object_id"] = i + 1
                    all_detections.append(det)
                progress_bar.set_postfix({"detections": len(all_detections)})

            self._save_detections(all_detections, paths["detections"])
            save_config_snapshot(self.config, paths["det_config"])
            logger.info(f"{sequence_name}: {len(all_detections)} detections "
                        f"across {len(image_files)} images")
            logger.info(f"Saved to: {paths['detections']}")
            logger.info(f"Inference time: "
                        f"{timedelta(seconds=int(time.time() - start_time))}")

        if self.config.generate_embeddings:
            log_section("GENERATING EMBEDDINGS")
            from embedding import IcebergEmbeddingsConfig, IcebergEmbeddingsTrainer
            trainer = IcebergEmbeddingsTrainer(
                IcebergEmbeddingsConfig(dataset=self.dataset))
            trainer.load_for_inference()
            for seq_name, paths in sequences.items():
                trainer.generate_embeddings(
                    image_dir=paths["images"],
                    detection_file=paths["detections"],
                    output_path=paths["det_embeddings"],
                )

        log_section("Detection complete")

    def _load_filter(self):
        """Load the land mask and image dimensions if filtering is enabled.

        Masks are inputs and live at DATA_DIR/<dataset root>/mask.<ext>.
        """
        mask, img_width, img_height = None, None, None
        if not (self.config.filter_masked_regions or self.config.edge_tolerance > 0):
            return mask, img_width, img_height

        # Image dimensions from the first image of the first sequence
        first_seq = next(iter(get_sequences(self.dataset).values()))
        image_ext = get_image_ext(first_seq["images"])
        sample_path = sorted(first_seq["images"].glob(f"*{image_ext}"))[0]
        img_height, img_width = cv2.imread(str(sample_path)).shape[:2]

        if self.config.filter_masked_regions:
            mask_dir = DATA_DIR / Path(self.dataset).parts[0]
            mask_file = None
            for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]:
                candidate = mask_dir / f"mask{ext}"
                if candidate.exists():
                    mask_file = candidate
                    break
            if mask_file is None:
                logger.warning(f"Mask file not found in {mask_dir}")
            else:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE).astype(bool)
                logger.info(f"Loaded mask {img_width}x{img_height} "
                            f"(ratio threshold {self.config.mask_ratio_threshold})")

        if self.config.edge_tolerance > 0:
            logger.info(f"Edge filtering enabled ({self.config.edge_tolerance}px)")
        return mask, img_width, img_height

    def _is_valid_detection(self, box, score, mask, img_width, img_height):
        """Apply all inline filters (cheapest first). Returns False to reject."""
        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin

        if score < self.config.confidence_threshold:
            return False

        if self.config.min_iceberg_size > 0 and width * height < self.config.min_iceberg_size:
            return False

        if self.config.edge_tolerance > 0:
            if (xmin <= self.config.edge_tolerance
                    or ymin <= self.config.edge_tolerance
                    or xmax >= img_width - self.config.edge_tolerance
                    or ymax >= img_height - self.config.edge_tolerance):
                return False

        if mask is not None and self.config.filter_masked_regions:
            left, top = int(max(0, xmin)), int(max(0, ymin))
            right, bottom = int(min(xmax, img_width)), int(min(ymax, img_height))
            if right <= left or bottom <= top:
                return False

            # Clamp to mask bounds (mask may differ in size)
            left = max(0, min(left, mask.shape[1] - 1))
            right = max(left + 1, min(right, mask.shape[1]))
            top = max(0, min(top, mask.shape[0] - 1))
            bottom = max(top + 1, min(bottom, mask.shape[0]))

            submask = mask[top:bottom, left:right]
            if submask.size > 0:
                water_ratio = 1 - np.count_nonzero(submask) / float(submask.size)
                if water_ratio > self.config.mask_ratio_threshold:
                    return False

        return True

    def _run_multi_scale_sliding_window_prediction(self, img_path, mask=None,
                                                   img_width=None, img_height=None):
        """Multi-scale detection; large scaled images use overlapping windows.

        Returns detections in (x, y, width, height) format after inline
        filtering, NMS, and nested-detection removal.
        """
        original_img = Image.open(img_path).convert("RGB")
        original_size = original_img.size
        if img_width is None or img_height is None:
            img_width, img_height = original_size

        scale_detections = []

        for scale in self.config.scales:
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled_img = original_img.resize(new_size, Image.LANCZOS)
            use_windows = max(new_size) > self.config.window_size[0]

            if use_windows:
                img_array = np.array(scaled_img)
                h, w = img_array.shape[:2]
                win_w, win_h = self.config.window_size
                step_x = int(win_w * (1 - self.config.overlap))
                step_y = int(win_h * (1 - self.config.overlap))

                for y in range(0, h, step_y):
                    if y + win_h > h:
                        y = h - win_h  # Snap to edge
                    for x in range(0, w, step_x):
                        if x + win_w > w:
                            x = w - win_w  # Snap to edge

                        window = img_array[y:y + win_h, x:x + win_w]
                        window_tensor = self.transform(
                            Image.fromarray(window)).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            predictions = self.model(window_tensor)

                        boxes = predictions[0]["boxes"].cpu().numpy()
                        scores = predictions[0]["scores"].cpu().numpy()
                        if len(boxes) > 0:
                            boxes[:, [0, 2]] = (boxes[:, [0, 2]] + x) / scale
                            boxes[:, [1, 3]] = (boxes[:, [1, 3]] + y) / scale
                            for box, score in zip(boxes, scores):
                                if self._is_valid_detection(box, score, mask,
                                                            img_width, img_height):
                                    scale_detections.append({"box": box, "score": score})
            else:
                img_tensor = self.transform(scaled_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    predictions = self.model(img_tensor)

                boxes = predictions[0]["boxes"].cpu().numpy()
                scores = predictions[0]["scores"].cpu().numpy()
                if len(boxes) > 0:
                    boxes[:, [0, 2]] /= scale
                    boxes[:, [1, 3]] /= scale
                    for box, score in zip(boxes, scores):
                        if self._is_valid_detection(box, score, mask,
                                                    img_width, img_height):
                            scale_detections.append({"box": box, "score": score})

        merged = self._nms(scale_detections)
        merged = self._remove_nested_detections(merged)
        return self._convert_to_detection_format(merged)

    def _nms(self, detections):
        """Greedy Non-Maximum Suppression on 'box'/'score' detections."""
        if not detections:
            return []
        detections.sort(key=lambda x: x["score"], reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [
                det for det in detections
                if calculate_iou(best["box"], det["box"]) < self.config.box_nms_thresh
            ]
        return keep

    def _remove_nested_detections(self, detections):
        """Remove detections nested inside other detections.

        High-confidence detections (score > box_nms_thresh) are always kept;
        when a low-confidence detection contains a high-confidence one, the
        low-confidence one is removed; among low-confidence pairs the higher
        score wins.
        """
        if not detections:
            return []

        detections = sorted(detections, key=lambda x: x["score"], reverse=True)
        keep = []

        for current in detections:
            is_high_confidence = current["score"] > self.config.box_nms_thresh
            should_keep = True
            remove_indices = []

            for i, kept_det in enumerate(keep):
                kept_is_high_confidence = kept_det["score"] > self.config.box_nms_thresh
                current_in_kept = self._calculate_intersection_ratio(
                    smaller_box=current["box"], larger_box=kept_det["box"])
                kept_in_current = self._calculate_intersection_ratio(
                    smaller_box=kept_det["box"], larger_box=current["box"])

                if current_in_kept >= self.config.box_nms_thresh:
                    if kept_is_high_confidence:
                        should_keep = False
                        break
                    elif is_high_confidence:
                        remove_indices.append(i)
                    else:
                        should_keep = False  # Both low: kept has higher score
                        break
                elif kept_in_current >= self.config.box_nms_thresh:
                    if is_high_confidence:
                        remove_indices.append(i)
                    elif kept_is_high_confidence:
                        should_keep = False
                        break
                    else:
                        should_keep = False
                        break

            for i in sorted(remove_indices, reverse=True):
                keep.pop(i)
            if should_keep:
                keep.append(current)

        return keep

    def _calculate_intersection_ratio(self, smaller_box, larger_box):
        """Fraction of smaller_box contained in larger_box (not IoU)."""
        x1_s, y1_s, x2_s, y2_s = smaller_box
        x1_l, y1_l, x2_l, y2_l = larger_box

        x1_i, y1_i = max(x1_s, x1_l), max(y1_s, y1_l)
        x2_i, y2_i = min(x2_s, x2_l), min(y2_s, y2_l)
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        smaller_box_area = (x2_s - x1_s) * (y2_s - y1_s)
        if smaller_box_area == 0:
            return 0.0
        return intersection_area / smaller_box_area

    def _convert_to_detection_format(self, detections):
        """Convert corner boxes to (x, y, width, height) output format."""
        formatted = []
        for det in detections:
            xmin, ymin, xmax, ymax = det["box"]
            formatted.append({
                "x": xmin, "y": ymin,
                "width": xmax - xmin, "height": ymax - ymin,
                "score": det["score"],
            })
        return formatted

    def _save_detections(self, detections, detections_file):
        """Write detections in MOT format (sorted)."""
        rows = [
            (det["image"], det["object_id"], det["x"], det["y"],
             det["width"], det["height"], det["score"])
            for det in detections
        ]
        write_mot_file(rows, detections_file)
