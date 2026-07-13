"""Iceberg appearance embeddings (DINOv2 + metric learning).

Pipeline: pretrained DINOv2 ViT backbone (timm) with BN-neck and linear
projection, identity-based PK-sampling, NT-Xent loss with Multi-Similarity
mining, and square-context crops. Model selection uses Rank-1/Rank-5/mAP
retrieval on held-out sequences, mirroring the downstream tracking task.

One shared model (models/embedding.pt) is trained on the training dataset
and used for embedding generation on all datasets. Exported format:
{f"{frame}_{id}": Tensor[feature_dim]} per sequence (see helpers.emb_key).

Training schedule: Stage 1 trains the head with a frozen backbone; Stage 2
unfreezes the last N transformer blocks with a lower backbone LR.
"""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pytorch_metric_learning import losses, miners, samplers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from helpers import (
    MODELS_DIR, get_image_ext, get_sequences, load_icebergs_by_frame,
    log_config, log_section, parse_annotations, resolve_device,
    save_config_snapshot,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IcebergEmbeddingsConfig:
    """All hyperparameters for embedding training and generation."""

    # Data
    dataset: str = "???"
    val_dataset: Optional[str] = None  # Held-out sequences for retrieval eval
    embed_source: str = "detections"   # 'detections' or 'ground_truth' (embed command)

    # Backbone (timm identifier; base/large DINOv2 variants are stronger but slower)
    backbone_name: str = "vit_small_patch14_dinov2.lvd142m"
    img_size: int = 224          # Must be divisible by patch size (14)
    feature_dim: int = 256       # Output dimension after projection head

    # Batch construction (PK-sampling: P identities x K crops)
    identities_per_batch: int = 16
    samples_per_identity: int = 4
    min_samples_per_identity: int = 2  # Drop identities without positive partners

    # Two-stage training schedule
    stage1_epochs: int = 5             # Head warmup, backbone frozen
    stage2_epochs: int = 25            # Joint fine-tuning
    unfreeze_last_n_blocks: int = 3

    # Optimization
    stage1_lr: float = 1e-3
    stage2_backbone_lr: float = 5e-5
    stage2_head_lr: float = 1e-4
    weight_decay: float = 1e-4

    # Loss / mining
    ntxent_temperature: float = 0.07   # Lower = sharper negative pushing
    miner_epsilon: float = 0.1

    # Early stopping (on validation Rank-1)
    patience: int = 8
    min_delta: float = 0.002

    # Crop preprocessing
    bbox_context: float = 0.15         # Context margin around bbox (fraction)

    # Hardware (None = auto-detect)
    num_workers: int = 4
    device: Optional[str] = None

    @property
    def batch_size(self) -> int:
        return self.identities_per_batch * self.samples_per_identity


# ============================================================================
# CROPPING (shared by training and inference)
# ============================================================================

def square_context_crop(
    img: Image.Image,
    bbox: Tuple[float, float, float, float],
    target_size: int,
    context: float = 0.15,
) -> Image.Image:
    """Square crop around a bbox with a context margin, resized to target_size.

    Used instead of letterbox padding: black borders create artificial
    gradients the model overfits to, and DINOv2 was pretrained without
    padding artifacts. Crops near image edges may be non-square before the
    resize, which handles them consistently.
    """
    x, y, w, h = bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + context)

    left = max(0, cx - side / 2.0)
    top = max(0, cy - side / 2.0)
    right = min(img.width, left + side)
    bottom = min(img.height, top + side)

    crop = img.crop((left, top, right, bottom))
    return crop.resize((target_size, target_size), Image.Resampling.LANCZOS)


import os as _os

# Per-WORKER cache of decoded full frames. Crops of one frame are drawn
# consecutively, so a small cache captures almost all the re-decode saving;
# the total memory cost is this * num_workers, so keep it small. Override with
# DRIFT_IMAGE_CACHE if frames are small and RAM is plentiful.
_IMAGE_CACHE_SIZE = int(_os.environ.get("DRIFT_IMAGE_CACHE", "4"))


@lru_cache(maxsize=_IMAGE_CACHE_SIZE)
def _load_full_image(path: str) -> Image.Image:
    """LRU-cached full-frame loader (each DataLoader worker has its own cache;
    total RAM cost is _IMAGE_CACHE_SIZE * num_workers decoded frames)."""
    return Image.open(path).convert("RGB")


# ============================================================================
# DATASETS
# ============================================================================

class IcebergIdentityDataset(Dataset):
    """Identity-labeled crops for metric learning.

    One "identity" is one iceberg track in one sequence; labels are globally
    unique across sequences. Identities with fewer than
    `min_samples_per_identity` crops are dropped (no positive partner in a
    batch means no InfoNCE signal). Exposes `self.labels` for
    pytorch_metric_learning's MPerClassSampler.
    """

    def __init__(
        self,
        sequences: List[Tuple[Any, Any]],
        image_ext: str = "jpg",
        target_size: int = 224,
        bbox_context: float = 0.15,
        transform: Optional[T.Compose] = None,
        min_samples_per_identity: int = 2,
    ):
        """
        Args:
            sequences: List of (gt_file, image_dir) tuples, typically built
                from get_sequences() paths.
        """
        self.target_size = target_size
        self.bbox_context = bbox_context
        self.image_ext = image_ext
        self.transform = transform or _default_train_transform()

        samples: List[Dict[str, Any]] = []
        labels: List[int] = []
        global_id = 0

        for gt_file, image_dir in sequences:
            gt_file, image_dir = str(gt_file), str(image_dir)
            if not os.path.exists(gt_file):
                logger.warning(f"Skipping: gt file not found at {gt_file}")
                continue

            icebergs_by_frame = load_icebergs_by_frame(gt_file)
            local_to_global: Dict[int, int] = {}
            n_crops_before = len(samples)

            for frame_name, icebergs in icebergs_by_frame.items():
                img_path = os.path.join(image_dir, f"{frame_name}.{image_ext}")
                for local_id, data in icebergs.items():
                    if local_id not in local_to_global:
                        local_to_global[local_id] = global_id
                        global_id += 1
                    label = local_to_global[local_id]
                    samples.append({
                        "image_path": img_path,
                        "bbox": data["bbox"],
                        "label": label,
                    })
                    labels.append(label)

            seq_name = os.path.basename(os.path.dirname(image_dir))
            logger.info(f"  {seq_name}: {len(local_to_global)} identities, "
                        f"{len(samples) - n_crops_before} crops")

        counts: Dict[int, int] = defaultdict(int)
        for lbl in labels:
            counts[lbl] += 1
        keep_idx = [i for i, lbl in enumerate(labels)
                    if counts[lbl] >= min_samples_per_identity]
        self.samples = [samples[i] for i in keep_idx]
        self.labels = [labels[i] for i in keep_idx]

        n_kept = len(set(self.labels))
        logger.info(f"  Total: {len(self.samples)} crops across {n_kept} identities "
                    f"(dropped {len(set(labels)) - n_kept} short identities)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = _load_full_image(s["image_path"])
        crop = square_context_crop(img, s["bbox"], self.target_size, self.bbox_context)
        if self.transform is not None:
            crop = self.transform(crop)
        return crop, s["label"]


class IcebergInferenceDataset(Dataset):
    """Detection crops for embedding generation.

    Emits (crop_tensor, unique_name) with unique_name = "{frame}_{id}",
    matching the key format consumed by tracking.py. Uses the same
    square-context crop as training for preprocessing parity.
    """

    def __init__(
        self,
        detections: List[Dict[str, Any]],
        full_frame_dir: str,
        target_size: int = 224,
        bbox_context: float = 0.15,
        image_ext: str = "jpg",
        transform: Optional[T.Compose] = None,
    ):
        self.detections = detections
        self.full_frame_dir = full_frame_dir
        self.target_size = target_size
        self.bbox_context = bbox_context
        self.image_ext = image_ext
        self.transform = transform or _default_eval_transform()

    def __len__(self) -> int:
        return len(self.detections)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        det = self.detections[idx]
        unique_name = f"{det['frame']}_{det['id']}"
        img_path = os.path.join(self.full_frame_dir,
                                f"{det['frame']}.{self.image_ext}")
        try:
            img = _load_full_image(img_path)
            bbox = (det["bb_left"], det["bb_top"], det["bb_width"], det["bb_height"])
            crop = square_context_crop(img, bbox, self.target_size, self.bbox_context)
            if self.transform is not None:
                crop = self.transform(crop)
            return crop, unique_name
        except Exception as e:
            logger.error(f"Error loading detection {idx} ({unique_name}): {e}")
            black = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
            if self.transform is not None:
                black = self.transform(black)
            return black, f"error_{idx}"


# ============================================================================
# TRANSFORMS
# ============================================================================

# DINOv2 was trained with standard ImageNet normalization
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _default_train_transform() -> T.Compose:
    """Modest augmentation preserving identity; avoids aggressive color
    jitter on near-monochrome ice imagery."""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0)], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _default_eval_transform() -> T.Compose:
    """Deterministic preprocessing for evaluation and embedding generation."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ============================================================================
# MODEL
# ============================================================================

class IcebergEmbedder(nn.Module):
    """DINOv2 backbone -> BN-neck -> linear projection -> L2 normalization.

    Output embeddings are L2-normalized so cosine similarity equals the dot
    product, as assumed by tracking.get_appearance_similarity. The BN-neck
    with frozen bias follows Luo et al. ("Bag of Tricks" for re-ID).
    """

    def __init__(self, config: IcebergEmbeddingsConfig):
        super().__init__()
        self.config = config

        self.backbone = timm.create_model(
            config.backbone_name,
            pretrained=True,
            num_classes=0,  # Remove classification head -> pooled features
            img_size=config.img_size,
        )
        backbone_dim = self.backbone.num_features

        self.bn_neck = nn.BatchNorm1d(backbone_dim)
        self.bn_neck.bias.requires_grad_(False)
        self.projection = nn.Linear(backbone_dim, config.feature_dim)

        if not hasattr(self.backbone, "blocks"):
            logger.warning("Backbone has no `.blocks` attribute; "
                           "unfreeze_last_n_blocks will be a no-op.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.bn_neck(feats)
        embed = self.projection(feats)
        return F.normalize(embed, dim=1)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unfreeze_last_blocks(self, n: int) -> None:
        """Unfreeze the last n transformer blocks plus the final norm layer."""
        if not hasattr(self.backbone, "blocks"):
            return
        blocks = self.backbone.blocks
        n = min(n, len(blocks))
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
        self.backbone.train()
        logger.info(f"  Unfroze last {n} transformer blocks")

    def trainable_param_groups(
        self, backbone_lr: float, head_lr: float
    ) -> List[Dict[str, Any]]:
        """Optimizer param groups with separate backbone and head LRs."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = (list(self.bn_neck.parameters())
                       + list(self.projection.parameters()))
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})
        return groups


# ============================================================================
# RETRIEVAL EVALUATION
# ============================================================================

class RetrievalEvaluator:
    """Rank-1 / Rank-5 / mAP retrieval evaluation on consecutive frame pairs.

    For each iceberg in frame t, ranks all candidates in frame t+1 by cosine
    similarity and checks where the ground-truth match lands. This mirrors
    the downstream tracking task and drives model selection.
    """

    def __init__(
        self,
        sequences: List[Tuple[Any, Any]],
        config: IcebergEmbeddingsConfig,
        image_ext: str = "jpg",
    ):
        self.sequences = [(str(gt), str(img)) for gt, img in sequences]
        self.config = config
        self.image_ext = image_ext
        self.transform = _default_eval_transform()

    @torch.no_grad()
    def evaluate(self, model: IcebergEmbedder) -> Dict[str, float]:
        """Run retrieval evaluation across all consecutive frame pairs."""
        model.eval()
        device = next(model.parameters()).device

        total_queries = 0
        rank1_hits = 0
        rank5_hits = 0
        ap_scores: List[float] = []

        for gt_file, image_dir in self.sequences:
            if not os.path.exists(gt_file):
                continue
            icebergs_by_frame = load_icebergs_by_frame(gt_file)
            frame_names = sorted(icebergs_by_frame.keys())

            for f_t, f_tp1 in zip(frame_names[:-1], frame_names[1:]):
                ids_t = list(icebergs_by_frame[f_t].keys())
                ids_tp1 = list(icebergs_by_frame[f_tp1].keys())
                if not ids_t or not ids_tp1:
                    continue

                emb_t = self._encode_frame(
                    model, device, image_dir, f_t, icebergs_by_frame[f_t], ids_t)
                emb_tp1 = self._encode_frame(
                    model, device, image_dir, f_tp1, icebergs_by_frame[f_tp1], ids_tp1)

                sim = emb_t @ emb_tp1.T  # Cosine similarity (L2-normalized rows)
                order = torch.argsort(sim, dim=1, descending=True)
                id_to_pos_tp1 = {iid: pos for pos, iid in enumerate(ids_tp1)}

                for qi, query_id in enumerate(ids_t):
                    if query_id not in id_to_pos_tp1:
                        continue  # No ground-truth match in frame t+1
                    rank = order[qi].tolist().index(id_to_pos_tp1[query_id])

                    total_queries += 1
                    if rank == 0:
                        rank1_hits += 1
                    if rank < 5:
                        rank5_hits += 1
                    ap_scores.append(1.0 / (rank + 1))  # Single-positive AP

        if total_queries == 0:
            return {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0, "n_queries": 0}
        return {
            "rank1": rank1_hits / total_queries,
            "rank5": rank5_hits / total_queries,
            "mAP": float(np.mean(ap_scores)),
            "n_queries": total_queries,
        }

    def _encode_frame(self, model, device, image_dir, frame_name, icebergs, ids):
        """Encode every iceberg in one frame into a [N, feature_dim] tensor."""
        img_path = os.path.join(image_dir, f"{frame_name}.{self.image_ext}")
        img = _load_full_image(img_path)
        crops = [
            self.transform(square_context_crop(
                img, icebergs[iid]["bbox"],
                self.config.img_size, self.config.bbox_context))
            for iid in ids
        ]
        batch = torch.stack(crops, dim=0).to(device)
        return model(batch)


# ============================================================================
# TRAINER
# ============================================================================

class IcebergEmbeddingsTrainer:
    """Two-stage training orchestration plus embedding generation.

    Tracks validation Rank-1 (if val_dataset is set) and keeps the best
    checkpoint at models/embedding.pt; without validation, the latest
    checkpoint is saved each epoch.
    """

    def __init__(self, config: IcebergEmbeddingsConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.model_path = MODELS_DIR / "embedding.pt"
        # Resolved lazily: get_image_ext needs a real image directory
        self.image_ext: Optional[str] = None

        self.model: Optional[IcebergEmbedder] = None
        self.train_dataset: Optional[IcebergIdentityDataset] = None
        self.train_loader: Optional[DataLoader] = None
        self.evaluator: Optional[RetrievalEvaluator] = None

        self.loss_fn = losses.NTXentLoss(temperature=config.ntxent_temperature)
        self.miner = miners.MultiSimilarityMiner(epsilon=config.miner_epsilon)

        self.history: Dict[str, List[float]] = {
            "train_losses": [], "val_rank1": [], "val_rank5": [],
            "val_mAP": [], "stage_boundaries": [],
        }

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def _build_datasets(self) -> None:
        log_section("BUILDING DATASETS")

        train_sequences = get_sequences(self.config.dataset)
        train_pairs = [(paths["ground_truth"], paths["images"])
                       for paths in train_sequences.values()]
        if not train_pairs:
            raise RuntimeError(f"No sequences found under {self.config.dataset}.")

        self.image_ext = get_image_ext(train_pairs[0][1])
        logger.info(f"Image extension: {self.image_ext}")

        logger.info(f"Training sequences ({len(train_pairs)}):")
        self.train_dataset = IcebergIdentityDataset(
            sequences=train_pairs,
            image_ext=self.image_ext,
            target_size=self.config.img_size,
            bbox_context=self.config.bbox_context,
            transform=_default_train_transform(),
            min_samples_per_identity=self.config.min_samples_per_identity,
        )

        if self.config.val_dataset is not None:
            val_sequences = get_sequences(self.config.val_dataset)
            val_pairs = [(paths["ground_truth"], paths["images"])
                         for paths in val_sequences.values()]
            logger.info(f"Validation sequences ({len(val_pairs)}): "
                        f"{', '.join(val_sequences.keys())}")
            self.evaluator = RetrievalEvaluator(
                sequences=val_pairs, config=self.config, image_ext=self.image_ext)

    def _build_dataloader(self) -> None:
        assert self.train_dataset is not None
        sampler = samplers.MPerClassSampler(
            labels=self.train_dataset.labels,
            m=self.config.samples_per_identity,
            batch_size=self.config.batch_size,
            length_before_new_iter=len(self.train_dataset),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

    def _build_model(self) -> None:
        log_section("BUILDING MODEL")
        logger.info(f"Backbone: {self.config.backbone_name}")
        self.model = IcebergEmbedder(self.config).to(self.device)
        n_total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total parameters: {n_total / 1e6:.1f}M")

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def _train_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        assert self.model is not None and self.train_loader is not None
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for crops, labels in self.train_loader:
            crops = crops.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            embeddings = self.model(crops)
            hard_pairs = self.miner(embeddings, labels)
            loss = self.loss_fn(embeddings, labels, hard_pairs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def _validate(self) -> Optional[Dict[str, float]]:
        if self.evaluator is None or self.model is None:
            return None
        return self.evaluator.evaluate(self.model)

    def _run_stage(self, stage_name, num_epochs, optimizer, early_stop,
                   best_metric) -> float:
        """Run one training stage; returns the best validation Rank-1 seen."""
        assert self.model is not None
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch(optimizer)
            self.history["train_losses"].append(train_loss)

            val_metrics = self._validate()
            if val_metrics is not None:
                self.history["val_rank1"].append(val_metrics["rank1"])
                self.history["val_rank5"].append(val_metrics["rank5"])
                self.history["val_mAP"].append(val_metrics["mAP"])
                logger.info(
                    f"[{stage_name}] epoch {epoch:3d} | loss {train_loss:.4f} | "
                    f"rank1 {val_metrics['rank1']:.4f} | "
                    f"rank5 {val_metrics['rank5']:.4f} | "
                    f"mAP {val_metrics['mAP']:.4f} | "
                    f"n_queries {val_metrics['n_queries']}")

                current = val_metrics["rank1"]
                if current > best_metric + self.config.min_delta:
                    best_metric = current
                    epochs_no_improve = 0
                    torch.save(self.model.state_dict(), self.model_path)
                    logger.info(f"  New best Rank-1: {best_metric:.4f} (saved)")
                else:
                    epochs_no_improve += 1
                    if early_stop and epochs_no_improve >= self.config.patience:
                        logger.info(f"  Early stopping after {epochs_no_improve} "
                                    f"epochs without Rank-1 improvement.")
                        break
            else:
                logger.info(f"[{stage_name}] epoch {epoch:3d} | loss {train_loss:.4f}")
                # No validation set: save every epoch as the latest checkpoint
                torch.save(self.model.state_dict(), self.model_path)

        return best_metric

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Build everything, run both stages, evaluate, plot training curves."""
        log_config(self.config, title="Iceberg Embedding Configuration")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._build_datasets()
        self._build_dataloader()
        self._build_model()
        assert self.model is not None

        best_rank1 = 0.0

        if self.config.stage1_epochs > 0:
            log_section("STAGE 1: head warmup (backbone frozen)")
            self.model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.stage1_lr,
                weight_decay=self.config.weight_decay,
            )
            best_rank1 = self._run_stage(
                "stage1", self.config.stage1_epochs, optimizer,
                early_stop=False, best_metric=best_rank1)
            self.history["stage_boundaries"].append(
                len(self.history["train_losses"]))

        if self.config.stage2_epochs > 0:
            log_section("STAGE 2: joint fine-tuning")
            self.model.unfreeze_last_blocks(self.config.unfreeze_last_n_blocks)
            optimizer = torch.optim.AdamW(
                self.model.trainable_param_groups(
                    backbone_lr=self.config.stage2_backbone_lr,
                    head_lr=self.config.stage2_head_lr,
                ),
                weight_decay=self.config.weight_decay,
            )
            best_rank1 = self._run_stage(
                "stage2", self.config.stage2_epochs, optimizer,
                early_stop=True, best_metric=best_rank1)

        results = self._final_evaluation()
        save_config_snapshot(self.config, MODELS_DIR / "embedding_config.yaml")
        return results

    # ------------------------------------------------------------------ #
    # Evaluation, plotting, and embedding generation
    # ------------------------------------------------------------------ #

    def _final_evaluation(self) -> Dict[str, Any]:
        """Evaluate the best checkpoint on the validation set."""
        if not self.model_path.exists():
            logger.warning("No checkpoint to evaluate.")
            return {}

        log_section("FINAL EVALUATION")
        assert self.model is not None
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device))
        metrics = self._validate()
        if metrics is not None:
            logger.info(f"Final Rank-1: {metrics['rank1']:.4f}")
            logger.info(f"Final Rank-5: {metrics['rank5']:.4f}")
            logger.info(f"Final mAP:    {metrics['mAP']:.4f}")
            logger.info(f"Queries:      {metrics['n_queries']}")
            return metrics
        logger.info("No validation set configured.")
        return {}


    @torch.no_grad()
    def generate_embeddings(self, image_dir, detection_file, output_path,
                            batch_size: int = 64) -> None:
        """Encode every annotation in one sequence and save the embedding dict.

        Args:
            image_dir: The sequence's images/ directory.
            detection_file: MOT-format annotation file (det.txt or gt.txt).
            output_path: Output .pt file (e.g. paths['det_embeddings']).
            batch_size: Inference batch size.
        """
        assert self.model is not None
        self.model.eval()
        if self.model_path.exists():
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device))

        image_dir, detection_file = str(image_dir), str(detection_file)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.image_ext is None:
            self.image_ext = get_image_ext(image_dir)

        detections = parse_annotations(detection_file)
        if not detections:
            logger.warning(f"No detections found in {detection_file}, skipping.")
            return

        dataset = IcebergInferenceDataset(
            detections=detections,
            full_frame_dir=image_dir,
            target_size=self.config.img_size,
            bbox_context=self.config.bbox_context,
            image_ext=self.image_ext,
            transform=_default_eval_transform(),
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        embeddings: Dict[str, torch.Tensor] = {}
        for crops, names in tqdm(loader, desc="Encoding detections"):
            crops = crops.to(self.device, non_blocking=True)
            feats = self.model(crops).cpu()  # Already L2-normalized
            for name, vec in zip(names, feats):
                embeddings[name] = vec

        torch.save(embeddings, output_path)
        logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")

    def load_for_inference(self) -> None:
        """Build the model and load the shared checkpoint (no training)."""
        if self.model is None:
            self._build_model()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {self.model_path}. "
                f"Train the embedding model first (train-embedding).")
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device))


def generate_dataset_embeddings(config: IcebergEmbeddingsConfig):
    """Generate embeddings for every sequence of a dataset (embed command).

    config.embed_source selects the annotation source:
        'detections'   det.txt -> detections/embeddings.pt (for tracking)
        'ground_truth' gt.txt  -> ground_truth/embeddings.pt (for GT
                       threshold derivation, see tracking.get_gt_thresholds)
    """
    if config.embed_source not in ("detections", "ground_truth"):
        raise ValueError(f"embed_source must be 'detections' or 'ground_truth', "
                         f"got '{config.embed_source}'")

    trainer = IcebergEmbeddingsTrainer(config)
    trainer.load_for_inference()

    sequences = get_sequences(config.dataset)
    for sequence_name, paths in sequences.items():
        if config.embed_source == "ground_truth":
            source, output = paths["ground_truth"], paths["gt_embeddings"]
        else:
            source, output = paths["detections"], paths["det_embeddings"]

        if not Path(source).exists():
            logger.warning(f"{sequence_name}: {source} not found, skipping")
            continue

        log_section(f"Generating embeddings: {sequence_name} ({config.embed_source})")
        trainer.generate_embeddings(paths["images"], source, output)