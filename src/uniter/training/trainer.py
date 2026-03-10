from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from uniter.config import AppConfig
from uniter.data.collate import RegionBatchCollator
from uniter.data.dataset import RegionBatch, RegionDataset, resolve_sample_splits
from uniter.data.manifest import RegionRecord, load_manifest
from uniter.models.multimodal import ModelOutputs, MultimodalRegionModel
from uniter.training.losses import (
    masked_accuracy,
    masked_cross_entropy_loss,
    segmentation_cross_entropy_loss,
    symmetric_contrastive_loss,
)
from uniter.utils.device import move_region_batch_to_device, resolve_device
from uniter.utils.logging import get_logger
from uniter.utils.seed import set_seed


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    alignment: torch.Tensor
    segmentation: torch.Tensor
    sentiment: torch.Tensor
    historical_sentiment: torch.Tensor
    lost_space: torch.Tensor
    segmentation_labeled_images: int
    segmentation_labeled_pixels: int
    sentiment_label_count: int
    historical_sentiment_label_count: int
    lost_space_label_count: int


class Trainer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.output_dir = config.output_dir
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.summary_dir = self.output_dir / "summaries"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.summary_dir / "history.json"
        self.training_state_path = self.output_dir / "training_state.json"
        self.best_summary_path = self.output_dir / "best_checkpoint.json"

        self.device = resolve_device(config.experiment.device)
        set_seed(config.experiment.seed)

        self.records = load_manifest(
            config.data.manifest_path,
            image_root=config.data.image_root,
            check_files=config.data.check_files_on_load,
        )
        split_map = resolve_sample_splits(self.records)

        self.collator = RegionBatchCollator(
            spatial_model_name=config.spatial_model.model_name,
            text_model_name=config.text_model.model_name,
            image_size=config.data.image_size,
            max_length=config.text_model.max_length,
            max_images_per_region=config.data.max_images_per_region,
            max_current_texts_per_region=config.data.max_current_texts_per_region,
            max_historical_texts_per_region=config.data.max_historical_texts_per_region,
            max_identity_texts_per_region=config.data.max_identity_texts_per_region,
            sentiment_ignore_index=config.sentiment.ignore_index,
            lost_space_ignore_index=config.lost_space.ignore_index,
            segmentation_ignore_index=config.spatial_supervision.ignore_index,
            segmentation_label_mapping=config.spatial_supervision.label_mapping,
        )
        self.train_loader = self._build_loader(split_map.get("train", []), shuffle=True)
        self.val_loader = self._build_loader(split_map.get("val", []), shuffle=False)

        self.model = MultimodalRegionModel(config).to(self.device)
        self.trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not self.trainable_parameters:
            raise RuntimeError("No trainable parameters found. Check the model freeze settings.")
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.use_amp = config.training.amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.start_epoch = 1
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.history = self._load_history()

        resume_path = config.training.resume_from
        if resume_path:
            self._resume_training(resume_path)

    def _build_loader(
        self,
        records: list[RegionRecord],
        *,
        shuffle: bool,
    ) -> DataLoader | None:
        if not records:
            return None
        dataset = RegionDataset(records)
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            drop_last=shuffle and len(dataset) > 1,
            num_workers=self.config.data.num_workers,
            collate_fn=self.collator,
        )

    def train(self) -> None:
        if self.train_loader is None:
            raise RuntimeError("Training split is empty. Please provide at least one train record.")

        self.logger.info("Starting training on %s with %d records.", self.device, len(self.records))
        patience_counter = 0
        completed_epoch = self.start_epoch - 1
        stop_reason = "completed"
        self._write_training_state(
            status="running",
            current_epoch=completed_epoch,
            stop_reason=None,
        )

        for epoch in range(self.start_epoch, self.config.training.epochs + 1):
            train_metrics = self._run_epoch(epoch, training=True)
            self.logger.info("Epoch %d train metrics: %s", epoch, train_metrics)

            val_metrics: dict[str, float] | None = None
            if self.val_loader is not None:
                val_metrics = self._run_epoch(epoch, training=False)
                self.logger.info("Epoch %d val metrics: %s", epoch, val_metrics)

            monitor_value = self._resolve_monitor_value(train_metrics, val_metrics)
            improved = self._is_improved(monitor_value)
            if improved:
                patience_counter = 0
                self.best_metric = monitor_value
                self.best_epoch = epoch
                if self.config.training.save_best:
                    self._save_checkpoint(
                        epoch,
                        checkpoint_path=self.checkpoint_dir / "best.pt",
                        is_best=True,
                        monitor_value=monitor_value,
                    )
                    self._write_best_summary()
            elif monitor_value is not None:
                patience_counter += 1

            self._write_summary(
                epoch,
                train_metrics,
                val_metrics,
                monitor_value=monitor_value,
                is_best=improved,
            )
            if epoch % self.config.training.save_every == 0:
                self._save_checkpoint(epoch, monitor_value=monitor_value)

            completed_epoch = epoch
            self._write_training_state(
                status="running",
                current_epoch=completed_epoch,
                stop_reason=None,
            )
            if (
                self.config.training.early_stopping_patience is not None
                and patience_counter >= self.config.training.early_stopping_patience
            ):
                stop_reason = (
                    "early_stopping:"
                    f"{self.config.training.monitor_metric}:"
                    f"{self.config.training.early_stopping_patience}"
                )
                self.logger.info(
                    "Stopping early at epoch %d after %d epochs without improvement.",
                    epoch,
                    patience_counter,
                )
                break

        final_status = "early_stopped" if stop_reason.startswith("early_stopping") else "completed"
        self._write_training_state(
            status=final_status,
            current_epoch=completed_epoch,
            stop_reason=stop_reason,
        )

    def _run_epoch(self, epoch: int, *, training: bool) -> dict[str, float]:
        loader = self.train_loader if training else self.val_loader
        if loader is None:
            return {}

        self.model.train(mode=training)
        total_loss = 0.0
        total_alignment_loss = 0.0
        total_segmentation_loss = 0.0
        total_sentiment_loss = 0.0
        total_historical_sentiment_loss = 0.0
        total_lost_space_loss = 0.0
        total_alignment_gap = 0.0
        total_ifi = 0.0
        total_mdi = 0.0
        total_iai = 0.0
        total_sentiment_correct = 0
        total_sentiment_labels = 0
        total_historical_sentiment_correct = 0
        total_historical_sentiment_labels = 0
        total_lost_space_correct = 0
        total_lost_space_labels = 0
        total_segmentation_labeled_images = 0
        total_segmentation_labeled_pixels = 0
        num_batches = 0
        mdi_batches = 0
        iai_batches = 0
        sentiment_batches = 0
        historical_sentiment_batches = 0
        lost_space_batches = 0
        segmentation_batches = 0

        context = torch.enable_grad if training else torch.no_grad
        with context():
            for step, batch in enumerate(loader, start=1):
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                batch = move_region_batch_to_device(batch, self.device)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(batch)
                    losses = self._compute_loss(outputs, batch)

                if training:
                    if self.use_amp:
                        self.scaler.scale(losses.total).backward()
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(
                            self.trainable_parameters,
                            max_norm=self.config.training.grad_clip_norm,
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        losses.total.backward()
                        clip_grad_norm_(
                            self.trainable_parameters,
                            max_norm=self.config.training.grad_clip_norm,
                        )
                        self.optimizer.step()

                total_loss += float(losses.total.detach().cpu())
                total_alignment_loss += float(losses.alignment.detach().cpu())
                total_segmentation_loss += float(losses.segmentation.detach().cpu())
                total_sentiment_loss += float(losses.sentiment.detach().cpu())
                total_historical_sentiment_loss += float(
                    losses.historical_sentiment.detach().cpu()
                )
                total_lost_space_loss += float(losses.lost_space.detach().cpu())
                total_alignment_gap += float(outputs.alignment_gap.mean().detach().cpu())
                total_ifi += float(outputs.ifi.mean().detach().cpu())
                total_segmentation_labeled_images += losses.segmentation_labeled_images
                total_segmentation_labeled_pixels += losses.segmentation_labeled_pixels
                if losses.segmentation_labeled_images > 0:
                    segmentation_batches += 1
                if losses.sentiment_label_count > 0:
                    sentiment_batches += 1
                if losses.historical_sentiment_label_count > 0:
                    historical_sentiment_batches += 1
                if losses.lost_space_label_count > 0:
                    lost_space_batches += 1

                if outputs.sentiment_logits is not None:
                    correct, label_count = masked_accuracy(
                        outputs.sentiment_logits,
                        batch.targets["sentiment_label"],
                        ignore_index=self.config.sentiment.ignore_index,
                    )
                    total_sentiment_correct += correct
                    total_sentiment_labels += label_count

                if outputs.historical_sentiment_logits is not None:
                    correct, label_count = masked_accuracy(
                        outputs.historical_sentiment_logits,
                        batch.targets["historical_sentiment_label"],
                        ignore_index=self.config.sentiment.ignore_index,
                    )
                    total_historical_sentiment_correct += correct
                    total_historical_sentiment_labels += label_count

                if outputs.lost_space_logits is not None:
                    correct, label_count = masked_accuracy(
                        outputs.lost_space_logits,
                        batch.targets["lost_space_label"],
                        ignore_index=self.config.lost_space.ignore_index,
                    )
                    total_lost_space_correct += correct
                    total_lost_space_labels += label_count

                valid_mdi = outputs.mdi[~torch.isnan(outputs.mdi)]
                if valid_mdi.numel() > 0:
                    total_mdi += float(valid_mdi.mean().detach().cpu())
                    mdi_batches += 1

                valid_iai = outputs.iai[~torch.isnan(outputs.iai)]
                if valid_iai.numel() > 0:
                    total_iai += float(valid_iai.mean().detach().cpu())
                    iai_batches += 1

                num_batches += 1

                if step % self.config.training.log_every == 0:
                    self.logger.info(
                        (
                            "epoch=%d split=%s step=%d loss=%.4f alignment=%.4f "
                            "segmentation=%.4f sentiment=%.4f historical_sentiment=%.4f "
                            "lost_space=%.4f labeled_regions=%d historical_labeled_regions=%d "
                            "lost_space_labeled_regions=%d segmentation_labeled_images=%d"
                        ),
                        epoch,
                        "train" if training else "val",
                        step,
                        float(losses.total.detach().cpu()),
                        float(losses.alignment.detach().cpu()),
                        float(losses.segmentation.detach().cpu()),
                        float(losses.sentiment.detach().cpu()),
                        float(losses.historical_sentiment.detach().cpu()),
                        float(losses.lost_space.detach().cpu()),
                        losses.sentiment_label_count,
                        losses.historical_sentiment_label_count,
                        losses.lost_space_label_count,
                        losses.segmentation_labeled_images,
                    )

        divisor = max(num_batches, 1)
        return {
            "loss": total_loss / divisor,
            "alignment_loss": total_alignment_loss / divisor,
            "segmentation_loss": total_segmentation_loss / max(segmentation_batches, 1),
            "sentiment_loss": total_sentiment_loss / max(sentiment_batches, 1),
            "historical_sentiment_loss": (
                total_historical_sentiment_loss / max(historical_sentiment_batches, 1)
            ),
            "lost_space_loss": total_lost_space_loss / max(lost_space_batches, 1),
            "alignment_gap": total_alignment_gap / divisor,
            "ifi": total_ifi / divisor,
            "mdi": total_mdi / max(mdi_batches, 1),
            "iai": total_iai / max(iai_batches, 1),
            "sentiment_accuracy": total_sentiment_correct / max(total_sentiment_labels, 1),
            "sentiment_label_count": float(total_sentiment_labels),
            "historical_sentiment_accuracy": (
                total_historical_sentiment_correct / max(total_historical_sentiment_labels, 1)
            ),
            "historical_sentiment_label_count": float(total_historical_sentiment_labels),
            "lost_space_accuracy": total_lost_space_correct / max(total_lost_space_labels, 1),
            "lost_space_label_count": float(total_lost_space_labels),
            "segmentation_labeled_images": float(total_segmentation_labeled_images),
            "segmentation_labeled_pixels": float(total_segmentation_labeled_pixels),
        }

    def _compute_loss(self, outputs: ModelOutputs, batch: RegionBatch) -> LossBreakdown:
        alignment_loss = symmetric_contrastive_loss(
            outputs.image_embeddings,
            outputs.text_embeddings,
            temperature=self.config.alignment.temperature,
        )
        segmentation_result = segmentation_cross_entropy_loss(
            outputs.segmentation_logits,
            batch.segmentation_labels,
            ignore_index=self.config.spatial_supervision.ignore_index,
        )
        sentiment_loss = alignment_loss.new_tensor(0.0)
        historical_sentiment_loss = alignment_loss.new_tensor(0.0)
        lost_space_loss = alignment_loss.new_tensor(0.0)
        sentiment_label_count = 0
        historical_sentiment_label_count = 0
        lost_space_label_count = 0

        if outputs.sentiment_logits is not None:
            sentiment_result = masked_cross_entropy_loss(
                outputs.sentiment_logits,
                batch.targets["sentiment_label"],
                ignore_index=self.config.sentiment.ignore_index,
                label_smoothing=self.config.losses.sentiment_label_smoothing,
            )
            sentiment_loss = sentiment_result.loss
            sentiment_label_count = sentiment_result.valid_count

        if outputs.historical_sentiment_logits is not None:
            historical_sentiment_result = masked_cross_entropy_loss(
                outputs.historical_sentiment_logits,
                batch.targets["historical_sentiment_label"],
                ignore_index=self.config.sentiment.ignore_index,
                label_smoothing=self.config.losses.sentiment_label_smoothing,
            )
            historical_sentiment_loss = historical_sentiment_result.loss
            historical_sentiment_label_count = historical_sentiment_result.valid_count

        if outputs.lost_space_logits is not None:
            lost_space_result = masked_cross_entropy_loss(
                outputs.lost_space_logits,
                batch.targets["lost_space_label"],
                ignore_index=self.config.lost_space.ignore_index,
                label_smoothing=self.config.losses.sentiment_label_smoothing,
            )
            lost_space_loss = lost_space_result.loss
            lost_space_label_count = lost_space_result.valid_count

        total_loss = (
            alignment_loss * self.config.losses.alignment_weight
            + segmentation_result.loss * self.config.losses.segmentation_weight
            + sentiment_loss * self.config.losses.sentiment_weight
            + historical_sentiment_loss * self.config.losses.historical_sentiment_weight
            + lost_space_loss * self.config.losses.lost_space_weight
        )
        return LossBreakdown(
            total=total_loss,
            alignment=alignment_loss,
            segmentation=segmentation_result.loss,
            sentiment=sentiment_loss,
            historical_sentiment=historical_sentiment_loss,
            lost_space=lost_space_loss,
            segmentation_labeled_images=segmentation_result.labeled_images,
            segmentation_labeled_pixels=segmentation_result.labeled_pixels,
            sentiment_label_count=sentiment_label_count,
            historical_sentiment_label_count=historical_sentiment_label_count,
            lost_space_label_count=lost_space_label_count,
        )

    def _resolve_monitor_value(
        self,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
    ) -> float | None:
        split_name, metric_name = self.config.training.monitor_metric.split(".", maxsplit=1)
        if split_name == "val" and val_metrics is not None and metric_name in val_metrics:
            return val_metrics[metric_name]
        if split_name == "train" and metric_name in train_metrics:
            return train_metrics[metric_name]
        fallback_metrics = val_metrics if val_metrics is not None else train_metrics
        if metric_name in fallback_metrics:
            self.logger.warning(
                "Requested monitor metric '%s' was unavailable; falling back to '%s.%s'.",
                self.config.training.monitor_metric,
                "val" if val_metrics is not None else "train",
                metric_name,
            )
            return fallback_metrics[metric_name]
        return None

    def _is_improved(self, monitor_value: float | None) -> bool:
        if monitor_value is None:
            return False
        if self.best_metric is None:
            return True
        if self.config.training.maximize_monitor_metric:
            return monitor_value > self.best_metric
        return monitor_value < self.best_metric

    def _resume_training(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        payload = torch.load(path, map_location=self.device)
        if not isinstance(payload, dict) or "model" not in payload:
            raise RuntimeError(f"Invalid training checkpoint: {path}")

        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        scaler_state = payload.get("scaler")
        if scaler_state and self.use_amp:
            self.scaler.load_state_dict(scaler_state)

        epoch = int(payload.get("epoch", 0))
        self.start_epoch = epoch + 1
        best_metric = payload.get("best_metric")
        self.best_metric = float(best_metric) if best_metric is not None else self.best_metric
        best_epoch = payload.get("best_epoch")
        self.best_epoch = int(best_epoch) if best_epoch is not None else self.best_epoch
        self.logger.info("Resumed training from %s at epoch %d.", path, epoch)

    def _save_checkpoint(
        self,
        epoch: int,
        *,
        checkpoint_path: Path | None = None,
        is_best: bool = False,
        monitor_value: float | None = None,
    ) -> None:
        destination = checkpoint_path or self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": asdict(self.config),
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "monitor_metric": self.config.training.monitor_metric,
                "monitor_value": monitor_value,
                "is_best": is_best,
            },
            destination,
        )
        self.logger.info("Saved checkpoint to %s", destination)

    def _write_summary(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float] | None,
        *,
        monitor_value: float | None,
        is_best: bool,
    ) -> None:
        payload: dict[str, Any] = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "monitor_metric": self.config.training.monitor_metric,
            "monitor_value": monitor_value,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "is_best": is_best,
        }
        summary_path = self.summary_dir / f"epoch_{epoch:03d}.json"
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.history.append(payload)
        self.history_path.write_text(
            json.dumps(self.history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_best_summary(self) -> None:
        payload = {
            "best_checkpoint": str((self.checkpoint_dir / "best.pt").resolve()),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "monitor_metric": self.config.training.monitor_metric,
        }
        self.best_summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_training_state(
        self,
        *,
        status: str,
        current_epoch: int,
        stop_reason: str | None,
    ) -> None:
        payload = {
            "status": status,
            "current_epoch": current_epoch,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "monitor_metric": self.config.training.monitor_metric,
            "resume_from": self.config.training.resume_from,
            "stop_reason": stop_reason,
        }
        self.training_state_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []
