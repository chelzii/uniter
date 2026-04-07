from __future__ import annotations

import json
import math
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
from uniter.inference.judgement import (
    fuse_region_judgement,
    judge_region_metrics,
    map_rule_level_to_class_index,
)
from uniter.models.multimodal import ModelOutputs, MultimodalRegionModel
from uniter.training.losses import (
    masked_accuracy,
    masked_cross_entropy_loss,
    masked_symmetric_contrastive_loss,
    segmentation_cross_entropy_loss,
    symmetric_contrastive_loss,
)
from uniter.utils.device import move_region_batch_to_device, resolve_device
from uniter.utils.logging import get_logger
from uniter.utils.seed import set_seed


def _non_empty_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_record_data_summary(records: list[RegionRecord]) -> dict[str, Any]:
    split_summary: dict[str, dict[str, int]] = {
        split: {
            "sample_count": 0,
            "image_count": 0,
            "current_text_count": 0,
            "historical_text_count": 0,
            "identity_text_count": 0,
            "current_sentiment_record_label_count": 0,
            "historical_sentiment_record_label_count": 0,
            "lost_space_label_count": 0,
            "sentiment_label_count": 0,
            "historical_sentiment_label_count": 0,
        }
        for split in ("train", "val", "test")
    }
    parent_region_ids: set[str] = set()
    bootstrap_views: set[str] = set()
    for record in records:
        bucket = split_summary.setdefault(
            record.split,
            {
                "sample_count": 0,
                "image_count": 0,
                "current_text_count": 0,
                "historical_text_count": 0,
                "identity_text_count": 0,
                "current_sentiment_record_label_count": 0,
                "historical_sentiment_record_label_count": 0,
                "lost_space_label_count": 0,
                "sentiment_label_count": 0,
                "historical_sentiment_label_count": 0,
            },
        )
        bucket["sample_count"] += 1
        bucket["image_count"] += len(record.image_paths)
        bucket["current_text_count"] += len(record.current_texts)
        bucket["historical_text_count"] += len(record.historical_texts)
        bucket["identity_text_count"] += len(record.identity_texts)
        bucket["current_sentiment_record_label_count"] += sum(
            int(label is not None) for label in record.current_sentiment_labels
        )
        bucket["historical_sentiment_record_label_count"] += sum(
            int(label is not None) for label in record.historical_sentiment_labels
        )
        bucket["lost_space_label_count"] += int(record.targets.lost_space_label is not None)
        bucket["sentiment_label_count"] += int(record.targets.sentiment_label is not None)
        bucket["historical_sentiment_label_count"] += int(
            record.targets.historical_sentiment_label is not None
        )
        parent_region_id = _non_empty_string(record.metadata.get("parent_region_id"))
        if parent_region_id is not None:
            parent_region_ids.add(parent_region_id)
        bootstrap_view = _non_empty_string(record.metadata.get("bootstrap_view"))
        if bootstrap_view is not None:
            bootstrap_views.add(bootstrap_view)

    empty_labels = []
    lost_space_labels = sum(
        split_summary[split]["lost_space_label_count"] for split in split_summary
    )
    if lost_space_labels == 0:
        empty_labels.append("lost_space_label")

    current_sentiment_labels = sum(
        split_summary[split]["sentiment_label_count"]
        + split_summary[split]["current_sentiment_record_label_count"]
        for split in split_summary
    )
    if current_sentiment_labels == 0:
        empty_labels.append("sentiment_label")

    historical_sentiment_labels = sum(
        split_summary[split]["historical_sentiment_label_count"]
        + split_summary[split]["historical_sentiment_record_label_count"]
        for split in split_summary
    )
    if historical_sentiment_labels == 0:
        empty_labels.append("historical_sentiment_label")
    is_single_region_bootstrap = (
        len(parent_region_ids) == 1
        and len(records) > 1
        and len(bootstrap_views) >= 1
    )
    return {
        "record_count": len(records),
        "split_summary": split_summary,
        "parent_region_ids": sorted(parent_region_ids),
        "bootstrap_views": sorted(bootstrap_views),
        "is_single_region_bootstrap": is_single_region_bootstrap,
        "empty_labels": empty_labels,
    }


@dataclass(slots=True)
class LossBreakdown:
    total: torch.Tensor
    alignment: torch.Tensor
    identity: torch.Tensor
    segmentation: torch.Tensor
    spatial_proxy: torch.Tensor
    sentiment: torch.Tensor
    historical_sentiment: torch.Tensor
    lost_space: torch.Tensor
    lost_space_consistency: torch.Tensor
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
        self.data_summary_path = self.summary_dir / "data_summary.json"

        self.device = resolve_device(config.experiment.device)
        set_seed(config.experiment.seed)

        self.records = load_manifest(
            config.data.manifest_path,
            image_root=config.data.image_root,
            check_files=config.data.check_files_on_load,
        )
        split_map = resolve_sample_splits(self.records)
        self.data_summary = build_record_data_summary(self.records)

        self.collator = RegionBatchCollator(
            random_sample=True,
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
        self.eval_collator = RegionBatchCollator(
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
        self.train_loader = self._build_loader(
            split_map.get("train", []),
            shuffle=True,
            collator=self.collator,
        )
        self.val_loader = self._build_loader(
            split_map.get("val", []),
            shuffle=False,
            collator=self.eval_collator,
        )

        self.model = MultimodalRegionModel(config).to(self.device)
        self._initialize_classifier_priors(split_map.get("train", []))
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
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=self.use_amp,
        )

        self.start_epoch = 1
        self.best_metric: float | None = None
        self.best_epoch: int | None = None

        resume_path = config.training.resume_from
        if resume_path is None:
            self._reset_training_artifacts()
        self.history = self._load_history()
        if resume_path:
            self._resume_training(resume_path)

    def _build_loader(
        self,
        records: list[RegionRecord],
        *,
        shuffle: bool,
        collator: RegionBatchCollator,
    ) -> DataLoader | None:
        if not records:
            return None
        dataset = RegionDataset(records)
        return DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.config.data.num_workers,
            collate_fn=collator,
        )

    def _initialize_classifier_priors(self, train_records: list[RegionRecord]) -> None:
        if not train_records:
            return
        self._initialize_head_bias_from_labels(
            head=getattr(self.model, "current_sentiment_head", None),
            num_classes=self.config.sentiment.num_classes,
            labels=self._collect_classification_labels(
                train_records,
                record_label_attr="current_sentiment_labels",
                region_label_attr="sentiment_label",
            ),
            head_name="current_sentiment",
        )
        self._initialize_head_bias_from_labels(
            head=getattr(self.model, "historical_sentiment_head", None),
            num_classes=self.config.sentiment.num_classes,
            labels=self._collect_classification_labels(
                train_records,
                record_label_attr="historical_sentiment_labels",
                region_label_attr="historical_sentiment_label",
            ),
            head_name="historical_sentiment",
        )
        self._initialize_head_bias_from_labels(
            head=getattr(self.model, "lost_space_head", None),
            num_classes=self.config.lost_space.num_classes,
            labels=[record.targets.lost_space_label for record in train_records],
            head_name="lost_space",
        )

    def _collect_classification_labels(
        self,
        records: list[RegionRecord],
        *,
        record_label_attr: str,
        region_label_attr: str,
    ) -> list[int | None]:
        flattened: list[int | None] = []
        for record in records:
            record_labels = getattr(record, record_label_attr)
            if record_labels:
                flattened.extend(record_labels)
                continue
            flattened.append(getattr(record.targets, region_label_attr))
        return flattened

    def _initialize_head_bias_from_labels(
        self,
        *,
        head: torch.nn.Module | None,
        num_classes: int,
        labels: list[int | None],
        head_name: str,
    ) -> None:
        if head is None or num_classes < 2:
            return
        classifier = getattr(head, "classifier", None)
        linear = classifier[-1] if isinstance(classifier, torch.nn.Sequential) and classifier else None
        if not isinstance(linear, torch.nn.Linear):
            return

        counts = torch.ones(num_classes, dtype=torch.float32)
        labeled_count = 0
        for label in labels:
            if label is None or label < 0 or label >= num_classes:
                continue
            counts[int(label)] += 1.0
            labeled_count += 1
        if labeled_count == 0:
            return

        priors = counts / counts.sum()
        with torch.no_grad():
            linear.bias.copy_(priors.log().to(device=linear.bias.device, dtype=linear.bias.dtype))
        self.logger.info(
            "Initialized %s head bias from %d train labels: %s",
            head_name,
            labeled_count,
            [round(float(value), 4) for value in priors.tolist()],
        )

    def _reset_training_artifacts(self) -> None:
        for path in self.checkpoint_dir.glob("*.pt"):
            path.unlink(missing_ok=True)
        for path in self.summary_dir.glob("epoch_*.json"):
            path.unlink(missing_ok=True)
        for path in (
            self.history_path,
            self.training_state_path,
            self.best_summary_path,
            self.data_summary_path,
        ):
            path.unlink(missing_ok=True)

    def train(self) -> None:
        if self.train_loader is None:
            raise RuntimeError("Training split is empty. Please provide at least one train record.")

        self.logger.info("Starting training on %s with %d records.", self.device, len(self.records))
        self._log_data_summary()
        patience_counter = 0
        completed_epoch = self.start_epoch - 1
        stop_reason = "completed"
        self._write_training_state(
            status="running",
            current_epoch=completed_epoch,
            stop_reason=None,
        )
        try:
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
        except Exception as exc:
            failure_reason = f"failed:{type(exc).__name__}:{exc}"
            self._write_training_state(
                status="failed",
                current_epoch=completed_epoch,
                stop_reason=failure_reason,
            )
            raise

        final_status = "early_stopped" if stop_reason.startswith("early_stopping") else "completed"
        self._write_training_state(
            status=final_status,
            current_epoch=completed_epoch,
            stop_reason=stop_reason,
        )

    def _log_data_summary(self) -> None:
        self.data_summary_path.write_text(
            json.dumps(self.data_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        for split in ("train", "val", "test"):
            split_bucket = self.data_summary["split_summary"].get(split, {})
            self.logger.info(
                "Data summary [%s]: samples=%d images=%d current_texts=%d historical_texts=%d "
                "identity_texts=%d current_text_labels=%d historical_text_labels=%d "
                "lost_space_labels=%d sentiment_labels=%d "
                "historical_sentiment_labels=%d",
                split,
                split_bucket.get("sample_count", 0),
                split_bucket.get("image_count", 0),
                split_bucket.get("current_text_count", 0),
                split_bucket.get("historical_text_count", 0),
                split_bucket.get("identity_text_count", 0),
                split_bucket.get("current_sentiment_record_label_count", 0),
                split_bucket.get("historical_sentiment_record_label_count", 0),
                split_bucket.get("lost_space_label_count", 0),
                split_bucket.get("sentiment_label_count", 0),
                split_bucket.get("historical_sentiment_label_count", 0),
            )
        if self.data_summary["is_single_region_bootstrap"]:
            self.logger.warning(
                "Current manifest is a single-region bootstrap split. "
                "It supports prototype validation, not multi-region comparison."
            )
        if self.data_summary["empty_labels"]:
            self.logger.warning(
                "These supervision targets are currently empty in the manifest: %s",
                ", ".join(self.data_summary["empty_labels"]),
            )
        self.logger.info("Persisted data summary to %s", self.data_summary_path)

    def _resolve_sentiment_supervision(
        self,
        *,
        region_logits: torch.Tensor | None,
        region_targets: torch.Tensor,
        record_logits: torch.Tensor | None,
        record_targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if (
            record_logits is not None
            and record_targets is not None
            and torch.any(record_targets != self.config.sentiment.ignore_index)
        ):
            return record_logits, record_targets
        return region_logits, region_targets

    def _run_epoch(self, epoch: int, *, training: bool) -> dict[str, float]:
        loader = self.train_loader if training else self.val_loader
        if loader is None:
            return {}

        self.model.train(mode=training)
        total_loss = 0.0
        total_alignment_loss = 0.0
        total_identity_loss = 0.0
        total_segmentation_loss = 0.0
        total_spatial_proxy_loss = 0.0
        total_sentiment_loss = 0.0
        total_historical_sentiment_loss = 0.0
        total_lost_space_loss = 0.0
        total_lost_space_consistency_loss = 0.0
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
        total_rule_lost_space_correct = 0
        total_rule_lost_space_labels = 0
        total_final_lost_space_correct = 0
        total_final_lost_space_labels = 0
        total_rule_model_agreement = 0
        total_rule_model_compared = 0
        total_rule_model_gap = 0.0
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
                    if not losses.total.requires_grad:
                        self.logger.warning(
                            "Skipping optimizer step at epoch=%d step=%d because the total loss "
                            "has no trainable gradient. Check batch size and label availability.",
                            epoch,
                            step,
                        )
                        num_batches += 1
                        continue
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
                total_identity_loss += float(losses.identity.detach().cpu())
                total_segmentation_loss += float(losses.segmentation.detach().cpu())
                total_spatial_proxy_loss += float(losses.spatial_proxy.detach().cpu())
                total_sentiment_loss += float(losses.sentiment.detach().cpu())
                total_historical_sentiment_loss += float(
                    losses.historical_sentiment.detach().cpu()
                )
                total_lost_space_loss += float(losses.lost_space.detach().cpu())
                total_lost_space_consistency_loss += float(
                    losses.lost_space_consistency.detach().cpu()
                )
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

                sentiment_logits, sentiment_targets = self._resolve_sentiment_supervision(
                    region_logits=outputs.sentiment_logits,
                    region_targets=batch.targets["sentiment_label"],
                    record_logits=outputs.current_sentiment_record_logits,
                    record_targets=batch.current_sentiment_labels,
                )
                if sentiment_logits is not None:
                    correct, label_count = masked_accuracy(
                        sentiment_logits,
                        sentiment_targets,
                        ignore_index=self.config.sentiment.ignore_index,
                    )
                    total_sentiment_correct += correct
                    total_sentiment_labels += label_count

                (
                    historical_sentiment_logits,
                    historical_sentiment_targets,
                ) = self._resolve_sentiment_supervision(
                    region_logits=outputs.historical_sentiment_logits,
                    region_targets=batch.targets["historical_sentiment_label"],
                    record_logits=outputs.historical_sentiment_record_logits,
                    record_targets=batch.historical_sentiment_labels,
                )
                if historical_sentiment_logits is not None:
                    correct, label_count = masked_accuracy(
                        historical_sentiment_logits,
                        historical_sentiment_targets,
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

                    (
                        rule_correct,
                        rule_label_count,
                        final_correct,
                        final_label_count,
                        agreement_count,
                        compared_count,
                        gap_sum,
                    ) = self._compute_lost_space_judgement_metrics(outputs, batch)
                    total_rule_lost_space_correct += rule_correct
                    total_rule_lost_space_labels += rule_label_count
                    total_final_lost_space_correct += final_correct
                    total_final_lost_space_labels += final_label_count
                    total_rule_model_agreement += agreement_count
                    total_rule_model_compared += compared_count
                    total_rule_model_gap += gap_sum

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
                            "identity=%.4f segmentation=%.4f spatial_proxy=%.4f "
                            "sentiment=%.4f historical_sentiment=%.4f "
                            "lost_space=%.4f lost_space_consistency=%.4f "
                            "labeled_regions=%d historical_labeled_regions=%d "
                            "lost_space_labeled_regions=%d segmentation_labeled_images=%d"
                        ),
                        epoch,
                        "train" if training else "val",
                        step,
                        float(losses.total.detach().cpu()),
                        float(losses.alignment.detach().cpu()),
                        float(losses.identity.detach().cpu()),
                        float(losses.segmentation.detach().cpu()),
                        float(losses.spatial_proxy.detach().cpu()),
                        float(losses.sentiment.detach().cpu()),
                        float(losses.historical_sentiment.detach().cpu()),
                        float(losses.lost_space.detach().cpu()),
                        float(losses.lost_space_consistency.detach().cpu()),
                        losses.sentiment_label_count,
                        losses.historical_sentiment_label_count,
                        losses.lost_space_label_count,
                        losses.segmentation_labeled_images,
                    )

        divisor = max(num_batches, 1)
        model_lost_space_accuracy = total_lost_space_correct / max(total_lost_space_labels, 1)
        rule_lost_space_accuracy = (
            total_rule_lost_space_correct / max(total_rule_lost_space_labels, 1)
        )
        final_lost_space_accuracy = (
            total_final_lost_space_correct / max(total_final_lost_space_labels, 1)
        )
        rule_model_agreement = total_rule_model_agreement / max(total_rule_model_compared, 1)
        rule_model_gap = total_rule_model_gap / max(total_rule_model_compared, 1)
        selection_score = self._compute_selection_score(
            loss=total_loss / divisor,
            model_lost_space_accuracy=model_lost_space_accuracy,
            rule_lost_space_accuracy=rule_lost_space_accuracy,
            final_lost_space_accuracy=final_lost_space_accuracy,
            rule_model_agreement=rule_model_agreement,
            sentiment_accuracy=total_sentiment_correct / max(total_sentiment_labels, 1),
            historical_sentiment_accuracy=(
                total_historical_sentiment_correct / max(total_historical_sentiment_labels, 1)
            ),
            labeled_count=total_final_lost_space_labels,
            compared_count=total_rule_model_compared,
            sentiment_labeled_count=total_sentiment_labels,
            historical_sentiment_labeled_count=total_historical_sentiment_labels,
        )

        return {
            "loss": total_loss / divisor,
            "alignment_loss": total_alignment_loss / divisor,
            "identity_loss": total_identity_loss / divisor,
            "segmentation_loss": total_segmentation_loss / max(segmentation_batches, 1),
            "spatial_proxy_loss": total_spatial_proxy_loss / divisor,
            "sentiment_loss": total_sentiment_loss / max(sentiment_batches, 1),
            "historical_sentiment_loss": (
                total_historical_sentiment_loss / max(historical_sentiment_batches, 1)
            ),
            "lost_space_loss": total_lost_space_loss / max(lost_space_batches, 1),
            "lost_space_consistency_loss": (
                total_lost_space_consistency_loss / max(lost_space_batches, 1)
            ),
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
            "lost_space_accuracy": model_lost_space_accuracy,
            "model_lost_space_accuracy": model_lost_space_accuracy,
            "rule_lost_space_accuracy": rule_lost_space_accuracy,
            "final_lost_space_accuracy": final_lost_space_accuracy,
            "lost_space_label_count": float(total_lost_space_labels),
            "rule_model_agreement": rule_model_agreement,
            "rule_model_gap": rule_model_gap,
            "selection_score": selection_score,
            "segmentation_labeled_images": float(total_segmentation_labeled_images),
            "segmentation_labeled_pixels": float(total_segmentation_labeled_pixels),
        }

    def _compute_loss(self, outputs: ModelOutputs, batch: RegionBatch) -> LossBreakdown:
        alignment_loss = symmetric_contrastive_loss(
            outputs.image_embeddings,
            outputs.text_embeddings,
            temperature=self.config.alignment.temperature,
        )
        identity_alignment_loss = masked_symmetric_contrastive_loss(
            torch.nn.functional.normalize(
                outputs.image_embeddings + outputs.text_embeddings,
                dim=-1,
            ),
            outputs.identity_embeddings,
            temperature=self.config.alignment.temperature,
            valid_mask=outputs.identity_region_mask,
        )
        segmentation_result = segmentation_cross_entropy_loss(
            outputs.segmentation_logits,
            batch.segmentation_labels,
            ignore_index=self.config.spatial_supervision.ignore_index,
        )
        sentiment_loss = alignment_loss.new_tensor(0.0)
        historical_sentiment_loss = alignment_loss.new_tensor(0.0)
        lost_space_loss = alignment_loss.new_tensor(0.0)
        lost_space_consistency_loss = alignment_loss.new_tensor(0.0)
        spatial_proxy_loss = alignment_loss.new_tensor(0.0)
        sentiment_label_count = 0
        historical_sentiment_label_count = 0
        lost_space_label_count = 0

        sentiment_logits, sentiment_targets = self._resolve_sentiment_supervision(
            region_logits=outputs.sentiment_logits,
            region_targets=batch.targets["sentiment_label"],
            record_logits=outputs.current_sentiment_record_logits,
            record_targets=batch.current_sentiment_labels,
        )
        if sentiment_logits is not None:
            sentiment_result = masked_cross_entropy_loss(
                sentiment_logits,
                sentiment_targets,
                ignore_index=self.config.sentiment.ignore_index,
                label_smoothing=self.config.losses.sentiment_label_smoothing,
            )
            sentiment_loss = sentiment_result.loss
            sentiment_label_count = sentiment_result.valid_count

        (
            historical_sentiment_logits,
            historical_sentiment_targets,
        ) = self._resolve_sentiment_supervision(
            region_logits=outputs.historical_sentiment_logits,
            region_targets=batch.targets["historical_sentiment_label"],
            record_logits=outputs.historical_sentiment_record_logits,
            record_targets=batch.historical_sentiment_labels,
        )
        if historical_sentiment_logits is not None:
            historical_sentiment_result = masked_cross_entropy_loss(
                historical_sentiment_logits,
                historical_sentiment_targets,
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
            rule_targets = self._build_rule_targets(outputs)
            rule_consistency_result = masked_cross_entropy_loss(
                outputs.lost_space_logits,
                rule_targets,
                ignore_index=self.config.lost_space.ignore_index,
                label_smoothing=self.config.losses.sentiment_label_smoothing,
            )
            lost_space_consistency_loss = rule_consistency_result.loss

        if outputs.spatial_proxy_score is not None:
            spatial_proxy_loss = torch.nan_to_num(
                outputs.spatial_proxy_score,
                nan=0.0,
            ).mean()

        total_loss = (
            alignment_loss * self.config.losses.alignment_weight
            + identity_alignment_loss * self.config.losses.identity_weight
            + segmentation_result.loss * self.config.losses.segmentation_weight
            + spatial_proxy_loss * self.config.losses.spatial_proxy_weight
            + sentiment_loss * self.config.losses.sentiment_weight
            + historical_sentiment_loss * self.config.losses.historical_sentiment_weight
            + lost_space_loss * self.config.losses.lost_space_weight
            + lost_space_consistency_loss * self.config.losses.lost_space_consistency_weight
        )
        return LossBreakdown(
            total=total_loss,
            alignment=alignment_loss,
            identity=identity_alignment_loss,
            segmentation=segmentation_result.loss,
            spatial_proxy=spatial_proxy_loss,
            sentiment=sentiment_loss,
            historical_sentiment=historical_sentiment_loss,
            lost_space=lost_space_loss,
            lost_space_consistency=lost_space_consistency_loss,
            segmentation_labeled_images=segmentation_result.labeled_images,
            segmentation_labeled_pixels=segmentation_result.labeled_pixels,
            sentiment_label_count=sentiment_label_count,
            historical_sentiment_label_count=historical_sentiment_label_count,
            lost_space_label_count=lost_space_label_count,
        )

    def _build_rule_targets(self, outputs: ModelOutputs) -> torch.Tensor:
        targets: list[int] = []
        class_names = self.config.lost_space.class_names
        ignore_index = self.config.lost_space.ignore_index
        for region_index in range(outputs.ifi.shape[0]):
            judgement = judge_region_metrics(
                ifi=self._optional_float(outputs.ifi, region_index),
                mdi=self._optional_float(outputs.mdi, region_index),
                alignment_gap=self._optional_float(outputs.alignment_gap, region_index),
                iai=self._optional_float(outputs.iai, region_index),
                config=self.config,
                single_region_bootstrap=self.data_summary["is_single_region_bootstrap"],
            )
            mapped = map_rule_level_to_class_index(judgement.final_level, class_names)
            targets.append(ignore_index if mapped is None else mapped)
        return torch.tensor(
            targets,
            device=outputs.ifi.device,
            dtype=torch.long,
        )

    def _compute_lost_space_judgement_metrics(
        self,
        outputs: ModelOutputs,
        batch: RegionBatch,
    ) -> tuple[int, int, int, int, int, int, float]:
        rule_correct = 0
        rule_label_count = 0
        final_correct = 0
        final_label_count = 0
        agreement_count = 0
        compared_count = 0
        gap_sum = 0.0
        class_names = self.config.lost_space.class_names
        ignore_index = self.config.lost_space.ignore_index

        for region_index in range(outputs.ifi.shape[0]):
            rule = judge_region_metrics(
                ifi=self._optional_float(outputs.ifi, region_index),
                mdi=self._optional_float(outputs.mdi, region_index),
                alignment_gap=self._optional_float(outputs.alignment_gap, region_index),
                iai=self._optional_float(outputs.iai, region_index),
                config=self.config,
                single_region_bootstrap=self.data_summary["is_single_region_bootstrap"],
            )
            logits = (
                outputs.lost_space_logits[region_index]
                if outputs.lost_space_logits is not None
                else None
            )
            fused = fuse_region_judgement(
                rule_judgement=rule,
                model_logits=logits,
                iai=self._optional_float(outputs.iai, region_index),
                class_names=class_names,
                config=self.config,
            )
            target = int(batch.targets["lost_space_label"][region_index].detach().cpu().item())
            if target != ignore_index:
                rule_index = map_rule_level_to_class_index(rule.final_level, class_names)
                if rule_index is not None:
                    rule_correct += int(rule_index == target)
                    rule_label_count += 1
                if fused.final_index is not None:
                    final_correct += int(fused.final_index == target)
                    final_label_count += 1
            if fused.model_pred_index is not None:
                rule_index = map_rule_level_to_class_index(rule.final_level, class_names)
                if rule_index is not None:
                    agreement_count += int(fused.model_pred_index == rule_index)
                    compared_count += 1
                    gap_sum += abs(float(fused.model_pred_index) - float(rule_index))

        return (
            rule_correct,
            rule_label_count,
            final_correct,
            final_label_count,
            agreement_count,
            compared_count,
            gap_sum,
        )

    def _optional_float(self, tensor: torch.Tensor, index: int) -> float | None:
        value = float(tensor[index].detach().cpu().item())
        if math.isnan(value):
            return None
        return value

    def _compute_selection_score(
        self,
        *,
        loss: float,
        model_lost_space_accuracy: float,
        rule_lost_space_accuracy: float,
        final_lost_space_accuracy: float,
        rule_model_agreement: float,
        sentiment_accuracy: float,
        historical_sentiment_accuracy: float,
        labeled_count: int,
        compared_count: int,
        sentiment_labeled_count: int,
        historical_sentiment_labeled_count: int,
    ) -> float:
        weighted_components: list[tuple[float, float]] = []
        if labeled_count > 0:
            weighted_components.extend(
                [
                    (final_lost_space_accuracy, 0.45),
                    (model_lost_space_accuracy, 0.15),
                    (rule_lost_space_accuracy, 0.15),
                ]
            )
        if compared_count > 0:
            weighted_components.append((rule_model_agreement, 0.1))
        if sentiment_labeled_count > 0:
            weighted_components.append((sentiment_accuracy, 0.075))
        if historical_sentiment_labeled_count > 0:
            weighted_components.append((historical_sentiment_accuracy, 0.075))
        weighted_components.append((1.0 / (1.0 + max(loss, 0.0)), 0.05))
        if not weighted_components:
            return 1.0 / (1.0 + max(loss, 0.0))
        numerator = sum(value * weight for value, weight in weighted_components)
        denominator = sum(weight for _, weight in weighted_components)
        return numerator / max(denominator, 1e-6)

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
