from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from uniter.config import AppConfig
from uniter.data.collate import RegionBatchCollator
from uniter.data.dataset import RegionBatch, RegionDataset, resolve_sample_splits
from uniter.data.manifest import RegionRecord, load_manifest
from uniter.models.multimodal import ModelOutputs, MultimodalRegionModel
from uniter.utils.device import move_region_batch_to_device, resolve_device
from uniter.utils.logging import get_logger


class InferenceRuntime:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.output_dir = config.output_dir
        self.device = resolve_device(config.experiment.device)
        self.records = load_manifest(
            config.data.manifest_path,
            image_root=config.data.image_root,
            check_files=config.data.check_files_on_load,
        )
        self.split_map = resolve_sample_splits(self.records)
        self.split_by_region = {record.region_id: record.split for record in self.records}
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
        self.model = MultimodalRegionModel(config).to(self.device)

    def resolve_records(self, split: str) -> list[RegionRecord]:
        if split == "all":
            return self.records
        return self.split_map.get(split, [])

    def build_loader(
        self,
        records: list[RegionRecord],
        *,
        batch_size: int | None = None,
    ) -> DataLoader:
        return DataLoader(
            RegionDataset(records),
            batch_size=batch_size or self.config.data.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.data.num_workers,
            collate_fn=self.collator,
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        payload = torch.load(path, map_location=self.device)
        state_dict = (
            payload["model"]
            if isinstance(payload, dict) and "model" in payload
            else payload
        )
        self.model.load_state_dict(state_dict)
        self.logger.info("Loaded checkpoint from %s", path)

    def prepare_model(
        self,
        *,
        checkpoint_path: str | Path | None,
        allow_random_init: bool = False,
    ) -> None:
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif self.config.inference.require_checkpoint and not allow_random_init:
            raise RuntimeError(
                "Inference requires a checkpoint by default. Pass --checkpoint or explicitly "
                "allow random initialization."
            )
        else:
            self.logger.warning(
                "No checkpoint was provided. Inference will use the model's current "
                "initialized weights."
            )
        self.model.eval()

    def iterate_region_batches(
        self,
        *,
        split: str,
        checkpoint_path: str | Path | None = None,
        batch_size: int | None = None,
        allow_random_init: bool = False,
    ) -> Iterator[tuple[RegionBatch, ModelOutputs]]:
        records = self.resolve_records(split)
        if not records:
            raise RuntimeError(f"No records found for split '{split}'.")
        self.prepare_model(
            checkpoint_path=checkpoint_path,
            allow_random_init=allow_random_init,
        )
        loader = self.build_loader(records, batch_size=batch_size)
        with torch.no_grad():
            for batch in loader:
                batch = move_region_batch_to_device(batch, self.device)
                yield batch, self.model(batch)
