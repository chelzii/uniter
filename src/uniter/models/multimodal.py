from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from uniter.config import AppConfig
from uniter.metrics.alignment import cosine_alignment_gap
from uniter.metrics.identity import compute_iai
from uniter.metrics.meaning import compute_mdi
from uniter.metrics.spatial import (
    build_adaptive_target_profiles,
    build_historical_plan_targets,
    compute_class_ratios,
    compute_ifi,
    compute_soft_class_ratios,
    compute_spatial_proxy_score,
    reduce_to_label_groups,
)
from uniter.models.classification import ClassificationHead
from uniter.models.pooling import aggregate_by_region, aggregate_by_region_with_counts
from uniter.models.projection import ProjectionHead
from uniter.models.sentiment import SentimentHead
from uniter.models.spatial import SpatialEncoder
from uniter.models.text import TextEncoder


@dataclass(slots=True)
class ModelOutputs:
    image_embeddings: torch.Tensor
    text_embeddings: torch.Tensor
    image_logits: torch.Tensor
    segmentation_logits: torch.Tensor
    satellite_region_mask: torch.Tensor
    current_text_features: torch.Tensor
    historical_text_features: torch.Tensor | None
    identity_text_features: torch.Tensor | None
    current_sentiment_record_logits: torch.Tensor | None
    historical_sentiment_record_logits: torch.Tensor | None
    sentiment_logits: torch.Tensor | None
    historical_sentiment_logits: torch.Tensor | None
    historical_region_mask: torch.Tensor
    mdi_sentiment_mask: torch.Tensor
    lost_space_logits: torch.Tensor | None
    identity_embeddings: torch.Tensor | None
    identity_region_mask: torch.Tensor
    alignment_gap: torch.Tensor
    ifi: torch.Tensor
    mdi: torch.Tensor
    iai: torch.Tensor
    spatial_proxy_score: torch.Tensor | None = None


class MultimodalRegionModel(nn.Module):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_encoder = SpatialEncoder(
            config.spatial_model.model_name,
            freeze_encoder=config.spatial_model.freeze_encoder,
            train_decode_head=config.spatial_model.train_decode_head,
        )
        self.text_encoder = TextEncoder(
            config.text_model.model_name,
            pooling=config.text_model.pooling,
            freeze_encoder=config.text_model.freeze_encoder,
        )

        self.image_projector = ProjectionHead(
            input_dim=self.spatial_encoder.num_labels,
            output_dim=config.alignment.embed_dim,
            dropout=config.alignment.projection_dropout,
        )
        self.text_projector = ProjectionHead(
            input_dim=self.text_encoder.hidden_size,
            output_dim=config.alignment.embed_dim,
            dropout=config.alignment.projection_dropout,
        )
        self.identity_projector = (
            ProjectionHead(
                input_dim=self.text_encoder.hidden_size,
                output_dim=config.alignment.embed_dim,
                dropout=config.identity.projection_dropout,
            )
            if config.identity.enabled
            else None
        )
        self.current_sentiment_head = (
            SentimentHead(
                input_dim=self.text_encoder.hidden_size,
                num_classes=config.sentiment.num_classes,
                dropout=config.sentiment.dropout,
            )
            if config.sentiment.enabled
            else None
        )
        self.historical_sentiment_head = (
            SentimentHead(
                input_dim=self.text_encoder.hidden_size,
                num_classes=config.sentiment.num_classes,
                dropout=config.sentiment.dropout,
            )
            if config.sentiment.enabled
            else None
        )
        self.lost_space_head = (
            ClassificationHead(
                input_dim=config.alignment.embed_dim * 2 + 6,
                num_classes=config.lost_space.num_classes,
                dropout=config.lost_space.dropout,
            )
            if config.lost_space.enabled
            else None
        )

    def _encode_optional_text_branch(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        region_index: torch.Tensor | None,
        num_regions: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        device = next(self.parameters()).device
        if input_ids is None or attention_mask is None or region_index is None:
            return (
                None,
                torch.zeros(num_regions, device=device, dtype=torch.bool),
                None,
            )
        branch_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_branch_features, counts = aggregate_by_region_with_counts(
            branch_features,
            region_index,
            num_regions,
        )
        return pooled_branch_features, counts > 0, branch_features

    def _aggregate_image_branch(
        self,
        image_logits: torch.Tensor,
        segmentation_logits: torch.Tensor,
        image_region_index: torch.Tensor,
        image_mask: torch.Tensor,
        num_regions: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = image_logits.device
        if not torch.any(image_mask):
            return (
                torch.zeros(
                    num_regions,
                    image_logits.shape[-1],
                    device=device,
                    dtype=image_logits.dtype,
                ),
                torch.zeros(num_regions, device=device, dtype=torch.bool),
                torch.zeros(
                    num_regions,
                    segmentation_logits.shape[1],
                    device=device,
                    dtype=torch.float32,
                ),
            )

        selected_image_logits = image_logits[image_mask]
        selected_region_index = image_region_index[image_mask]
        pooled_logits, counts = aggregate_by_region_with_counts(
            selected_image_logits,
            selected_region_index,
            num_regions,
        )
        class_ratios = compute_class_ratios(
            segmentation_logits[image_mask],
            selected_region_index,
            num_regions,
        )
        return pooled_logits, counts > 0, class_ratios

    def forward(self, batch) -> ModelOutputs:
        num_regions = len(batch.region_ids)

        image_logits, segmentation_logits = self.spatial_encoder(batch.pixel_values)
        region_image_logits = aggregate_by_region(
            image_logits,
            batch.image_region_index,
            num_regions,
        )
        street_image_mask = ~batch.image_is_satellite
        satellite_image_mask = batch.image_is_satellite
        (
            street_region_image_logits,
            street_region_mask,
            street_class_ratios,
        ) = self._aggregate_image_branch(
            image_logits,
            segmentation_logits,
            batch.image_region_index,
            street_image_mask,
            num_regions,
        )
        (
            satellite_region_image_logits,
            satellite_region_mask,
            satellite_class_ratios,
        ) = self._aggregate_image_branch(
            image_logits,
            segmentation_logits,
            batch.image_region_index,
            satellite_image_mask,
            num_regions,
        )
        street_image_embeddings = F.normalize(
            self.image_projector(street_region_image_logits + region_image_logits * 0.1),
            dim=-1,
        )
        satellite_image_embeddings = F.normalize(
            self.image_projector(satellite_region_image_logits),
            dim=-1,
        )
        image_embeddings = street_image_embeddings.clone()
        if torch.any(satellite_region_mask):
            satellite_mix = 0.65 * street_image_embeddings + 0.35 * satellite_image_embeddings
            image_embeddings[satellite_region_mask] = F.normalize(
                satellite_mix[satellite_region_mask],
                dim=-1,
            )

        text_features = self.text_encoder(
            input_ids=batch.current_input_ids,
            attention_mask=batch.current_attention_mask,
        )
        region_text_features = aggregate_by_region(
            text_features,
            batch.current_region_index,
            num_regions,
        )
        text_embeddings = F.normalize(self.text_projector(region_text_features), dim=-1)
        current_sentiment_record_logits = None
        sentiment_logits = None
        if self.current_sentiment_head is not None:
            current_sentiment_record_logits = self.current_sentiment_head(text_features)
            sentiment_logits = aggregate_by_region(
                current_sentiment_record_logits,
                batch.current_region_index,
                num_regions,
            )

        (
            historical_text_features,
            historical_region_mask,
            historical_record_features,
        ) = self._encode_optional_text_branch(
            batch.historical_input_ids,
            batch.historical_attention_mask,
            batch.historical_region_index,
            num_regions,
        )
        historical_sentiment_record_logits = None
        historical_sentiment_logits = None
        if (
            self.historical_sentiment_head is not None
            and historical_record_features is not None
            and batch.historical_region_index is not None
        ):
            historical_sentiment_record_logits = self.historical_sentiment_head(
                historical_record_features
            )
            historical_sentiment_logits = aggregate_by_region(
                historical_sentiment_record_logits,
                batch.historical_region_index,
                num_regions,
            )
        identity_text_features, identity_region_mask, _ = self._encode_optional_text_branch(
            batch.identity_input_ids,
            batch.identity_attention_mask,
            batch.identity_region_index,
            num_regions,
        )
        identity_embeddings = (
            F.normalize(self.identity_projector(identity_text_features), dim=-1)
            if self.identity_projector is not None and identity_text_features is not None
            else None
        )

        street_label_groups = reduce_to_label_groups(
            street_class_ratios,
            id2label=self.spatial_encoder.id2label,
            label_groups=self.config.spatial_model.label_groups,
        )
        satellite_label_groups = reduce_to_label_groups(
            satellite_class_ratios,
            id2label=self.spatial_encoder.id2label,
            label_groups=self.config.spatial_model.label_groups,
        )
        per_image_region_index = torch.arange(
            segmentation_logits.shape[0],
            device=batch.image_region_index.device,
            dtype=batch.image_region_index.dtype,
        )
        per_image_class_ratios = compute_class_ratios(
            segmentation_logits,
            per_image_region_index,
            segmentation_logits.shape[0],
        )
        per_image_label_groups = reduce_to_label_groups(
            per_image_class_ratios,
            id2label=self.spatial_encoder.id2label,
            label_groups=self.config.spatial_model.label_groups,
        )
        street_per_image_group_ratios = {
            name: values[street_image_mask]
            for name, values in per_image_label_groups.items()
        }
        satellite_per_image_group_ratios = {
            name: values[satellite_image_mask]
            for name, values in per_image_label_groups.items()
        }
        street_soft_class_ratios = compute_soft_class_ratios(
            segmentation_logits[street_image_mask],
            batch.image_region_index[street_image_mask],
            num_regions,
        )
        satellite_soft_class_ratios = compute_soft_class_ratios(
            segmentation_logits[satellite_image_mask],
            batch.image_region_index[satellite_image_mask],
            num_regions,
        )
        street_soft_group_ratios = reduce_to_label_groups(
            street_soft_class_ratios,
            id2label=self.spatial_encoder.id2label,
            label_groups=self.config.spatial_model.label_groups,
        )
        satellite_soft_group_ratios = reduce_to_label_groups(
            satellite_soft_class_ratios,
            id2label=self.spatial_encoder.id2label,
            label_groups=self.config.spatial_model.label_groups,
        )
        soft_per_image_class_ratios = compute_soft_class_ratios(
            segmentation_logits,
            per_image_region_index,
            segmentation_logits.shape[0],
        )
        soft_per_image_label_groups = reduce_to_label_groups(
            soft_per_image_class_ratios,
            id2label=self.spatial_encoder.id2label,
            label_groups=self.config.spatial_model.label_groups,
        )
        street_soft_per_image_group_ratios = {
            name: values[street_image_mask]
            for name, values in soft_per_image_label_groups.items()
        }
        satellite_soft_per_image_group_ratios = {
            name: values[satellite_image_mask]
            for name, values in soft_per_image_label_groups.items()
        }
        historical_plan_targets = build_historical_plan_targets(
            metadata=batch.metadata,
            device=street_class_ratios.device,
        )

        ifi = compute_ifi(
            street_group_ratios=street_label_groups,
            street_target_profile=build_adaptive_target_profiles(
                base_profile=self.config.metrics.ifi_street_target_profile,
                metadata=batch.metadata,
                device=street_class_ratios.device,
                image_type="street",
            ),
            street_weights=self.config.metrics.ifi_street_weights,
            satellite_group_ratios=(
                satellite_label_groups if torch.any(satellite_region_mask) else None
            ),
            satellite_target_profile=build_adaptive_target_profiles(
                base_profile=self.config.metrics.ifi_satellite_target_profile,
                metadata=batch.metadata,
                device=street_class_ratios.device,
                image_type="satellite",
            ),
            satellite_weights=self.config.metrics.ifi_satellite_weights,
            cross_view_weights=self.config.metrics.ifi_cross_view_weights,
            per_image_street_group_ratios=(
                street_per_image_group_ratios if torch.any(street_image_mask) else None
            ),
            street_image_region_index=(
                batch.image_region_index[street_image_mask]
                if torch.any(street_image_mask)
                else None
            ),
            street_image_point_ids=(
                [
                    point_id
                    for point_id, is_satellite in zip(
                        batch.image_point_ids,
                        batch.image_is_satellite.tolist(),
                        strict=True,
                    )
                    if not is_satellite
                ]
                if torch.any(street_image_mask)
                else None
            ),
            street_image_view_directions=(
                [
                    direction
                    for direction, is_satellite in zip(
                        batch.image_view_directions,
                        batch.image_is_satellite.tolist(),
                        strict=True,
                    )
                    if not is_satellite
                ]
                if torch.any(street_image_mask)
                else None
            ),
            street_image_longitudes=(
                [
                    longitude
                    for longitude, is_satellite in zip(
                        batch.image_longitudes,
                        batch.image_is_satellite.tolist(),
                        strict=True,
                    )
                    if not is_satellite
                ]
                if torch.any(street_image_mask)
                else None
            ),
            street_image_latitudes=(
                [
                    latitude
                    for latitude, is_satellite in zip(
                        batch.image_latitudes,
                        batch.image_is_satellite.tolist(),
                        strict=True,
                    )
                    if not is_satellite
                ]
                if torch.any(street_image_mask)
                else None
            ),
            per_image_satellite_group_ratios=(
                satellite_per_image_group_ratios if torch.any(satellite_image_mask) else None
            ),
            satellite_image_region_index=(
                batch.image_region_index[satellite_image_mask]
                if torch.any(satellite_image_mask)
                else None
            ),
            historical_plan_targets=historical_plan_targets,
        )
        spatial_proxy_score = compute_spatial_proxy_score(
            street_group_ratios=street_soft_group_ratios,
            street_target_profile=build_adaptive_target_profiles(
                base_profile=self.config.metrics.ifi_street_target_profile,
                metadata=batch.metadata,
                device=street_class_ratios.device,
                image_type="street",
            ),
            street_weights=self.config.metrics.ifi_street_weights,
            satellite_group_ratios=(
                satellite_soft_group_ratios if torch.any(satellite_image_mask) else None
            ),
            satellite_target_profile=build_adaptive_target_profiles(
                base_profile=self.config.metrics.ifi_satellite_target_profile,
                metadata=batch.metadata,
                device=street_class_ratios.device,
                image_type="satellite",
            ),
            satellite_weights=self.config.metrics.ifi_satellite_weights,
            cross_view_weights=self.config.metrics.ifi_cross_view_weights,
            per_image_street_group_ratios=(
                street_soft_per_image_group_ratios if torch.any(street_image_mask) else None
            ),
            street_image_region_index=(
                batch.image_region_index[street_image_mask]
                if torch.any(street_image_mask)
                else None
            ),
            per_image_satellite_group_ratios=(
                satellite_soft_per_image_group_ratios
                if torch.any(satellite_image_mask)
                else None
            ),
            satellite_image_region_index=(
                batch.image_region_index[satellite_image_mask]
                if torch.any(satellite_image_mask)
                else None
            ),
            historical_plan_targets=historical_plan_targets,
        )
        mdi_device_type = region_text_features.device.type
        with torch.amp.autocast(device_type=mdi_device_type, enabled=False):
            mdi = compute_mdi(
                F.normalize(region_text_features.float(), dim=-1),
                (
                    F.normalize(historical_text_features.float(), dim=-1)
                    if historical_text_features is not None
                    else None
                ),
                normalizer=self.config.metrics.mdi_normalizer,
                mode=self.config.metrics.mdi_mode,
                sentiment_weight=self.config.metrics.mdi_sentiment_weight,
                current_sentiment_logits=(
                    sentiment_logits.float() if sentiment_logits is not None else None
                ),
                historical_sentiment_logits=(
                    historical_sentiment_logits.float()
                    if historical_sentiment_logits is not None
                    else None
                ),
                current_sentiment_record_logits=(
                    current_sentiment_record_logits.float()
                    if current_sentiment_record_logits is not None
                    else None
                ),
                current_sentiment_region_index=batch.current_region_index,
                historical_sentiment_record_logits=(
                    historical_sentiment_record_logits.float()
                    if historical_sentiment_record_logits is not None
                    else None
                ),
                historical_sentiment_region_index=batch.historical_region_index,
                historical_mask=historical_region_mask,
                sentiment_mode_mask=(
                    historical_region_mask
                    & batch.targets["sentiment_label"].ne(self.config.sentiment.ignore_index)
                    & batch.targets["historical_sentiment_label"].ne(
                        self.config.sentiment.ignore_index
                    )
                ),
            )
        mdi_sentiment_mask = (
            historical_region_mask
            & batch.targets["sentiment_label"].ne(self.config.sentiment.ignore_index)
            & batch.targets["historical_sentiment_label"].ne(self.config.sentiment.ignore_index)
        )
        alignment_gap = cosine_alignment_gap(image_embeddings, text_embeddings)
        iai = compute_iai(
            identity_embeddings,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            normalizer=self.config.metrics.iai_normalizer,
            identity_mask=identity_region_mask,
            metadata=batch.metadata,
        )
        lost_space_logits = None
        if self.lost_space_head is not None:
            lost_space_features = torch.cat(
                [
                    image_embeddings,
                    text_embeddings,
                    ifi.unsqueeze(-1),
                    torch.nan_to_num(mdi, nan=0.0).unsqueeze(-1),
                    alignment_gap.unsqueeze(-1),
                    historical_region_mask.to(dtype=ifi.dtype).unsqueeze(-1),
                    torch.nan_to_num(iai, nan=0.0).unsqueeze(-1),
                    identity_region_mask.to(dtype=ifi.dtype).unsqueeze(-1),
                ],
                dim=-1,
            )
            lost_space_logits = self.lost_space_head(lost_space_features)

        return ModelOutputs(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            image_logits=region_image_logits,
            segmentation_logits=segmentation_logits,
            satellite_region_mask=satellite_region_mask,
            current_text_features=region_text_features,
            historical_text_features=historical_text_features,
            identity_text_features=identity_text_features,
            current_sentiment_record_logits=current_sentiment_record_logits,
            historical_sentiment_record_logits=historical_sentiment_record_logits,
            sentiment_logits=sentiment_logits,
            historical_sentiment_logits=historical_sentiment_logits,
            historical_region_mask=historical_region_mask,
            mdi_sentiment_mask=mdi_sentiment_mask,
            lost_space_logits=lost_space_logits,
            identity_embeddings=identity_embeddings,
            identity_region_mask=identity_region_mask,
            alignment_gap=alignment_gap,
            ifi=ifi,
            mdi=mdi,
            iai=iai,
            spatial_proxy_score=spatial_proxy_score,
        )
