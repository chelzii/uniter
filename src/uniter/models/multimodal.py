from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from uniter.config import AppConfig
from uniter.metrics.alignment import cosine_alignment_gap
from uniter.metrics.identity import compute_iai
from uniter.metrics.meaning import compute_mdi
from uniter.metrics.spatial import compute_class_ratios, compute_ifi, reduce_to_label_groups
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
    current_text_features: torch.Tensor
    historical_text_features: torch.Tensor | None
    identity_text_features: torch.Tensor | None
    sentiment_logits: torch.Tensor | None
    historical_sentiment_logits: torch.Tensor | None
    historical_region_mask: torch.Tensor
    lost_space_logits: torch.Tensor | None
    identity_embeddings: torch.Tensor | None
    identity_region_mask: torch.Tensor
    alignment_gap: torch.Tensor
    ifi: torch.Tensor
    mdi: torch.Tensor
    iai: torch.Tensor


class MultimodalRegionModel(nn.Module):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_encoder = SpatialEncoder(
            config.spatial_model.model_name,
            freeze_encoder=config.spatial_model.freeze_encoder,
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
        self.sentiment_head = (
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
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        device = next(self.parameters()).device
        if input_ids is None or attention_mask is None or region_index is None:
            return (
                None,
                torch.zeros(num_regions, device=device, dtype=torch.bool),
            )
        branch_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_branch_features, counts = aggregate_by_region_with_counts(
            branch_features,
            region_index,
            num_regions,
        )
        return pooled_branch_features, counts > 0

    def forward(self, batch) -> ModelOutputs:
        num_regions = len(batch.region_ids)

        image_logits, segmentation_logits = self.spatial_encoder(batch.pixel_values)
        region_image_logits = aggregate_by_region(
            image_logits,
            batch.image_region_index,
            num_regions,
        )
        image_embeddings = F.normalize(self.image_projector(region_image_logits), dim=-1)

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
        sentiment_logits = (
            self.sentiment_head(region_text_features)
            if self.sentiment_head is not None
            else None
        )

        historical_text_features, historical_region_mask = self._encode_optional_text_branch(
            batch.historical_input_ids,
            batch.historical_attention_mask,
            batch.historical_region_index,
            num_regions,
        )
        historical_sentiment_logits = (
            self.sentiment_head(historical_text_features)
            if self.sentiment_head is not None and historical_text_features is not None
            else None
        )
        identity_text_features, identity_region_mask = self._encode_optional_text_branch(
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

        class_ratios = compute_class_ratios(
            segmentation_logits,
            batch.image_region_index,
            num_regions,
        )
        label_groups = reduce_to_label_groups(
            class_ratios,
            id2label=self.spatial_encoder.id2label,
            label_groups=self.config.spatial_model.label_groups,
        )

        ifi = compute_ifi(
            label_groups,
            target_profile=self.config.metrics.ifi_target_profile,
            weights=self.config.metrics.ifi_weights,
        )
        mdi = compute_mdi(
            F.normalize(region_text_features, dim=-1),
            (
                F.normalize(historical_text_features, dim=-1)
                if historical_text_features is not None
                else None
            ),
            normalizer=self.config.metrics.mdi_normalizer,
            current_sentiment_logits=sentiment_logits,
            historical_sentiment_logits=historical_sentiment_logits,
            historical_mask=historical_region_mask,
        )
        alignment_gap = cosine_alignment_gap(image_embeddings, text_embeddings)
        iai = compute_iai(
            identity_embeddings,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            normalizer=self.config.metrics.iai_normalizer,
            identity_mask=identity_region_mask,
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
            current_text_features=region_text_features,
            historical_text_features=historical_text_features,
            identity_text_features=identity_text_features,
            sentiment_logits=sentiment_logits,
            historical_sentiment_logits=historical_sentiment_logits,
            historical_region_mask=historical_region_mask,
            lost_space_logits=lost_space_logits,
            identity_embeddings=identity_embeddings,
            identity_region_mask=identity_region_mask,
            alignment_gap=alignment_gap,
            ifi=ifi,
            mdi=mdi,
            iai=iai,
        )
