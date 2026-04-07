from __future__ import annotations

import torch

from uniter.models.huggingface import load_text_model
from uniter.models.pooling import masked_mean_pool


class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        *,
        pooling: str = "mean",
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = load_text_model(model_name, use_safetensors=False)
        self.pooling = pooling

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        return masked_mean_pool(last_hidden_state, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(outputs.last_hidden_state, attention_mask)
