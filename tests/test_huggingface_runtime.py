from __future__ import annotations

from torch import nn

from uniter.models.huggingface import load_tokenizer
from uniter.models.spatial import SpatialEncoder


def test_load_tokenizer_prefers_local_cache(monkeypatch) -> None:
    import uniter.models.huggingface as huggingface

    calls: list[dict[str, object]] = []

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            calls.append({"model_name": model_name, **kwargs})
            return {"model_name": model_name, **kwargs}

    monkeypatch.setattr(
        huggingface,
        "try_to_load_from_cache",
        lambda model_name, filename: "/tmp/cached" if filename == "config.json" else None,
    )
    monkeypatch.setattr(huggingface, "AutoTokenizer", _FakeTokenizer)
    load_tokenizer.cache_clear()

    tokenizer = load_tokenizer("fake-model")

    assert tokenizer["model_name"] == "fake-model"
    assert calls[0]["local_files_only"] is True


def test_spatial_encoder_keeps_decode_head_trainable_when_backbone_is_frozen(
    monkeypatch,
) -> None:
    import uniter.models.spatial as spatial_module

    class _FakeSegFormer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.decode_head = nn.Linear(2, 2)
            self.config = type(
                "Config",
                (),
                {"num_labels": 2, "id2label": {0: "road", 1: "building"}},
            )()

    monkeypatch.setattr(
        spatial_module,
        "load_segformer",
        lambda model_name, **kwargs: _FakeSegFormer(),
    )

    encoder = SpatialEncoder(
        "fake-model",
        freeze_encoder=True,
        train_decode_head=True,
    )

    assert all(not parameter.requires_grad for parameter in encoder.model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in encoder.model.decode_head.parameters())
