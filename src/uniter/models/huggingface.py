from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from huggingface_hub import try_to_load_from_cache
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from transformers import SegformerForSemanticSegmentation

LOGGER = logging.getLogger(__name__)


def disable_safetensors_conversion() -> None:
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


def prepare_huggingface_runtime() -> None:
    disable_safetensors_conversion()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def _offline_env_enabled() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _has_local_cache(model_name: str) -> bool:
    for filename in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        cached = try_to_load_from_cache(model_name, filename)
        if isinstance(cached, str):
            return True
    return False


def _load_pretrained(factory: Any, model_name: str, **kwargs: Any) -> Any:
    prepare_huggingface_runtime()
    prefer_local_files = _offline_env_enabled() or _has_local_cache(model_name)
    if prefer_local_files:
        try:
            return factory.from_pretrained(model_name, local_files_only=True, **kwargs)
        except OSError:
            if _offline_env_enabled():
                raise
            LOGGER.info(
                "Local Hugging Face cache miss for %s; retrying with online lookup.",
                model_name,
            )
    return factory.from_pretrained(
        model_name,
        local_files_only=_offline_env_enabled(),
        **kwargs,
    )


@lru_cache(maxsize=None)
def load_image_processor(model_name: str) -> Any:
    return _load_pretrained(AutoImageProcessor, model_name)


@lru_cache(maxsize=None)
def load_tokenizer(model_name: str) -> Any:
    return _load_pretrained(AutoTokenizer, model_name)


def load_text_model(model_name: str, **kwargs: Any) -> Any:
    return _load_pretrained(AutoModel, model_name, **kwargs)


def load_segformer(model_name: str, **kwargs: Any) -> Any:
    return _load_pretrained(SegformerForSemanticSegmentation, model_name, **kwargs)
