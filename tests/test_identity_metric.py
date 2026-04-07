from __future__ import annotations

import torch

from uniter.metrics.identity import build_identity_attribute_vector, compute_iai


def test_compute_iai_is_clamped_to_unit_interval() -> None:
    image_embeddings = torch.tensor([[1.0, 0.0]])
    text_embeddings = torch.tensor([[1.0, 0.0]])
    identity_embeddings = torch.tensor([[-1.0, 0.0]])

    iai = compute_iai(
        identity_embeddings,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        normalizer=0.5,
        identity_mask=torch.tensor([True]),
    )

    assert torch.allclose(iai, torch.tensor([1.0]))


def test_compute_iai_uses_identity_text_entropy_when_available() -> None:
    image_embeddings = torch.tensor([[1.0, 0.0]])
    text_embeddings = torch.tensor([[1.0, 0.0]])
    identity_embeddings = torch.tensor([[1.0, 0.0]])

    iai = compute_iai(
        identity_embeddings,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        normalizer=1.0,
        identity_mask=torch.tensor([True]),
        metadata=[
            {
                "_selected_identity_texts": [
                    "宗教遗产侧翼线性界面",
                    "游客支线通达与驻留节点",
                    "沿街复合型服务与小型消费边界",
                ]
            }
        ],
    )

    assert float(iai[0].item()) > 0.2


def test_compute_iai_remains_lower_for_coherent_single_identity() -> None:
    image_embeddings = torch.tensor([[1.0, 0.0]])
    text_embeddings = torch.tensor([[1.0, 0.0]])
    identity_embeddings = torch.tensor([[1.0, 0.0]])

    iai = compute_iai(
        identity_embeddings,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        normalizer=1.0,
        identity_mask=torch.tensor([True]),
        metadata=[
            {
                "_selected_identity_texts": [
                    "宗教遗产核心界面",
                    "历史文化遗产边界",
                ]
            }
        ],
    )

    assert float(iai[0].item()) < 0.45


def test_build_identity_attribute_vector_preserves_structured_identity_axes() -> None:
    attribute_vector = build_identity_attribute_vector(
        {
            "_selected_identity_texts": [
                "宗教遗产侧翼线性界面与历史边界",
                "游客支线通达与驻留节点",
                "沿街复合型服务与缓冲型过渡界面",
            ]
        }
    )

    assert attribute_vector is not None
    assert attribute_vector["function_religious_heritage"] > 0.0
    assert attribute_vector["function_commercial_service"] > 0.0
    assert attribute_vector["interface_boundary_edge"] > 0.0
    assert attribute_vector["interface_node_flow"] > 0.0
    assert attribute_vector["structure_linear_edge"] > 0.0
    assert attribute_vector["identity_conflict_score"] > 0.0
