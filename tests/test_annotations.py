from __future__ import annotations

import json
from pathlib import Path

from uniter.data.annotations import import_annotations_into_manifest


def test_import_annotations_into_manifest_updates_parent_region_targets(tmp_path: Path) -> None:
    manifest_path = tmp_path / "regions.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "region_id": "region_train",
                        "split": "train",
                        "image_paths": ["a.jpg"],
                        "current_texts": ["x"],
                        "metadata": {"parent_region_id": "region_a"},
                        "targets": {},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "region_id": "region_val",
                        "split": "val",
                        "image_paths": ["b.jpg"],
                        "current_texts": ["y"],
                        "metadata": {"parent_region_id": "region_a"},
                        "targets": {},
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    annotations_root = tmp_path / "annotations"
    (annotations_root / "region_a" / "current_sentiment").mkdir(parents=True)
    (annotations_root / "region_a" / "historical_sentiment").mkdir(parents=True)
    (annotations_root / "region_a" / "lost_space").mkdir(parents=True)
    (annotations_root / "region_a" / "current_sentiment" / "labels.csv").write_text(
        "region_id,sentiment_label\nregion_a,2\nregion_a,1\n",
        encoding="utf-8",
    )
    (annotations_root / "region_a" / "historical_sentiment" / "labels.csv").write_text(
        "region_id,historical_sentiment_label\nregion_a,2\nregion_a,2\n",
        encoding="utf-8",
    )
    (annotations_root / "region_a" / "lost_space" / "label.json").write_text(
        json.dumps({"region_id": "region_a", "lost_space_label_id": 2}, ensure_ascii=False),
        encoding="utf-8",
    )

    output_path, summary = import_annotations_into_manifest(
        manifest_path=manifest_path,
        annotation_root=annotations_root,
    )

    updated_records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["updated_record_count"] == 2
    assert updated_records[0]["targets"]["sentiment_label"] == 2
    assert updated_records[0]["targets"]["historical_sentiment_label"] == 2
    assert updated_records[0]["targets"]["lost_space_label"] == 2
    assert updated_records[1]["targets"]["sentiment_label"] == 2


def test_import_annotations_into_manifest_aligns_text_level_sentiment_labels(tmp_path: Path) -> None:
    manifest_path = tmp_path / "regions.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "region_id": "region_train",
                "split": "train",
                "image_paths": ["a.jpg"],
                "current_texts": ["text a", "text b", "text c"],
                "historical_texts": ["hist a", "hist b"],
                "metadata": {"parent_region_id": "region_a"},
                "targets": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    annotations_root = tmp_path / "annotations"
    (annotations_root / "region_a" / "current_sentiment").mkdir(parents=True)
    (annotations_root / "region_a" / "historical_sentiment").mkdir(parents=True)
    (annotations_root / "region_a" / "current_sentiment" / "labels.csv").write_text(
        (
            "region_id,text,sentiment_label\n"
            "region_a,text a,2\n"
            "region_a,text c,1\n"
        ),
        encoding="utf-8",
    )
    (annotations_root / "region_a" / "historical_sentiment" / "labels.csv").write_text(
        (
            "region_id,text,historical_sentiment_label\n"
            "region_a,hist b,0\n"
        ),
        encoding="utf-8",
    )

    output_path, summary = import_annotations_into_manifest(
        manifest_path=manifest_path,
        annotation_root=annotations_root,
    )

    updated_record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert updated_record["current_sentiment_labels"] == [2, None, 1]
    assert updated_record["historical_sentiment_labels"] == [None, 0]
    assert summary["matched_text_label_count"] == 3


def test_import_annotations_matches_texts_after_punctuation_normalization(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "regions.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "region_id": "region_train",
                "split": "train",
                "image_paths": ["a.jpg"],
                "current_texts": ["卧龙寺要预约吗?", "全程步行不到1公里,太好逛啦"],
                "historical_texts": [],
                "metadata": {"parent_region_id": "region_a"},
                "targets": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    annotations_root = tmp_path / "annotations"
    (annotations_root / "region_a" / "current_sentiment").mkdir(parents=True)
    (annotations_root / "region_a" / "current_sentiment" / "labels.csv").write_text(
        (
            "region_id,text,sentiment_label\n"
            "region_a,卧龙寺要预约吗？,1\n"
            "region_a,全程步行不到1公里，太好逛啦,2\n"
        ),
        encoding="utf-8",
    )

    output_path, summary = import_annotations_into_manifest(
        manifest_path=manifest_path,
        annotation_root=annotations_root,
    )

    updated_record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert updated_record["current_sentiment_labels"] == [1, 2]
    assert summary["matched_text_label_count"] == 2


def test_import_annotations_matches_texts_with_containment_and_fuzzy_fallbacks(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "regions.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "region_id": "region_train",
                "split": "train",
                "image_paths": ["a.jpg"],
                "current_texts": [
                    "西安citywalk路线 | 从书院门到卧龙禅寺。西安周末去哪 西安旅游 西安citywalk 书院门 卧龙禅寺",
                    "卧龙禅寺名字的由来有️人知道吗",
                    (
                        "西安小众免费景点,秋冬首选这里 闭眼冲!。来西安好几天,基本都是阴雨天气,"
                        "离开的时候是个大晴天,很开心选了这么个好地方,在古城里的卧龙禅寺;靠近书院门"
                        "以及永宁门附近,有种大隐隐于市的味道,整个寺庙不大不小,但就很惬意,"
                        "大晴天和金黄的银杏叶真的超级有感觉!"
                    ),
                ],
                "historical_texts": [],
                "metadata": {"parent_region_id": "region_a"},
                "targets": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    annotations_root = tmp_path / "annotations"
    (annotations_root / "region_a" / "current_sentiment").mkdir(parents=True)
    (annotations_root / "region_a" / "current_sentiment" / "labels.csv").write_text(
        (
            "region_id,text,sentiment_label\n"
            "region_a,#西安周末去哪 #西安旅游 #西安citywalk #书院门 #卧龙禅寺,1\n"
            "region_a,卧龙禅寺名字的由来???人知道吗,2\n"
            "region_a,来西安好几天，基本都是阴雨天气，离开的时候是个大晴天，很开心选了这么个好地方，在古城里的卧龙禅寺；靠近书院门以及永宁门附近，有种大隐隐于市的味道，整个寺庙不大不小，但就很惬意，大晴天和金黄的银杏叶真的超级有感觉！,0\n"
        ),
        encoding="utf-8",
    )

    output_path, summary = import_annotations_into_manifest(
        manifest_path=manifest_path,
        annotation_root=annotations_root,
    )

    updated_record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert updated_record["current_sentiment_labels"] == [1, 2, 0]
    assert summary["matched_text_label_count"] == 3
