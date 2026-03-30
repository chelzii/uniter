#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable


TARGET_KEYWORDS = [
    "卧龙寺",
    "卧龙禅寺",
    "卧龙",
    "开通西巷",
    "三学街",
    "书院门",
    "碑林",
    "柏树林",
    "文昌门",
    "永宁门",
    "南门",
]

NEARBY_CONTEXT_KEYWORDS = [
    "城墙",
    "钟楼",
    "大差市",
    "citywalk",
    "路线",
    "步行",
    "地铁",
    "公交",
    "导航",
    "地址",
    "位置",
    "门票",
    "预约",
    "开放",
    "开放时间",
    "关门",
]

SPACE_SCENE_KEYWORDS = [
    "玉兰",
    "银杏",
    "落叶",
    "花期",
    "红墙",
    "古刹",
    "建筑",
    "碑",
    "钟",
    "文物",
    "院落",
    "大殿",
    "殿前",
    "树",
    "树叶",
    "景",
    "光影",
    "拍照",
    "机位",
]

ATMOSPHERE_KEYWORDS = [
    "安静",
    "清幽",
    "清净",
    "闹中取静",
    "人少",
    "人多",
    "小众",
    "有韵味",
    "氛围",
    "市井",
    "平和",
    "静下",
    "治愈",
    "体验",
    "不起眼",
    "低调",
    "方便",
    "交通便利",
]

RITUAL_IDENTITY_KEYWORDS = [
    "香火",
    "斋饭",
    "法会",
    "祈福",
    "求财",
    "事业",
    "挂单",
    "戒律",
    "诵经",
    "灵验",
    "上香",
]

OTHER_SITE_KEYWORDS = [
    "广仁寺",
    "青龙寺",
    "八仙宫",
    "八仙观",
    "大兴善寺",
    "大兴国禅寺",
    "香积寺",
    "大雁塔",
    "云居寺",
    "罔极寺",
    "弥陀古寺",
]

LOW_INFO_EXACT = {
    "转发微博",
    "关注",
    "关注转发",
    "转发",
    "谢谢",
    "谢谢您",
    "谢谢啦",
    "好滴",
    "好的",
    "好的谢谢",
    "好的谢谢!",
    "好的谢谢！",
    "好嘞谢谢!",
    "好嘞谢谢！",
    "嗯嗯",
    "喔喔",
    "噫",
    "哇喔",
    "好",
    "美",
    "1",
    "？",
    "?",
    "没有",
    "对的",
    "去啦",
    "幸运",
    "想去",
    "走!",
    "走！",
    "太美了",
    "好看!",
    "好看！",
}

LOW_INFO_PATTERNS = [
    re.compile(
        r"^(好看|漂亮|真棒|厉害|赞|太好看了|好美|真好看啊|好好看.*|拍的真棒|"
        r"精彩分享|感谢分享.*|赞感谢分享.*|欢迎常来|来交作业)$"
    ),
    re.compile(r"^(周末愉快|周四愉快|开心的周末|周末闲情雅致|平安喜乐.*|寒至.*|立冬安康.*)$"),
    re.compile(r"^此微博已在.*优质微博推荐.*$"),
    re.compile(r"^此微博.*添加位置.*$"),
    re.compile(r"^恭喜，您的微博获得.*微博榜.*$"),
    re.compile(r"^常常与你互动.*$"),
]

OFF_TOPIC_PATTERNS = [
    re.compile(r"股市|新股票|回本|职务侵占|距你睡觉还剩|葫芦头|牛油果手串|小橘子黄了|手机链|阿拉善"),
    re.compile(r"空手把锄头|富贵粪坑求"),
    re.compile(r"何海霞图书馆|通用大语言模型|AI写作|文章用词生动但不晦涩|适配大众阅读需求"),
]

MEDIA_INTERACTION_PATTERNS = [
    re.compile(r"取图|商用|注明出处|老师礼貌拿图|自用请自行取图|不授权任何形式商用"),
    re.compile(r"你用.+拍的"),
    re.compile(r"第[0-9一二三四五六七八九十]+张"),
    re.compile(r"私你"),
    re.compile(r"刷到了|刷到"),
    re.compile(r"主页"),
    re.compile(r"跟我拍"),
]

LOW_VALUE_FRAGMENT_PATTERNS = [
    re.compile(r"^(卧龙禅寺|卧龙寺|是卧龙禅寺|书院门哦|同机位|玉兰唉[,，]?|山西文物重地|卧龙被猫镇了|陕师大玉兰花|有卧龙必有凤雏)$"),
]

GENERIC_POETIC_PATTERNS = [
    re.compile(r"^山城烟雨,古刹钟声.*卧龙寺里听风吟[。.!！?？]?$"),
    re.compile(r"^宝相花配古刹,静中寻禅意[。.!！?？]?$"),
    re.compile(r"^禅钟悠扬,古木参天.*宁静致远[。.!！?？]?$"),
    re.compile(r"^念佛人心清净.*成佛无非心净定经[。.!！?？]?$"),
    re.compile(r"^卧龙禅寺,古韵悠长,心灵之旅[。.!！?？]?$"),
    re.compile(r"^卧龙禅寺,历史悠久,静谧之地[。.!！?？]?$"),
    re.compile(r"^卧龙禅寺真美,古风十足.*心灵宁静[。.!！?？]?$"),
    re.compile(r"^好美的画面,玉兰花下的禅寺.*期待你的美图分享.*$"),
    re.compile(r"^触景生情.*感悟人生.*$"),
    re.compile(r"^卧龙寺藏了多少风水密码[。.!！?？]?$"),
    re.compile(r"^在卧龙禅寺,连空气都弥漫着宁静与智慧的气息[。.!！?？]?$"),
    re.compile(r"^这个真的是非常的治愈啊[。.!！?？]?$"),
]

EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]+")
URL_RE = re.compile(r"https?://\S+")
AT_RE = re.compile(r"@[\w\-\u4e00-\u9fff]+")
MULTI_SPACE_RE = re.compile(r"\s+")
MULTI_PUNCT_RE = re.compile(r"([!！?？,.，。~～])\1+")
TEXT_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean single-region data workspace into training-ready data.")
    parser.add_argument(
        "--input-dir",
        default="datasets/data_workspace",
        help="Path to the raw data workspace.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/data_workspace_cleaned",
        help="Path to write the cleaned training-ready dataset.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"Expected a single-region manifest in {path}, got {len(lines)} lines.")
    return json.loads(lines[0])


def write_manifest(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def normalize_text(text: str) -> str:
    value = html.unescape(unicodedata.normalize("NFKC", text or ""))
    value = value.replace("\u200b", " ").replace("\ufeff", " ")
    value = value.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    value = URL_RE.sub(" ", value)
    value = value.replace("网页链接", " ")
    value = AT_RE.sub(" ", value)
    value = value.replace("#", " ")
    value = EMOJI_RE.sub(" ", value)
    value = MULTI_SPACE_RE.sub(" ", value).strip()
    value = MULTI_PUNCT_RE.sub(r"\1", value)
    return value.strip(" '\"")


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def build_cleaned_text(row: dict[str, str]) -> str:
    title = normalize_text(row.get("title", ""))
    text = normalize_text(row.get("text", ""))
    if row.get("source_type") == "note":
        if title and text and title not in text:
            return f"{title}。{text}"
        return text or title
    return text


def relevance_score(text: str, row: dict[str, str]) -> int:
    score = 0
    if contains_any(text, TARGET_KEYWORDS):
        score += 3
    if contains_any(text, NEARBY_CONTEXT_KEYWORDS):
        score += 2
    if contains_any(text, SPACE_SCENE_KEYWORDS):
        score += 2
    if contains_any(text, ATMOSPHERE_KEYWORDS):
        score += 2
    if contains_any(text, RITUAL_IDENTITY_KEYWORDS):
        score += 1
    if row.get("source_type") == "note":
        score += 3
    if len(text) >= 20:
        score += 1
    return score


def target_keyword_hits(text: str) -> int:
    return sum(text.count(keyword) for keyword in TARGET_KEYWORDS)


def other_site_hits(text: str) -> int:
    return sum(text.count(keyword) for keyword in OTHER_SITE_KEYWORDS)


def classify_record(row: dict[str, str], cleaned_text: str) -> str:
    if not cleaned_text:
        return "drop_empty"
    if cleaned_text in LOW_INFO_EXACT:
        return "drop_low_info_exact"
    if any(pattern.match(cleaned_text) for pattern in LOW_INFO_PATTERNS):
        return "drop_social_boilerplate"
    if any(pattern.search(cleaned_text) for pattern in OFF_TOPIC_PATTERNS):
        return "drop_off_topic"
    if any(pattern.search(cleaned_text) for pattern in MEDIA_INTERACTION_PATTERNS):
        return "drop_media_interaction"
    if any(pattern.match(cleaned_text) for pattern in LOW_VALUE_FRAGMENT_PATTERNS):
        return "drop_low_value_fragment"
    if any(pattern.match(cleaned_text) for pattern in GENERIC_POETIC_PATTERNS):
        return "drop_generic_poetic"
    if not TEXT_CHAR_RE.search(cleaned_text):
        return "drop_symbol_only"
    if len(cleaned_text) <= 2:
        return "drop_too_short"
    if contains_any(cleaned_text, OTHER_SITE_KEYWORDS) and not contains_any(cleaned_text, TARGET_KEYWORDS):
        return "drop_other_site_only"
    if row.get("source_type") == "note":
        if other_site_hits(cleaned_text) >= 2:
            return "drop_mixed_multi_site_note"
        if other_site_hits(cleaned_text) >= 1 and re.search(r"(1️⃣|2️⃣|3️⃣|p1-p5|p6-p10|pick哪一个|上午去的|下午去的)", cleaned_text):
            return "drop_mixed_multi_site_note"
    if contains_any(cleaned_text, OTHER_SITE_KEYWORDS) and contains_any(cleaned_text, TARGET_KEYWORDS):
        if not contains_any(cleaned_text, ATMOSPHERE_KEYWORDS + SPACE_SCENE_KEYWORDS + RITUAL_IDENTITY_KEYWORDS):
            return "drop_mixed_site_low_signal"
    if len(cleaned_text) <= 4 and relevance_score(cleaned_text, row) < 2:
        return "drop_too_short"
    if row.get("source_platform") == "weibo" and row.get("source_type") == "tree_reply":
        if target_keyword_hits(cleaned_text) == 0 and not contains_any(
            cleaned_text,
            SPACE_SCENE_KEYWORDS + ATMOSPHERE_KEYWORDS + RITUAL_IDENTITY_KEYWORDS + NEARBY_CONTEXT_KEYWORDS,
        ):
            return "drop_low_relevance"
        if target_keyword_hits(cleaned_text) == 0 and len(cleaned_text) <= 8:
            return "drop_too_short"
    if relevance_score(cleaned_text, row) >= 2:
        return "keep"
    return "drop_low_relevance"


def copy_dataset_assets(input_dir: Path, output_dir: Path) -> None:
    for relative in ["images", "historical_texts", "identity_texts", "split_map.json"]:
        src = input_dir / relative
        dst = output_dir / relative
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_cleaned_text_file(path: Path, cleaned_texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text in cleaned_texts:
            handle.write(text + "\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    manifest = read_manifest(input_dir / "regions.jsonl")
    metadata = read_json(input_dir / "metadata" / "kaitong_west_lane.json")
    raw_rows = read_csv_rows(input_dir / "current_texts" / "current_text_index.csv")

    processed_rows: list[dict[str, object]] = []
    kept_texts: list[str] = []
    reason_counts: Counter[str] = Counter()
    kept_by_platform: Counter[str] = Counter()
    kept_by_source_type: Counter[str] = Counter()
    seen_cleaned_texts: set[str] = set()

    for row in raw_rows:
        cleaned_text = build_cleaned_text(row)
        reason = classify_record(row, cleaned_text)
        keep_for_training = reason == "keep"
        if keep_for_training and cleaned_text in seen_cleaned_texts:
            reason = "drop_duplicate_normalized"
            keep_for_training = False
        if keep_for_training:
            seen_cleaned_texts.add(cleaned_text)
            kept_texts.append(cleaned_text)
            kept_by_platform[row["source_platform"]] += 1
            kept_by_source_type[row["source_type"]] += 1
        reason_counts[reason] += 1

        processed_row = dict(row)
        processed_row["cleaned_text"] = cleaned_text
        processed_row["keep_for_training"] = "true" if keep_for_training else "false"
        processed_row["drop_reason"] = "" if keep_for_training else reason
        processed_rows.append(processed_row)

    if not kept_texts:
        raise ValueError("Cleaning removed every current text record; training-ready output would be empty.")

    copy_dataset_assets(input_dir, output_dir)

    cleaned_index_path = output_dir / "current_texts" / "current_text_index_cleaned.csv"
    fieldnames = list(processed_rows[0].keys())
    write_csv_rows(cleaned_index_path, fieldnames, processed_rows)
    write_cleaned_text_file(
        output_dir / "current_texts" / "kaitong_west_lane" / "kaitong_west_lane.txt",
        kept_texts,
    )

    cleaned_manifest = deepcopy(manifest)
    cleaned_manifest["current_texts"] = kept_texts
    write_manifest(output_dir / "regions.jsonl", cleaned_manifest)

    cleaned_metadata = deepcopy(metadata)
    cleaned_metadata["current_text_count"] = len(kept_texts)
    cleaned_metadata["raw_current_text_count"] = len(raw_rows)
    cleaned_metadata["cleaning_report"] = "current_texts/cleaning_report.json"
    cleaned_metadata["notes"] = (
        "清洗版统计说明：current_text_count 为训练保留的去重后 current_texts 条数；"
        "raw_current_text_count 为原始索引 current_text_index.csv 的总记录数。"
        "清洗后逐条文本见 current_texts/current_text_index_cleaned.csv，训练入口见 regions.jsonl。"
        "其余 image/historical/identity 资源继承自原始工作区。"
    )
    write_json(output_dir / "metadata" / "kaitong_west_lane.json", cleaned_metadata)

    report = {
        "region_id": manifest["region_id"],
        "raw_current_text_count": len(raw_rows),
        "kept_current_text_count": len(kept_texts),
        "dropped_current_text_count": len(raw_rows) - len(kept_texts),
        "keep_rate": round(len(kept_texts) / len(raw_rows), 4),
        "drop_reason_counts": dict(reason_counts),
        "kept_by_platform": dict(kept_by_platform),
        "kept_by_source_type": dict(kept_by_source_type),
        "sample_kept_texts": kept_texts[:20],
    }
    write_json(output_dir / "current_texts" / "cleaning_report.json", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
