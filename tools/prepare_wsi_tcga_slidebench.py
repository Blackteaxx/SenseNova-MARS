#!/usr/bin/env python3
"""Convert MultiPathQA TCGA SlideBench rows to SenseNova WSI RL data."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CSV_PATH = Path(
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/"
    "WSI-Nav/gigapixel-goblin/data/multipathqa/MultiPathQA.csv"
)
DEFAULT_DICOM_ROOT = Path(
    "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/"
    "data/multipathqa_tcga_dicom"
)
DEFAULT_OUTPUT_ROOT = Path("data/wsi_tcga_slidebench")

SYSTEM_PROMPT = """# Role
你是一个用于病理全切片图像问答的逐步推理助手。

你会先观察整张 WSI 的缩略图；如果需要查看细节，可以调用工具裁剪局部区域。每一轮你必须二选一：
1. 在 <tool_call>...</tool_call> 中发起一次具体工具调用；
2. 或在 <answer>...</answer> 中给出最终选项编号。

输出格式必须严格遵守：

继续观察时：
<thinking>用简短中文说明当前判断和下一步要看的区域。</thinking>
<tool_call>{"name":"wsi_zoom_in_tool","arguments":{"bbox_2d":[x1,y1,x2,y2],"label":"区域说明"}}</tool_call>

准备回答时：
<thinking>用简短中文总结证据。</thinking>
<answer>选项编号</answer>
"""


@dataclass(frozen=True)
class ConvertResult:
    train_jsonl: Path
    val_jsonl: Path
    train_manifest: Path
    val_manifest: Path
    train_count: int
    val_count: int


def parse_options(raw: str) -> list[str]:
    value = ast.literal_eval(raw)
    if not isinstance(value, list):
        raise ValueError(f"options must be a list, got {type(value).__name__}")
    return [str(item) for item in value]


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def sort_key(row: dict[str, str]) -> tuple[int, int | str]:
    benchmark_id = row.get("benchmark_id", "")
    try:
        return (0, int(benchmark_id))
    except ValueError:
        return (1, benchmark_id)


def build_prompt(question: str, options: list[str]) -> str:
    option_lines = "\n".join(f"{idx}. {option}" for idx, option in enumerate(options, start=1))
    return (
        "<image>\n"
        f"Question: {question}\n\n"
        "Options:\n"
        f"{option_lines}\n\n"
        "You may inspect the whole-slide thumbnail and call tools when needed.\n"
        "When ready, return only the final option number in <answer>...</answer>."
    )


def build_sample(row: dict[str, str], dicom_root: Path) -> dict[str, Any]:
    options = parse_options(row["options"])
    file_id = row["file_id"]
    series_dir = dicom_root / "wsi" / file_id
    benchmark_id = str(row.get("benchmark_id", ""))
    return {
        "index": benchmark_id,
        "prompt": [
            {
                "role": "user",
                "content": build_prompt(row["prompt"], options),
            }
        ],
        "reward_model": {
            "ground_truth": str(row["answer"]).strip(),
        },
        "multi_modal_data": {
            "wsi": {
                "file_id": file_id,
                "image_path": row.get("image_path", ""),
                "series_dir": str(series_dir),
            }
        },
        "extra_info": {
            "runtime_generated_initial_image": True,
            "benchmark_name": row.get("benchmark_name", ""),
            "benchmark_id": benchmark_id,
            "file_id": file_id,
        },
    }


def load_tcga_slidebench_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [
        row
        for row in rows
        if row.get("benchmark_name") == "tcga_slidebench" and truthy(row.get("is_valid", ""))
    ]
    return sorted(rows, key=sort_key)


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def manifest_entry(split: str, output_root: Path, count: int) -> dict[str, Any]:
    root = output_root / split
    return {
        "root": str(root),
        "annotation": str(root / "data.jsonl"),
        "length": count,
        "repeat_time": 1,
        "reward_fn": ["em_score_numeric_mcq", "format_score"],
        "unused_reward_fn": [],
        "input_template": {
            "name": "general",
            "arguments": {
                "system_prompt": SYSTEM_PROMPT,
                "format_instruction": "",
                "add_image_path": False,
            },
        },
        "comment": f"wsi_tcga_slidebench_{split}",
    }


def write_manifest(path: Path, split: str, output_root: Path, count: int) -> None:
    path.write_text(
        json.dumps({f"wsi_tcga_slidebench_{split}": manifest_entry(split, output_root, count)}, ensure_ascii=False, indent=2)
        + "\n"
    )


def convert(
    csv_path: Path,
    dicom_root: Path,
    output_root: Path,
    train_size: int = 160,
    manifest_dir: Path = Path("."),
) -> ConvertResult:
    rows = load_tcga_slidebench_rows(csv_path)
    samples = [build_sample(row, dicom_root) for row in rows]
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]

    train_jsonl = output_root / "train" / "data.jsonl"
    val_jsonl = output_root / "val" / "data.jsonl"
    write_jsonl(train_jsonl, train_samples)
    write_jsonl(val_jsonl, val_samples)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = manifest_dir / "train_wsi_tcga_slidebench.json"
    val_manifest = manifest_dir / "test_wsi_tcga_slidebench.json"
    write_manifest(train_manifest, "train", output_root, len(train_samples))
    write_manifest(val_manifest, "val", output_root, len(val_samples))

    return ConvertResult(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        train_count=len(train_samples),
        val_count=len(val_samples),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=Path("."))
    parser.add_argument("--train-size", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = convert(
        csv_path=args.csv_path,
        dicom_root=args.dicom_root,
        output_root=args.output_root,
        train_size=args.train_size,
        manifest_dir=args.manifest_dir,
    )
    print(f"Wrote {result.train_count} train samples to {result.train_jsonl}")
    print(f"Wrote {result.val_count} val samples to {result.val_jsonl}")
    print(f"Wrote train manifest to {result.train_manifest}")
    print(f"Wrote val manifest to {result.val_manifest}")


if __name__ == "__main__":
    main()
