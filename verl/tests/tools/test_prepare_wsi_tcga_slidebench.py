import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "prepare_wsi_tcga_slidebench.py"
    spec = importlib.util.spec_from_file_location("prepare_wsi_tcga_slidebench", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark_name",
                "benchmark_id",
                "image_path",
                "answer",
                "options",
                "image_exists",
                "patch_exists",
                "is_valid",
                "metric_type",
                "file_id",
                "prompt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_parse_options_accepts_python_literal_and_json():
    module = _load_module()

    assert module.parse_options("['A', 'B']") == ["A", "B"]
    assert module.parse_options('["A", "B"]') == ["A", "B"]


def test_normalize_answer_to_letter_accepts_number_and_letter():
    module = _load_module()

    assert module.normalize_answer_to_letter("1") == "A"
    assert module.normalize_answer_to_letter("3") == "C"
    assert module.normalize_answer_to_letter("d") == "D"


def test_convert_writes_stable_train_and_val_jsonl(tmp_path):
    module = _load_module()
    csv_path = tmp_path / "MultiPathQA.csv"
    dicom_root = tmp_path / "dicom"
    output_root = tmp_path / "out"
    rows = [
        {
            "benchmark_name": "tcga_slidebench",
            "benchmark_id": "2",
            "image_path": "slide-b.svs",
            "answer": "1",
            "options": "['A', 'B']",
            "image_exists": "true",
            "patch_exists": "true",
            "is_valid": "true",
            "metric_type": "mcq",
            "file_id": "file-b",
            "prompt": "Question B?",
        },
        {
            "benchmark_name": "tcga_slidebench",
            "benchmark_id": "1",
            "image_path": "slide-a.svs",
            "answer": "2",
            "options": "['C', 'D']",
            "image_exists": "true",
            "patch_exists": "true",
            "is_valid": "true",
            "metric_type": "mcq",
            "file_id": "file-a",
            "prompt": "Question A?",
        },
        {
            "benchmark_name": "other",
            "benchmark_id": "3",
            "image_path": "slide-c.svs",
            "answer": "1",
            "options": "['E', 'F']",
            "image_exists": "true",
            "patch_exists": "true",
            "is_valid": "true",
            "metric_type": "mcq",
            "file_id": "file-c",
            "prompt": "Question C?",
        },
    ]
    _write_csv(csv_path, rows)

    result = module.convert(
        csv_path=csv_path,
        dicom_root=dicom_root,
        output_root=output_root,
        train_size=1,
        manifest_dir=tmp_path,
    )

    train_rows = [json.loads(line) for line in (output_root / "train" / "data.jsonl").read_text().splitlines()]
    val_rows = [json.loads(line) for line in (output_root / "val" / "data.jsonl").read_text().splitlines()]
    train_meta = json.loads(result.train_manifest.read_text())
    val_meta = json.loads(result.val_manifest.read_text())

    assert train_rows[0]["extra_info"]["benchmark_id"] == "1"
    assert val_rows[0]["extra_info"]["benchmark_id"] == "2"
    assert train_rows[0]["reward_model"]["ground_truth"] == "B"
    assert train_rows[0]["multi_modal_data"]["wsi"] == {
        "file_id": "file-a",
        "image_path": "slide-a.svs",
        "series_dir": str(dicom_root / "wsi" / "file-a"),
    }
    assert "A. C" in train_rows[0]["prompt"][0]["content"]
    assert "B. D" in train_rows[0]["prompt"][0]["content"]
    assert "<answer>...</answer>" in train_rows[0]["prompt"][0]["content"]
    assert train_meta["wsi_tcga_slidebench_train"]["length"] == 1
    assert val_meta["wsi_tcga_slidebench_val"]["length"] == 1
    assert train_meta["wsi_tcga_slidebench_train"]["reward_fn"] == ["em_score_mcq", "format_score"]
