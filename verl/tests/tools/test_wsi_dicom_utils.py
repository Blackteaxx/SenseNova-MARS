from pathlib import Path

import pytest
from PIL import Image

from verl.tools.wsi_dicom_utils import bbox_2d_to_level0_region, resize_long_side, resolve_dicom_anchor


def test_resolve_dicom_anchor_uses_explicit_anchor(tmp_path: Path):
    anchor = tmp_path / "b.dcm"
    anchor.write_text("x")

    assert resolve_dicom_anchor(series_dir=tmp_path, dicom_anchor_path=anchor) == anchor


def test_resolve_dicom_anchor_picks_sorted_dcm(tmp_path: Path):
    (tmp_path / "b.dcm").write_text("x")
    (tmp_path / "a.dcm").write_text("x")
    (tmp_path / "ignore.txt").write_text("x")

    assert resolve_dicom_anchor(tmp_path).name == "a.dcm"


def test_resolve_dicom_anchor_rejects_missing_series(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="DICOM series directory does not exist"):
        resolve_dicom_anchor(tmp_path / "missing")


def test_resolve_dicom_anchor_rejects_empty_series(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No .dcm files found"):
        resolve_dicom_anchor(tmp_path)


def test_bbox_to_region_converts_global_relative():
    assert bbox_2d_to_level0_region([0, 0, 1000, 1000], 2000, 1000) == (0, 0, 2000, 1000)
    assert bbox_2d_to_level0_region([250, 100, 750, 900], 2000, 1000) == (500, 100, 1500, 900)


@pytest.mark.parametrize(
    "bbox",
    [
        [10, 10, 10, 20],
        [-1, 0, 10, 20],
        [0, 0, 1001, 20],
        [0, 0, 20],
        ["a", 0, 10, 20],
    ],
)
def test_bbox_validation_rejects_bad_values(bbox):
    with pytest.raises(ValueError):
        bbox_2d_to_level0_region(bbox, 1000, 1000)


def test_resize_long_side_preserves_aspect_ratio():
    image = Image.new("RGB", (2000, 1000), "white")

    resized = resize_long_side(image, 1000)

    assert resized.size == (1000, 500)


def test_resize_long_side_does_not_upscale():
    image = Image.new("RGB", (500, 250), "white")

    resized = resize_long_side(image, 1000)

    assert resized is image
