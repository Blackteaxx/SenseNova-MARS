import math
from pathlib import Path
from typing import Any

from PIL import Image


def resolve_dicom_anchor(
    series_dir: str | Path | None = None,
    dicom_anchor_path: str | Path | None = None,
) -> Path:
    if dicom_anchor_path:
        anchor = Path(dicom_anchor_path)
        if not anchor.is_file():
            raise FileNotFoundError(f"DICOM anchor does not exist: {anchor}")
        return anchor

    if not series_dir:
        raise ValueError("series_dir or dicom_anchor_path is required")

    series_path = Path(series_dir)
    if not series_path.is_dir():
        raise FileNotFoundError(f"DICOM series directory does not exist: {series_path}")

    anchors = sorted(series_path.glob("*.dcm"))
    if not anchors:
        raise FileNotFoundError(f"No .dcm files found in DICOM series directory: {series_path}")
    return anchors[0]


def bbox_2d_to_level0_region(
    bbox: list[Any] | tuple[Any, ...],
    slide_width: int,
    slide_height: int,
    *,
    relative_coord_max: float = 1000.0,
) -> tuple[int, int, int, int]:
    if slide_width <= 0 or slide_height <= 0:
        raise ValueError(f"Slide dimensions must be positive, got {slide_width}x{slide_height}")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox_2d must be a list of four numeric values")

    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox_2d values must be numeric: {bbox}") from exc

    if any(math.isnan(value) or math.isinf(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"bbox_2d values must be finite: {bbox}")

    if min(x1, y1, x2, y2) < 0 or max(x1, y1, x2, y2) > relative_coord_max:
        raise ValueError(f"bbox_2d values must be in [0, {relative_coord_max}]: {bbox}")

    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"bbox_2d must satisfy x1 < x2 and y1 < y2: {bbox}")

    abs_x1 = round(x1 / relative_coord_max * slide_width)
    abs_y1 = round(y1 / relative_coord_max * slide_height)
    abs_x2 = round(x2 / relative_coord_max * slide_width)
    abs_y2 = round(y2 / relative_coord_max * slide_height)

    abs_x1 = max(0, min(slide_width, abs_x1))
    abs_y1 = max(0, min(slide_height, abs_y1))
    abs_x2 = max(0, min(slide_width, abs_x2))
    abs_y2 = max(0, min(slide_height, abs_y2))

    if abs_x1 >= abs_x2 or abs_y1 >= abs_y2:
        raise ValueError(f"bbox_2d maps to an empty level-0 region: {bbox}")

    return abs_x1, abs_y1, abs_x2, abs_y2


def resize_long_side(image: Image.Image, max_long_side: int) -> Image.Image:
    if max_long_side <= 0:
        raise ValueError(f"max_long_side must be positive, got {max_long_side}")

    width, height = image.size
    long_side = max(width, height)
    if long_side <= max_long_side:
        return image

    scale = max_long_side / long_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def open_dicom_slide(
    series_dir: str | Path | None = None,
    dicom_anchor_path: str | Path | None = None,
):
    anchor = resolve_dicom_anchor(series_dir=series_dir, dicom_anchor_path=dicom_anchor_path)
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError("openslide is required for WSI DICOM loading") from exc

    return openslide.OpenSlide(str(anchor))


def load_wsi_thumbnail(
    *,
    series_dir: str | Path | None = None,
    dicom_anchor_path: str | Path | None = None,
    long_side: int = 1024,
) -> Image.Image:
    slide = open_dicom_slide(series_dir=series_dir, dicom_anchor_path=dicom_anchor_path)
    try:
        if getattr(slide, "level_count", 0) <= 0:
            raise RuntimeError("OpenSlide returned no levels for DICOM WSI")
        if not getattr(slide, "level_dimensions", None):
            raise RuntimeError("OpenSlide returned no level dimensions for DICOM WSI")
        thumbnail = slide.get_thumbnail((long_side, long_side)).convert("RGB")
        return thumbnail
    finally:
        close = getattr(slide, "close", None)
        if close is not None:
            close()
