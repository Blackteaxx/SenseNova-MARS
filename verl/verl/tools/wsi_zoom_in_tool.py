import logging
import os
from typing import Any, Optional
from uuid import uuid4

from PIL import Image

from . import wsi_dicom_utils
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WSIZoomInTool(BaseTool):
    """Zoom into a DICOM WSI using global-relative 0..1000 coordinates."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.relative_coord_max = float(config.get("relative_coord_max", 1000.0))
        self.crop_long_side = int(config.get("crop_long_side", 1024))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = dict(kwargs.get("create_kwargs", {}) or {})
        self._instance_dict[instance_id] = {
            "series_dir": create_kwargs.get("series_dir"),
            "dicom_anchor_path": create_kwargs.get("dicom_anchor_path"),
            "file_id": create_kwargs.get("file_id"),
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        bbox = parameters.get("bbox_2d") or parameters.get("bbox")
        instance_data = self._instance_dict.get(instance_id, {})

        try:
            wsi_dicom_utils.bbox_2d_to_level0_region(
                bbox,
                1000,
                1000,
                relative_coord_max=self.relative_coord_max,
            )
        except ValueError as exc:
            return (
                ToolResponse(text=f"Error: {exc}"),
                -0.05,
                {"success": False, "error": "invalid_bbox"},
            )

        try:
            slide = wsi_dicom_utils.open_dicom_slide(
                series_dir=instance_data.get("series_dir"),
                dicom_anchor_path=instance_data.get("dicom_anchor_path"),
            )
        except Exception as exc:
            logger.warning("Failed to open DICOM WSI: %s", exc)
            return (
                ToolResponse(text=f"Error: {exc}"),
                0.0,
                {"success": False, "error": "execution_error", "error_type": "execution_error"},
            )

        try:
            slide_width, slide_height = slide.dimensions
            x1, y1, x2, y2 = wsi_dicom_utils.bbox_2d_to_level0_region(
                bbox,
                slide_width,
                slide_height,
                relative_coord_max=self.relative_coord_max,
            )

            crop = slide.read_region((x1, y1), 0, (x2 - x1, y2 - y1))
            if not isinstance(crop, Image.Image):
                raise RuntimeError(f"read_region returned {type(crop).__name__}, expected PIL.Image")

            crop = crop.convert("RGB")
            crop = wsi_dicom_utils.resize_long_side(crop, self.crop_long_side)
            return (
                ToolResponse(text="Here is the zoomed WSI crop:", image=[crop]),
                0.0,
                {
                    "success": True,
                    "bbox_2d": bbox,
                    "region": [x1, y1, x2, y2],
                    "file_id": instance_data.get("file_id"),
                },
            )
        except Exception as exc:
            logger.warning("Failed to crop DICOM WSI: %s", exc)
            return (
                ToolResponse(text=f"Error: {exc}"),
                0.0,
                {"success": False, "error": "execution_error", "error_type": "execution_error"},
            )
        finally:
            close = getattr(slide, "close", None)
            if close is not None:
                close()

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
