import pytest
from PIL import Image

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.wsi_zoom_in_tool import WSIZoomInTool


class FakeSlide:
    dimensions = (2000, 1000)
    level_count = 1
    level_dimensions = ((2000, 1000),)

    def __init__(self):
        self.closed = False
        self.read_args = None

    def read_region(self, location, level, size):
        self.read_args = (location, level, size)
        return Image.new("RGBA", size, "red")

    def close(self):
        self.closed = True


def make_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "wsi_zoom_in_tool",
                "description": "Zoom into a WSI DICOM series.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {"type": "array"},
                        "label": {"type": "string"},
                    },
                    "required": ["bbox_2d"],
                },
            },
        }
    )


@pytest.mark.asyncio
async def test_wsi_zoom_tool_returns_crop_image(monkeypatch):
    fake_slide = FakeSlide()

    def fake_open_dicom_slide(**kwargs):
        assert kwargs["series_dir"] == "/series"
        assert kwargs["dicom_anchor_path"] == "/series/a.dcm"
        return fake_slide

    monkeypatch.setattr("verl.tools.wsi_zoom_in_tool.wsi_dicom_utils.open_dicom_slide", fake_open_dicom_slide)
    tool = WSIZoomInTool(
        config={"type": "native", "relative_coord_max": 1000.0, "crop_long_side": 1024},
        tool_schema=make_schema(),
    )

    instance_id, _ = await tool.create(
        create_kwargs={
            "series_dir": "/series",
            "dicom_anchor_path": "/series/a.dcm",
            "file_id": "file-1",
        }
    )
    response, reward, stats = await tool.execute(instance_id, {"bbox_2d": [250, 100, 750, 900]})

    assert reward == 0.0
    assert stats["success"] is True
    assert response.text == "Here is the zoomed WSI crop:"
    assert response.image and response.image[0].mode == "RGB"
    assert response.image[0].size == (1000, 800)
    assert fake_slide.read_args == ((500, 100), 0, (1000, 800))
    assert fake_slide.closed is True


@pytest.mark.asyncio
async def test_wsi_zoom_tool_rejects_invalid_bbox_with_negative_reward():
    tool = WSIZoomInTool(config={"type": "native"}, tool_schema=make_schema())
    instance_id, _ = await tool.create(create_kwargs={"series_dir": "/series"})

    response, reward, stats = await tool.execute(instance_id, {"bbox_2d": [10, 10, 10, 20]})

    assert reward < 0
    assert stats["success"] is False
    assert stats["error"] == "invalid_bbox"
    assert response.text and response.text.startswith("Error:")


@pytest.mark.asyncio
async def test_wsi_zoom_tool_reports_dicom_open_error(monkeypatch):
    def fake_open_dicom_slide(**kwargs):
        raise RuntimeError("cannot open")

    monkeypatch.setattr("verl.tools.wsi_zoom_in_tool.wsi_dicom_utils.open_dicom_slide", fake_open_dicom_slide)
    tool = WSIZoomInTool(config={"type": "native"}, tool_schema=make_schema())
    instance_id, _ = await tool.create(create_kwargs={"series_dir": "/series"})

    response, reward, stats = await tool.execute(instance_id, {"bbox_2d": [0, 0, 1000, 1000]})

    assert reward == 0.0
    assert stats["success"] is False
    assert stats["error"] == "execution_error"
    assert stats["error_type"] == "execution_error"
    assert response.text == "Error: cannot open"
