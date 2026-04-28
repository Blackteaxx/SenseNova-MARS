from PIL import Image

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


def test_prepare_wsi_runtime_inputs_loads_thumbnail_and_injects_tool_kwargs(monkeypatch):
    loop = object.__new__(ToolAgentLoop)
    thumbnail = Image.new("RGB", (64, 32), "white")
    seen = {}

    def fake_load_wsi_thumbnail(**kwargs):
        seen.update(kwargs)
        return thumbnail

    monkeypatch.setattr("verl.experimental.agent_loop.tool_agent_loop.load_wsi_thumbnail", fake_load_wsi_thumbnail)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Question"}]}]
    wsi_data = {
        "series_dir": "/series",
        "dicom_anchor_path": "/series/a.dcm",
        "file_id": "file-1",
        "image_path": "slide.svs",
    }

    out_messages, image_data, tools_kwargs, use_initial_prompt_ids = loop._prepare_wsi_runtime_inputs(
        messages=messages,
        multi_modal_data={"wsi": wsi_data},
        tools_kwargs={},
    )

    assert out_messages is messages
    assert image_data == [thumbnail]
    assert use_initial_prompt_ids is False
    assert seen == {"series_dir": "/series", "dicom_anchor_path": "/series/a.dcm", "long_side": 1024}
    assert tools_kwargs["wsi_zoom_in_tool"]["create_kwargs"] == wsi_data


def test_prepare_wsi_runtime_inputs_preserves_existing_tool_kwargs(monkeypatch):
    loop = object.__new__(ToolAgentLoop)
    thumbnail = Image.new("RGB", (64, 32), "white")
    monkeypatch.setattr(
        "verl.experimental.agent_loop.tool_agent_loop.load_wsi_thumbnail",
        lambda **kwargs: thumbnail,
    )
    tools_kwargs = {
        "wsi_zoom_in_tool": {
            "create_kwargs": {
                "existing": "kept",
            }
        }
    }

    _, _, updated, _ = loop._prepare_wsi_runtime_inputs(
        messages=[],
        multi_modal_data={"wsi": {"series_dir": "/series"}},
        tools_kwargs=tools_kwargs,
    )

    assert updated is not tools_kwargs
    assert updated["wsi_zoom_in_tool"]["create_kwargs"]["existing"] == "kept"
    assert updated["wsi_zoom_in_tool"]["create_kwargs"]["series_dir"] == "/series"
    assert "series_dir" not in tools_kwargs["wsi_zoom_in_tool"]["create_kwargs"]


def test_prepare_wsi_runtime_inputs_converts_text_image_marker(monkeypatch):
    loop = object.__new__(ToolAgentLoop)
    thumbnail = Image.new("RGB", (64, 32), "white")
    monkeypatch.setattr(
        "verl.experimental.agent_loop.tool_agent_loop.load_wsi_thumbnail",
        lambda **kwargs: thumbnail,
    )

    messages = [{"role": "user", "content": "<image>\nQuestion"}]
    out_messages, image_data, _, use_initial_prompt_ids = loop._prepare_wsi_runtime_inputs(
        messages=messages,
        multi_modal_data={"wsi": {"series_dir": "/series"}},
        tools_kwargs={},
    )

    assert out_messages is not messages
    assert out_messages[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "\nQuestion"},
    ]
    assert image_data == [thumbnail]
    assert use_initial_prompt_ids is False


def test_prepare_wsi_runtime_inputs_noops_for_non_wsi_data():
    loop = object.__new__(ToolAgentLoop)
    tools_kwargs = {"other_tool": {"create_kwargs": {"x": 1}}}

    messages, image_data, updated, use_initial_prompt_ids = loop._prepare_wsi_runtime_inputs(
        messages=[],
        multi_modal_data={},
        tools_kwargs=tools_kwargs,
    )

    assert messages == []
    assert image_data is None
    assert updated is tools_kwargs
    assert use_initial_prompt_ids is True
