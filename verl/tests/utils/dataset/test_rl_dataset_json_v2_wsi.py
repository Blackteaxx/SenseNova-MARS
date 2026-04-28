import torch

from verl.utils.dataset.rl_dataset_json_v2 import RLHFJSONDatasetV2


class _FakeProcessor:
    image_processor = object()

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "serialized prompt"

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class _FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def test_json_dataset_preserves_wsi_multi_modal_data_without_loading_images():
    dataset = object.__new__(RLHFJSONDatasetV2)
    processor = _FakeProcessor()
    dataset.dataframe = [
        {
            "id": "sample-1",
            "prompt": [{"role": "user", "content": "<image>\nQuestion"}],
            "multi_modal_data": {
                "wsi": {
                    "file_id": "file-1",
                    "series_dir": "/series",
                    "image_path": "slide.svs",
                }
            },
            "reward_model": {"ground_truth": "1"},
            "extra_info": {},
        }
    ]
    dataset.prompt_key = "prompt"
    dataset.image_key = "image"
    dataset.video_key = "video"
    dataset.processor = processor
    dataset.tokenizer = _FakeTokenizer()
    dataset.apply_chat_template_kwargs = {}
    dataset.tool_schemas = None
    dataset.image_patch_size = 16
    dataset.return_multi_modal_inputs = False
    dataset.max_prompt_length = 8
    dataset.truncation = "error"
    dataset.return_full_prompt = False
    dataset.image_search_title_list_key = "image_search_title_list"
    dataset.image_search_thumbnail_list_key = "image_search_thumbnail_list"
    dataset.image_search_summary_key = "image_search_summary"
    dataset.image_search_max_results = 3
    dataset.need_tools_kwargs = False

    row = dataset[0]

    assert row["multi_modal_data"] == {
        "wsi": {
            "file_id": "file-1",
            "series_dir": "/series",
            "image_path": "slide.svs",
        }
    }
    assert processor.call_kwargs["images"] is None
