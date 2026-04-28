# WSI DICOM RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small-scale TCGA SlideBench WSI DICOM RL training loop in SenseNova-MARS using the existing ToolAgentLoop, a new WSI zoom tool, numeric MCQ reward, and conversion scripts.

**Architecture:** Keep SenseNova's rollout/token/logprob machinery intact. Extend ToolAgentLoop only at the multimodal lazy-loading boundary so `multi_modal_data.wsi` produces a runtime thumbnail and tool create kwargs; put all DICOM series IO and crop logic in a new focused WSI tool/helper; keep close-ended reward as a small addition to the existing tool reward functions.

**Tech Stack:** Python, verl/SenseNova agent loop, OpenSlide, PIL, Qwen3-VL processor, pytest, YAML tool configs, JSON/JSONL dataset manifests.

---

## File Structure

- Create `verl/verl/tools/wsi_dicom_utils.py`
  - Pure and low-IO helpers: DICOM anchor resolution, OpenSlide wrapper helpers, thumbnail generation, relative bbox validation/conversion, image resizing.
- Create `verl/verl/tools/wsi_zoom_in_tool.py`
  - `WSIZoomInTool` native tool. Owns WSI DICOM crop execution and tool reward/error semantics.
- Modify `verl/verl/experimental/agent_loop/tool_agent_loop.py`
  - Add WSI lazy-loading support in `run()`.
  - Inject WSI metadata into `tools_kwargs["wsi_zoom_in_tool"]["create_kwargs"]`.
  - Disable dataset precomputed prompt ids for runtime-generated WSI thumbnails.
- Reward reuses existing `em_score_mcq`; the data converter normalizes CSV numeric answers to `A/B/C/D`.
- Keep the one-off MultiPathQA converter outside the SenseNova source tree, for example at `../data/SenseNova-example/scripts/prepare_wsi_tcga_slidebench.py`.
- Create `config/tool_config/tools_wsi_train.yaml`
  - Tool registry config for `wsi_zoom_in_tool`.
- Create `config/tool_config/tools_wsi_val.yaml`
  - Validation tool config, initially identical to train.
- Create `train_wsi_tcga_slidebench.sh`
  - Small-scale GRPO entrypoint for WSI DICOM training.
- Create tests:
  - `verl/tests/tools/test_wsi_dicom_utils.py`
  - `verl/tests/tools/test_wsi_zoom_in_tool.py`
  - `verl/tests/experimental/agent_loop/test_wsi_lazy_loading.py`

## Task 1: MCQ Reward Convention

**Files:**
- No SenseNova reward source-code changes required.
- Data converter normalizes CSV answers before writing JSONL.

- [ ] **Step 1: Reuse existing reward**

Use the existing reward function:

```text
em_score_mcq
```

The converter should write:

```json
"reward_fn": ["em_score_mcq", "format_score"]
```

- [ ] **Step 2: Normalize answer labels in the data script**

Convert CSV numeric answers to letters:

```text
1 -> A
2 -> B
3 -> C
4 -> D
```

- [ ] **Step 3: Render options with letters**

```text
A. {options[0]}
B. {options[1]}
```

- [ ] **Step 4: Verify no numeric reward is registered**

```bash
python3 - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location("reward_tool", "verl/verl/utils/reward_score/tool.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert "em_score_mcq" in mod.compute_score_fns
assert "em_score_numeric_mcq" not in mod.compute_score_fns
PY
```

## Task 2: WSI DICOM Helpers

**Files:**
- Create: `verl/verl/tools/wsi_dicom_utils.py`
- Test: `verl/tests/tools/test_wsi_dicom_utils.py`

- [ ] **Step 1: Write helper tests**

Tests should avoid requiring real OpenSlide data. Use temp dirs and monkeypatch fake slide objects.

Cover:

```python
def test_resolve_dicom_anchor_picks_sorted_dcm(tmp_path):
    (tmp_path / "b.dcm").write_text("x")
    (tmp_path / "a.dcm").write_text("x")
    assert resolve_dicom_anchor(tmp_path).name == "a.dcm"

def test_bbox_to_region_converts_global_relative():
    assert bbox_2d_to_level0_region([0, 0, 1000, 1000], 2000, 1000) == (0, 0, 2000, 1000)
    assert bbox_2d_to_level0_region([250, 100, 750, 900], 2000, 1000) == (500, 100, 1500, 900)

def test_bbox_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        bbox_2d_to_level0_region([10, 10, 10, 20], 1000, 1000)
    with pytest.raises(ValueError):
        bbox_2d_to_level0_region([-1, 0, 10, 20], 1000, 1000)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd /Users/hutu/codes/WSI-Nav/SenseNova-MARS/verl
uv run pytest tests/tools/test_wsi_dicom_utils.py -q
```

Expected: FAIL because module is missing.

- [ ] **Step 3: Implement helpers**

Implement:

```python
resolve_dicom_anchor(series_dir: str | Path, dicom_anchor_path: str | Path | None = None) -> Path
bbox_2d_to_level0_region(bbox: list, slide_width: int, slide_height: int) -> tuple[int, int, int, int]
resize_long_side(image: Image.Image, max_long_side: int) -> Image.Image
load_wsi_thumbnail(series_dir=None, dicom_anchor_path=None, long_side=1024) -> Image.Image
```

Import `openslide` inside functions so pure unit tests do not need the package unless thumbnail/crop is exercised.

- [ ] **Step 4: Run helper tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/verl/tools/wsi_dicom_utils.py verl/tests/tools/test_wsi_dicom_utils.py
git commit -m "feat(tools): add wsi dicom helpers"
```

## Task 3: WSI Zoom Tool

**Files:**
- Create: `verl/verl/tools/wsi_zoom_in_tool.py`
- Test: `verl/tests/tools/test_wsi_zoom_in_tool.py`

- [ ] **Step 1: Write tool tests**

Use monkeypatch to replace the OpenSlide-opening helper with a fake slide:

```python
class FakeSlide:
    dimensions = (2000, 1000)
    level_count = 1
    level_dimensions = ((2000, 1000),)
    def read_region(self, location, level, size):
        assert location == (500, 100)
        assert size == (1000, 800)
        return Image.new("RGBA", size, "red")
    def close(self):
        pass
```

Cover:

```python
async def test_wsi_zoom_tool_returns_crop_image(...)
async def test_wsi_zoom_tool_rejects_invalid_bbox_with_negative_reward(...)
async def test_wsi_zoom_tool_reports_missing_series_as_execution_error(...)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd /Users/hutu/codes/WSI-Nav/SenseNova-MARS/verl
uv run pytest tests/tools/test_wsi_zoom_in_tool.py -q
```

Expected: FAIL because tool is missing.

- [ ] **Step 3: Implement `WSIZoomInTool`**

Implement BaseTool methods:

```python
class WSIZoomInTool(BaseTool):
    async def create(self, instance_id=None, **kwargs):
        # Store series_dir, dicom_anchor_path, file_id from create_kwargs.

    async def execute(self, instance_id, parameters, **kwargs):
        # Validate bbox_2d.
        # Resolve/open DICOM anchor.
        # Convert bbox to level-0 region.
        # read_region(location=(x1, y1), level=0, size=(w, h)).
        # Convert RGBA to RGB, resize long side, return ToolResponse(image=[crop]).
```

Model-caused argument errors return `ToolResponse(text="Error: ..."), -0.05, {"success": False, "error": "invalid_bbox"}`.

Environment errors return text response, `0.0`, and `{"success": False, "error": "...", "error_type": "execution_error"}` where practical.

- [ ] **Step 4: Run tool tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/verl/tools/wsi_zoom_in_tool.py verl/tests/tools/test_wsi_zoom_in_tool.py
git commit -m "feat(tools): add wsi zoom tool"
```

## Task 4: ToolAgentLoop WSI Lazy Loading

**Files:**
- Modify: `verl/verl/experimental/agent_loop/tool_agent_loop.py`
- Test: `verl/tests/experimental/agent_loop/test_wsi_lazy_loading.py`

- [ ] **Step 1: Write focused tests**

Avoid full Ray rollout. Unit-test helper methods added to `ToolAgentLoop`.

Expected helper boundary:

```python
messages, image_data, tools_kwargs, use_initial_prompt_ids = loop._prepare_wsi_runtime_inputs(
    messages=messages,
    multi_modal_data={"wsi": {"series_dir": "/x", "file_id": "f"}},
    tools_kwargs={},
    runtime_generated_initial_image=True,
)
```

Test:

1. `load_wsi_thumbnail` is called.
2. returned `image_data` contains one PIL image.
3. `tools_kwargs["wsi_zoom_in_tool"]["create_kwargs"]` contains WSI metadata.
4. `use_initial_prompt_ids` is False.
5. non-WSI path remains unchanged.

- [ ] **Step 2: Run tests and verify failure**

```bash
cd /Users/hutu/codes/WSI-Nav/SenseNova-MARS/verl
uv run pytest tests/experimental/agent_loop/test_wsi_lazy_loading.py -q
```

Expected: FAIL because helper is missing.

- [ ] **Step 3: Implement ToolAgentLoop helper and integrate into `run()`**

Add a small method, not a large inline block:

```python
def _prepare_wsi_runtime_inputs(self, messages, multi_modal_data, tools_kwargs):
    wsi_data = multi_modal_data.get("wsi")
    if not wsi_data:
        return messages, None, tools_kwargs, True
    image = load_wsi_thumbnail(...)
    tools_kwargs = copy.deepcopy(tools_kwargs or {})
    create_kwargs = tools_kwargs.setdefault("wsi_zoom_in_tool", {}).setdefault("create_kwargs", {})
    create_kwargs.update(wsi_data)
    return messages, [image], tools_kwargs, False
```

In `run()`:

1. Check WSI before `image_paths`.
2. Store the returned thumbnail list in the local `image_data` variable.
3. Store the updated `tools_kwargs` in the local `tools_kwargs` variable before constructing `AgentData`.
4. If WSI exists, set `initial_prompt_ids = None`.
5. Preserve existing image path lazy loading for non-WSI samples.

The state flow must be explicit:

```python
messages, wsi_image_data, tools_kwargs, use_initial_prompt_ids = self._prepare_wsi_runtime_inputs(
    messages=messages,
    multi_modal_data=multi_modal_data,
    tools_kwargs=tools_kwargs,
)
if wsi_image_data is not None:
    image_data = wsi_image_data
...
if not use_initial_prompt_ids:
    initial_prompt_ids = None
```

Then `AgentData(image_data=image_data, tools_kwargs=tools_kwargs, initial_prompt_ids=initial_prompt_ids, ...)` carries the runtime thumbnail into `_handle_pending_state()`.

No separate `_handle_pending_state()` rewrite is needed beyond preserving the existing fallback branch. The existing code already does the required runtime tokenization when `agent_data.initial_prompt_ids is None`:

```python
raw_prompt = self.processor.apply_chat_template(
    agent_data.messages,
    tools=tool_schemas,
    add_generation_prompt=True,
    tokenize=False,
    **self.apply_chat_template_kwargs,
)
model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
```

This satisfies the spec requirement: WSI samples skip dataset precomputed prompt ids and use the runtime-generated thumbnail for processor tokenization. Position ids and final padded tensors are produced later by `AgentLoopWorkerBase._postprocess()` from `AgentLoopOutput.prompt_ids`, `response_ids`, `attention_mask`, and loaded images; do not try to mutate the original dataset batch tensors inside `ToolAgentLoop`.

- [ ] **Step 4: Run lazy-loading tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/verl/experimental/agent_loop/tool_agent_loop.py verl/tests/experimental/agent_loop/test_wsi_lazy_loading.py
git commit -m "feat(agent): load wsi thumbnails at runtime"
```

## Task 5: MultiPathQA Conversion Script

**Files:**
- External data script: `../data/SenseNova-example/scripts/prepare_wsi_tcga_slidebench.py`
- SenseNova source-tree change: `train_wsi_tcga_slidebench.sh` should call that script through `DATA_PREP_SCRIPT`.

- [ ] **Step 1: Write conversion tests**

Use a temp CSV with fields:

```text
benchmark_name,benchmark_id,image_path,answer,options,image_exists,patch_exists,is_valid,metric_type,file_id,prompt
```

Test:

1. Only `tcga_slidebench` and truthy `is_valid` rows are kept.
2. Options parse from Python-list string.
3. Output has `reward_model.ground_truth`.
4. Output has `multi_modal_data.wsi.series_dir`.
5. Output has `extra_info.runtime_generated_initial_image = true`.
6. Fixed split sizes work for a small requested split.

- [ ] **Step 2: Run tests and verify failure**

```bash
python3 /Users/hutu/codes/WSI-Nav/data/SenseNova-example/scripts/prepare_wsi_tcga_slidebench.py --help
```

Expected: FAIL because script is missing.

- [ ] **Step 3: Implement converter**

CLI:

```bash
python ../data/SenseNova-example/scripts/prepare_wsi_tcga_slidebench.py \
  --csv-path /path/MultiPathQA.csv \
  --dicom-root /mnt/.../multipathqa_tcga_dicom \
  --output-root data/wsi_tcga_slidebench \
  --train-size 160
```

Outputs:

```text
data/wsi_tcga_slidebench/train/data.jsonl
data/wsi_tcga_slidebench/val/data.jsonl
train_wsi_tcga_slidebench.json
```

The manifest JSON should point to generated JSONL files and set:

```json
"reward_fn": ["em_score_mcq", "format_score"]
```

Every converted row must write:

```json
"extra_info": {
  "runtime_generated_initial_image": true,
  "benchmark_name": "tcga_slidebench",
  "benchmark_id": "...",
  "file_id": "..."
}
```

Prompt text must follow the spec template exactly:

```text
<image>
Question: {prompt}

Options:
A. {options[0]}
B. {options[1]}
...

You may inspect the whole-slide thumbnail and call tools when needed.
When ready, return only the final option letter in <answer>...</answer>.
```

`options` should parse Python-list strings and JSON-list strings; preserve the CSV option order and render letter labels. `answer` should be normalized from the CSV 1-based number into `A/B/C/D` before writing `reward_model.ground_truth`.

- [ ] **Step 4: Run converter tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add train_wsi_tcga_slidebench.sh
git commit -m "chore(data): keep wsi converter outside source tree"
```

## Task 6: WSI Tool Configs and Training Script

**Files:**
- Create: `config/tool_config/tools_wsi_train.yaml`
- Create: `config/tool_config/tools_wsi_val.yaml`
- Create: `train_wsi_tcga_slidebench.sh`

- [ ] **Step 1: Add tool config**

Define native tool:

```yaml
tools:
  - class_name: verl.tools.wsi_zoom_in_tool.WSIZoomInTool
    config:
      type: native
      relative_coord_max: 1000.0
      thumbnail_long_side: 1024
      crop_long_side: 1024
      timeout_seconds: 30
      use_smart_resize: true
    tool_schema:
      type: function
      function:
        name: wsi_zoom_in_tool
        description: "Zoom in on a whole-slide image DICOM series using global-relative 0..1000 coordinates."
        parameters:
          type: object
          properties:
            bbox_2d:
              type: array
              items:
                type: number
              minItems: 4
              maxItems: 4
            label:
              type: string
          required:
            - bbox_2d
```

- [ ] **Step 2: Add small-scale training script**

Base it on `train_multi_node.sh`, but set:

```bash
TOOL_CONFIG_TRAIN="$(dirname $0)/config/tool_config/tools_wsi_train.yaml"
TOOL_CONFIG_VAL="$(dirname $0)/config/tool_config/tools_wsi_val.yaml"
train_files=$(dirname $0)/train_wsi_tcga_slidebench.json
val_files=$(dirname $0)/train_wsi_tcga_slidebench.json
data.return_raw_chat=True
data.train_batch_size=${TRAIN_BATCH_SIZE:-16}
data.val_batch_size=${VAL_BATCH_SIZE:-16}
actor_rollout_ref.rollout.n=${ROLLOUT_N:-2}
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_ASSISTANT_TURNS:-3}
```

Do not configure LLM judge reward kwargs because WSI first version uses only numeric EM.

- [ ] **Step 3: Validate YAML loads**

Run:

```bash
cd /Users/hutu/codes/WSI-Nav/SenseNova-MARS
PYTHONPATH=verl uv run python - <<'PY'
from verl.tools.utils.tool_registry import initialize_tools_from_config
for path in ["config/tool_config/tools_wsi_train.yaml", "config/tool_config/tools_wsi_val.yaml"]:
    tools = initialize_tools_from_config(path)
    assert [t.name for t in tools] == ["wsi_zoom_in_tool"]
print("ok")
PY
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add config/tool_config/tools_wsi_train.yaml config/tool_config/tools_wsi_val.yaml train_wsi_tcga_slidebench.sh
git commit -m "feat(config): add wsi rl training entrypoint"
```

## Task 7: End-to-End Local Verification

**Files:**
- No new files unless fixes are needed.

- [ ] **Step 1: Run focused tests**

```bash
cd /Users/hutu/codes/WSI-Nav/SenseNova-MARS/verl
uv run pytest \
  tests/tools/test_wsi_dicom_utils.py \
  tests/tools/test_wsi_zoom_in_tool.py \
  tests/experimental/agent_loop/test_wsi_lazy_loading.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run converter on remote-mounted/local-visible paths if available**

If the remote DolphinFS path is not locally mounted, skip locally and run on `cpu-jump` later.

```bash
python ../data/SenseNova-example/scripts/prepare_wsi_tcga_slidebench.py \
  --csv-path /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/WSI-Nav/gigapixel-goblin/data/multipathqa/MultiPathQA.csv \
  --dicom-root /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/data/multipathqa_tcga_dicom \
  --output-root data/wsi_tcga_slidebench \
  --train-size 160
```

Expected: 160 train rows, 37 val rows, manifest written.

- [ ] **Step 3: Run DICOM smoke test on `cpu-jump`**

Use generated JSONL and a tiny helper command to instantiate thumbnail/crop for 1 sample. Expected: OpenSlide opens DICOM anchor, thumbnail and crop are generated.

- [ ] **Step 4: Commit any verification fixes**

Commit only if code changes were needed:

```bash
git add <changed files>
git commit -m "fix(wsi): address smoke verification issues"
```
