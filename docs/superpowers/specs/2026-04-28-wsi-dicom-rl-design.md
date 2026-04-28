# WSI DICOM RL 训练闭环设计

日期：2026-04-28

## 背景与目标

本设计的目标是在 `SenseNova-MARS` 中搭建一个小规模 WSI 强化学习训练闭环，用于训练 Qwen3-VL 在 TCGA SlideBench close-ended 任务上的 WSI 导航能力。

第一版只复刻 `gigapixel-goblin` 中 GIANT 的核心交互协议，不复用 GIANT 的完整 OpenAI-compatible runtime。训练流程仍由 SenseNova/verl 管理 rollout、token、logprob、tool response mask 和 PPO/GRPO 更新。

目标数据来自远端：

```text
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/WSI-Nav/gigapixel-goblin/data/multipathqa/MultiPathQA.csv
```

WSI 图像使用 TCGA DICOM series，而不是 SVS。DICOM series 根目录为：

```text
/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/data/multipathqa_tcga_dicom/wsi/<file_id>/*.dcm
```

`MultiPathQA.csv` 中的 `image_path` 仍会保留在样本中，但它只作为 provenance 字段记录原始 slide 名称，不参与 WSI 打开。运行时 WSI 解析优先级为：

```text
series_dir -> dicom_anchor_path -> file_id 推导的 series_dir
```

不得从 `image_path` 的 `.svs` 文件名去读取图像。

第一版成功标准：

1. 从 `tcga_slidebench` 构造固定划分的 SenseNova 训练/验证数据。
2. rollout 开始时动态从 DICOM series 生成 whole-slide thumbnail。
3. 模型可以调用 WSI crop tool，在 0..1000 global-relative 坐标系中裁剪 WSI 区域。
4. 最终答案用 `<answer>A/B/C/D</answer>` 表示选项字母。
5. reward 复用现有字母选项 EM 和格式分。
6. 可以跑通一次小规模 GRPO 训练与验证闭环。

## 非目标

第一版不做以下事情：

1. 不复用 `gigapixel-goblin` 的 `OpenAICompatRuntime`。
2. 不把训练轨迹直接写回 GIANT 的 benchmark artifact 格式。
3. 不使用 LLM judge 作为训练 reward。
4. 不做 open-ended VQA reward。
5. 不预生成 thumbnail/crop 图片作为训练输入。
6. 不改造 vLLM/OpenAI tool parser；继续使用 SenseNova 当前 prompt-level tool-call 解析。

## 总体架构

整体仍然沿用 SenseNova 当前 agent loop 框架：

```text
RLHFJSONDatasetV2
  -> ToolAgentLoop
  -> SGLang rollout server
  -> WSIZoomInTool
  -> ToolRewardManager
  -> GRPO/PPO update
```

关键改动分为四层：

1. 数据转换：在 `scripts/prepare_wsi_tcga_slidebench.py` 放置 MultiPathQA 到 SenseNova JSON/JSONL 的数据准备脚本；它是训练辅助脚本，不属于 `verl` 核心库源码。
2. AgentLoop：复用 `ToolAgentLoop`，只扩展 lazy loading 支持 `multi_modal_data.wsi`。
3. Tool：新增 WSI DICOM crop tool。
4. Reward：复用现有 `em_score_mcq`，转换脚本将 CSV 中的 1-based 答案编号规范化为 `A/B/C/D`。

## 数据设计

### 输入数据

从 `MultiPathQA.csv` 中筛选：

```text
benchmark_name == "tcga_slidebench"
is_valid == True
```

远端 CSV 当前应有 197 条 `tcga_slidebench` 样本。第一版采用固定划分：

```text
train: 160
val:   37
```

划分规则应可复现。推荐按 `benchmark_id` 的数值或字符串稳定排序后切分，避免随机 seed 改变导致实验不可复现。

远端 CSV 当前字段为：

```text
benchmark_name, benchmark_id, image_path, answer, options,
image_exists, patch_exists, is_valid, metric_type, file_id, prompt
```

字段映射如下：

| CSV 字段 | SenseNova 字段 | 说明 |
| --- | --- | --- |
| `benchmark_name` | `extra_info.benchmark_name` | 只接受 `tcga_slidebench` |
| `benchmark_id` | `extra_info.benchmark_id` / `index` | 用于稳定排序、追踪样本 |
| `prompt` | `prompt[0].content` | 作为问题文本 |
| `options` | `prompt[0].content` | 解析 Python/JSON list 后展开为 `A/B/C/D` 选项 |
| `answer` | `reward_model.ground_truth` | CSV 中的 1-based 选项编号会转换成 `A/B/C/D` |
| `file_id` | `multi_modal_data.wsi.file_id` | 用于定位 DICOM series |
| `image_path` | `multi_modal_data.wsi.image_path` | 仅记录原始 SVS 名称，不用于读取 |

选项展开模板为：

```text
Question: {prompt}

Options:
A. {options[0]}
B. {options[1]}
...

You may inspect the whole-slide thumbnail and call tools when needed.
When ready, return only the final option letter in <answer>...</answer>.
```

### 输出数据格式

转换后每条样本保持 SenseNova 现有结构，尽量不引入不必要的新字段。

示例结构：

```json
{
  "prompt": [
    {
      "role": "user",
      "content": "<image>\nQuestion: ...\n\nOptions:\nA. ...\nB. ...\nC. ...\nD. ...\n\nReturn the final option letter in <answer>...</answer>."
    }
  ],
  "reward_model": {
    "ground_truth": "3"
  },
  "multi_modal_data": {
    "wsi": {
      "file_id": "fa75eee1-cb31-46bc-a814-7baee5847d64",
      "image_path": "TCGA-HC-7080-01Z-00-DX1.svs",
      "series_dir": "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/data/multipathqa_tcga_dicom/wsi/fa75eee1-cb31-46bc-a814-7baee5847d64"
    }
  },
  "extra_info": {
    "benchmark_name": "tcga_slidebench",
    "benchmark_id": "2270",
    "file_id": "fa75eee1-cb31-46bc-a814-7baee5847d64"
  }
}
```

`options` 和 `answer` 不需要额外传给 tool：

1. `options` 只展开到 prompt 文本中。
2. `answer` 只进入 `reward_model.ground_truth`。
3. WSI tool 只需要 DICOM series 信息和模型给出的 bbox。

### Meta 配置

新增训练配置 JSON，例如：

```text
train_wsi_tcga_slidebench.json
```

其中 train split 使用：

```json
"reward_fn": ["em_score_mcq", "format_score"]
```

val split 使用同样 reward，保证训练与验证口径一致。

## AgentLoop 设计

### 复用原则

继续使用现有 `ToolAgentLoop`：

```text
verl/verl/experimental/agent_loop/tool_agent_loop.py
```

不新增单独的 `wsi_tool_agent`，避免复制状态机逻辑。现有状态机已经满足需求：

```text
PENDING -> GENERATING -> PROCESSING_TOOLS -> GENERATING -> TERMINATED
```

需要扩展的位置是 `run()` 中的 multi-modal lazy loading：

```python
multi_modal_data = kwargs.get("multi_modal_data", {})
image_paths = multi_modal_data.get("image_paths", None)
```

新增分支：

```python
wsi_data = multi_modal_data.get("wsi", None)
if wsi_data:
    image_data = [load_wsi_thumbnail(wsi_data)]
    inject_wsi_create_kwargs(tools_kwargs, wsi_data)
elif image_paths:
    ...
```

### Initial Thumbnail

rollout 开始时，AgentLoop 从 `multi_modal_data.wsi` 解析 DICOM series：

1. 找到 `series_dir`。
2. 选择一个稳定的 `.dcm` anchor。
3. 用 OpenSlide 打开 anchor。
4. 生成 whole-slide thumbnail。
5. 将 thumbnail 作为第一张 image 放入 `image_data`。

这样 dataset 不需要预生成普通图片，符合运行时生成 thumbnail 的设计。

### Prompt Tokenization 注意点

现有训练路径会在 dataset 阶段预先生成 `input_ids`。但如果 initial thumbnail 在 AgentLoop runtime 才生成，dataset 阶段没有真实 image，不能正确计算 Qwen3-VL 的视觉 token 与 mrope position ids。

第一版需要在设计上明确处理这一点。推荐方案是：

1. WSI 样本在 dataset 阶段仍保留 `<image>` placeholder 和 raw chat。
2. 对 WSI 数据禁用复用 dataset 预计算的 `initial_prompt_ids`。
3. 在 `ToolAgentLoop._handle_pending_state()` 中，当存在 runtime-generated WSI thumbnail 时，使用 processor 重新 apply chat template 并 processor tokenize。

这会让 WSI 样本走与 validation 类似的 runtime prompt 构造路径，确保 image token 与 position ids 来自真实 thumbnail。

如果后续发现训练框架强依赖 dataset `input_ids` 的 shape，可以先用小规模 smoke test 验证，并在必要时为 WSI dataset 增加专用标记，例如：

```json
"extra_info": {
  "runtime_generated_initial_image": true
}
```

AgentLoop 看到该标记后忽略 dataset 预计算 prompt ids。

具体实现边界：

1. `scripts/prepare_wsi_tcga_slidebench.py` 为 WSI 样本写入 `extra_info.runtime_generated_initial_image = true`。
2. `ToolAgentLoop.run()` 看到该标记和 `multi_modal_data.wsi` 后，不使用 kwargs 中的 `input_ids` 作为 `initial_prompt_ids`。
3. `_handle_pending_state()` 在 WSI 模式下用 runtime 生成的 thumbnail 重新调用 processor，得到新的 prompt ids，并写入 `agent_data.prompt_ids`。
4. 不需要回写原始 batch 的 `input_ids`、`attention_mask`、`position_ids`。AgentLoop 的 `_postprocess()` 会使用 `AgentLoopOutput.prompt_ids/response_ids/response_mask` 重新组装 rollout batch。

这一路径必须用 smoke test 验证，防止训练 worker 仍然隐式依赖 dataset 阶段的 prompt tensor shape。

### Thumbnail 与图像尺寸

第一版使用显式尺寸配置，避免 processor 行为随默认值变化：

```text
wsi_thumbnail_long_side: 1024
wsi_crop_long_side: 1024
image_patch_size: 16
```

thumbnail 生成时保持宽高比，最长边不超过 `wsi_thumbnail_long_side`。crop 返回时保持宽高比，最长边不超过 `wsi_crop_long_side`，并继续交给 Qwen3-VL processor 按 `image_patch_size=16` 处理。

如果 tool config 中提供 `min_pixels` / `max_pixels`，WSI tool 应与现有 `ImageZoomInTool` 一样在返回前做 smart resize，保证尺寸落在 Qwen3-VL 可接受范围内。

### Tool kwargs 注入

AgentLoop 负责将 WSI metadata 注入到 tool create kwargs：

```python
tools_kwargs.setdefault("wsi_zoom_in_tool", {})
tools_kwargs["wsi_zoom_in_tool"].setdefault("create_kwargs", {})
tools_kwargs["wsi_zoom_in_tool"]["create_kwargs"].update({
    "series_dir": ...,
    "dicom_anchor_path": ...,
    "file_id": ...,
})
```

模型不需要也不应该看到 `series_dir` 或 `file_id`。模型只输出 bbox。

## Tool 设计

### 新增工具

新增文件：

```text
verl/verl/tools/wsi_zoom_in_tool.py
```

类名建议：

```text
WSIZoomInTool
```

工具 schema 名称：

```text
wsi_zoom_in_tool
```

参数：

```json
{
  "bbox_2d": [x1, y1, x2, y2],
  "label": "optional"
}
```

`bbox_2d` 使用 GIANT 风格的 global-relative 0..1000 坐标系。`label` 只用于可读日志，不参与 crop。

### DICOM 打开策略

tool 在 `create()` 中接收：

```python
create_kwargs = {
    "series_dir": ".../wsi/<file_id>",
    "dicom_anchor_path": ".../*.dcm",
    "file_id": "..."
}
```

如果 `dicom_anchor_path` 缺失，则从 `series_dir` 中按文件名排序选择第一个 `.dcm`。OpenSlide 4+ 打开 anchor 后会发现同 series sibling files。

该假设必须在实现前通过 smoke test 确认：

```python
slide = openslide.OpenSlide(dicom_anchor_path)
assert slide.level_count > 0
assert slide.level_dimensions
thumbnail = slide.get_thumbnail((1024, 1024))
```

如果目标环境无法通过单个 anchor 读取 DICOM series，则第一版实现必须先停下修环境或改 DICOM reader 策略，不能静默 fallback 到 SVS。

### Crop 逻辑

tool 在 `execute()` 中：

1. 校验 `bbox_2d` 是 4 个数。
2. 校验坐标范围在 `0..1000`。
3. 校验 `x1 < x2` 且 `y1 < y2`。
4. 读取 slide width/height。
5. 将 relative bbox 转成 level-0 region：

```text
x1_abs = round(x1 / 1000 * slide_width)
y1_abs = round(y1 / 1000 * slide_height)
x2_abs = round(x2 / 1000 * slide_width)
y2_abs = round(y2 / 1000 * slide_height)
```

6. 使用 OpenSlide read_region 或等价封装读取 crop。
7. 将 crop resize 到适合 Qwen3-VL 的尺寸范围。
8. 返回：

```python
ToolResponse(
    text="Here is the zoomed WSI crop:",
    image=[cropped_image],
)
```

### 错误与 tool reward

模型责任错误返回负 tool reward，例如 `-0.05`：

1. bbox JSON 格式不合法。
2. bbox 缺失。
3. bbox 坐标越界。
4. bbox 面积为 0 或太小。

环境或数据错误返回 text error，并在 `tool_stats.error_type` 中标记为 execution error；这类错误不应被当作模型格式错误惩罚，但应在日志中明显暴露：

1. DICOM series 不存在。
2. OpenSlide 无法打开。
3. 读 region 失败。

tool 应尽量把 DICOM 打开和 crop 读取放入可控执行块中，并提供 `timeout_seconds` 配置。若读取超时或文件损坏，tool 应返回 execution error，而不是让 rollout worker 长时间 hang。

现有 `format_score` 会根据负 tool reward 扣掉格式分，因此无需额外把 tool reward 加到最终 reward。

这一点依赖 `verl/verl/utils/reward_score/tool.py` 中的 `_has_model_caused_tool_error()` 和 `compute_format_score()`：负 `tool_rewards` 会被视为模型造成的 tool 参数错误，从而使 `format_score` 返回 0。实现时需要保持 `ToolAgentLoop` 把 `tool_rewards` 和 `tool_stats` 写入 `extra_fields/non_tensor_batch`。

## Reward 设计

继续使用：

```text
verl/verl/workers/reward_manager/tool.py
```

复用已有 reward function：

```text
em_score_mcq
```

数据转换阶段负责把 CSV 中的 1-based 数字答案转换成字母答案：

```text
1 -> A
2 -> B
3 -> C
4 -> D
```

reward 逻辑继续使用现有 MCQ 抽取逻辑：

1. 取最后一个 assistant turn。
2. 去掉 thinking 结束标签之前内容。
3. 优先从最后一个 `<answer>...</answer>` 中抽答案。
4. 从答案中抽 `A-D` 选项字母。
5. 与 `reward_model.ground_truth` 做 EM 比较。

示例：

```text
ground_truth = "C"
<answer>C</answer>              -> 1
<answer>Option C</answer>       -> 1
<answer>C. Gleason pattern</answer> -> 1
<answer>B</answer>              -> 0
<answer>The answer is option C, not D</answer> -> 1
missing answer tag              -> 0
```

最终训练 reward：

```text
score = em_score_mcq + format_score
```

其中 `format_score` 默认 `0.5`。第一版最大分数为 `1.5`。

## 配置设计

新增 tool config：

```text
config/tool_config/tools_wsi_train.yaml
config/tool_config/tools_wsi_val.yaml
```

仅包含 `wsi_zoom_in_tool`，避免 text/image search tool 干扰第一版训练。

新增训练脚本或配置：

```text
train_wsi_tcga_slidebench.sh
train_wsi_tcga_slidebench.json
```

关键配置：

```text
data.custom_cls.path="pkg://verl.utils.dataset.rl_dataset_json_v2"
data.custom_cls.name="RLHFJSONDatasetV2"
data.return_raw_chat=True
data.tool_config_path=config/tool_config/tools_wsi_train.yaml
actor_rollout_ref.rollout.multi_turn.tool_config_path=config/tool_config/tools_wsi_train.yaml
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
reward_model.reward_manager=tool
reward_model.reward_kwargs.format_score=0.5
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
```

小规模闭环建议先使用较低并发和较小 batch，确认 DICOM IO、rollout 和 reward 都能稳定跑通后再扩大。

## 测试与验证

### 数据转换验证

检查：

1. 总数为 197。
2. train 为 160，val 为 37。
3. 每条样本 `reward_model.ground_truth` 是有效选项字母。
4. 每条样本的 `series_dir` 存在。
5. 每条 prompt 包含所有 options。

### Tool smoke test

对 3 条固定样本测试：

1. 能选择 DICOM anchor。
2. 能打开 DICOM WSI。
3. 能生成 initial thumbnail。
4. 能确认 `level_count > 0` 且 `level_dimensions` 非空。
5. 对 `[0, 0, 1000, 1000]` 能返回整图近似 crop 或有效 crop。
6. 对一个中间 bbox 能返回 PIL image。
7. 非法 bbox 返回负 reward。
8. 损坏或不存在的 `series_dir` 返回 execution error，不导致进程崩溃。

### AgentLoop smoke test

构造 1 条样本，使用 mock 或短 rollout：

1. `multi_modal_data.wsi` 能触发 runtime thumbnail。
2. `wsi_zoom_in_tool` 的 create kwargs 能拿到 WSI metadata。
3. tool 返回 image 后，tool observation token 的 `response_mask` 为 0。
4. final answer token 的 reward 能落到最后有效 response token。

### RL smoke train

先用小规模参数跑通：

```text
train subset: 16 或 32
val subset: 8
rollout.n: 2 或 4
max_turns: 2 到 3
```

通过标准：

1. rollout 不因 DICOM IO 崩溃。
2. 至少出现成功 tool call。
3. reward extra info 中有 `em_score_mcq` 和 `format_score`。
4. val 能输出 accuracy-like 指标。
5. 训练能完成至少一个 update step。
6. 当某条样本发生 DICOM execution error 时，错误能被记录到 `tool_stats`，训练脚本能明确失败或跳过，而不是无日志挂起。

## 风险与缓解

### Runtime thumbnail 与 dataset 预计算 prompt ids 冲突

风险：dataset 阶段没有真实 thumbnail，预计算的 `input_ids` 与 runtime image token 不一致。

缓解：WSI 样本在 AgentLoop 中重新构造 prompt ids，不复用 dataset `input_ids`。

### DICOM IO 慢

风险：rollout 多并发打开 DICOM series，导致 IO 压力大。

缓解：

1. 第一版降低并发。
2. tool 内可加轻量 anchor/path cache。
3. 后续再考虑 thumbnail cache，但第一版不预生成。

### OpenSlide 版本兼容

风险：远端环境 OpenSlide 版本不足，无法读取 DICOM WSI。

缓解：先写 smoke test 检查 OpenSlide 能否打开 anchor `.dcm`。如果失败，优先修环境，而不是改训练逻辑。

### 坐标语义偏差

风险：模型使用 0..1000 global-relative bbox，但 crop 逻辑如果按当前 crop-relative 理解，会偏离 GIANT 协议。

缓解：第一版明确所有 bbox 都是 whole-slide global-relative 坐标，不是当前 crop-relative 坐标。

## 实施顺序

1. 在 workspace 数据目录中准备 MultiPathQA 转换脚本与 meta 配置。
2. 新增 `WSIZoomInTool` 和 WSI thumbnail/crop helper。
3. 扩展 `ToolAgentLoop` lazy loading 支持 `multi_modal_data.wsi`。
4. 复用 `em_score_mcq`，转换脚本负责数字答案到字母答案的规范化。
5. 新增 WSI tool config 与训练脚本。
6. 跑数据转换验证。
7. 跑 tool smoke test。
8. 跑 1 条样本 AgentLoop smoke test。
9. 跑小规模 RL smoke train。
