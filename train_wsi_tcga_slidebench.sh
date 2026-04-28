#!/bin/bash
# Small-scale WSI DICOM GRPO loop for TCGA SlideBench.

set -ex

cd "$(dirname "$0")"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
TRAIN_FILES=${TRAIN_FILES:-"$(pwd)/train_wsi_tcga_slidebench.json"}
VAL_FILES=${VAL_FILES:-"$(pwd)/test_wsi_tcga_slidebench.json"}
TOOL_CONFIG_TRAIN=${TOOL_CONFIG_TRAIN:-"$(pwd)/config/tool_config/tools_wsi_train.yaml"}
TOOL_CONFIG_VAL=${TOOL_CONFIG_VAL:-"$(pwd)/config/tool_config/tools_wsi_val.yaml"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-37}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-3}
AGENT_WORKERS=${AGENT_WORKERS:-8}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
NNODES=${NNODES:-1}
TP_SIZE=${TP_SIZE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-5}
EXP_NAME=${EXP_NAME:-wsi_tcga_slidebench_qwen3vl8b_smoke}
START_RAY=${START_RAY:-1}

export CODE_DIR="$(pwd)/verl"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export MAX_PIXELS=${MAX_PIXELS:-2097152}
export MIN_PIXELS=${MIN_PIXELS:-65536}
export SGL_ENABLE_JIT_DEEPGEMM=0

if [ "${PREPARE_DATA:-0}" = "1" ]; then
    python3 tools/prepare_wsi_tcga_slidebench.py \
        --csv-path "${MULTIPATHQA_CSV:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/WSI-Nav/gigapixel-goblin/data/multipathqa/MultiPathQA.csv}" \
        --dicom-root "${DICOM_ROOT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangquan/code/hutu/data/multipathqa_tcga_dicom}" \
        --output-root "${OUTPUT_ROOT:-data/wsi_tcga_slidebench}" \
        --manifest-dir "$(pwd)" \
        --train-size "${TRAIN_SIZE:-160}"
fi

if [ "$START_RAY" = "1" ]; then
    ray stop --force 2>/dev/null || true
    ray start --head --port="${MASTER_PORT:-6379}" --dashboard-host=0.0.0.0 --dashboard-port="${DASHBOARD_PORT:-8265}" --disable-usage-stats
fi

mkdir -p "rollout_data/$EXP_NAME/train" "rollout_data/$EXP_NAME/validation" logs

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.custom_cls.path="pkg://verl.utils.dataset.rl_dataset_json_v2" \
    data.custom_cls.name="RLHFJSONDatasetV2" \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.dataloader_num_workers=4 \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    data.image_key=image \
    data.image_patch_size=16 \
    data.tool_config_path="$TOOL_CONFIG_TRAIN" \
    +data.val_tool_config_path="$TOOL_CONFIG_VAL" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.freeze_vision_tower=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.5}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side="right" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_TRAIN" \
    +actor_rollout_ref.rollout.multi_turn.wsi_thumbnail_long_side="${WSI_THUMBNAIL_LONG_SIDE:-1024}" \
    +actor_rollout_ref.rollout.multi_turn.val_tool_config_path="$TOOL_CONFIG_VAL" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers="$AGENT_WORKERS" \
    reward_model.reward_manager=tool \
    reward_model.launch_reward_fn_async=True \
    +reward_model.reward_kwargs.log_num_round=True \
    +reward_model.reward_kwargs.format_score=0.5 \
    +reward_model.val_reward_kwargs.log_num_round=True \
    +reward_model.val_reward_kwargs.format_score=0.5 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.resume_mode="auto" \
    trainer.logger="${TRAINER_LOGGER:-['console']}" \
    trainer.project_name='wsi_nav_rl' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.rollout_data_dir="$(pwd)/rollout_data/$EXP_NAME/train" \
    trainer.validation_data_dir="$(pwd)/rollout_data/$EXP_NAME/validation" \
    trainer.total_epochs="$TOTAL_EPOCHS" 2>&1 | tee "logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
