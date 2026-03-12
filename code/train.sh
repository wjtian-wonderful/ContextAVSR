#!/bin/bash  # Add shebang to specify shell interpreter  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
export PYTHONWARNINGS="ignore::DeprecationWarning:librosa.core.audio"
export PYTHONWARNINGS="ignore::FutureWarning:librosa.core.audio"
export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTHONWARNINGS="ignore"

export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
export MPI_NUM_THREADS=8
VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-""}
# 计算 GPU 数量（通过逗号分割后的元素个数）
if [ -z "$VISIBLE_DEVICES" ]; then
    # 如果未设置，默认使用所有可用 GPU（通过 nvidia-smi 获取）
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
else
    # 按逗号分割并统计数量
    GPU_COUNT=$(echo "$VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi


export NPROC_PER_NODE=${GPU_COUNT}
export FPS_MAX_FRAMES=40
export MASTER_PORT=7845
export ENABLE_AUDIO_OUTPUT=0
export USE_AUDIO_IN_VIDEO=True
export ENABLE_VIDEO_OUTPUT=0
export VIDEO_MAX_PIXELS=50176
export FORCE_QWENVL_VIDEO_READER=torchvision
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' 
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

python -m torch.distributed.launch \
    -m swift.cli.sft \
    --model "Qwen/Qwen2.5-Omni-7B" \
    --dataset   "../data/test_train.jsonl" \
    --torch_dtype bfloat16 \
    --val_dataset "../data/test_val.jsonl" \
    --train_type lora   \
    --lora_rank 8   \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --learning_rate 1e-4  \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --gradient_accumulation_steps 2 \
    --output_dir lm_output/exp1 