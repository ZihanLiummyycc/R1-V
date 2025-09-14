#!/usr/bin/env bash
set -Eeuo pipefail

cd /root/R1-V
export PYTHONPATH=./

# 激活 conda 环境
source /root/anaconda3/etc/profile.d/conda.sh
conda activate zihan-env

# 环境变量
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 启动训练
accelerate launch \
  --num_processes=1 \
  --num_machines=1 \
  --mixed_precision=bf16 \
  /root/R1-V/src/r1-v/src/open_r1/sft_normal.py \
  --model_name_or_path /root/models/my-model \
  --dataset_name /root/R1-V/datasets/sports_train_data_small \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing True \
  --bf16 True \
  --logging_steps 50 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --save_strategy steps \
  --save_steps 1000 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
  --greater_is_better False \
  --save_total_limit 2 \
  --output_dir /root/outputs/sft_sports_train_data_small \
  --report_to tensorboard
