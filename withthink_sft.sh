export OUT_WITH_THINK="/root/outputs/sft_sports_with_think_lora_full"
export DATA_WITH_THINK="/ai/cunchu/root/R1-V/datasets/sports_train_data_with_think"

accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 \
  "$SFT" \
  --model_name_or_path "$MODEL" \
  --trust_remote_code True \
  --dataset_name "$DATA_WITH_THINK" \
  --dataset_train_split "$DS_TRAIN_SPLIT" \
  --dataset_test_split "$DS_TEST_SPLIT" \
  --use_peft True \
  --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --attn_implementation sdpa \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --gradient_checkpointing True \
  --bf16 True \
  --logging_steps 50 \
  --eval_strategy steps --eval_steps 1000 \
  --save_strategy steps --save_steps 1000 --save_total_limit 3 \
  --report_to tensorboard \
  --max_seq_length 4096 \
  --output_dir "$OUT_WITH_THINK"