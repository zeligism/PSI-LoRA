#!/bin/bash
# Runs RoBERTa on GLUE tasks.
# Try to change only optim, lora, and lr.

# Environment setup
export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=5

# Experiment configuration
model=roberta
task=cola    # mnli, mrpc, cola, rte, sts-b, etc.
optim=sgd  # sgd or adamw
lora=oplora_scaled  # lora, oplora, oplora_proj, oplora_fista, oplora_scaled, precond_lora, svdlora, svdlora_fista, or full
num_train_epochs=3

# Optim
lr=1e-5  # SGD (~5e-3), AdamW (~1e-5)
momentum=0.9
adam_beta1=0.9
adam_beta2=0.999
adam_epsilon=1e-8
weight_decay=0.0
max_grad_norm=1.0

# LoRA
lora_rank=8
# OPLoRA params
oplora_iters=0
oplora_lmbd=1e-2
# Scaled OPLoRA params
lr=0.5
oplora_damping=1e-5
oplora_beta1=0.9
oplora_beta2=0.99
oplora_kfac_metric=true
oplora_metric_power=0.5
lr_for_non_lora=1e-4

# SVDLoRA + AdamW
# optim=adamw
# lora=svdlora
# lr=1e-4

# PrecondLoRA + SGD
# optim=sgd
# lora=precond_lora
# lr=5.0001e-3

# Default command
cmd=(
  python src/run_glue.py \
    --model_name_or_path roberta-base \
    --task_name $task \
    --target_modules "query, value" \
    --do_train \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --max_seq_length 128 \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 10 \
    --num_train_epochs $num_train_epochs \
    --bf16 \
    --lr_scheduler_type "linear" \
    --warmup_ratio 0.00 \
    --eval_strategy steps \
    --save_strategy steps \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ignore_mismatched_sizes \
    --save_total_limit 1
)

# ================ OPTIM ================
if [[ $optim == "sgd" ]]; then
  cmd+=(
    --optim sgd \
    --learning_rate $lr \
    --momentum $momentum \
    --weight_decay $weight_decay \
    --max_grad_norm $max_grad_norm
  )
  optim_id="sgd(lr=${lr}_momentum=${momentum}_wd=${weight_decay}_clip=${max_grad_norm})"

elif [[ $optim == "adamw" ]]; then
  cmd+=(
    --optim adamw_torch \
    --learning_rate $lr \
    --adam_beta1 $adam_beta1 \
    --adam_beta2 $adam_beta2 \
    --adam_epsilon $adam_epsilon \
    --weight_decay $weight_decay \
    --max_grad_norm $max_grad_norm
  )
  optim_id="adamw(lr=${lr}_betas=${adam_beta1},${adam_beta2}_eps=${adam_epsilon}_wd=${weight_decay}_clip=${max_grad_norm})"
fi

# ================ LORA ================
if [[ $lora == "lora" ]]; then
  cmd+=(
    --use_lora \
    --lora_rank $lora_rank \
    --lora_alpha $lora_rank
  )
  lora_id="lora(rank=${lora_rank})"

elif [[ $lora == "svdlora" ]]; then
  cmd+=(
    --use_lora \
    --lora_rank $lora_rank \
    --lora_alpha $lora_rank \
    --use_svdlora
  )
  lora_id="svdlora(rank=${lora_rank})"

elif [[ $lora == "svdlora_fista" ]]; then
  cmd+=(
    --use_lora \
    --lora_rank $lora_rank \
    --lora_alpha $lora_rank \
    --use_svdlora \
    --use_svdlora_fista
  )
  lora_id="svdlora_fista(rank=${lora_rank})"

elif [[ $lora == "oplora" ]]; then
  cmd+=(
    --use_lora \
    --lora_rank $lora_rank \
    --lora_alpha $lora_rank \
    --use_oplora \
    --oplora_lookahead_iters $oplora_iters \
    --oplora_lmbd $oplora_lmbd
  )
  lora_id="oplora(rank=${lora_rank}_iters=${oplora_iters}_lmbd=${oplora_lmbd})"

elif [[ $lora == "oplora_proj" ]]; then
  cmd+=(
    --use_lora \
    --lora_rank $lora_rank \
    --lora_alpha $lora_rank \
    --use_oplora \
    --use_oplora_proj \
    --oplora_lookahead_iters $oplora_iters \
    --oplora_lmbd $oplora_lmbd
  )
  lora_id="oplora_proj(rank=${lora_rank}_iters=${oplora_iters}_lmbd=${oplora_lmbd})"

elif [[ $lora == "oplora_fista" ]]; then
  cmd+=(
    --use_lora \
    --lora_rank $lora_rank \
    --lora_alpha $lora_rank \
    --use_oplora \
    --use_oplora_fista \
    --oplora_lookahead_iters $oplora_iters \
    --oplora_lmbd $oplora_lmbd
  )
  lora_id="oplora_fista(rank=${lora_rank}_iters=${oplora_iters}_lmbd=${oplora_lmbd})"

elif [[ $lora == "oplora_scaled" ]]; then
  cmd+=(
    --use_lora \
    --lora_rank $lora_rank \
    --lora_alpha $lora_rank \
    --use_oplora \
    --use_oplora_scaled \
    --oplora_lookahead_iters $oplora_iters \
    --oplora_lmbd $oplora_lmbd \
    --oplora_damping $oplora_damping \
    --oplora_beta1 $oplora_beta1 \
    --oplora_beta2 $oplora_beta2 \
    --oplora_metric_power $oplora_metric_power \
    --learning_rate_for_non_lora $lr_for_non_lora
  )
  if [[ $oplora_kfac_metric == "true" ]]; then
    cmd+=(--oplora_kfac_metric)
  fi
  lora_id="oplora_scaled(rank=${lora_rank}_iters=${oplora_iters}_lmbd=${oplora_lmbd}_damping=${oplora_damping}_beta1=${oplora_beta1}_beta2=${oplora_beta2}_metricpow=${oplora_metric_power}_kfac=${oplora_kfac_metric}_lr_non_lora=${lr_for_non_lora})"

elif [[ $lora == "precond_lora" ]]; then
  cmd+=(
    --use_lora \
    --lora_rank $lora_rank \
    --lora_alpha $lora_rank \
    --use_precond_lora \
    --oplora_lmbd $oplora_lmbd
  )
  lora_id="precond_lora(rank=${lora_rank}_lmbd=${oplora_lmbd})"

elif [[ $lora == "full" ]]; then
  lora_id=full

else
  echo "Unknown LoRA method: $lora"
  exit 1
fi

# Create output dir and add output dir args
exp_name="${model}_${task}_${optim}_${lora}"
method_id="${optim_id}_${lora_id}"
output_dir="output/${exp_name}/${method_id}"
cmd+=(--output_dir "${output_dir}" --logging_dir "${output_dir}/logs/")

# # Quick skip if results already exist
# if [[ -s "${output_dir}/all_results.json" ]]; then
#   echo "Run already completed. Skipping: ${output_dir}"
#   exit 0
# fi

# Execute command
echo ""
echo "Running: $output_dir"
echo "+ ${cmd[@]}"
"${cmd[@]}"
