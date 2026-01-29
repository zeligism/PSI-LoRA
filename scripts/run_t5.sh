#!/bin/bash
# Runs T5 on SQuAD v2.

# Environment setup
export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=0

source "scripts/gpu_count.sh"
RUNNER=(torchrun --standalone --nproc_per_node "$(gpu_count)")

# Experiment configuration
model=t5
task=squadv2
optim=adamw  # sgd or adamw
lora=lora  # lora, oplora, oplora_proj, oplora_fista, oplora_scaled, precond_lora, svdlora, svdlora_fista, or full
num_train_epochs=2

# Optim

lr=1e-4  # SGD (~5e-3), AdamW (~1e-5)
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
# lr=0.05
oplora_damping=1e-5
oplora_beta1=0.9
oplora_beta2=0.9
oplora_kfac_metric=true
oplora_metric_power=0.5
lr_for_non_lora=1e-3

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
  "${RUNNER[@]}" src/run_seq2seq_qa.py \
    --model_name_or_path google-t5/t5-small \
    --dataset_name squad_v2 \
    --target_modules "q,v" \
    --context_column context \
    --question_column question \
    --answer_column answers \
    --do_train \
    --do_eval \
    --lr_scheduler_type "linear" \
    --evaluation_strategy steps \
    --save_strategy steps \
    --eval_steps 2000 \
    --save_steps 4000 \
    --logging_steps 50 \
    --per_device_train_batch_size 12 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --num_train_epochs $num_train_epochs \
    --warmup_ratio 0.03 \
    --bf16 \
    --report_to tensorboard \
    --version_2_with_negative \
    --predict_with_generate \
    --generation_max_length 16 \
    --generation_num_beams 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_f1 \
    --greater_is_better True \
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

# Quick skip if results already exist
if [[ -s "${output_dir}/all_results.json" ]]; then
  echo "Run already completed. Skipping: ${output_dir}"
  exit 0
fi

# Execute command
echo ""
echo "Running: $output_dir"
echo "+ ${cmd[@]}"
"${cmd[@]}"
