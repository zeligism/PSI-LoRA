#!/usr/bin/env bash
# Ablation: Scaled OPLoRA metric-specific parameters
# TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=5 bash scripts/run_ablation_roberta_oplora_scaled_metric.sh

set -euo pipefail

# Experiment configuration
EXPERIMENT="ablation_oplora_scaled_metric"
RUN_CMD="python src/run_glue.py"
EXTRA_ARGS="$*"

# Default scaled OPLoRA params
OPLORA_ITERS=0
OPLORA_LMBD=1e-2
OPLORA_BETA1=0.9
OPLORA_BETA2=0.99
LR_FOR_NON_LORA=1e-4

# Ablation sweeps
LRS=(0.1 1.0 10.0)
KFAC_METRICS=(true false)
METRIC_POWERS=(0.25 0.5)
OPLORA_DAMPINGS=(1e-3 1e-1 10)
SEEDS=(0)


for seed in "${SEEDS[@]}"; do
  for lr in "${LRS[@]}"; do
    for kfac in "${KFAC_METRICS[@]}"; do
      for METRIC_POWER in "${METRIC_POWERS[@]}"; do
        for OPLORA_DAMPING in "${OPLORA_DAMPINGS[@]}"; do
          echo "========================================"
          run_id="seed=${seed}_lr=${lr}_kfac=${kfac}_damping=${OPLORA_DAMPING}_metricpow=${METRIC_POWER}"
          echo "Running run: ${run_id}"

          output_dir="output/${EXPERIMENT}/${run_id}"

          # A quick check to skip already completed runs
          if [[ -s "${output_dir}/all_results.json" ]]; then
            echo "Skipping completed run: ${run_id}"
            echo "----------------------------------------"
            continue
          fi

          # removing --overwrite_output_dir to continue from last checkpoint
          cmd=(
            ${RUN_CMD}
            --model_name_or_path "roberta-base"
            --task_name "mnli"
            --target_modules "query, value"
            --do_train
            --per_device_train_batch_size 32
            --per_device_eval_batch_size 128
            --max_seq_length 128
            --eval_steps 500
            --save_steps 500
            --logging_steps 10
            --num_train_epochs 3
            --bf16
            --lr_scheduler_type "linear"
            --warmup_ratio 0.00
            --eval_strategy steps
            --save_strategy steps
            --report_to tensorboard
            --ignore_mismatched_sizes
            --save_total_limit 1
            --optim "sgd"
            --learning_rate "${lr}"
            --weight_decay 0.0
            --max_grad_norm 1.0
            --use_lora
            --lora_rank 8
            --lora_alpha 8
            --use_oplora
            --use_oplora_scaled
            --oplora_lookahead_iters "${OPLORA_ITERS}"
            --oplora_lmbd "${OPLORA_LMBD}"
            --oplora_damping "${OPLORA_DAMPING}"
            --oplora_beta1 "${OPLORA_BETA1}"
            --oplora_beta2 "${OPLORA_BETA2}"
            --oplora_metric_power "${METRIC_POWER}"
            --learning_rate_for_non_lora "${LR_FOR_NON_LORA}"
            --seed "${seed}"
            --output_dir "${output_dir}"
            --logging_dir "${output_dir}/logs/"
          )

          if [[ "${kfac}" == "true" ]]; then
            cmd+=(--oplora_kfac_metric)
          fi
          if [[ -n "${EXTRA_ARGS}" ]]; then
            cmd+=(${EXTRA_ARGS})
          fi

          echo "+ ${cmd[@]}"
          "${cmd[@]}"
          echo "Finished run: ${run_id}"
          echo "----------------------------------------"
        done
      done
    done
  done
done
