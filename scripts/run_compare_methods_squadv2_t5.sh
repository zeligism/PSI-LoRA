#!/usr/bin/env bash
# Runs coarse hyperparameter tuning for baselines on SQuADv2 with T5.

set -euo pipefail

# Experiment configuration
EXPERIMENT="compare_methods_squadv2_t5"
RUN_CMD="python src/run_seq2seq_qa.py"
EXTRA_ARGS="$*"

SEEDS=(0 1 2)
# METHODS=("lora" "full" "svdlora" "oplora" "oplora_proj" "oplora_scaled" "precond_lora")
METHODS=("lora" "full" "oplora_scaled")
LORA_RANKS=(8)

# - no saving models, an interrupted run has to restart from scratch
cmd=(
  ${RUN_CMD} \
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
    --save_strategy no \
    --eval_steps 2000 \
    --logging_steps 50 \
    --per_device_train_batch_size 12 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --num_train_epochs 2 \
    --warmup_ratio 0.03 \
    --bf16 \
    --report_to tensorboard \
    --version_2_with_negative \
    --predict_with_generate \
    --generation_max_length 16 \
    --generation_num_beams 1 \
    --save_total_limit 1
)

shuffle_array() {
  printf '%s\n' "$@" | shuf
}

mapfile -t SEEDS_SHUFFLED < <(shuffle_array "${SEEDS[@]}")
mapfile -t METHODS_SHUFFLED < <(shuffle_array "${METHODS[@]}")
mapfile -t LORA_RANKS_SHUFFLED < <(shuffle_array "${LORA_RANKS[@]}")

for seed in "${SEEDS_SHUFFLED[@]}"; do

    # Method
    sgd_args=(
      --optim sgd \
      --momentum 0.9
    )
    adam_args=(
      --optim adamw_torch \
      --adam_beta1 0.9 \
      --adam_beta2 0.99 \
      --adam_epsilon 1e-8
    )

  for lora_rank in "${LORA_RANKS_SHUFFLED[@]}"; do
    for method in "${METHODS_SHUFFLED[@]}"; do
      # Set method-specific hyperparameters
      case $method in
        "lora")
          lrs=(2e-4 5e-4 1e-3)
          method_args=(
            --use_lora \
            --lora_rank "$lora_rank" \
            --lora_alpha "$lora_rank" \
            "${adam_args[@]}"
          )
          ;;
        "full")
          lrs=(2e-5 5e-5 1e-4)
          method_args=(
            "${adam_args[@]}"
          )
          ;;
        "svdlora")
          lrs=(1e-4)
          method_args=(
            --use_lora \
            --lora_rank "$lora_rank" \
            --lora_alpha "$lora_rank" \
            --use_svdlora \
            "${adam_args[@]}"
          )
          ;;
        "precond_lora")
          lrs=(2e-3)
          method_args=(
            "${sgd_args[@]}"
            --use_lora \
            --lora_rank "$lora_rank" \
            --lora_alpha "$lora_rank" \
            --use_precond_lora \
            --oplora_lmbd 1e-2
          )
          ;;
        "oplora")
          lrs=()  # XXX
          method_args=(
            "${sgd_args[@]}"
            --use_lora \
            --lora_rank "$lora_rank" \
            --lora_alpha "$lora_rank" \
            --use_oplora \
            --oplora_lookahead_iters "0" \
            --oplora_lmbd "1e-2" \
            --learning_rate_for_non_lora "5e-3"
          )
          ;;
        "oplora_proj")
          lrs=(5e-3)
          method_args=(
            "${sgd_args[@]}"
            --use_lora \
            --lora_rank "$lora_rank" \
            --lora_alpha "$lora_rank" \
            --use_oplora \
            --use_oplora_proj \
            --oplora_lookahead_iters "0" \
            --oplora_lmbd "1e-2"
          )
          ;;
        "oplora_scaled")
          lrs=(0.2 0.5 1.0)
          method_args=(
            "${sgd_args[@]}"
            --use_lora \
            --lora_rank "$lora_rank" \
            --lora_alpha "$lora_rank" \
            --use_oplora \
            --use_oplora_scaled \
            --oplora_lookahead_iters "0" \
            --oplora_lmbd "1e-2" \
            --oplora_beta1 "0.9" \
            --oplora_beta2 "0.99" \
            --oplora_damping "1e-5" \
            --oplora_kfac_metric "true" \
            --oplora_metric_power "0.5" \
            --learning_rate_for_non_lora "1e-4"
          )
          ;;
        *)
          echo "Unknown method: $method"
          exit 1
          ;;
      esac

      for lr in "${lrs[@]}"; do
        echo "========================================"
        run_id="seed=${seed},method=${method},rank=${lora_rank},lr=${lr}"

        output_dir="output/${EXPERIMENT}/${run_id}"
        if [[ -s "${output_dir}/all_results.json" ]]; then
          # echo "Skipping completed run: ${run_id}"
          # echo "----------------------------------------"
          continue
        fi
        if [[ -d "${output_dir}" ]]; then
          echo "Skipping non-empty directory: ${run_id}"
          echo "----------------------------------------"
          continue
        fi

        # Run command
        echo "Running run: ${run_id}"
        set +e  # don't exit on failure
        "${cmd[@]}" \
          --learning_rate "$lr" \
          --seed "$seed" \
          --output_dir "${output_dir}" \
          --logging_dir "${output_dir}/logs/" \
          "${method_args[@]}" \
          ${EXTRA_ARGS}
        run_status=$?
        set -e

        if [[ $run_status -eq 0 ]]; then
          echo "Completed run: ${run_id}"
        else
          echo "Run failed (continuing): ${run_id}"
        fi
        echo "----------------------------------------"

      done
    done
  done
done
