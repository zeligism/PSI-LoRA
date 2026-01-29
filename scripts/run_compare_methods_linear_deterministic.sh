#!/usr/bin/env bash
# Train on linear dataset with a full-batch gradient and no momentum

set -euo pipefail

# Experiment configuration
TASK=linear
EXPERIMENT="compare_methods_linear_deterministic"
RUN_CMD="python -m src.run batch_size=10000 momentum=0.0"
EXTRA_ARGS="$*"

# Baseline method arguments
SVDLORA_ARGS="method=svdlora optimizer=sgd lr=0.5"
FULL_ARGS="method=full optimizer=sgd lr=0.5"
LORA_SGD_ARGS="method=lora optimizer=sgd lr=0.01"
LORA_ADAM_ARGS="method=lora optimizer=adamw lr=0.5"
PRECOND_LORA_SGD_ARGS="method=lora optimizer=sgd_precond_lora lr=0.48"  # 0.5 diverges
PRECOND_LORA_ADAM_ARGS="method=lora optimizer=adamw_precond_lora lr=0.5"
OPLORA_1_ARGS="method=lora optimizer=oplora oplora_iters=1 lr=0.5"
OPLORA_2_ARGS="method=lora optimizer=oplora oplora_iters=2 lr=0.5"
OPLORA_8_ARGS="method=lora optimizer=oplora oplora_iters=8 lr=0.5"
OPLORA_32_ARGS="method=lora optimizer=oplora oplora_iters=32 lr=0.5"
OPLORA_128_ARGS="method=lora optimizer=oplora oplora_iters=128 lr=0.5"

# Loops
SEEDS=(0 1 2 3 4)
METHODS=(
  "${SVDLORA_ARGS}"
  "${FULL_ARGS}"
  "${LORA_SGD_ARGS}"
  "${LORA_ADAM_ARGS}"
  "${PRECOND_LORA_SGD_ARGS}"
  "${PRECOND_LORA_ADAM_ARGS}"
  "${OPLORA_1_ARGS}"
  "${OPLORA_2_ARGS}"
  "${OPLORA_8_ARGS}"
  "${OPLORA_32_ARGS}"
  "${OPLORA_128_ARGS}"
)

# Run experiments
for seed in "${SEEDS[@]}"; do
  for METHOD_ARGS in "${METHODS[@]}"; do
    echo "========================================"
    echo "Running seed=${seed} with method args: ${METHOD_ARGS}"
    CMD="${RUN_CMD} seed=${seed} experiment=${EXPERIMENT} task=${TASK} ${METHOD_ARGS} ${EXTRA_ARGS} continue_training=true"
    echo "+ ${CMD}"
    eval "${CMD}"
    echo "Finished seed=${seed} with method args: ${METHOD_ARGS}"
    echo "----------------------------------------"
  done
done
