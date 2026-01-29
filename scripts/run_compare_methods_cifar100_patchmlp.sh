#!/usr/bin/env bash
# Train on CIFAR-100 dataset with PatchMLP

set -euo pipefail

# Experiment configuration
TASK=cifar100_patchmlp
EXPERIMENT="compare_methods_cifar100_patchmlp"
RUN_CMD="python -m src.run"
EXTRA_ARGS="$*"

# Baseline method arguments
SVDLORA_ARGS="method=svdlora optimizer=sgd lr=0.005"
FULL_SGD_ARGS="method=full optimizer=sgd lr=0.005"
FULL_ADAM_ARGS="method=full optimizer=adamw lr=0.001"
LORA_SGD_ARGS="method=lora optimizer=sgd lr=0.01"
LORA_ADAM_ARGS="method=lora optimizer=adamw lr=0.001"
PRECOND_LORA_SGD_ARGS="method=lora optimizer=sgd_precond_lora lr=0.005"
PRECOND_LORA_ADAM_ARGS="method=lora optimizer=adamw_precond_lora lr=0.0005"
OPLORA_1_ARGS="method=lora optimizer=oplora oplora_iters=1 lr=0.01"
OPLORA_2_ARGS="method=lora optimizer=oplora oplora_iters=2 lr=0.01"
OPLORA_8_ARGS="method=lora optimizer=oplora oplora_iters=8 lr=0.01"
OPLORA_PROJ_1_ARGS="method=lora optimizer=oplora_proj oplora_iters=1 lr=0.01"
OPLORA_PROJ_2_ARGS="method=lora optimizer=oplora_proj oplora_iters=2 lr=0.01"
OPLORA_PROJ_8_ARGS="method=lora optimizer=oplora_proj oplora_iters=8 lr=0.01"

# Loops
SEEDS=(0 1 2)
METHODS=(
  "${SVDLORA_ARGS}"
  "${FULL_SGD_ARGS}"
  "${FULL_ADAM_ARGS}"
  "${LORA_SGD_ARGS}"
  "${LORA_ADAM_ARGS}"
  "${PRECOND_LORA_SGD_ARGS}"
  "${PRECOND_LORA_ADAM_ARGS}"
  "${OPLORA_1_ARGS}"
  "${OPLORA_2_ARGS}"
  "${OPLORA_8_ARGS}"
  "${OPLORA_PROJ_1_ARGS}"
  "${OPLORA_PROJ_2_ARGS}"
  "${OPLORA_PROJ_8_ARGS}"
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
