#!/usr/bin/env bash
# Train on linear dataset with a full-batch gradient and no momentum

set -euo pipefail

# Experiment configuration
TASK=linear
EXPERIMENT="compare_momentum_rank_linear_stochastic"
RUN_CMD="python -m src.run batch_size=64 momentum=0.75"
EXTRA_ARGS="$*"

# Baseline method arguments
SVDLORA_ARGS="method=svdlora optimizer=sgd lr=0.1"
OPLORA_1_RANKMULT_1_ARGS="method=lora optimizer=oplora oplora_iters=1 oplora_rank_mult=1 lr=0.1"
OPLORA_1_RANKMULT_2_ARGS="method=lora optimizer=oplora oplora_iters=1 oplora_rank_mult=2 lr=0.1"
OPLORA_1_RANKMULT_4_ARGS="method=lora optimizer=oplora oplora_iters=1 oplora_rank_mult=4 lr=0.1"
OPLORA_2_RANKMULT_1_ARGS="method=lora optimizer=oplora oplora_iters=2 oplora_rank_mult=1 lr=0.1"
OPLORA_2_RANKMULT_2_ARGS="method=lora optimizer=oplora oplora_iters=2 oplora_rank_mult=2 lr=0.1"
OPLORA_2_RANKMULT_4_ARGS="method=lora optimizer=oplora oplora_iters=2 oplora_rank_mult=4 lr=0.1"
OPLORA_8_RANKMULT_1_ARGS="method=lora optimizer=oplora oplora_iters=8 oplora_rank_mult=1 lr=0.1"
OPLORA_8_RANKMULT_2_ARGS="method=lora optimizer=oplora oplora_iters=8 oplora_rank_mult=2 lr=0.1"
OPLORA_8_RANKMULT_4_ARGS="method=lora optimizer=oplora oplora_iters=8 oplora_rank_mult=4 lr=0.1"

# Loops
SEEDS=(0 1 2 3 4)
METHODS=(
  "${SVDLORA_ARGS}"
  "${OPLORA_1_RANKMULT_1_ARGS}"
  "${OPLORA_1_RANKMULT_2_ARGS}"
  "${OPLORA_1_RANKMULT_4_ARGS}"
  "${OPLORA_2_RANKMULT_1_ARGS}"
  "${OPLORA_2_RANKMULT_2_ARGS}"
  "${OPLORA_2_RANKMULT_4_ARGS}"
  "${OPLORA_8_RANKMULT_1_ARGS}"
  "${OPLORA_8_RANKMULT_2_ARGS}"
  "${OPLORA_8_RANKMULT_4_ARGS}"
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