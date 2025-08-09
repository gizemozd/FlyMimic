#!/bin/bash
set -e

SEEDS=(0)
CONFIGS=(
  train_arm
  train_arm_damp
  train_arm_damp_stiff
  train_arm_stiff
)

export TRAIN_SCRIPT="train_muscle.py"

# Build the list of commands
for SEED in "${SEEDS[@]}"; do
  for CONFIG in "${CONFIGS[@]}"; do
    python $TRAIN_SCRIPT seed=$SEED --config-path ../flymimic/config --config-name $CONFIG &
  done
done
