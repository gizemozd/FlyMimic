#!/bin/bash
set -e

SEEDS=(0)
CONFIGS=(
  train_arm
  train_arm_damp
  train_arm_damp_stiff
  train_arm_stiff
)

# Set to "muscle" or "torque" (default: muscle)
MODE="${1:-muscle}"

if [ "$MODE" = "torque" ]; then
  export TRAIN_SCRIPT="train_torque.py"
else
  export TRAIN_SCRIPT="train_muscle.py"
fi

# Build the list of commands
for SEED in "${SEEDS[@]}"; do
  for CONFIG in "${CONFIGS[@]}"; do
    python $TRAIN_SCRIPT seed=$SEED --config-path ../flymimic/config --config-name $CONFIG &
  done
done
