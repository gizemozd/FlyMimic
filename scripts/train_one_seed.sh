#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <seed> [mode]"
    echo "  mode: muscle (default) or torque"
    exit 1
fi

SEED=$1
MODE="${2:-muscle}"

CONFIGS=(
  train_arm
  train_arm_damp
  train_arm_damp_stiff
  train_arm_stiff
)

if [ "$MODE" = "torque" ]; then
  TRAIN_SCRIPT="train_torque.py"
else
  TRAIN_SCRIPT="train_muscle.py"
fi

for CONFIG in "${CONFIGS[@]}"; do
  python $TRAIN_SCRIPT seed=$SEED --config-path ../flymimic/config --config-name $CONFIG &
done

wait
