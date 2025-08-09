#!/bin/bash

# Make sure script exits on any error
set -e
# Path to your Python training script
SCRIPT=$1
SEEDS=(0 1)
# Seeds to run with
MODEL_PATHS=(
  "./logs/ppo_muscle_arm_only_seed_0_2025-07-29_22-33-26.zip"
  "./logs/ppo_muscle_arm_only_seed_1_2025-07-30_06-59-24.zip"
  # "./logs/ppo_muscle_arm_only_seed_4_2025-07-30_11-30-12.zip"
  # "./logs/ppo_muscle_arm_only_seed_5_2025-07-30_11-30-12.zip"
  # "./assets/models/fly_new/best_combined_arm_damping_stiff_cvt3"
)

# Loop over seeds and model paths
# enumerate over the array SEEDS

for i in "${!SEEDS[@]}"; do
  SEED=${SEEDS[$i]}
  MODEL_PATH=${MODEL_PATHS[$i]}
  LOG_PATH=${LOG_PATHS[$i]}
  echo "Running with seed: $SEED and model path: $MODEL_PATH"
  # Run the script with the seed and model path as arguments
  python "$SCRIPT" \
  --seed "$SEED" \
  --tot_ts 20000000 \
  --xml_path "./assets/models/fly_new/best_combined_arm_cvt3" \
  --exp "arm_only" \
  --load_model "$MODEL_PATH" &
done

wait

echo "All seeds are done"
