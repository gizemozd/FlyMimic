#!/bin/bash

$HOME/miniconda3/bin/conda init bash
. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate flymimic

# Check if two arguments are given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_seed> <end_seed>"
    exit
fi

start_seed=$1
end_seed=$2

for seed in $(seq $start_seed $end_seed)
do
    # Submit a job for each seed
    sbatch --job-name=ppo_training_$seed \
           --output=ppo_training_${seed}_%j.out \
           --error=ppo_training_${seed}_%j.err \
           --partition=gpu \
           --ntasks=1 \
           --cpus-per-task=20 \
           --mem=50G \
           --time=5:00:00 \
           train_one_seed.sh $seed
done