from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def process_event_files(log_dir, save=False, plot=True):
    """
    Process TensorBoard event files to extract evaluation metrics.
    Args:
        log_dir (str): Directory containing TensorBoard event files.
        save (bool): If True, save the processed data to a .npz file.
        plot (bool): If True, plot the evaluation metrics.
    Returns:
        steps (list): List of training steps.
        mean_reward (list): List of mean rewards.
        mean_ep_len (list): List of mean episode lengths.
    """
    # Load event files
    event_files = list(Path(log_dir).glob("events.out.tfevents.*"))

    if not event_files:
        raise FileNotFoundError(f"No event files found in {log_dir}")

    # Load both and concatenate
    event_files.sort()  # Ensure consistent order
    steps = []
    mean_reward = []
    mean_ep_len = []
    for file in event_files:
        file = str(file)
        print(f"Loading events from {file}")
        event_acc = EventAccumulator(file)
        event_acc.Reload()
        steps.extend([e.step for e in event_acc.Scalars("eval/mean_reward")])
        mean_reward.extend([e.value for e in event_acc.Scalars("eval/mean_reward")])
        mean_ep_len.extend([e.value for e in event_acc.Scalars("eval/mean_ep_length")])

    if save:
        np.savez(
            f"{log_dir}/eval_data.npz",
            steps=np.array(steps),
            mean_reward=np.array(mean_reward),
            mean_ep_len=np.array(mean_ep_len),
        )
        print(f"Data saved to {log_dir}/eval_data.npz")

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(steps, mean_reward, label="Mean Reward")
        plt.plot(steps, mean_ep_len, label="Mean Episode Length")
        plt.xlabel("Training Step")
        plt.ylabel("Mean Value")
        plt.title("Evaluation Metrics over Time: " + log_dir)
        plt.legend()
        plt.grid(True)
        plt.show()

    return steps, mean_reward, mean_ep_len


if __name__ == "__main__":
    log_dirs = [
        "../../logs/ppo_muscle_seed_arm_0_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_1_2025-08-09_23-28-29_1/",
        "../../logs/ppo_muscle_seed_arm_2_2025-08-09_23-28-29_1/",
        "../../logs/ppo_muscle_seed_arm_3_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_4_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_damping_0_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_damping_1_2025-08-09_23-28-29_1/",
        "../../logs/ppo_muscle_seed_arm_damping_2_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_damping_3_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_damping_4_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_damping_stiff_0_2025-08-09_23-28-29_1/",
        "../../logs/ppo_muscle_seed_arm_damping_stiff_1_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_damping_stiff_2_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_damping_stiff_3_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_damping_stiff_4_2025-08-09_23-28-29_1/",
        "../../logs/ppo_muscle_seed_arm_stiff_0_2025-08-09_23-28-31_1/",
        "../../logs/ppo_muscle_seed_arm_stiff_1_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_stiff_2_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_stiff_3_2025-08-09_23-28-30_1/",
        "../../logs/ppo_muscle_seed_arm_stiff_4_2025-08-09_23-28-30_1/",
    ]

    for log_dir in log_dirs:
        print(f"Processing log directory: {log_dir}")
        try:
            process_event_files(log_dir, save=True, plot=True)
        except Exception as e:
            print(f"Error processing {log_dir}: {e}")
