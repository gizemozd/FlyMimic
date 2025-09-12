import sys
import os
import numpy as np
from pathlib import Path

from flymimic.tasks.fly.mocap_tracking_muscle import mocap_tracking_muscle
from flymimic.envs.dmcontrol_wrapper import DMControlGymWrapper


def test_play_mode_rewards(
    clip="0002",
    model_name="best_combined_arm_cvt3",
    max_steps=20,
):
    """
    Test that play mode gives perfect rewards (1.0).

    Args:
        clip: Motion capture clip to use
        model_name: MuJoCo model name
        max_steps: Maximum steps to test

    Returns:
        dict: Test results including rewards and statistics
    """

    # Create environment in play mode
    dm_env = mocap_tracking_muscle(
        clip=clip,
        test=True,  # Start from beginning
        play=True,  # Enable play mode - robot follows mocap exactly
        model_name=model_name
    )

    env = DMControlGymWrapper(dm_env, seed=0)

    # Reset environment
    obs, _ = env.reset()

    rewards = []
    # Run steps and collect rewards
    for step in range(max_steps):
        action = np.zeros(env.action_space.shape[0])  # Zero action
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)


    # Analyze results
    # Velocity will not work since it is physics.forward()
    assert all([np.isclose(reward, 2/3, atol=1e-1) for reward in rewards])
