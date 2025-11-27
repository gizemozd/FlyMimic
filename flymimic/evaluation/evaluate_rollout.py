# If on SSH unset DISPLAY; export MUJOCO_GL=egl
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import yaml
from stable_baselines3 import PPO

from flymimic.envs.dmcontrol_wrapper import DMControlGymWrapper
from flymimic.tasks.fly.mocap_tracking_muscle import mocap_tracking_muscle


@dataclass
class EvalConfig:
    model_path: Path
    xml_name: str
    video_path: Path
    save_path: Optional[Path] = None
    play: bool = False
    record_video: bool = True
    rollout_steps: int = 210
    camera_id: int = 0
    fps: int = 30
    render_every_n_steps: int = 1
    use_viewer: bool = False
    show_plot: bool = True
    figures_dir: Path = Path("figures")
    videos_dir: Path = Path("videos")
    data_dir: Path = Path("rollout_data")
    headless: bool = False


def _ensure_dirs(cfg: EvalConfig) -> None:
    cfg.videos_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_path:
        cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.video_path.parent.mkdir(parents=True, exist_ok=True)


def _select_actuator_names(physics) -> Tuple[List[str], int]:
    """Extract actuator names and count from physics model."""
    nu = physics.model.nu
    names = [physics.model.id2name(i, "actuator") for i in range(nu)]
    return names, nu


def _get_observation_names(dm_env) -> List[str]:
    """Get observation names from the DM environment."""
    observation_spec = dm_env.observation_spec()
    return [
        (obs_name, observation_spec[obs_name].shape)
        for obs_name in observation_spec.keys()
    ]


def _set_tendon_colors(physics, actuator_names: List[str], action: np.ndarray) -> None:
    """Color tendons by action value; ignore missing tendons gracefully."""
    for i, act in enumerate(action):
        red = float(np.clip(act, 0.0, 1.0))
        blue = 1.0 - red
        rgba = [red, 0.0, blue, 1.0]
        tendon_name = f"{actuator_names[i]}_tendon"
        try:
            physics.named.model.tendon_rgba[tendon_name] = rgba
        except (KeyError, IndexError):
            # Not all actuators must have a *_tendon entry
            print(f"Tendon {tendon_name} not found, skipping color update")


def plot_rollout(
    obs_arr: np.ndarray,
    act_arr: np.ndarray,
    rew_arr: np.ndarray,
    actuator_names: List[str],
    fig_path: Path,
    show: bool = True,
    joint_labels: Optional[List[str]] = None,
) -> None:
    """Make and save rollout plots with improved visualization."""
    if joint_labels is None:
        joint_labels = [
            "LFCoxa_yaw",
            "LFCoxa_pitch",
            "LFCoxa_roll",
            "LFTrochanter_yaw",
            "LFTrochanter_pitch",
            "LFTrochanter_roll",
            "LFTibia_pitch",
        ]

    fig, axs = plt.subplots(4, 1, figsize=(12, 13))
    fig.suptitle(f"Rollout Analysis - {fig_path.stem}", fontsize=16)

    # Heuristic: first half = angles, second half = velocities
    dims = len(joint_labels) * 2
    half = dims // 2

    # Angles
    for i in range(min(half, len(joint_labels))):
        axs[0].plot(obs_arr[:, i], label=joint_labels[i], linewidth=1.5)
    axs[0].set_title("Joint Angles (rad)")
    axs[0].set_xlabel("Time steps")
    axs[0].set_ylabel("Angle (rad)")
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axs[0].grid(True, alpha=0.3)

    # Velocities
    for i in range(half, dims):
        label = (
            joint_labels[i - half]
            if (i - half) < len(joint_labels)
            else f"vel_{i-half}"
        )
        axs[1].plot(obs_arr[:, i], label=label, linewidth=1.5)
    axs[1].set_title("Joint Velocities (rad/s)")
    axs[1].set_xlabel("Time steps")
    axs[1].set_ylabel("Velocity (rad/s)")
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axs[1].grid(True, alpha=0.3)

    # Actions
    cmap = plt.get_cmap("tab20")  # Better colormap for more colors
    for i in range(act_arr.shape[1]):
        label = actuator_names[i] if i < len(actuator_names) else f"act{i}"
        # Truncate long labels for readability
        display_label = label[:20] + "..." if len(label) > 20 else label
        axs[2].plot(
            act_arr[:, i], label=display_label, color=cmap(i % 20), linewidth=1.5
        )
    axs[2].set_title("Actions")
    axs[2].set_xlabel("Time steps")
    axs[2].set_ylabel("Action values")
    axs[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)
    axs[2].grid(True, alpha=0.3)

    # Rewards
    axs[3].plot(rew_arr, label="Reward", linewidth=2, alpha=0.8)
    axs[3].plot(
        np.cumsum(rew_arr),
        label="Cumulative Reward",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )
    axs[3].set_title("Reward")
    axs[3].set_xlabel("Time steps")
    axs[3].set_ylabel("Reward")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    # Add summary statistics as text
    total_reward = np.sum(rew_arr)
    mean_reward = np.mean(rew_arr)
    axs[3].text(
        0.02,
        0.98,
        f"Total: {total_reward:.2f}\nMean: {mean_reward:.4f}",
        transform=axs[3].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {fig_path}")
    except Exception as e:
        print(f"Failed to save plot to {fig_path}: {e}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_rollout(
    path: Path,
    actions: np.ndarray,
    action_key_names: List[str],
    observations: np.ndarray,
    observation_key_names: List[str],
    rewards: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save rollout data to pkl file with optional metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    action_dict = {name: actions[:, i] for i, name in enumerate(action_key_names)}
    i = 0
    observation_dict = {}

    for name, name_shape in observation_key_names:
        observation_dict[name] = observations[:, i : i + name_shape[0]]
        i += name_shape[0]

    # Prepare data to save
    save_data = {
        "actions": action_dict,
        "observations": observation_dict,
        "rewards": rewards,
    }

    # Add metadata if provided
    if metadata:
        save_data.update(metadata)

    try:
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Rollout data saved to {path}")
    except Exception as e:
        print(f"Failed to save rollout data to {path}: {e}")
        raise


def select_xml_name_from_model_path(model_path: Path) -> str:
    name = model_path.stem

    if "arm_damping_stiff" in name:
        return "best_combined_arm_damping_stiff_cvt3"
    if "arm_stiff" in name:
        return "best_combined_arm_stiff_cvt3"
    if "arm_damping" in name:
        return "best_combined_arm_damping_cvt3"
    return "best_combined_arm_cvt3"


def evaluate_rollout(cfg: EvalConfig) -> Dict[str, Any]:
    """Run rollout, optionally record video and viewer, return arrays & paths."""
    # Validate inputs
    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {cfg.model_path}")
    _ensure_dirs(cfg)

    if cfg.headless:
        os.environ["MUJOCO_GL"] = "egl"
        print("Running in headless mode")
    else:
        os.environ["MUJOCO_GL"] = "glfw"
        print("Running in windowed mode")

    from dm_control import mujoco

    try:
        # Load model
        print("Loading PPO model...")
        model = PPO.load(str(cfg.model_path))

        # Env + physics
        print(f"Creating environment with model: {cfg.xml_name}")
        dm_env = mocap_tracking_muscle(
            test=True, play=cfg.play, model_name=cfg.xml_name
        )
        env = DMControlGymWrapper(dm_env)
        physics = dm_env.physics

        # Prepare camera & writer
        writer = None
        camera = None
        if cfg.record_video:
            try:
                camera = mujoco.Camera(
                    physics, height=480, width=640, camera_id=cfg.camera_id
                )
                writer = imageio.get_writer(str(cfg.video_path), fps=cfg.fps)
                print(f"Recording rollout to: {cfg.video_path}")
            except Exception as e:
                print(f"Failed to setup video recording: {e}")
                cfg.record_video = False

        # Actuator names
        actuator_names, nu = _select_actuator_names(physics)

        # Rollout buffers
        obs_buf, act_buf, rew_buf = [], [], []

        # Reset
        obs, _ = env.reset()
        episode_count = 0

        for step in range(cfg.rollout_steps):
            try:
                action, _ = model.predict(obs, deterministic=True)

                # Color tendons by action if not in "play" mode
                if not cfg.play:
                    _set_tendon_colors(physics, actuator_names, action)

                obs, reward, terminated, truncated, _ = env.step(action)

                # Slice: keep angles & vels & muscle dynamics
                obs_slice = obs[12:]
                obs_buf.append(obs_slice.copy())
                act_buf.append(action.copy())
                rew_buf.append(reward)

                if writer and camera and (step % cfg.render_every_n_steps == 0):
                    try:
                        frame = camera.render()
                        writer.append_data(frame)
                    except Exception as e:
                        print(f"Failed to render frame at step {step}: {e}")

                if terminated or truncated:
                    episode_count += 1
                    print(f"Episode {episode_count} completed at step {step}")
                    obs, _ = env.reset()

            except Exception as e:
                print(f"Error during rollout at step {step}: {e}")
                break

        if writer:
            try:
                writer.close()
                print("✅ Rollout recording complete!")
            except Exception as e:
                print(f"Error closing video writer: {e}")

        # Convert to arrays
        obs_arr = np.asarray(obs_buf)
        act_arr = np.asarray(act_buf)
        rew_arr = np.asarray(rew_buf)

        print(
            f"Rollout completed: {len(obs_arr)} steps, "
            f"total reward: {np.sum(rew_arr):.3f}, "
            f"mean reward: {np.mean(rew_arr):.3f}"
        )

        # Plot
        safe_name = f"{cfg.model_path.stem}_{'play' if cfg.play else 'no_play'}"
        fig_path = cfg.figures_dir / f"rollout_plot_{safe_name}.png"
        plot_rollout(
            obs_arr, act_arr, rew_arr, actuator_names, fig_path, show=cfg.show_plot
        )

        # Save pkl with metadata
        if cfg.save_path:
            metadata = {
                "model_path": str(cfg.model_path),
                "xml_name": cfg.xml_name,
                "rollout_steps": cfg.rollout_steps,
                "play_mode": cfg.play,
                "actuator_names": actuator_names,
                "total_reward": np.sum(rew_arr),
                "episode_count": episode_count,
            }
            save_rollout(
                cfg.save_path,
                actions=act_arr,
                action_key_names=actuator_names,
                observations=obs_arr,
                observation_key_names=_get_observation_names(dm_env)[4:],
                rewards=rew_arr,
                metadata=metadata,
            )

        # Optional: viewer
        if cfg.use_viewer:

            from dm_control.viewer import launch

            class Policy:
                def __init__(self, model, env):
                    self.model = model
                    self.env = env
                    self.obs, _ = env.reset()

                def __call__(self, physics):
                    action, _ = self.model.predict(self.obs, deterministic=True)
                    self.obs, _, term, trunc, _ = self.env.step(action)
                    if term or trunc:
                        self.obs, _ = self.env.reset()
                    return action

            print("Launching viewer...")
            launch(dm_env, policy=Policy(model, env))

        return {
            "observations": obs_arr,
            "actions": act_arr,
            "rewards": rew_arr,
            "figure_path": fig_path,
            "video_path": cfg.video_path if cfg.record_video else None,
            "save_path": cfg.save_path,
            "actuator_names": actuator_names,
            "total_reward": np.sum(rew_arr),
            "mean_reward": np.mean(rew_arr),
            "episode_count": episode_count,
        }

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        print(f"Failed to load config from {config_path}: {e}")
        return {}


def create_config_from_dict(
    config_dict: Dict[str, Any], model_path: Path, play: bool = False
) -> EvalConfig:
    """Create EvalConfig from dictionary, filling in required fields."""
    # Determine XML name if not specified
    xml_name = config_dict.get("xml_name") or select_xml_name_from_model_path(
        model_path
    )

    # Generate paths
    safe_name = f"{model_path.stem}_{'play' if play else 'no_play'}"
    videos_dir = Path(config_dict.get("videos_dir", "videos"))
    figures_dir = Path(config_dict.get("figures_dir", "figures"))
    data_dir = Path(config_dict.get("data_dir", "rollout_data"))

    video_path = videos_dir / f"rollout_{safe_name}.mp4"
    save_path = data_dir / f"{safe_name}.pkl"

    return EvalConfig(
        model_path=model_path,
        xml_name=xml_name,
        video_path=video_path,
        save_path=save_path,
        play=play,
        record_video=config_dict.get("record_video", True),
        rollout_steps=config_dict.get("rollout_steps", 210),
        camera_id=config_dict.get("camera_id", 0),
        fps=config_dict.get("fps", 30),
        render_every_n_steps=config_dict.get("render_every_n_steps", 1),
        use_viewer=config_dict.get("use_viewer", False),
        show_plot=config_dict.get("show_plot", True),
        figures_dir=figures_dir,
        videos_dir=videos_dir,
        data_dir=data_dir,
        headless=config_dict.get("headless", False),
    )


if __name__ == "__main__":

    config_dict = load_config_from_yaml("../config/eval_config.yaml")

    models = list(Path("../../logs").glob("*.zip"))
    if not models:
        print("No model files found. Please specify .zip files.")
        exit(1)

    # Determine play modes
    play_modes = [True, False]

    # Evaluate each model
    results = []
    for model_path in models:
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            continue

        for play in play_modes:
            try:
                print(f"Evaluating {model_path.name} with play={play}")
                cfg = create_config_from_dict(config_dict, model_path, play)
                result = evaluate_rollout(cfg)
                results.append(result)
                print(f"Completed {model_path.name} (play={play})")
            except Exception as e:
                print(f"Failed to evaluate {model_path.name} (play={play}): {e}")
                continue

    print(f"Evaluation complete! Processed {len(results)} rollouts.")
