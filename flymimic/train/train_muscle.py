""" Train PPO on the muscle model."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from flymimic.envs.dmcontrol_wrapper import DMControlGymWrapper
from flymimic.tasks.fly.mocap_tracking_muscle import mocap_tracking_muscle


class EpisodeRewardCallback(BaseCallback):
    """Log per-episode reward to WandB during training rollouts."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                wandb.log(
                    {"train/episode_reward": info["episode"]["r"]},
                    step=self.num_timesteps,
                )
        return True


def _to_namespace(obj):
    """Allow cfg to be a dot access obj."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


# Base directory for flymimic
flymimic_dir = Path(__file__).resolve().parents[2]


def train(cfg):
    cfg = _to_namespace(cfg)
    seed = cfg.seed
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Base logging directories
    logs_dir = flymimic_dir / "logs"
    eval_logs_dir = logs_dir / "eval"
    ckpt_dir = logs_dir / "ckpts"
    best_model_dir = logs_dir / f"best_muscle_seed_{cfg.exp}_{seed}_v3"
    logs_dir.mkdir(exist_ok=True)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # Training environment
    train_env = Monitor(
        DMControlGymWrapper(mocap_tracking_muscle(model_name=cfg.xml_name), seed=seed),
        str(logs_dir),
    )
    activation = cfg.policy.activation_fn
    if isinstance(activation, str):
        activation = eval(activation)

    # Policy kwargs
    policy_kwargs = dict(
        # FIXME: get this from config
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
        activation_fn=activation,
    )

    # WandB
    wandb_cfg = getattr(cfg, "wandb", None)
    run_name = cfg.log_file_name or f"muscle_{cfg.exp}_seed{seed}_{timestamp}"
    wandb.init(
        project=wandb_cfg.project if wandb_cfg else "flymimic",
        entity=wandb_cfg.entity if wandb_cfg and wandb_cfg.entity != "null" else None,
        tags=list(wandb_cfg.tags) if wandb_cfg and hasattr(wandb_cfg, "tags") else ["muscle"],
        name=run_name,
        config={
            "model": "muscle",
            "xml_name": cfg.xml_name,
            "exp": cfg.exp,
            "seed": seed,
            "tot_ts": cfg.tot_ts,
            "learning_rate": cfg.train.learning_rate,
            "n_steps": cfg.train.n_steps,
            "batch_size": cfg.train.batch_size,
            "n_epochs": cfg.train.n_epochs,
            "activation_fn": cfg.policy.activation_fn,
        },
        sync_tensorboard=True,
    )

    # Model creation or loading
    if cfg.load_model:
        print(f"Loading model from {cfg.load_model}")
        model = PPO.load(cfg.load_model, env=train_env, tensorboard_log=str(logs_dir))
        log_file_name = cfg.log_file_name or Path(cfg.load_model).stem
    else:
        log_file_name = f"ppo_muscle_seed_{cfg.exp}_{seed}_{timestamp}"
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=cfg.train.learning_rate,
            verbose=cfg.train.verbose,
            tensorboard_log=str(logs_dir),
            policy_kwargs=policy_kwargs,
            n_steps=cfg.train.n_steps,
            batch_size=cfg.train.batch_size,
            n_epochs=cfg.train.n_epochs,
        )

    # Evaluation environment
    eval_env = Monitor(
        DMControlGymWrapper(mocap_tracking_muscle(model_name=cfg.xml_name), seed=seed),
        str(eval_logs_dir),
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(logs_dir),
        eval_freq=cfg.callbacks.eval_freq,
        deterministic=False,
        render=False,
        n_eval_episodes=cfg.callbacks.n_eval_episodes,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.callbacks.save_freq,
        save_path=str(ckpt_dir),
        name_prefix=f"ppo_mocap_agent_muscle_seed_{cfg.exp}_{seed}_{timestamp}",
    )
    wandb_callback = WandbCallback(verbose=0)
    episode_reward_callback = EpisodeRewardCallback()
    callbacks = CallbackList(
        [eval_callback, checkpoint_callback, wandb_callback, episode_reward_callback]
    )

    # Training
    model.learn(
        total_timesteps=cfg.tot_ts,
        callback=callbacks,
        tb_log_name=log_file_name,
        reset_num_timesteps=not bool(cfg.load_model),
    )

    # Save final model
    model.save(str(logs_dir / log_file_name))
    wandb.finish()
