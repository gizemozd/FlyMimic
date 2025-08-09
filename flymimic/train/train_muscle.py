""" Train PPO on the muscle model."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from flymimic.envs.dmcontrol_wrapper import DMControlGymWrapper
from flymimic.tasks.fly.mocap_tracking_muscle import mocap_tracking_muscle


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
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Training
    model.learn(
        total_timesteps=cfg.tot_ts,
        callback=callbacks,
        tb_log_name=log_file_name,
        reset_num_timesteps=not bool(cfg.load_model),
    )

    # Save final model
    model.save(str(logs_dir / log_file_name))
