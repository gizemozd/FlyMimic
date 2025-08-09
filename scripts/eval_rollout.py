import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from flymimic.evaluation import evaluate_rollout as eval


@hydra.main(
    version_base=None, config_path="../flymimic/config", config_name="eval_config"
)
def main(cfg: DictConfig):
    # Headless rendering mode
    if cfg.headless:
        os.environ["MUJOCO_GL"] = "egl"  # or 'osmesa' if needed
        print("Running in headless mode (MUJOCO_GL=egl)")

    model_path = Path(cfg.model_path)
    xml_name = cfg.xml_name or eval.select_xml_name_from_model_path(model_path)

    eval_cfg = eval.EvalConfig(
        model_path=model_path,
        xml_name=xml_name,
        video_path=Path(cfg.videos_dir)
        / f"rollout_{model_path.stem}_{'play' if cfg.play else 'no_play'}.mp4",
        save_path=Path(cfg.data_dir)
        / f"{model_path.stem}_{'play' if cfg.play else 'no_play'}.npz",
        play=cfg.play,
        record_video=cfg.record_video,
        rollout_steps=cfg.rollout_steps,
        camera_id=cfg.camera_id,
        fps=cfg.fps,
        render_every_n_steps=cfg.render_every_n_steps,
        use_viewer=cfg.use_viewer,
        show_plot=cfg.show_plot,
        figures_dir=Path(cfg.figures_dir),
        videos_dir=Path(cfg.videos_dir),
        data_dir=Path(cfg.data_dir),
    )

    result = eval.evaluate_rollout(eval_cfg)
    print(f"Evaluation complete. Figure saved at: {result['figure_path']}")
    if result["video_path"]:
        print(f"Video saved at: {result['video_path']}")
    if result["save_path"]:
        print(f"Rollout data saved at: {result['save_path']}")


if __name__ == "__main__":
    main()
    # # Loop over all zip files and run eval with overrides
    # for model_path in Path(".").glob("*.zip"):
    #     main(overrides=[
    #         f"model_path={model_path}",
    #         "play=false",
    #         "headless=true"
    #     ])
