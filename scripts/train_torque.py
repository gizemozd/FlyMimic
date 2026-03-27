import hydra
from omegaconf import DictConfig

from flymimic.train.train_torque import train


@hydra.main(
    config_path="../flymimic/config", config_name="train_torque", version_base=None
)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
