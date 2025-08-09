import hydra
from omegaconf import DictConfig

from flymimic.train.train_muscle import train


@hydra.main(
    config_path="../flymimic/config", config_name="train_arm", version_base=None
)
def main(cfg: DictConfig):
    # Pass cfg straight through
    train(cfg)


if __name__ == "__main__":
    main()
