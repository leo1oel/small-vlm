from pathlib import Path

from omegaconf import DictConfig

import hydra

config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")  # type: ignore
def main(cfg: DictConfig) -> None:  # type: ignore
   print(cfg)  # type: ignore


if __name__ == "__main__":
    main()
