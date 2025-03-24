from pathlib import Path

import hydra
from omegaconf import DictConfig

config_path: Path = Path(__file__).resolve().parent.parent.parent / "conf"

@hydra.main(version_base=None, config_path=str(config_path), config_name="config")  # type: ignore
def main(cfg: DictConfig) -> None:
    print(cfg)


if __name__ == "__main__":
    main()
