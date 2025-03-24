from pathlib import Path
from typing import dict, Any

from omegaconf import DictConfig

import hydra

current_dir: Path = Path(__file__).resolve().parent
config_path: Path = current_dir.parent / "conf"


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> Dict[str, float]:
    features_path = extract_features(cfg)

    metrics = linear_probe(cfg, features_path)

    return metrics


if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
