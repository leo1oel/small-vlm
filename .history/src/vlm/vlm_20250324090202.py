from pathlib import Path
from typing import Any

from omegaconf import DictConfig

import hydra

current_dir: Path = Path(__file__).resolve().parent
config_path: Path = current_dir.parent / "conf"


@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> dict[str, float]:
    result: dict[str, float] = {}  # 示例返回值
    return result


if __name__ == "__main__":
    main()
