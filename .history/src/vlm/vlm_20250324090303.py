from pathlib import Path

from omegaconf import DictConfig

import hydra

current_dir: Path = Path(__file__).resolve().parent
config_path: Path = current_dir.parent / "conf"

def run(cfg: DictConfig) -> dict[str, float]:
    # 你的实际逻辑
    result: dict[str, float] = {}  # 示例返回值
    return result

@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
