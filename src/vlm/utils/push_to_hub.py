import json
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from transformers import AutoConfig

from ..models import get_dynamic_vlm_class


def push_to_hub(pretrained: str, repo_name: str):
    try:
        pretrained_path = Path(pretrained)
        templates_path = Path("templates")

        if not pretrained_path.exists():
            raise FileNotFoundError(f"Model path not found: {pretrained_path}")
        if not templates_path.exists():
            raise FileNotFoundError(f"Templates path not found: {templates_path}")

        try:
            parent_llm_class, parent_casual_llm_class, base_model_path = get_dynamic_vlm_class(
                str(pretrained_path)
            )
        except Exception as e:
            raise e

        _apply_templates(
            pretrained_path, parent_llm_class, parent_casual_llm_class, base_model_path
        )

        _copy_directories(templates_path, pretrained_path)

        _update_config(pretrained_path)

        # processor = VLMProcessor.from_pretrained(
        #     pretrained,
        # )
        # VLMForCasualLM, _ = get_dynamic_vlm(pretrained)
        # model: VLMForCasualLM = VLMForCasualLM.from_pretrained(
        #     pretrained, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
        # )

        # processor.push_to_hub(repo_name)
        # model.push_to_hub(repo_name)

        return True

    except Exception as e:
        raise e
        return False


def _apply_templates(
    pretrained_path: Path, parent_llm_class: str, parent_casual_llm_class: str, base_model_path: str
):
    env = Environment(loader=FileSystemLoader("templates"))

    _render_template(
        env,
        "modeling_vlm.py.j2",
        {
            "parent_class": parent_llm_class.__name__,
            "casual_parent_class": parent_casual_llm_class.__name__,
        },
        pretrained_path / "modeling_vlm.py",
    )

    _render_template(env, "processing_vlm.py.j2", {}, pretrained_path / "processing_vlm.py")

    parent_config_class = AutoConfig.from_pretrained(
        base_model_path, trust_remote_code=True
    ).__class__

    _render_template(
        env,
        "configuration_vlm.py.j2",
        {
            "parent_class": parent_config_class.__name__,
        },
        pretrained_path / "configuration_vlm.py",
    )


def _render_template(env: Environment, template_name: str, context: dict, output_path: Path):
    try:
        template = env.get_template(template_name)
        output = template.render(**context)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(output)
    except Exception as e:
        raise e


def _copy_directories(templates_path: Path, pretrained_path: Path):
    try:
        for directory in ["connectors", "visual_encoders"]:
            source = templates_path / directory
            target = pretrained_path / directory

            if source.exists():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            else:
                print(f"Source directory does not exist: {source}")
    except Exception as e:
        raise e


def _update_config(pretrained_path: Path):
    config_path = pretrained_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = json.load(f)

        config["auto_map"] = {
            "AutoConfig": "configuration_vlm.VLMConfig",
            "AutoModel": "modeling_vlm.VLMForCasualLM",
            "AutoProcessor": "processing_vlm.VLMProcessor",
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    except Exception as e:
        raise e
