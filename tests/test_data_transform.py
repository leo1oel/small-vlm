from unittest.mock import MagicMock, patch

import torch
from icecream import ic
from PIL import Image

from vlm.config.config_schema import AppConfig


# 加载配置
def load_test_config():
    import hydra

    with hydra.initialize(version_base=None, config_path="../src/vlm/config"):
        cfg = hydra.compose(config_name="test_config")
        return cfg


def test_transform_function():
    cfg: AppConfig = load_test_config()  # pyright: ignore

    mock_encoder = MagicMock()
    mock_encoder.token_size = 5
    mock_encoder.preprocessor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}  # pyright: ignore

    mock_llm = MagicMock()

    with (
        patch("vlm.models.model.VLM._build_visual_encoder", return_value=mock_encoder),
        patch(
            "vlm.models.language_models.HFLLMLanguageModel._build_language_model",
            return_value=mock_llm,
        ),
        patch("vlm.models.model.VLM._build_connector", return_value=MagicMock()),
    ):
        from vlm.models import VLM

        model = VLM(cfg.model, cfg.trainer)

        test_image = Image.new("RGB", (224, 224), color=(73, 109, 137))
        # test text transformation
        test_conversations = [
            {"from": "human", "value": "Describe the image <image>. \n"},
            {"from": "assistant", "value": "This is a blue image."},
            {"from": "human", "value": "Describe the image. \n"},
        ]
        tokenizer = model.language_model.tokenizer
        ic(tokenizer)
        ic(test_conversations)
        conversation = []
        for item in test_conversations:
            role = "user" if item["from"] == "human" else "assistant"
            conversation.append({"role": role, "content": item["value"]})
        ic(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))  # pyright: ignore
        ic(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False))  # pyright: ignore
        ic(
            tokenizer.apply_chat_template(  # pyright: ignore
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                padding=False,
                truncation=True,
            )
        )
        transform_text = model.language_model.transform
        text_and_label = transform_text(test_conversations, 5, False)
        ic(text_and_label[0])
        ic(text_and_label[1])

        # test data transformation
        test_conversations_str = '[{"from": "human", "value": "Describe the image <image>. \n"},{"from": "assistant", "value": "This is a blue image."},{"from": "human", "value": "Describe the image. \n"}]'
        test_item = {"image": test_image, "text": test_conversations_str}
        result = model.transform(test_item)  # pyright: ignore
        assert isinstance(result["image"], torch.Tensor)
        assert isinstance(result["text"], torch.Tensor)
        assert isinstance(result["label"], torch.Tensor)
        assert result["image"].shape == (3, 224, 224)  # pyright: ignore
        assert result["label"].shape[0] - result["text"].shape[0] == 4  # pyright: ignore
