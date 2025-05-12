# import string
# from types import SimpleNamespace
from typing import Any

# import pandas as pd
import torch
from PIL import Image

# from transformers import StoppingCriteria
from transformers.image_processing_utils import BaseImageProcessor
from transformers.tokenization_utils import PreTrainedTokenizer

# from vlmeval.dataset import DATASET_TYPE
# from vlmeval.smp import cn_string
# from vlmeval.vlm import BaseModel

# from ..models import VLM


def expand2square(pil_img: Image.Image, background_color: tuple):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images: list, image_processor: BaseImageProcessor, model_cfg: Any):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == "pad":
        for image in images:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors="pt")["pixel_values"]
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    image_token_index: int,
    return_tensors: str | None = None,
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X: list, sep: int):
        return [ele for sublist in zip(X, [sep] * len(X), strict=False) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


# class KeywordsStoppingCriteria(StoppingCriteria):
#     def __init__(self, keywords: list, tokenizer: PreTrainedTokenizer, input_ids: list):
#         self.keywords: list = keywords
#         self.keyword_ids: list = []
#         self.max_keyword_len: int = 0
#         for keyword in keywords:
#             cur_keyword_ids = tokenizer(keyword).input_ids
#             if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
#                 cur_keyword_ids = cur_keyword_ids[1:]
#             if len(cur_keyword_ids) > self.max_keyword_len:
#                 self.max_keyword_len = len(cur_keyword_ids)
#             self.keyword_ids.append(torch.tensor(cur_keyword_ids))
#         self.tokenizer: PreTrainedTokenizer = tokenizer
#         self.start_len: int = input_ids.shape[1]

#     def call_for_batch(self, output_ids: torch.LongTensor, **_: Any) -> bool:
#         offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
#         self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
#         for keyword_id in self.keyword_ids:
#             truncated_output_ids = output_ids[0, -keyword_id.shape[0] :]
#             if torch.equal(truncated_output_ids, keyword_id):
#                 return True
#         outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
#         for keyword in self.keywords:
#             if keyword in outputs:
#                 return True
#         return False

#     @override
#     def __call__(
#         self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
#     ) -> bool:
#         outputs = []
#         for i in range(output_ids.shape[0]):
#             outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
#         return all(outputs)


# class VLMGenerator(BaseModel):
#     INSTALL_REQ: bool = True
#     INTERLEAVE: bool = True

#     def __init__(self, model_path: str, **kwargs: Any):
#         super().__init__(model_path, **kwargs)
#         self.system_prompt: str = (
#             "A chat between a curious human and an artificial intelligence assistant. "
#             "The assistant gives helpful, detailed, and polite answers to the human's questions. "
#         )
#         self.stop_str: str = "</s>"
#         self.model: VLM = VLM.from_pretrained(model_path, device="cpu")

#         self.model = self.model.cuda()
#         self.tokenizer: PreTrainedTokenizer = self.model.language_model.tokenizer
#         self.image_processor: BaseImageProcessor = self.model..visual_encoder.preprocessor
#         self.conv_mode: str = "llava_v1"
#         kwargs_default = dict(
#             do_sample=False,
#             temperature=0,
#             max_new_tokens=2048,
#             top_p=None,
#             num_beams=1,
#             use_cache=True,
#         )
#         kwargs_default.update(kwargs)
#         self.kwargs: dict = kwargs_default

#     @override
#     def use_custom_prompt(self, dataset: Any):
#         assert dataset is not None
#         if DATASET_TYPE(dataset) == "MCQ":
#             return True
#         return False

#     @override
#     def build_prompt(self, line: list, dataset: Any = None):
#         assert self.use_custom_prompt(dataset)
#         assert dataset is None or isinstance(dataset, str)
#         tgt_path = self.dump_image(line, dataset)

#         question = line["question"]
#         hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
#         if hint is not None:
#             question = hint + "\n" + question

#         options = {
#             cand: line[cand]
#             for cand in string.ascii_uppercase
#             if cand in line and not pd.isna(line[cand])
#         }
#         for key, item in options.items():
#             question += f"\n{key}. {item}"
#         prompt = question

#         if len(options):
#             prompt += (
#                 "\n请直接回答选项字母。"
#                 if cn_string(prompt)
#                 else "\nAnswer with the option's letter from the given choices directly."
#             )
#         else:
#             prompt += (
#                 "\n请直接回答问题。" if cn_string(prompt) else "\nAnswer the question directly."
#             )

#         message = [dict(type="image", value=s) for s in tgt_path]
#         message.append(dict(type="text", value=prompt))
#         return message

#     def concat_tilist(self, message: list):
#         text, images = "", []
#         for item in message:
#             if item["type"] == "text":
#                 text += item["value"]
#             elif item["type"] == "image":
#                 text += " <image> "
#                 images.append(item["value"])
#         return text, images

#     def chat_inner(self, message: list, _: Any = None):
#         image_token_index: int = self.model.model_config.language_model.image_token_index

#         prompt = self.system_prompt
#         images = []
#         for utter in message:
#             prompt += "USER: " if utter["role"] == "user" else "ASSISTANT: "
#             content, images_sub = self.concat_tilist(utter["content"])
#             prompt += content
#             images.extend(images_sub)
#             prompt += " " if utter["role"] == "user" else self.stop_str
#         assert message[-1]["role"] == "user", message
#         prompt += "ASSISTANT: "

#         images = [Image.open(s).convert("RGB") for s in images]
#         args = SimpleNamespace(image_aspect_ratio="pad")
#         image_tensor = process_images(images, self.image_processor, args).to(
#             "cuda", dtype=torch.float16
#         )

#         input_ids = (
#             tokenizer_image_token(prompt, self.tokenizer, image_token_index, return_tensors="pt")
#             .unsqueeze(0)
#             .cuda()
#         )
#         keywords = [self.stop_str]
#         stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 input_ids,
#                 images=image_tensor,
#                 stopping_criteria=[stopping_criteria],
#                 **self.kwargs,
#             )
#         output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#         return output

#     @override
#     def generate_inner(self, message: list, _: Any = None):
#         image_token_index: int = self.model.model_config.language_model.image_token_index

#         # Support interleave text and image
#         content, images = self.concat_tilist(message)

#         images = [Image.open(s).convert("RGB") for s in images]
#         args = SimpleNamespace(image_aspect_ratio="pad")
#         if images:
#             image_tensor = process_images(images, self.image_processor, args).to(
#                 "cuda", dtype=torch.float16
#             )
#         else:
#             image_tensor = None

#         prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

#         input_ids = (
#             tokenizer_image_token(prompt, self.tokenizer, image_token_index, return_tensors="pt")
#             .unsqueeze(0)
#             .cuda()
#         )
#         keywords = [self.stop_str]
#         stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
#         with torch.inference_mode():
#             output_ids = self.model.generate(
#                 input_ids,
#                 images=image_tensor,
#                 stopping_criteria=[stopping_criteria],
#                 **self.kwargs,
#             )

#         output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#         return output
