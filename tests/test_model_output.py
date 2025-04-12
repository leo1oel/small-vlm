# import torch

# from vlm.models.model import VLM

# model = VLM.load_from_checkpoint(
#     "/pasteur2/u/yuhuiz/yiming/small-vlm/outputs/2025-04-12/00-33-13/checkpoints/model.pt"
# )

# model.eval()

# language_model = model.language_model
# tokenizer = language_model.tokenizer

# # text = "a b a b a b"

# # input = tokenizer(text, return_tensors="pt")
# input = {
#     "input_ids": torch.tensor([[1, 263, 18449, 734, 29872, 1134, 734, 29872, 9909, 411, 3244]]),
#     "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
# }

# outputs = language_model(input_ids=input["input_ids"], attention_mask=input["attention_mask"])

# print(outputs)

# output_ids = outputs.argmax(dim=-1)

# print(output_ids)

# output_text = tokenizer.decode(output_ids[0])

# print(output_text)

# output_ids = language_model.language_model.generate(
#     input_ids=input["input_ids"],
#     attention_mask=input["attention_mask"],
#     max_new_tokens=10,
#     do_sample=False,
# )

# print(output_ids)

# print(tokenizer.decode(output_ids[0]))
