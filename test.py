from transformers import pipeline
import torch

pipe = pipeline(
    model="bigscience/bloom",
    max_new_tokens=2,
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
)
# pipe = pipeline(
#     model="bigscience/bigscience-small-testing",
#     max_new_tokens=2,
#     model_kwargs={
#         "device_map": "auto",
#         "torch_dtype": torch.bfloat16,
#     }
# )
# print(pipe.model.device)


print(pipe("Math exercise - answers:\n34+10=44\n54+20="))
