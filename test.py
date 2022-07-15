from transformers import pipeline
import torch

pipe = pipeline(
    model="bigscience/bloom",
    max_new_tokens=1,
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
)

print(pipe("I enjoy walking with my cute dog"))
