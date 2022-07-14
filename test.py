from transformers import pipeline
import torch

pipe = pipeline(
    model="bigscience/bigscience-small-testing",
    device=0,
    max_new_tokens=20,
    torch_dtype=torch.bfloat16,
)
pipe.model.to(torch.bfloat16)

print(pipe("I enjoy walking with my cute dog"))
