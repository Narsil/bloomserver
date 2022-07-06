from safetensors.torch import save_file, load_file, load
from huggingface_hub import hf_hub_download
import torch

MODEL_ID = "bigscience/bloom"

for (local, filename) in [
    ("bloom-embedding.bin", "pytorch_model_00001-of-00072.bin"),
    ("bloom-h.1.bin", "pytorch_model_00002-of-00072.bin"),
    ("bloom-final.bin", "pytorch_model_00072-of-00072.bin"),
]:
    filename = hf_hub_download(MODEL_ID, filename=filename)
    data = torch.load(filename, map_location="cpu")

    # Need to copy since that call mutates the tensors to numpy
    save_file(data.copy(), local)
