from safetensors.torch import save_file, load_file, load
from huggingface_hub import hf_hub_download
import torch


def convert_350m():
    filename = hf_hub_download("bigscience/bloom-350m", filename="pytorch_model.bin")
    data = torch.load(filename, map_location="cpu")

    # Need to copy since that call mutates the tensors to numpy
    save_file(data.copy(), "bloom-350m.bin")

def convert_full():
    MODEL_ID = "bigscience/bloom"
    filenames = [
        (f"bloom-h.{i}.bin", f"pytorch_model_000{i+1:02d}-of-00072.bin")
        for i in range(1, 72)
    ]
    for (local, filename) in [
        ("bloom-embedding.bin", "pytorch_model_00001-of-00072.bin"),
        ("bloom-final.bin", "pytorch_model_00072-of-00072.bin"),
    ] + filenames:
        filename = hf_hub_download(MODEL_ID, filename=filename)
        data = torch.load(filename, map_location="cpu")

        # Need to copy since that call mutates the tensors to numpy
        save_file(data.copy(), local)

convert_350m()
