from safetensors.torch import save_file
# from huggingface_hub import hf_hub_download
from transformers.utils import cached_file
import torch
import tqdm
import os

DIRECTORY = "weights"

def convert_350m():
    filename = cached_file("bigscience/bloom-560m", filename="pytorch_model.bin")
    data = torch.load(filename, map_location="cpu")

    # Need to copy since that call mutates the tensors to numpy
    save_file(data.copy(), os.path.join(DIRECTORY, "bloom-350m.bin"))


def convert_testing():
    filename = cached_file(
        "bigscience/bigscience-small-testing", filename="pytorch_model.bin"
    )
    data = torch.load(filename, map_location="cpu")

    # Need to copy since that call mutates the tensors to numpy
    save_file(data.copy(), os.path.join(DIRECTORY, "bloom-testing.bin"))


def convert_full():
    MODEL_ID = "bigscience/bloom"
    filenames = [
        (f"bloom-h.{i}.bin", f"pytorch_model_000{i+2:02d}-of-00072.bin")
        for i in range(0, 70)
    ]
    for (local, filename) in tqdm.tqdm(
        [
            ("bloom-embedding.bin", "pytorch_model_00001-of-00072.bin"),
            ("bloom-final.bin", "pytorch_model_00072-of-00072.bin"),
        ]
        + filenames
    ):
        if os.path.exists(local):
            continue
        filename = cached_file(MODEL_ID, filename=filename)
        data = torch.load(filename, map_location="cpu")

        save_file(data.copy(), os.path.join(DIRECTORY, local))

if __name__ == "__main__":
    convert_testing()
    convert_350m()
    convert_full()
