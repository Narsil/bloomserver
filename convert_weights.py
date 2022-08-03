from safetensors.torch import save_file, load_file, load
# from huggingface_hub import hf_hub_download
from transformers.utils import hf_bucket_url, cached_path
import torch
import tqdm
import os

DIRECTORY = "weights"

def hf_hub_download(model_id, filename):
    shard_url = hf_bucket_url(
            model_id,
            filename=filename,
        )
    cached_filename = cached_path(
            shard_url,
    )
    return cached_filename


def convert_350m():
    filename = hf_hub_download("bigscience/bloom-560m", filename="pytorch_model.bin")
    data = torch.load(filename, map_location="cpu")

    # Need to copy since that call mutates the tensors to numpy
    save_file(data.copy(), os.path.join(DIRECTORY, "bloom-350m.bin"))


def convert_testing():
    filename = hf_hub_download(
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
        filename = hf_hub_download(MODEL_ID, filename=filename)
        data = torch.load(filename, map_location="cpu")

        save_file(data.copy(), os.path.join(DIRECTORY, local))

if __name__ == "__main__":
    convert_testing()
    convert_350m()
    convert_full()
