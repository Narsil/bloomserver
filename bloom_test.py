import asyncio
import base64
import datetime
import functools
import io
import time
import json
import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoConfig,
)
from transformers.utils.hub import (
    hf_bucket_url,
    cached_path,
)
from transformers.models.bloom.modeling_bloom import BloomBlock, build_alibi_tensor
from torch import nn
from torch.nn import functional as F


from queue import Queue, Empty
from threading import Thread, Lock
import functools
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
from PIL import Image

from transformers.pipelines import Pipeline, PipelineException

from flask import Flask, jsonify, make_response, request

MODEL_ID = "bigscience/bloom"
num_threads = 14
LAYERS_PER_THREAD = 5
embeddings_filename = "pytorch_model_00001-of-00072.bin"
layer_template_filename = "pytorch_model_000{:02d}-of-00072.bin"
final_filename = "pytorch_model_00072-of-00072.bin"


def load(filename, device):
    print(f"[{datetime.datetime.now()}] Loading {filename}")
    shard_url = hf_bucket_url(MODEL_ID, filename=filename)
    cached_filename = cached_path(
        shard_url,
    )
    print(f"[{datetime.datetime.now()}] Found {cached_filename}")
    result = torch.load(cached_filename, map_location=device)
    print(f"[{datetime.datetime.now()}] Loaded {cached_filename}")
    return result


weights = load(embeddings_filename, "cuda:0")
input_ids = torch.Tensor([[0, 1, 2, 3, 4, 5]]).long().cuda()
input_embeds = F.embedding(
    input_ids,
    weights["word_embeddings.weight"],
    3,
)

np.save("input_embeds.npy", input_embeds.cpu().float())

# with open("word_embeddings.npy", "wb") as f:
#     np.save(f, weights["word_embeddings.weight"].float().numpy())
