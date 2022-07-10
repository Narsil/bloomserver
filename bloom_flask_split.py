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


from queue import Queue, Empty
from threading import Thread, Lock
import functools
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np
from PIL import Image

from transformers.pipelines import Pipeline, PipelineException

from flask import Flask, jsonify, make_response, request


MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
QUEUE_SIZE = int(os.environ.get("QUEUE_SIZE", 2 * MAX_BATCH_SIZE))
PADDING_IDX = 3
BIG = True
mutex = Lock()

if BIG:
    MODEL_ID = "bigscience/bloom"
    num_threads = 14
    LAYERS_PER_THREAD = 5
    embeddings_filename = "pytorch_model_00001-of-00072.bin"
    layer_template_filename = "pytorch_model_000{:02d}-of-00072.bin"
    final_filename = "pytorch_model_00072-of-00072.bin"
else:
    MODEL_ID = "bigscience/bigscience-small-testing"
    num_threads = 1
    LAYERS_PER_THREAD = 2
    embeddings_filename = "pytorch_model.bin"
    layer_template_filename = "pytorch_model.bin"
    final_filename = "pytorch_model.bin"

logger = logging.getLogger(__file__)


def normalize_payload(payload: str) -> Tuple[Any, Dict]:
    # payload = bpayload.decode("utf-8")

    # We used to accept raw strings, we need to maintain backward compatibility
    try:
        payload = json.loads(payload)
    except Exception:
        pass

    parameters: Dict[str, Any] = {}
    if isinstance(payload, dict) and "inputs" in payload:
        inputs = payload["inputs"]
        parameters = payload.get("parameters", {})
    else:
        inputs = payload
    return inputs, parameters


def padding(items, config, dtype):
    max_length = max([input_ids.shape[1] for input_ids, _ in items])
    batch_size = len(items)
    input_ids = (
        torch.zeros((batch_size, max_length), dtype=torch.int32, device="cuda:0")
        + PADDING_IDX
    )
    attention_mask = torch.zeros((batch_size, max_length)).long().to(device="cuda:0")
    alibi = build_alibi_tensor(max_length, config.n_head, dtype)
    rqs = []
    for i, (small_input_ids, rq) in enumerate(items):
        length = small_input_ids.shape[1]
        input_ids[i, -length:] = small_input_ids
        attention_mask[i, -length:] = 1
        rqs.append(rq)
    return (input_ids, attention_mask, alibi, rqs)


def load(filename, device):
    print(f"[{datetime.datetime.now()}] Loading {filename}")
    shard_url = hf_bucket_url(MODEL_ID, filename=filename)
    cached_filename = cached_path(
        shard_url,
    )
    print(f"[{datetime.datetime.now()}] Found {cached_filename}")
    result =  torch.load(cached_filename, map_location=device)
    print(f"[{datetime.datetime.now()}] Loaded {cached_filename}")
    return result



def thread1(q, follow_queue):
    print("Loading thread1 (0)")
    config = AutoConfig.from_pretrained(MODEL_ID)
    device = "cuda:0"

    weights = load(embeddings_filename, device)
    word_embeddings = nn.Embedding(config.vocab_size, config.n_embed).to(device=device, dtype=torch.bfloat16).eval()
    word_embeddings.weight = nn.Parameter(weights["word_embeddings.weight"])

    word_embeddings_layernorm = (
        nn.LayerNorm(config.n_embed, eps=config.layer_norm_epsilon).to(device=device, dtype=torch.bfloat16).eval()
    )
    word_embeddings_layernorm.weight = nn.Parameter(
        weights["word_embeddings_layernorm.weight"]
    )
    word_embeddings_layernorm.bias = nn.Parameter(
        weights["word_embeddings_layernorm.bias"]
    )

    print("Loaded thread1 (0)")
    while True:
        with torch.no_grad():
            print("Loop thread1")
            items = []
            while len(items) < MAX_BATCH_SIZE:
                print("Loop items")
                if len(items) > 0:
                    try:
                        print("Waiting without blocking")
                        item = q.get(False)
                        print("Go new ids not blocking")
                    except Empty:
                        break
                else:
                    print("Waiting for new incoming ids")
                    item = q.get()
                    print("Go new ids")
                items.append(item)
                print("end loop items")

            print(f"Doing a batch request on {len(items)} element", flush=True)
            input_ids, attention_mask, alibi, rqs = padding(
                items, config, word_embeddings.weight.dtype
            )

            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(device=device)
            alibi = alibi.to(device=device)

            input_embeds = word_embeddings(input_ids)
            hidden_states = word_embeddings_layernorm(input_embeds)

            follow_queue.put_nowait((hidden_states, attention_mask, alibi, rqs))
            del hidden_states, attention_mask, alibi, rqs
            del items, input_embeds


def thread2(receive_queue, send_queue, thread_number):
    print(f"Loading thread2 ({thread_number})")
    start = datetime.datetime.now()
    device = f"cuda:{thread_number}"
    config = AutoConfig.from_pretrained(MODEL_ID)
    # TODO Load the actual layers
    layers = []
    for i in range(LAYERS_PER_THREAD):
        layer_number = (thread_number - 1) * LAYERS_PER_THREAD + i

        weights = load(layer_template_filename.format(layer_number + 2), device)
        block = BloomBlock(config, layer_number=layer_number).to(device=device, dtype=torch.bfloat16).eval()
        block.input_layernorm.weight = nn.Parameter(
            weights[f"h.{layer_number}.input_layernorm.weight"]
        )
        block.input_layernorm.bias = nn.Parameter(
            weights[f"h.{layer_number}.input_layernorm.bias"]
        )

        block.post_attention_layernorm.weight = nn.Parameter(
            weights[f"h.{layer_number}.post_attention_layernorm.weight"]
        )
        block.post_attention_layernorm.bias = nn.Parameter(
            weights[f"h.{layer_number}.post_attention_layernorm.bias"]
        )

        block.self_attention.query_key_value.weight = nn.Parameter(
            weights[f"h.{layer_number}.self_attention.query_key_value.weight"]
        )
        block.self_attention.query_key_value.bias = nn.Parameter(
            weights[f"h.{layer_number}.self_attention.query_key_value.bias"]
        )

        block.self_attention.dense.weight = nn.Parameter(
            weights[f"h.{layer_number}.self_attention.dense.weight"]
        )
        block.self_attention.dense.bias = nn.Parameter(
            weights[f"h.{layer_number}.self_attention.dense.bias"]
        )

        block.mlp.dense_h_to_4h.weight = nn.Parameter(
            weights[f"h.{layer_number}.mlp.dense_h_to_4h.weight"]
        )
        block.mlp.dense_h_to_4h.bias = nn.Parameter(
            weights[f"h.{layer_number}.mlp.dense_h_to_4h.bias"]
        )

        block.mlp.dense_4h_to_h.weight = nn.Parameter(
            weights[f"h.{layer_number}.mlp.dense_4h_to_h.weight"]
        )
        block.mlp.dense_4h_to_h.bias = nn.Parameter(
            weights[f"h.{layer_number}.mlp.dense_4h_to_h.bias"]
        )

        layers.append(block)
        print(f"Loaded layer {layer_number} thread2 ({thread_number}) in {datetime.datetime.now() - start}")
    print(f"Loaded thread2 ({thread_number}) in {datetime.datetime.now() - start}")
    while True:
        with torch.no_grad():
            hidden_states, attention_mask, alibi, rqs = receive_queue.get()
            print(f"Got hidden states thread2 ({thread_number})")

            hidden_states = hidden_states.to(device=device)
            attention_mask = attention_mask.to(device=device)
            alibi = alibi.to(device=device)

            for layer in layers:
                hidden_states = layer(
                    hidden_states, attention_mask=attention_mask, alibi=alibi
                )[0]
            send_queue.put_nowait((hidden_states, attention_mask, alibi, rqs))
            del hidden_states, attention_mask, alibi, rqs


def thread3(receive_queue, thread_number):
    print(f"Loading thread3 ({thread_number})")
    device = f"cuda:{thread_number}"
    config = AutoConfig.from_pretrained(MODEL_ID)

    weights = load(final_filename, device)
    ln_f = nn.LayerNorm(config.n_embed, eps=config.layer_norm_epsilon).to(device=device, dtype=torch.bfloat16).eval()
    ln_f.weight = nn.Parameter(weights["ln_f.weight"])
    ln_f.bias = nn.Parameter(weights["ln_f.bias"])
    weights = load(embeddings_filename, device)
    lm_head = (
        nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(device=device, dtype=torch.bfloat16).eval()
    )
    lm_head.weight = nn.Parameter(weights["word_embeddings.weight"])

    print(f"Loaded thread3 ({thread_number})")
    while True:
        with torch.no_grad():
            hidden_states, attention_mask, alibi, rqs = receive_queue.get()
            print(f"Got hidden states thread3 ({thread_number})")
            hidden_states = hidden_states.to(device=device)
            attention_mask = attention_mask.to(device=device)
            alibi = alibi.to(device=device)

            hidden_states = ln_f(hidden_states)
            logits = lm_head(hidden_states)

            logits = hidden_states

            for i in range(logits.shape[0]):
                rqs[i].put(logits[i : i + 1])

            del hidden_states, attention_mask, alibi, rqs


# def run_app(requests_queue, tokenizer):
app = Flask(__name__)

requests_queue = Queue()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

queues = [Queue() for i in range(num_threads + 1)]
requests_queue = Queue()
for i in range(0, num_threads):
    Thread(target=thread2, args=(queues[i], queues[i + 1], i + 1)).start()
Thread(target=thread3, args=(queues[num_threads], num_threads + 1)).start()
Thread(target=thread1, args=(requests_queue, queues[0])).start()


@app.route("/generate", methods=["POST"])
def generate():
    content = request.json
    device = "cpu"

    qsize = requests_queue.qsize()
    print("Queue size", qsize)
    if qsize >= QUEUE_SIZE:
        return make_response({"error": "queue full, try again later"}, 503)

    response_queue = Queue()
    inputs, parameters = normalize_payload(content)

    input_ids = tokenizer(inputs, return_tensors="pt")["input_ids"].to(device)

    for i in range(parameters.get("max_new_tokens", 3)):
        print("Putting ids in the queue", input_ids.shape)
        requests_queue.put((input_ids, response_queue))
        logits = response_queue.get()
        new_id = logits[:, -1:].argmax(dim=-1).to(device)
        input_ids = torch.cat([input_ids, new_id], dim=-1)

    answer = tokenizer.decode(input_ids[0].tolist())
    response = make_response(jsonify([{"generated_text": answer}]), 200)
    print("Answering ", answer)
    return response
