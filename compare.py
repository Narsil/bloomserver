import numpy as np
import torch


PARTS = {
    "baddbmm": [
        "sliced_alibi",
        "query_layer",
        "key_layer",
        "beta",
        "alpha",
        "matmul_result",
    ],
    "softmax": ["attention_probs"],
    "bmm": ["context_layer", "context_layer2", "dense", "residual", "dropout"],
    "mlp": ["init", "gelu", "output"],
}


for layer in range(70):
    for part_name, parts in PARTS.items():
        for name in parts:
            A = np.load(f"python_{part_name}_{name}_{layer}.npy")
            B = np.load(f"rust_{part_name}_{name}_{layer}.npy")
            print("part", part_name, "Name ", name, "layer", layer, np.allclose(A, B))
            if not  np.allclose(A, B):
                import ipdb;ipdb.set_trace()
