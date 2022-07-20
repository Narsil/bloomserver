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


for gen_step in range(20):
    for layer in range(70):
        for part_name, parts in PARTS.items():
            for name in parts:
                try:
                    A = np.load(f"{gen_step}_python_{part_name}_{name}_{layer}.npy")
                    B = np.load(f"{gen_step}_rust_{part_name}_{name}_{layer}.npy")
                except Exception:
                    continue
                print("gen_step", gen_step, "part", part_name, "Name ", name, "layer", layer, np.allclose(A, B))
                if not  np.allclose(A, B):
                    import ipdb;ipdb.set_trace()
