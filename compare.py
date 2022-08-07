import numpy as np
import torch


PARTS = {
    "baddbmm": [
        "sliced_alibi",
        "query_layer",
        "key_layer",
        "value_layer",
        "beta",
        "alpha",
        "matmul_result",
    ],
    "softmax": ["attention_mask", "attention_probs"],
    "bmm": ["value_layer", "attention_probs_reshaped", "context_layer", "context_layer2", "dense", "residual", "dropout"],
    "mlp": ["init", "gelu", "output"],
}


for gen_step in range(20):
    for layer in range(70):
        for part_name, parts in PARTS.items():
            for name in parts:
                try:
                    A = np.load(f"tensors/{gen_step}_python_{part_name}_{name}_{layer}.npy")
                    B = np.load(f"tensors/{gen_step}_rust_{part_name}_{name}_{layer}.npy")
                except Exception:
                    continue
                print("gen_step", gen_step, "part", part_name, "Name ", name, "layer", layer, np.allclose(A, B, rtol=1e-8, atol=1e-10) and A.shape == B.shape)
                if A.shape != B.shape:
                    import ipdb;ipdb.set_trace()
                if not  np.allclose(A, B, rtol=1e-8, atol=1e-10):
                    import ipdb;ipdb.set_trace()
                # torch.testing.assert_close(torch.from_numpy(A), torch.from_numpy(B))
                try:
                    torch.testing.assert_close(torch.from_numpy(A), torch.from_numpy(B))
                except Exception as e:
                    print(e)
