







import torch

def lora_forward(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:

    r = A.shape[0]

    W_lora = W + (alpha / r) * (B @ A)

    y = x @ W_lora

    return y