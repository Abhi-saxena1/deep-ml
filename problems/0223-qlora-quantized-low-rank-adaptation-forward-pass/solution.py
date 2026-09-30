import torch

def qlora_forward(
    x: torch.Tensor,
    quantized_W: torch.Tensor,
    scale: float,
    zero_point: float,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:

    r = A.shape[0]
    wd = quantized_W * scale + zero_point
    y = x @ wd + (alpha/r)* x @ B @ A

    return y 

    