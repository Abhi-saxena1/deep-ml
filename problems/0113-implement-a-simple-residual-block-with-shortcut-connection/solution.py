import torch

def residual_block(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    z= w1 @ x
    z= torch.relu(z)
    y = torch.relu(w2 @ z +x)
    return y

    