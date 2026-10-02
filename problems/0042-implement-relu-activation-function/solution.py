import torch

def relu(z: float) -> torch.Tensor:
    if z>= 0:
        return torch.tensor(z)
    else :
        return torch.tensor(0.0)
    