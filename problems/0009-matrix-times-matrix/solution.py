
import torch

def matrixmul(a, b) -> torch.Tensor:
    a = torch.as_tensor(a, dtype=torch.float)
    b = torch.as_tensor(b, dtype=torch.float)

    if a.size(1) != b.size(0):
        return torch.tensor(-1)

    return a @ b