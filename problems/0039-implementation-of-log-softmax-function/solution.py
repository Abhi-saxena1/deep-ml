import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    x = torch.tensor(scores, dtype =torch.float)
    x = x - torch.max(x)
    exp_x = torch.exp(x)
    ls = x -  torch.log(torch.sum(exp_x))
    return ls

