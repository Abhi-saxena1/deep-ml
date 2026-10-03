import torch

def layer_normalization(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:

    n =X.shape[-1]
    mean = torch.mean(X, dim=-1, keepdim=True)
    variance = torch.mean(torch.square(X - mean),dim =-1 , keepdim = True)
    norm = (X-mean)/torch.sqrt(variance + epsilon)   
    ss = gamma*norm + beta
    return ss

