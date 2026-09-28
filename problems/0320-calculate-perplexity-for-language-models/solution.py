import torch

def calculate_perplexity(probabilities: list[float]) -> float:
    p = torch.tensor(probabilities)
    x = torch.log(p)
    y = torch.sum(x)
    z = p.numel()
    y = y*-1/z
    y = torch.exp(y)

    return y.item()
    
   