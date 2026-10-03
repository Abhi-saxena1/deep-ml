import torch
def rmsnorm(x: torch.Tensor, g: torch.Tensor, eps: float = 1e-5):
    n = x.shape[-1]
    squared = torch.square(x)
    mean = torch.mean(squared, dim=-1, keepdim=True)
    rms = torch.sqrt(mean + eps)

    m = (x / rms) * g

    return m