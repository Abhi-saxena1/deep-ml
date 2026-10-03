import torch

def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10000.0
) -> torch.Tensor:

    seq_len, d = x.shape

    x_even = x[:, 0::2]
    x_odd = x[:, 1::2]

    i = torch.arange(d // 2, dtype=x.dtype)

    theta = 1 / (base ** (2 * i / d))

    angle = positions.to(x.dtype).unsqueeze(1) * theta.unsqueeze(0)

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    result = torch.empty_like(x)

    result[:, 0::2] = rotated_even
    result[:, 1::2] = rotated_odd

    return result