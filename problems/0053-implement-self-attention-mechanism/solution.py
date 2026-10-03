import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    dk = K.shape[-1]
    scores = (Q @ K.T) / torch.sqrt(torch.tensor(dk, dtype=Q.dtype))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output   
    
   