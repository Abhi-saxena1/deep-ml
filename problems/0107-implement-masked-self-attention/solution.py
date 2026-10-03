import torch

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q,K,V

    
def masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dk= K.shape[-1]
    scores = (Q @ K.T)/(dk ** 0.5)
    sm = scores + mask
    a = torch.softmax(sm, dim = -1)
    output = a @ V
    return output
    