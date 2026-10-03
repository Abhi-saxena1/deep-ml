import torch
import torch.nn.functional as F
from typing import Tuple

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Q = torch.matmul(X,W_q)
    K = torch.matmul(X,W_k)
    V = torch.matmul(X,W_v)
    return Q, K , V
   

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    dk = K.shape[-1]
    scores = Q @ K.T/torch.sqrt(torch.tensor(dk, dtype = Q.dtype))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output
   
    

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:

    d_model = Q.shape[-1]
    dk = (d_model // n_heads)
    Q_heads = Q.reshape(Q.shape[0], n_heads, dk).permute(1, 0, 2)
    K_heads = K.reshape(K.shape[0], n_heads, dk).permute(1, 0, 2)
    V_heads = V.reshape(V.shape[0], n_heads, dk).permute(1, 0, 2)
   
    outputs = []

    for h in range(n_heads):

        output = self_attention(

            Q_heads[h],
            K_heads[h],
            V_heads[h]
        )
        outputs.append(output)
    
   

    return torch.cat(outputs, dim = -1)
    
