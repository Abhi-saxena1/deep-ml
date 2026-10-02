import torch

def learned_positional_encoding(token_embeddings: torch.Tensor, position_embedding_table: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    l = token_embeddings.shape[1]
   
    h =  token_embeddings +position_embedding_table[start_pos:start_pos+l]
    return h
    