import torch
import torch.nn as nn

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi ={
            '<BOS>':0,
            "<EOS>":1
        }
        for i ,ch in enumerate(chars):
            self.stoi[ch] = i + 2
        self.itos={}
        for token, index in self.stoi.items():
            self.itos[index] = token
        self.vocab_size= len(self.stoi)
        self.embedding = nn.Embedding(
            self.vocab_size,16
        )
    def encode(self ,text: str) -> torch.Tensor:
        indices = [self.stoi['<BOS>']]
        for ch in text:
            
            indices.append(self.stoi[ch])
        indices.append(self.stoi["<EOS>"])
        return torch.tensor(indices,dtype = torch.long)
    def decode(self, indices:torch.Tensor)-> str:
        tokens =[]
        for index in indices:

            tokens.append(self.itos[int(index)])
        return "".join(tokens)   
            
            
            