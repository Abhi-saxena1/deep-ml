import torch

def sigmoid(z: float) -> float:
  sg = 1/(1+(torch.exp(torch.tensor(-z))))
  return sg.item()
    
   
