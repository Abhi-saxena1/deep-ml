import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
  x = torch.tensor(scores, dtype= torch.float)
  
  x = x - torch.max(x)
  exp_x = torch.exp(x)
  prob = exp_x/torch.sum(exp_x)
  return prob.tolist()


 
  




