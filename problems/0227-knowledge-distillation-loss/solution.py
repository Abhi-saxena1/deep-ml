import torch
import torch.nn.functional as F

def distillation_loss(
	student_logits: torch.Tensor,
	teacher_logits: torch.Tensor,
	temperature: float = 1.0
) -> torch.Tensor:
    
    p = torch.softmax(teacher_logits / temperature, dim=-1)
    q = torch.softmax(student_logits / temperature, dim=-1)

    kl = torch.sum(p * torch.log(p / q))

    loss = (temperature ** 2) * kl
	return loss
