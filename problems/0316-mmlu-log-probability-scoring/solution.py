import torch

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    l = torch.tensor(log_probs)
    y = torch.tensor(correct_answers)
    predictions = torch.argmax(l, dim=1)
    accuracy = torch.mean((predictions == y).float())
    
    rows = torch.arange(len(y))
    log_normalizer = torch.logsumexp(l, dim=1)
    z = l[rows, y]
    z = torch.exp(z - log_normalizer)
    z = torch.mean(z)
        
    
    return {
    'accuracy': accuracy.item(),
    'predictions':predictions.tolist(),
    'avg_correct_prob': z.item()
}







