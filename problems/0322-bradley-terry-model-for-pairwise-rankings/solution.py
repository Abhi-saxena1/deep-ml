import torch
from typing import List, Tuple

def fit_bradley_terry(
    comparisons: List[Tuple[int, int]],
    n_items: int,
    learning_rate: float = 0.5,
    n_iterations: int = 100
) -> torch.Tensor:

    beta = torch.zeros(n_items)

    for i in range(n_iterations):

        gradient = torch.zeros(n_items)

        for winner, loser in comparisons:

            difference = beta[winner] - beta[loser]
            probability = torch.sigmoid(difference)
            error = 1 - probability

            gradient[winner] += error
            gradient[loser] -= error

        beta += learning_rate * gradient

        beta -= beta.mean()

    return beta

    


                
    
