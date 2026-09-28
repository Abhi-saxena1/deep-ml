import torch

def elo_rating_update(ratings: dict, matches: list, k_factor: float) -> dict:
    for model_a, model_b, result in matches:
        RA = ratings[model_a] 
        RB = ratings[model_b] 
        EA = 1 / (1+10**( (RB - RA)/400))
        EB =  1 - EA
        if result == "a":



            SA = 1
            SB = 0

        elif result == "b":

            SA = 0
            SB = 1

        else:

            SA = 0.5
            SB = 0.5
        RA = RA + k_factor * (SA - EA)
        RB = RB + k_factor * (SB - EB)
        ratings[model_a] = torch.tensor(RA)
        ratings[model_b] = torch.tensor(RB)
       
    return ratings


   
    