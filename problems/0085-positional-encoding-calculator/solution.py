import torch

def pos_encoding(position: int, d_model: int):
    if position <= 0 or d_model <= 0:
        return -1

    encoding = []

    for pos in range(position):
        row = []

        for i in range(d_model):
            angle = pos / (10000 ** ((i // 2) * 2 / d_model))

            if i % 2 == 0:
                value = torch.sin(torch.tensor(angle))
            else:
                value = torch.cos(torch.tensor(angle))

            row.append(value)

        encoding.append(torch.stack(row))

    return torch.stack(encoding).to(torch.float16)
    
    



    