import torch


x = torch.tensor([
    [1, 2, 3],
    [-1, -2, -3]
]).unsqueeze(1).repeat(1, 2, 1)

print(x.view(4, 3))