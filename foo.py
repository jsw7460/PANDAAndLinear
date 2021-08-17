import torch
from fast_soft_sort.pytorch_ops import soft_rank
x = torch.tensor([
    [2.4, 3.0, 1.3]
])

y = soft_rank(x, regularization_strength=100)
print(y)