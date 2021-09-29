"""
LINEAR TRAIN에서 복사ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ
"""
import re

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from sorting import SortingSolver
from rl_with_attention import GraphEmbedding, AttentionModule, Glimpse, Pointer

EMBEDDING_SIZE = 128
BATCH_SIZE = 1
N = 256
FILENAME = "sorting-%d" % N

def kl_div(n_step):
    Solver = SortingSolver(8, EMBEDDING_SIZE, EMBEDDING_SIZE, N).to("cuda:0")  # Panda 모델
    with open("../sorting/" + FILENAME, "rb") as f:
        tmp = torch.load(f)
    Solver.load_state_dict(tmp.state_dict())

    input = np.zeros((N, 8))
    for i in range(N):
        x = "{0:b}".format(i)
        x = x.zfill(8)
        y = re.findall(r"\d", x)
        z = [int(_) for _ in y]
        input[i] = np.array(z)
    input = torch.from_numpy(input)
    input = input.to("cuda:0").float()

    rs = []

    x = torch.zeros((BATCH_SIZE, N, 8), device="cuda:0")
    for i in range(BATCH_SIZE):
        r = torch.randperm(N)
        rs.append(r)
        x[i] = input[r]
    x = x[:, :16, :]
    print(x)
    r = torch.stack(rs, dim=0).cuda()
    log_probs, actions, rl_distributions = Solver(x, ret_distributions=True)
    ret = torch.gather(r, 1, actions)

    actions = actions.squeeze()
    KL_calc = torch.nn.KLDivLoss()
    kl_divergence = 0

    if n_step == 1:
        for t in range(len(rl_distributions) - 1):
            # prev_distribution = rl_distributions[t].squeeze()
            # first_sampled_mask = actions[t]
            #
            # prev_distribution[first_sampled_mask] = 0
            # prev_distribution = prev_distribution / torch.sum(prev_distribution)
            # renormalized_distribution = torch.log(prev_distribution)
            #
            # next_distribution = rl_distributions[t+1]

            prev_distribution = rl_distributions[t].squeeze()
            first_sampled_mask = actions[t]
            prev_distribution[first_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)


            renormalized_distribution = torch.log(prev_distribution)

            print(renormalized_distribution, prev_distribution)
            next_distribution = rl_distributions[t+1]
            # print(renormalized_distribution, next_distribution)
            kl_divergence += KL_calc(renormalized_distribution, next_distribution)

        return kl_divergence / (len(rl_distributions)-1)

    if n_step == 3:
        for t in range(len(rl_distributions) - 3):
            prev_distribution = rl_distributions[t].squeeze()
            first_sampled_mask = actions[t]
            second_sampled_mask = actions[t+1]
            third_sampled_mask = actions[t+2]

            prev_distribution[first_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)
            prev_distribution[second_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)
            prev_distribution[third_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)

            # renormalized_distribution = torch.log(prev_distribution)

            next_distribution = rl_distributions[t+3]
            kl_divergence += KL_calc(torch.log(next_distribution), prev_distribution)
            # kl_divergence += KL_calc(renormalized_distribution, next_distribution)

        return kl_divergence / (len(rl_distributions) - 3)


if __name__ == "__main__":
    ret = 0
    for i in range(10):
        ret += kl_div(1)
    print(ret / 10)
