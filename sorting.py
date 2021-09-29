import re

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from scipy.stats import wasserstein_distance
from rl_with_attention import GraphEmbedding, AttentionModule, Glimpse, Pointer

EMBEDDING_SIZE = 128
BATCH_SIZE = 256
N = 256

FILENAME = "sorting-%d" % N

class SortingSolver(nn.Module):
    def __init__(self, input_dim, embedding_size, hidden_size, seq_len, n_head=4,
             c=10, ret_embedded_vector=False):
        super(SortingSolver, self).__init__()
        self.ret_embedded_vector = ret_embedded_vector
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.n_head = n_head
        self.C = c
        self.embedding = GraphEmbedding(input_dim, embedding_size)
        self.mha = AttentionModule(embedding_size, n_head, hidden_size)

        self.init_w = nn.Parameter(torch.Tensor(self.embedding_size))
        self.init_w.data.uniform_(-0.1, 0.1)
        self.glimpse = Glimpse(self.embedding_size, self.hidden_size, self.n_head)
        self.pointer = Pointer(self.embedding_size, self.hidden_size, 1, self.C)

        self.h_context_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.v_weight_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.h_query_embed = nn.Linear(self.embedding_size, self.embedding_size)
        self.memory_transform = nn.Linear(self.embedding_size, self.embedding_size)
        self.chosen_transform = nn.Linear(self.embedding_size, self.embedding_size)
        self.h1_transform = nn.Linear(self.embedding_size, self.embedding_size)
        self.h2_transform = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, inputs, argmax=False, ret_distributions=False):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        embedded, h, h_mean, h_bar, chosen_vector, left_vector, query = self._prepare(inputs)

        prev_chosen_indices = []
        prev_chosen_logprobs = []
        cumulated_distributions = []
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        for index in range(seq_len):
            i = index
            _, n_query = self.glimpse(query, h, mask)
            prob, _ = self.pointer(n_query, h, mask)        # [batch size x num_tasks]
            cumulated_distributions.append(prob)
            # cumulated_distributions.append(self.pointer(n_query, h, mask, ret_score=True))
            cat = Categorical(prob)
            if argmax:
                _, chosen = torch.max(prob, -1)
            chosen = cat.sample()               # [batch_size].
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)

            mask[[i for i in range(batch_size)], chosen] = True

            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_size)
            chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_vector = chosen_vector + self.chosen_transform(chosen_hs)
            left_vector = left_vector - self.memory_transform(chosen_hs)
            h1 = self.h1_transform(torch.tanh(chosen_vector))
            h2 = self.h2_transform(torch.tanh(left_vector))
            v_weight = self.v_weight_embed(chosen_hs)
            query = self.h_query_embed(h1 + h2 + v_weight)
        if ret_distributions:
            return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1), cumulated_distributions
        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)

    def fow2(self, inputs, argmax=False, ret_distributions=False, chosens=None):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        embedded, h, h_mean, h_bar, chosen_vector, left_vector, query = self._prepare(inputs)

        prev_chosen_indices = []
        prev_chosen_logprobs = []
        cumulated_distributions = []
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        for index in range(seq_len):
            i = index
            _, n_query = self.glimpse(query, h, mask)
            prob, _ = self.pointer(n_query, h, mask)  # [batch size x num_tasks]
            cumulated_distributions.append(prob)
            cat = Categorical(prob)
            # if argmax:
            #     _, chosen = torch.max(prob, -1)
            # chosen = cat.sample()  # [batch_size].
            chosen = chosens[:, i]
            logprobs = cat.log_prob(chosen)
            prev_chosen_indices.append(chosen)
            prev_chosen_logprobs.append(logprobs)

            mask[[i for i in range(batch_size)], chosen] = True

            cc = chosen.unsqueeze(1).unsqueeze(2).repeat(1, 1, self.embedding_size)
            chosen_hs = h.gather(1, cc).squeeze(1)
            chosen_vector = chosen_vector + self.chosen_transform(chosen_hs)
            left_vector = left_vector - self.memory_transform(chosen_hs)
            h1 = self.h1_transform(torch.tanh(chosen_vector))
            h2 = self.h2_transform(torch.tanh(left_vector))
            v_weight = self.v_weight_embed(chosen_hs)
            query = self.h_query_embed(h1 + h2 + v_weight)
        if ret_distributions:
            return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1), torch.stack(cumulated_distributions, dim=1)
        return torch.stack(prev_chosen_logprobs, 1), torch.stack(prev_chosen_indices, 1)

    def _prepare(self, inputs):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        embedded = self.embedding(inputs)
        h = self.mha(embedded)
        h_mean = h.mean(dim=1)
        h_bar = self.h_context_embed(h_mean)
        v_weight = self.v_weight_embed(self.init_w)
        chosen_vector = torch.zeros((batch_size, self.embedding_size))
        chosen_vector = chosen_vector.cuda()
        left_vector = self.memory_transform(h).sum(dim=1)
        h1 = self.h1_transform(torch.tanh(chosen_vector)) #H_o
        h2 = self.h2_transform(torch.tanh(left_vector)) #H_l
        query = self.h_query_embed(h1 + h2 + v_weight) #c
        return embedded, h, h_mean, h_bar, chosen_vector, left_vector, query


def main1():
    Solver = SortingSolver(8, EMBEDDING_SIZE, EMBEDDING_SIZE, N).to("cuda:0")  # Panda 모델

    # for name, param in Solver.named_parameters():
    #     if name.startswith("embedding") or name.startswith("mha"):
    #         pass
    #     else:
    #         param.requires_grad = False
    #     print(param.requires_grad)

    optimizer = torch.optim.Adam(Solver.parameters(), lr=1e-4)

    input = np.zeros((N, 8))
    for i in range(N):
        x = "{0:b}".format(i)
        x = x.zfill(8)
        y = re.findall(r"\d", x)
        z = [int(_) for _ in y]
        input[i] = np.array(z)
    input = torch.from_numpy(input)
    input = input.to("cuda:0").float()

    for epoch in range(100000):
        # print(epoch)
        rs = []
        x = torch.zeros((BATCH_SIZE, N, 8), device="cuda:0")
        for i in range(BATCH_SIZE):
            r = torch.randperm(N)
            rs.append(r)
            x[i] = input[r]
        x = x[:, :7, :]
        r = torch.stack(rs, dim=0).cuda()
        log_probs, actions = Solver(x)
        rewards = torch.zeros_like(actions, dtype=torch.float32)  # [BATCH_SIZE x N]
        ret = torch.gather(r, 1, actions)

        # rewards[:, 0] = actions[:, 0]
        # for j in range(15):
        #     rewards[:, j+1] = actions[:, j+1] - actions[:, j]

        rewards = torch.mean((ret[:, 1:] > ret[:, :-1]).float(), -1)
        rewards = rewards.unsqueeze(-1)

        loss = -torch.sum(((rewards - rewards.detach().mean()) * log_probs), dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(loss)
            print(actions)
            print(ret[0])
            with open("../sorting/" + FILENAME, "wb") as f:
                torch.save(Solver, f)

def main2():
    Solver = torch.load("../sorting/" + FILENAME).cuda()
    input = np.zeros((256, 8))
    for i in range(256):
        x = "{0:b}".format(i)
        x = x.zfill(8)
        y = re.findall(r"\d", x)
        z = [int(_) for _ in y]
        input[i] = np.array(z)
    input = torch.from_numpy(input)
    input = input.to("cuda:0").float()
    #
    rs = []
    x = torch.zeros((1, N, 8), device="cuda:0")
    for i in range(1):
        r = torch.randperm(N)
        rs.append(r)
        x[i] = input[r]
    x = x[:, :7, :]
    print(r[:7])
    # r = torch.stack(rs, dim=0).cuda()
    # # x = [1 x 16 x 8]
    x = x.repeat(3, 1, 1)
    ss1 = [0, 1, 2, 3, 4, 5, 6]
    ss2 = [2, 0, 1, 3, 4, 5, 6]
    ss3 = [6, 0, 1, 3, 4, 5, 2]
    chosens = torch.LongTensor(np.array([ss1, ss2, ss3], dtype=np.int32)).cuda()
    log_probs, actions, probs = Solver.fow2(x, chosens=chosens, ret_distributions=True)
    probs = probs.detach().cpu().numpy()
    print(probs[0][1])
    print(probs[2][1])
    a = np.random.uniform(0, 1, 100)
    a = a / a.sum()
    b = np.random.uniform(0, 1, 100)
    b = b / b.sum()
    print(0.5 * np.sum(np.abs(a - b)))
    print(0.5 * np.sum(np.abs((probs[0][1] - probs[1][1]))))


main2()