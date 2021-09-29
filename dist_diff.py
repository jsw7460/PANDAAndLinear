"""
LINEAR TRAIN에서 복사ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ
"""
import time
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import cy_heuristics as heu
from fast_soft_sort.pytorch_ops import soft_rank
from linearsolver import LinearSolver
from util import get_util_range, Datasets
from sched_solver import Solver

test_module = heu.test_RTA_LC
parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=32)
parser.add_argument("--num_procs", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument("--num_train_dataset", type=int, default=200000)
parser.add_argument("--num_test_dataset", type=int, default=50)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_step", type=int, default=100)
parser.add_argument("--use_deadline", action="store_true")
parser.add_argument("--range_l", type=str, default="3.00")
parser.add_argument("--range_r", type=str, default="3.00")
parser.add_argument("--use_cuda", action="store_true")
parser.add_argument("--load", type=int, default=-1)
parser.add_argument("--positive", action="store_true")

confidence = 0.05

args = parser.parse_args()
use_deadline = args.use_deadline
use_cuda = True

positive = False
DEBUG = False
if DEBUG:
    positive = True

fname = "LIN-p%d-t%d-d%d-l[%s, %s]" % (
            args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)


def kl_div(n_step):
    util_range = get_util_range(args.num_procs)

    trsets = []
    tesets = []
    on = False
    for util in util_range:
        on = False
        if util == args.range_l:
            on = True
        if on:
            if positive:
                load_file_name = "../Pandadata/tr/%d-%d/positive/%s"
            else:
                load_file_name = "../Pandadata/tr/%d-%d/%s"
            with open(load_file_name % (args.num_procs, args.num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                trsets.append(ts)
            with open("../Pandadata/te/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                tesets.append(ts)
        if util == args.range_r:
            break

    train_dataset = Datasets(trsets)
    test_dataset = Datasets(tesets)
    train_dataset.setlen(args.num_train_dataset)
    test_dataset.setlen(args.num_test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    eval_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    temp_fname = "localRL-p%d-t%d-d%d-l[%s, %s].torchmodel" % \
                 (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
    model = torch.load("../Pandamodels/localrlmodels/" + temp_fname).cuda()

    rl_model = Solver(
        args.num_procs,
        args.embedding_size,
        args.hidden_size,
        args.num_tasks,
        use_deadline=False,
        use_cuda=True, ret_embedded_vector=True,
    )
    rl_model.load_state_dict(model.state_dict())
    if use_cuda:
        model = model.cuda()
        rl_model = rl_model.cuda()

    rl_model = rl_model.eval()

    if use_cuda:
        rl_model = rl_model.to("cuda:0")

    ss = np.array(list(range(32)))
    ss2 = np.array(list(reversed(range(32))))
    guide = torch.LongTensor(np.array([ss, ss2], dtype=np.int32)).cuda()

    for epoch in range(args.num_epochs):
        loss_ = 0
        avg_hit = []
        for batch_idx, (_, sample_batch) in enumerate(train_loader):
            sample_batch = sample_batch[:2, :, :]
            _, actions, distributions = rl_model(sample_batch, guide=guide)
            break
        break

    a = distributions[0][5].detach().cpu().numpy()
    b = distributions[1][5].detach().cpu().numpy()

    print(0.5 * np.sum(np.abs(a - b)))

    exit(0)
    kl_div = 0
    KL_calc = torch.nn.KLDivLoss(reduction="batchmean")
    actions = actions.squeeze()




    # Timestep +1
    if n_step == 1:
        for t in range(args.num_tasks - 1):
            previous_distribution = distributions[t].squeeze()
            sampled_task = actions[t]
            previous_distribution[sampled_task] = 0
            # renormalized_distribution = previous_distribution
            renormalized_distribution = torch.log(previous_distribution / torch.sum(previous_distribution))

            next_distribution = distributions[t+1]

            kl_div += KL_calc(renormalized_distribution, next_distribution)
            # kl_div += torch.nn.KLDivLoss(size_average=False)(renormalized_distribution, next_distribution)
        return kl_div / (args.num_tasks-1)


    # Timestep +3
    elif n_step == 3:
        for t in range(args.num_tasks - 3):
            prev_distribution = distributions[t].squeeze()
            first_sampled_mask = actions[t]
            second_sampled_mask = actions[t+1]
            third_sampled_mask = actions[t+2]
            prev_distribution[first_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)
            prev_distribution[second_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)

            prev_distribution = prev_distribution.detach().cpu().numpy()

            # renormalized_distribution = torch.log(prev_distribution)
            rl_next_distribution = distributions[t+2]
            rl_next_distribution = rl_next_distribution.detach().cpu().numpy()
            kl_div += np.sum(np.abs(prev_distribution - rl_next_distribution))
            # kl_div += KL_calc(renormalized_distribution, rl_next_distribution)
        return kl_div / (args.num_tasks-3)


    # Timestep +5
    else:
        for t in range(args.num_tasks - 5):
            
            prev_distribution = distributions[t].squeeze()
            first_sampled_mask = actions[t]
            second_sampled_mask = actions[t+1]
            third_sampled_mask = actions[t+2]
            fourth_sampled_mask = actions[t+3]
            fifth_sampled_mask = actions[t+4]

            prev_distribution[first_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)
            prev_distribution[second_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)
            prev_distribution[third_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)
            prev_distribution[fourth_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)
            prev_distribution[fifth_sampled_mask] = 0
            prev_distribution = prev_distribution / torch.sum(prev_distribution)
            renormalized_distribution = torch.log(prev_distribution)
            rl_next_distribution = distributions[t+5]
            kl_div += KL_calc(renormalized_distribution, rl_next_distribution)
        return kl_div / (args.num_tasks - 5)

if __name__ == "__main__":
    store = []
    n_step = 3
    for t in range(10):
        x = kl_div(n_step)
        store.append(x.item())

    store = np.array(store)
    mean = np.mean(store)
    std = np.sqrt(np.var(store)) / np.sqrt(100)
    print("mean:{}, std:{}".format(mean, std))
    with open("kl/kl"+str(n_step), "w") as f:
        print("mean:{}, std:{}".format(mean, std), file=f)


