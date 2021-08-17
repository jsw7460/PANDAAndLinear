import numba
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.utils import shuffle
from sched_solver import Solver
import cy_heuristics as heu
from fast_soft_sort.pytorch_ops import soft_rank
from linearsolver import LinearSolver
from linearsolver import sample_gumbel, get_rank
import time

test_module = heu.test_RTA_LC
parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=48)
parser.add_argument("--num_procs", type=int, default=6)
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--num_train_dataset", type=int, default=200000)
parser.add_argument("--num_test_dataset", type=int, default=5000)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--lr", type=float, default=1.0 * 1e-2)
parser.add_argument("--lr_decay_step", type=int, default=100)
parser.add_argument("--use_deadline", action="store_true")
parser.add_argument("--range_l", type=str, default="4.50")
parser.add_argument("--range_r", type=str, default="4.50")
parser.add_argument("--use_cuda", action="store_true")

confidence = 0.05

args = parser.parse_args()
use_deadline = args.use_deadline


def get_util_range(num_proc):
    util = [str(x) for x in range(10, num_proc * 100, 10)]
    ret = []
    for x in util:
        if len(x) == 2:
            ret.append('0.' + x)
        else:
            ret.append(x[:len(x) - 2] + '.' + x[len(x) - 2:])
    return ret

class Datasets(Dataset):
    def __init__(self, l):
        super(Datasets, self).__init__()
        ret = []
        le = []
        for dd in l:
            ret.append(dd.data_set)
        self.data_set = np.vstack(ret)

    def setlen(self, newlen):
        self.data_set = shuffle(self.data_set)
        self.data_set = self.data_set[:newlen]

    def __len__(self):
        return self.data_set.shape[0]

    def __getitem__(self, idx):
        return idx, self.data_set[idx]

if __name__ == "__main__":
    util_range = get_util_range(args.num_procs)

    trsets = []
    tesets = []
    on = False
    for util in util_range:
        on = False
        if util == args.range_l:
            on = True
        if on:
            with open("../Pandadata/tr/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
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

    dist_file_name = "LIN-p%d-t%d-d%d-l[%s, %s]" % (
            args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)

    Distilation = LinearSolver(args.num_procs, args.num_tasks,
                               args.use_deadline, False)

    with open("../Pandamodels/linearmodels/" + dist_file_name + ".torchmodel", "rb") as f:
        tmp = torch.load(f)
    Distilation.load_state_dict(tmp.state_dict())
    Distilation.eval()

    start = time.time()
    distil_ret = []
    for i, batch in eval_loader:
        linear_score = Distilation(batch)
        gumbel_score = sample_gumbel(linear_score, sampling_number=5)      # [batch_size x num_gumbel_sample x num_tasks]
        gumbel_rank = get_rank(gumbel_score)        # [batch_size x num_gumbel_sample x num_tasks]
        val = 0
        for j, order in enumerate(gumbel_rank):     # j : ~batch size
            tmp_ret = []
            for k, order in enumerate(gumbel_rank[j]):        # k : ~ num_gumbel_sample
                tmp_ret.append(
                    test_module(batch[j].numpy(), args.num_procs, order, False, False)
                )
            # print(tmp_ret)
            val += max(tmp_ret)
        distil_ret.append(val)
    end = time.time()
    print("Gumbel sampling. Hit :", sum(distil_ret), end - start)

    rl_file_name = "RL-p%d-t%d-d%d-l[%s, %s]" % (
            args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)

    RLModel = Solver(
        args.num_procs,
        args.embedding_size,
        args.hidden_size,
        args.num_tasks,
        use_deadline=False,
        use_cuda=False
    )
    RLModel = RLModel.to("cpu")
    with open("../Pandamodels/rlmodels/" + rl_file_name + ".torchmodel", "rb") as f:
        tmp = torch.load(f)
    RLModel.load_state_dict(tmp.state_dict())
    RLModel.eval()

    SAMPLING_NUMBER = 1
    ret = []

    rlstart = time.time()
    for i, batch in eval_loader:
        _ret = []
        for _ in range(SAMPLING_NUMBER):
            store = []
            _, _, actions = RLModel(batch, argmax=True)
            for j, chosen in enumerate(actions.cpu().numpy()):
                order = np.zeros_like(chosen)
                for p in range(args.num_tasks):
                    order[chosen[p]] = args.num_tasks - p - 1
                store.append(test_module(batch[j].cpu().numpy(), args.num_procs, order, use_deadline, False))
            _ret.append(sum(store))
        ret.append(max(_ret))

    rlend = time.time()

    print("LIN TIME  : {:.3f}, RL TIME : {:.3f}, LIN GENERATES {}, RL GENERATES {}".format(
        end-start, rlend-rlstart, sum(distil_ret), sum(ret)
    ))
    print("RANGE : {}".format(args.range_l))


