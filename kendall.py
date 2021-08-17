"""
LINEAR TRAIN에서 복사ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ
"""
import numba
import pickle
import argparse
import math
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sched_solver import Solver
from sched import SchedT1Dataset
import cy_heuristics as heu
import sched_heuristic as py_heu
from sklearn.utils import shuffle
import scipy.stats
from fast_soft_sort.pytorch_ops import soft_rank, soft_sort
from linearsolver import LinearSolver
from scipy import stats

test_module = heu.test_RTA_LC
parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=32)
parser.add_argument("--num_procs", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=30000000)
parser.add_argument("--num_train_dataset", type=int, default=200000)
parser.add_argument("--num_test_dataset", type=int, default=256)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--lr", type=float, default=1.0 * 1e-2)
parser.add_argument("--lr_decay_step", type=int, default=100)
parser.add_argument("--use_deadline", action="store_true")
parser.add_argument("--range_l", type=str, default="3.20")
parser.add_argument("--range_r", type=str, default="3.20")
parser.add_argument("--use_cuda", action="store_true")

confidence = 0.05

args = parser.parse_args()
use_deadline = args.use_deadline
# test_module = heu.test_RTA_LC
use_cuda = True

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
            with open("tr/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                trsets.append(ts)
            with open("te/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
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
        batch_size = args.batch_size,
        shuffle = True,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    eval_loader = DataLoader(
        test_dataset,
        batch_size = args.batch_size,
        shuffle=False,
        pin_memory=True
    )


    def wrap(x):
        _sample, num_proc, use_deadline = x
        return heu.OPA(_sample, num_proc, None, use_deadline)

    with ProcessPoolExecutor(max_workers=4) as executor:
        inputs = []
        res_opa = np.zeros(len(test_dataset), dtype=int).tolist()
        for i, sample in test_dataset:
            inputs.append((sample, args.num_procs, use_deadline))
        for i, ret in tqdm(enumerate(executor.map(wrap, inputs))):
            res_opa[i] = ret
        opares = np.sum(res_opa)
    print("[before training][OPA generates %d]" % opares)

    temp_fname = "RL-p%d-t%d-d%d-l[%s, %s].torchmodel" % (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
    try:
        model = torch.load("models/" + temp_fname).cuda()
    except:
        raise AssertionError("Loading Error")

    rl_model = Solver(
        args.num_procs,
        args.embedding_size,
        args.hidden_size,
        args.num_tasks,
        use_deadline=False,
        use_cuda = True
    )
    rl_model.load_state_dict(model.state_dict())
    if args.use_cuda:
        model = model.cuda()
        rl_model = rl_model.cuda()

    rl_model = rl_model.eval()

    ret = []
    for i, _batch in eval_loader:
        if use_cuda:
            _batch = _batch.cuda()
        R, log_prob, actions = rl_model(_batch, argmax=True)
        for j, chosen in enumerate(actions.cpu().numpy()):
            order = np.zeros_like(chosen)
            for p in range(args.num_tasks):
                order[chosen[p]] = args.num_tasks - p - 1
            if use_cuda:
                ret.append(test_module(_batch[j].cpu().numpy(), args.num_procs, order, use_deadline, False))
            else:
                ret.append(test_module(_batch[j].numpy(), args.num_procs, order, use_deadline, False))
            rl_order = order

    print("[Before training][RL model generates %d]" % (np.sum(ret)))

    temp_fname = "LIN-p%d-t%d-d%d-l[%s, %s].torchmodel" % (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
    try :
        model = torch.load("linearmodels/" + temp_fname).cuda()
    except:
        raise AssertionError("Loading Error")
    linear_model = LinearSolver(args.num_procs, args.num_tasks,
                                args.use_deadline, args.use_cuda)

    linear_model.load_state_dict(model.state_dict())
    linear_model = linear_model.to("cuda:0")

    # EVALUATE
    linear_model.eval()
    lin_ret = []
    for i, _batch in eval_loader:
        if use_cuda:
            _batch = _batch.to("cuda:0")
        ev_linear_score = linear_model(_batch)

        _, ev_linear_score_idx = torch.sort(ev_linear_score, descending=True)
        np_linear_score = ev_linear_score_idx.cpu().detach().numpy()
        for j, chosen in enumerate(np_linear_score):
            order = np.zeros_like(chosen)
            for p in range(args.num_tasks):
                order[chosen[p]] = args.num_tasks - p - 1
            if use_cuda:
                lin_ret.append(test_module(_batch[j].cpu().numpy(), args.num_procs, order, use_deadline=False, ret_score=False))
            else:
                lin_ret.append(test_module(_batch[j].numpy(), args.num_procs, order, use_deadline, False))
            lin_order = order

    tau, p_value = stats.kendalltau(rl_order, lin_order)
    print(tau)
    print(p_value)