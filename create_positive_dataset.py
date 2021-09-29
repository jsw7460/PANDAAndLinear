import argparse
import pickle
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import cy_heuristics as heu
from sched_solver import Solver
from util import get_util_range, Datasets

test_module = heu.test_RTA_LC
parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=48)
parser.add_argument("--num_procs", type=int, default=6)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument("--num_train_dataset", type=int, default=2000)
parser.add_argument("--num_test_dataset", type=int, default=200)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--lr", type=float, default=1.0 * 1e-2)
parser.add_argument("--lr_decay_step", type=int, default=100)
parser.add_argument("--use_deadline", action="store_true")
parser.add_argument("--range_l", type=str, default="4.60")
parser.add_argument("--range_r", type=str, default="4.60")

confidence = 0.05

args = parser.parse_args()
use_deadline = args.use_deadline
use_cuda = False


if __name__ == "__main__":
    util_range = get_util_range(args.num_procs)

    trsets = []
    on = False
    for util in util_range:
        on = False
        if util == args.range_l:
            on = True
        if on:
            with open("../Pandadata/tr/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
                ts = pickle.load(f)
                trsets.append(ts)
        if util == args.range_r:
            break

    train_dataset = Datasets(trsets)
    train_dataset.setlen(args.num_train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False
    )

    temp_fname = "localRL-p%d-t%d-d%d-l[%s, %s].torchmodel" % \
                 (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
    model = torch.load("../Pandamodels/localrlmodels/" + temp_fname)

    rl_model = Solver(
        args.num_procs,
        args.embedding_size,
        args.hidden_size,
        args.num_tasks,
        use_deadline=False,
        use_cuda=False
    )

    rl_model.load_state_dict(model.state_dict())
    rl_model = rl_model.eval()
    positive_sample_idx = []
    # for step, (i, sample) in enumerate(train_loader):
    #     with torch.no_grad():
    #         _, _, actions = model(sample, argmax=True)
    #     for j, chosen in enumerate(actions.cpu().numpy()):
    #         order = np.zeros_like(chosen)
    #         for p in range(args.num_tasks):
    #             order[chosen[p]] = args.num_tasks - p - 1
    #         result = test_module(sample[j].numpy().squeeze(), args.num_procs, order, use_deadline, False)
    #         if result:
    #             positive_sample_idx.append(step * args.batch_size + j)
    #
    # positive_data = np.vstack([train_dataset.data_set[idx][np.newaxis, :] for idx in positive_sample_idx])
    # save_dir = "../Pandadata/tr/%d-%d/positive" % (args.num_procs, args.num_tasks)

    # Make label
    rl_label = []
    for batch_idx, (_, sample_batch) in enumerate(train_loader):
        print("Making %dth label...%d" % (batch_idx, (batch_idx + 1) * args.batch_size / args.num_train_dataset))
        with torch.no_grad():
            rewards, probs, action = model(sample_batch)
        rl_order = torch.zeros_like(action, dtype=torch.float)
        for i in range(rl_order.size(0)):  # batch size
            for j in range(rl_order.size(1)):  # num_tasks
                rl_order[i][action[i][j]] = args.num_tasks - j
        rl_order = rl_order.cpu().numpy()
        rl_label.append(rl_order)

    rl_label_numpy = np.vstack([batch for batch in rl_label])       # 이걸 저장해야 함
    save_dir = "../Pandadata/tr/%d-%d" % (args.num_procs, args.num_tasks)
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        pass
    with open(save_dir + "/" + args.range_l + "label", "wb") as f:
        pickle.dump(rl_label_numpy, f)
    # print("Total {} positive sample saved".format(len(positive_sample_idx)))
