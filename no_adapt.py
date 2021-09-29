import time
import math
import pickle
import numpy as np
import argparse
import scipy
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sched_solver import Solver
import cy_heuristics as heu
from util import Datasets, get_util_range

"""Adaptation없이 해당 구간에서 직접 바로 학습"""

parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=48)
parser.add_argument("--num_procs", type=int, default=6)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--num_train_dataset", type=int, default=100000)
parser.add_argument("--num_test_dataset", type=int, default=5000)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_step", type=int, default=100)
parser.add_argument("--use_deadline", action="store_true")
parser.add_argument("--range_l", type=str, default="4.60")
parser.add_argument("--range_r", type=str, default="4.60")
parser.add_argument("--use_cuda", action="store_true", default=True)

args = parser.parse_args()

confidence = 0.05
test_module = heu.test_RTA_LC
use_cuda = args.use_cuda

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


    def wrap(x):
        _sample, num_proc, use_deadline = x
        return heu.OPA(_sample, num_proc, None, use_deadline)


    with ProcessPoolExecutor(max_workers=10) as executor:
        inputs = []
        res_opa = np.zeros(len(test_dataset), dtype=int).tolist()
        for i, sample in test_dataset:
            inputs.append((sample, args.num_procs, args.use_deadline))
        for i, ret in tqdm(enumerate(executor.map(wrap, inputs))):
            res_opa[i] = ret
        opares = np.sum(res_opa)
    print("[before training][OPA generates %d]" % opares)

    rl_model = Solver(args.num_procs, args.embedding_size, args.hidden_size,
                      args.num_tasks, use_deadline=False, use_cuda=True)
    if args.use_cuda:
        rl_model.cuda()

    start = time.time()
    """Training Loop"""
    rl_model.train()
    # Make a baseline model
    bl_model = Solver(args.num_procs, args.embedding_size, args.hidden_size,
                      args.num_tasks, use_deadline=False, use_cuda=True)
    if args.use_cuda:
        bl_model.cuda()
    bl_model.load_state_dict(rl_model.state_dict())
    bl_model.eval()
    optimizer = optim.Adam(rl_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, gamma=0.9, last_epoch=-1)
    last_rl_model_sum = -1
    updates = 0
    noupdateinarow = 0
    _max = -1
    for epoch in range(args.num_epochs):
        loss_ = 0
        avg_hit = []
        for batch_idx, (_, sample_batch) in enumerate(train_loader):
            if use_cuda:
                sample_batch = sample_batch.cuda()
            num_samples = sample_batch.shape[0]
            optimizer.zero_grad()
            rewards, log_probs, action = rl_model(sample_batch)
            baseline, _bl_log_probs, _bl_action = bl_model(sample_batch, argmax=True)
            advantage = rewards - baseline
            if use_cuda:
                advantage = advantage.cuda()
            loss = -torch.sum((advantage * log_probs), dim=-1).mean()
            loss.backward()
            loss_ += loss.cpu().detach().numpy()
            avg_hit.append((rewards.cpu().detach().mean()))
            torch.nn.utils.clip_grad_norm_(rl_model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            updates += 1
            if use_cuda:
                diff = advantage.sum(dim=-1).detach().cpu().numpy()
            else:
                diff = advantage.sum(dim=-1).detach().numpy()
            D = diff.mean()
            S_D = 1e-10 + np.sqrt(((diff - D) ** 2).sum() / (1e-10 + num_samples - 1))
            tval = D / (S_D / (1e-10 + math.sqrt(1e-10 + num_samples)))
            p = scipy.stats.t.cdf(tval, num_samples)
            if (p >= 1. - 0.5 * confidence) or (p <= 0.5 * confidence):
                bl_model.load_state_dict(rl_model.state_dict())
            if updates % 100 == 0:
                end = time.time()
                rl_model.eval()
                ret = []
                for i, _batch in eval_loader:
                    if use_cuda:
                        _batch = _batch.cuda()
                    R, log_prob, actions = rl_model(_batch, argmax=True)
                    for j, chosen in enumerate(actions.cpu().numpy()):
                        order = np.zeros_like(chosen)
                        for k in range(args.num_tasks):
                            order[chosen[k]] = args.num_tasks - k - 1       # 중요할수록 숫자가 높다.
                        if use_cuda:
                            ret.append(test_module(_batch[j].cpu().numpy(), args.num_procs,
                                                   order, args.use_deadline, False))
                        else:
                            ret.append(test_module(_batch[j].numpy(), args.num_procs,
                                                   order, args.use_deadline, False))
                fname = "nonadaptRL-p%d-t%d-d%d-l[%s, %s]" % (args.num_procs, args.num_tasks,
                                                       int(args.use_deadline), args.range_l, args.range_r)
                rl_model_sum = np.sum(ret)

                elapsed = (end - start)
                minute = int(elapsed // 60)
                second = int(elapsed - 60 * minute)

                print("경과시간 : {}m {}s".format(minute, second))
                print("[consumed %d samples][at epoch %d][RL model generates %d][OPA generates %d]"
                      % (updates * args.batch_size, epoch, rl_model_sum, opares),
                      "log_probability\t", log_prob.cpu().detach().numpy().mean(), "avg_hit", np.mean(avg_hit))
                stop = False
                with open("log/nonadapt/" + fname, "a") as f:
                    print("[consumed %d samples][at epoch %d][RL model generates %d][OPA generates %d]"
                          % (updates * args.batch_size, epoch, rl_model_sum, opares),
                          "log_probability\t", log_prob.cpu().detach().numpy().mean(),
                          "avg_hit", np.mean(avg_hit), file=f)
                    if rl_model_sum == args.num_test_dataset:
                        print("total hit at epoch", epoch, file=f)
                        print("경과시간 : {}m {}s".format(minute, second), file=f)
                        print("total hit at epoch", epoch)
                        torch.save(rl_model, "../Pandamodels/nonadapt/" + fname + ".torchmodel")
                        print("SAVE SUCCESS")
                        stop = True

                    if rl_model_sum > _max:
                        noupdateinarow = 0
                        _max = rl_model_sum
                        torch.save(rl_model, "../Pandamodels/nonadapt/" + fname + ".torchmodel")
                        print("SAVE SUCCESS")
                    else:
                        noupdateinarow += 1
                    if noupdateinarow >= 20:
                        print("not update 20 times", epoch, file=f)
                        print("경과시간 : {}m {}s".format(minute, second), file=f)
                        print("not update m0 times", epoch)
                        torch.save(rl_model, "../Pandamodels/nonadapt/" + fname + ".torchmodel")
                        print("SAVE SUCCESS")
                        stop = True
                if stop:
                    raise NotImplementedError

                rl_model.train()
