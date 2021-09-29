import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from functools import wraps

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import cy_heuristics as heu
from linearsolver import LinearSolver
from linearsolver import sample_gumbel, get_rank
from sched_solver import Solver

test_module = heu.test_RTA_LC
parser = argparse.ArgumentParser()

parser.add_argument("--num_tasks", type=int, default=32)
parser.add_argument("--num_procs", type=int, default=4)
parser.add_argument("--num_test_dataset", type=int, default=2)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--use_deadline", action="store_true")
parser.add_argument("--range_l", type=str, default="3.10")
parser.add_argument("--range_r", type=str, default="3.10")
parser.add_argument("--use_cuda", action="store_true")

confidence = 0.05

args = parser.parse_args()
use_deadline = args.use_deadline

SAVING_FILE_NAME = "p%d-t%d-d%d-l[%s, %s]" \
                   % (args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)

NUM_TEST = 100
NET_COMPARE = False

def wrap(x):
    _sample, num_proc, use_deadline = x
    return heu.OPA(_sample, num_proc, None, use_deadline)


def dm_wrap(x):
    _sample, num_proc, use_deadline = x
    return heu.test_RTA_LC(_sample, num_proc, 1, use_deadline)


def timer(func):
    @wraps(func)
    def wrapper(*args):
        start = time.time()
        result = func(*args)
        end = time.time()
        print("Function {name}, Time : {time:.3f} with result {result}"
              .format(name=func.__name__, time=end-start, result=result))
        return result
    return wrapper


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


# @timer
def test_heu(eval_dataset, mode="OPA", ignore=False):
    if ignore:
        return 0
    with ProcessPoolExecutor(max_workers=10) as executor:
        inputs = []
        res_opa = np.zeros(len(eval_dataset), dtype=int).tolist()
        for i, sample in eval_dataset:
            inputs.append((sample, args.num_procs, use_deadline))
        for i, ret in tqdm(enumerate(executor.map(wrap, inputs))):
            res_opa[i] = ret
        opares = np.sum(res_opa)
    return opares


# @timer
def test_dm(eval_dataset):
    # with ProcessPoolExecutor(max_workers=1) as executor:
    inputs = []
    res_dm = np.zeros(len(eval_dataset), dtype=int).tolist()
    for i, sample in eval_dataset:
        inputs.append((sample, args.num_procs, use_deadline))
        # print(sample)
    # print(inputs[0])
    dm_wrap(inputs[0])
    # print("run")
    # for i, ret in tqdm(enumerate(executor.map(dm_wrap, inputs))):
    #     res_dm[i] = ret
    # operas = np.sum(res_dm)
    # return operas


# @timer
def test_gumbel(model, eval_loader, gumbel_number):
    ret = []
    val = 0
    for i, batch in eval_loader:
        with torch.no_grad():
            linear_score = model(batch, normalize=True)
        gumbel_score = sample_gumbel(linear_score, sampling_number=gumbel_number)
        gumbel_rank = get_rank(gumbel_score)  # [batch_size x num_gumbel_sample x num_tasks]
        for j, order in enumerate(gumbel_rank):  # j : ~batch size
            for k, orderd in enumerate(gumbel_rank[j]):  # k : ~ num_gumbel_sample
                x = test_module(batch[j].numpy(), args.num_procs, orderd, False, False)
                if x == 1:
                    val += 1
                    break
                else:
                    continue
    return val


# @timer
def test_global_reinforce(model, eval_loader):
    ret = []
    for i, batch in eval_loader:
        with torch.no_grad():
            _, _, actions = model(batch, argmax=True)
        for j, chosen in enumerate(actions.cpu().numpy()):
            order = np.zeros_like(chosen)
            for p in range(args.num_tasks):
                order[chosen[p]] = args.num_tasks - p - 1
            ret.append(test_module(batch[j].numpy(), args.num_procs, order, use_deadline, False))
    return sum(ret)


# @timer
def test_reinforce(model, eval_loader, ignore=False):
    if ignore:
        return 0
    ret = []
    for i, batch in eval_loader:
        with torch.no_grad():
            _, _, actions = model(batch, argmax=True)
        for j, chosen in enumerate(actions.cpu().numpy()):
            order = np.zeros_like(chosen)
            for p in range(args.num_tasks):
                order[chosen[p]] = args.num_tasks - p - 1
            ret.append(test_module(batch[j].numpy(), args.num_procs, order, use_deadline, False))
    return sum(ret)


def test_reinforce_sampling(model, eval_loader, ignore=False):
    if ignore:
        return 0
    ret = 0
    for i, batch in eval_loader:
        with torch.no_grad():
            actions = model(batch, argmax=False, multisampling=True)    # [batch_size x Num_sampling x seq_len]
        for idx in range(actions.size(0)):      # idx ~ batch_size
            for j, chosen in enumerate(actions[idx]):       # j ~ Num_sampling
                order = np.zeros_like(chosen)
                for p in range(args.num_tasks):
                    order[chosen[p]] = args.num_tasks - p - 1
                success = test_module(batch[idx].numpy(), args.num_procs, order, use_deadline, False)
                if success:
                    ret += 1
                    break

    return ret


# @timer
def test_distillation(model, eval_loader):
    ret = []
    for i, batch in eval_loader:
        with torch.no_grad():
            score = model(batch).detach().numpy()
        argsort = np.argsort(-score)
        for j, chosen in enumerate(argsort):
            order = np.zeros_like(chosen).squeeze()
            for p in range(args.num_tasks):
                order[chosen[p]] = args.num_tasks - p - 1
            ret.append(test_module(batch[j].numpy(), args.num_procs, order, use_deadline, False))
    return sum(ret)


util_range = get_util_range(args.num_procs)
tesets = []
on = False
for util in util_range:
    on = False
    if util == args.range_l:
        on = True
    if on:
        with open("../Pandadata/te/%d-%d/%s" % (args.num_procs, args.num_tasks, util), 'rb') as f:
            ts = pickle.load(f)
            tesets.append(ts)
    if util == args.range_r:
        break


def main(netcompare=False):
    test_dataset = Datasets(tesets)
    test_dataset.setlen(args.num_test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False
    )

    if netcompare:
        """SRD"""
        srd_results = []
        print("LINEAR")
        dist_file_name = "LIN-p%d-t%d-d%d-l[%s, %s]" % (
            args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
        Distillation = LinearSolver(args.num_procs, args.num_tasks,
                                    args.use_deadline, False)
        with open("../Pandamodels/linearmodels/" + dist_file_name + ".torchmodel", "rb") as f:
            tmp = torch.load(f, map_location=torch.device("cpu"))
        Distillation.load_state_dict(tmp.state_dict())
        Distillation.cpu()
        Distillation.eval()
        srd_value = test_distillation(Distillation, test_loader)
        srd_results.append(srd_value)
        print()

        """Ranknet model"""
        rknet_results = []
        print("RANKNET")
        rknet_file_name = "RKNET-p%d-t%d-d%d-l[%s, %s]" % (
            args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r
        )
        RanknetModel = LinearSolver(args.num_procs, args.num_tasks,
                                   args.use_deadline, False)
        with open("../Pandamodels/rknetmodels/" + rknet_file_name + ".torchmodel", "rb") as f:
            tmp = torch.load(f, map_location=torch.device("cpu"))
        RanknetModel.load_state_dict(tmp.state_dict())
        RanknetModel.eval()
        rknet_value = test_distillation(RanknetModel, test_loader)
        rknet_results.append(rknet_value)
        print()

        """Listnet model"""
        listnet_results = []
        print("LISTNET")
        listnet_file_name = "LISTNET-p%d-t%d-d%d-l[%s, %s]" % (
            args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r
        )
        listnet_model = LinearSolver(args.num_procs, args.num_tasks,
                                     args.use_deadline, False)
        with open("../Pandamodels/listnetmodels/" + listnet_file_name + ".torchmodel", "rb") as f:
            tmp = torch.load(f, map_location=torch.device("cpu"))
        listnet_model.load_state_dict(tmp.state_dict())
        listnet_model.eval()
        listnet_value = test_distillation(listnet_model, test_loader)
        listnet_results.append(listnet_value)
        print()

        return srd_value, rknet_value, listnet_value

    if not netcompare:
        opa_time, rl_time, rl_sampling_time, srd_time, g3_time, g5_time, g7_time, g10_time = 0, 0, 0, 0, 0, 0, 0, 0
        # """Heuristic Test"""
        heu_results = []
        print("OPA")
        start = time.time()
        heu_val = test_heu(test_dataset, "OPA", ignore=True)
        end = time.time()
        opa_time = end - start
        heu_results.append(heu_val)
        print()

        """Reinforcement Learning Model Test"""
        print("Local REINFORCE")
        rl_results = []
        rl_file_name = "localRL-p%d-t%d-d%d-l[%s, %s]" % (
            args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
        RLModel = Solver(args.num_procs, args.embedding_size, args.hidden_size,
                         args.num_tasks, use_deadline=False, use_cuda=False).cpu()
        with open("../Pandamodels/localrlmodels/" + rl_file_name + ".torchmodel", "rb") as f:
            tmp = torch.load(f)
        RLModel.load_state_dict(tmp.state_dict())
        RLModel.eval()
        start = time.time()
        rl_value = test_reinforce(RLModel, test_loader, ignore=True)
        end = time.time()
        rl_time = end-start
        rl_results.append(rl_value)
        print()

        """RL - Sampling Test"""
        print("RL-Sampling")
        rl_sampling_results = []
        start = time.time()
        rl_sampling_value = test_reinforce_sampling(RLModel, test_loader, ignore=True)
        print(rl_sampling_value)
        end = time.time()
        rl_sampling_time = end-start
        rl_sampling_results.append(rl_sampling_value)
        print()

        # """SRD Model Test"""
        srd_results = []
        print("LINEAR")
        dist_file_name = "LIN-p%d-t%d-d%d-l[%s, %s]" % (
            args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
        Distillation = LinearSolver(args.num_procs, args.num_tasks,
                                   args.use_deadline, False)
        with open("../Pandamodels/linearmodels/" + dist_file_name + ".torchmodel", "rb") as f:
            tmp = torch.load(f)
        Distillation.load_state_dict(tmp.state_dict())
        Distillation.cpu()
        Distillation.eval()
        start = time.time()
        srd_value = test_distillation(Distillation, test_loader)
        end=time.time()
        srd_time = end-start
        srd_results.append(srd_value)
        print()


        """Ranknet Model Test"""
        # print("RANKNET")
        # rknet_file_name = "RKNET-p%d-t%d-d%d-l[%s, %s]" % (
        #     args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r
        # )
        # RKModel = LinearSolver(args.num_procs, args.num_tasks,
        #                            args.use_deadline, False)
        # with open("../Pandamodels/rknetmodels/" + rknet_file_name + ".torchmodel", "rb") as f:
        #     tmp = torch.load(f)
        # RKModel.load_state_dict(tmp.state_dict())
        # RKModel.eval()
        # test_distillation(RKModel, test_loader)
        # print()

        """GumbelSearch Model Test"""
        print("GUMBELSEARCH")

        start = time.time()
        gum_3_value = test_gumbel(Distillation, test_loader, 3)
        end = time.time()
        g3_time = end-start

        start = time.time()
        gum5_value = test_gumbel(Distillation, test_loader, 5)
        end = time.time()
        g5_time = end-start

        start = time.time()
        gum7_value = test_gumbel(Distillation, test_loader, 7)
        end = time.time()
        g7_time = end-start

        gum_results = []
        start = time.time()
        gum_10_value = test_gumbel(Distillation, test_loader, 10)
        end = time.time()
        g10_time = end-start
        gum_results.append(gum_10_value)
        print()

        start = time.time()
        gum_15_value = test_gumbel(Distillation, test_loader, 15)
        end=time.time()
        g15_time = end-start

        start = time.time()
        gum_20_value = test_gumbel(Distillation, test_loader, 20)
        end = time.time()
        g20_time = end-start

        start = time.time()
        gum_30_value = test_gumbel(Distillation, test_loader, 30)
        end = time.time()
        g30_time = end-start

        # return opa_time, rl_time, rl_sampling_time, srd_time, g3_time, g5_time, g7_time, g10_time
        values = (heu_val, rl_value, rl_sampling_value, srd_value, gum_3_value, gum5_value, gum7_value, gum_10_value, gum_15_value, gum_20_value, gum_30_value)
        times = (opa_time, rl_time, rl_sampling_time, srd_time, g3_time, g5_time, g7_time, g10_time, g15_time, g20_time, g30_time)
        return values, times


def pca():
    test_dataset = Datasets(tesets)
    test_dataset.setlen(args.num_test_dataset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False
    )

    rl_file_name = "localRL-p%d-t%d-d%d-l[%s, %s]" % (
        args.num_procs, args.num_tasks, int(use_deadline), args.range_l, args.range_r)
    RLModel = Solver(args.num_procs, args.embedding_size, args.hidden_size,
                     args.num_tasks, use_deadline=False, use_cuda=False, ret_embedded_vector=True).cpu()
    with open("../Pandamodels/localrlmodels/" + rl_file_name + ".torchmodel", "rb") as f:
        tmp = torch.load(f)
    RLModel.load_state_dict(tmp.state_dict())
    RLModel.eval()
    for i, batch in test_loader:
        with torch.no_grad():
            _, actions, embedded_vectors = RLModel(batch)
        break

    embedded_vectors = embedded_vectors.numpy()
    actions = actions.detach().cpu().numpy()

    with open("pca/panda_emb", "wb") as f:
        pickle.dump(embedded_vectors, f)
    with open("pca/panda_actions", "wb") as f:
        pickle.dump(actions, f)


if __name__ == "__main__":
    if NET_COMPARE:
        srd_results, ranknet_results, listnet_results = [], [], []
        for i in range(NUM_TEST):
            srd_value, ranknet_value, listnet_value = main(NET_COMPARE)
            srd_results.append(srd_value)
            ranknet_results.append(ranknet_value)
            listnet_results.append(listnet_value)

        srd_results = np.array(srd_results)
        srd_avg = np.mean(srd_results)
        srd_std = np.sqrt(np.var(srd_results)) / np.sqrt(NUM_TEST)

        ranknet_results = np.array(ranknet_results)
        ranknet_avg = np.mean(ranknet_results)
        ranknet_std = np.sqrt(np.var(ranknet_results)) / np.sqrt(NUM_TEST)

        listnet_results = np.array(listnet_results)
        listnet_avg = np.mean(listnet_results)
        listnet_std = np.sqrt(np.var(listnet_results)) / np.sqrt(NUM_TEST)

        print("SRD :      {}, STD :  {:.2f}".format(srd_avg, srd_std))
        print("RANKNET :  {}, STD :  {:.2f}".format(ranknet_avg, ranknet_std))
        print("LISTNET :  {}, STD :  {:.2f}".format(listnet_avg, listnet_std))

        with open("netcompare/" + SAVING_FILE_NAME, "a") as f:
            print("SRD :      {}, STD :  {:.2f}".format(srd_avg, srd_std), file=f)
            print("RANKNET :  {}, STD :  {:.2f}".format(ranknet_avg, ranknet_std), file=f)
            print("LISTNET :  {}, STD :  {:.2f}".format(listnet_avg, listnet_std), file=f)

    else:
        OPA_v, RL_v, RLS_v, SRD_v, G3_v, G5_v, G7_v, G10_v, G15_v, G20_v, G30_v = [], [], [], [], [], [], [], [], [], [], []
        OPA_t, RL_t, RLS_t, SRD_t, G3_t, G5_t, G7_t, G10_t, G15_t, G20_t, G30_t = [], [], [], [], [], [], [], [], [], [], []
        for i in range(NUM_TEST):
            values, times = main()
            opav, rlv, rlsv, srdv, g3v, g5v, g7v, g10v, g15v, g20v, g30v = values
            opat, rlt, rlst, srdt, g3t, g5t, g7t, g10t, g15t, g20t, g30t = times

            OPA_v.append(opav)
            OPA_t.append(opat)

            RL_v.append(rlv)
            RL_t.append(rlt)

            RLS_v.append(rlsv)
            RLS_t.append(rlst)

            SRD_v.append(srdv)
            SRD_t.append(srdt)

            G3_v.append(g3v)
            G3_t.append(g3t)

            G5_v.append(g5v)
            G5_t.append(g5t)

            G7_v.append(g7v)
            G7_t.append(g7t)

            G10_v.append(g10v)
            G10_t.append(g10t)

            G15_v.append(g15v)
            G15_t.append(g15t)
            G20_v.append(g20v)
            G20_t.append(g20t)

            G30_v.append(g30v)
            G30_t.append(g30t)

        OPA_v_std = np.sqrt(np.var(np.array(OPA_v)))
        OPA_t_std = np.sqrt(np.var(np.array(OPA_t)))

        RL_v_std = np.sqrt(np.var(np.array(RL_v)))
        RL_t_std = np.sqrt(np.var(np.array(RL_t)))

        RLS_v_std = np.sqrt(np.var(np.array(RLS_v)))
        RLS_t_std = np.sqrt(np.var(np.array(RLS_t)))

        SRD_v_std = np.sqrt(np.var(np.array(SRD_v)))
        SRD_t_std = np.sqrt(np.var(np.array(SRD_t)))

        G3_v_std = np.sqrt(np.var(np.array(G3_v)))
        G3_t_std = np.sqrt(np.var(np.array(G3_t)))

        G5_v_std = np.sqrt(np.var(np.array(G5_v)))
        G5_t_std = np.sqrt(np.var(np.array(G5_t)))

        G7_v_std = np.sqrt(np.var(np.array(G7_v)))
        G7_t_std = np.sqrt(np.var(np.array(G7_t)))

        G10_v_std = np.sqrt(np.var(np.array(G10_v)))
        G10_t_std = np.sqrt(np.var(np.array(G10_t)))

        G15_v_std = np.sqrt(np.var(np.array(G15_v)))
        G15_t_std = np.sqrt(np.var(np.array(G15_t)))

        G20_v_std = np.sqrt(np.var(np.array(G20_v)))
        G20_t_std = np.sqrt(np.var(np.array(G20_t)))

        G30_v_std = np.sqrt(np.var(np.array(G30_v)))
        G30_t_std = np.sqrt(np.var(np.array(G30_t)))

        with open("result/0911/" + SAVING_FILE_NAME, "a") as f:
            print("OPA(v)   : avg:{}, m:{}, M:{}, std:{}".format(sum(OPA_v) / NUM_TEST, min(OPA_v), max(OPA_v), OPA_v_std),
                  file=f)
            print("RL(v)    : avg:{}, m:{}, M:{}, std:{}".format(sum(RL_v) / NUM_TEST, min(RL_v), max(RL_v), RL_v_std),
                  file=f)
            print("RLS(v)   : avg:{}, m:{}, M:{}, std:{}".format(sum(RLS_v) / NUM_TEST, min(RLS_v), max(RLS_v), RLS_v_std),
                  file=f)
            print("SRD(v)   : avg:{}, m:{}, M:{}, std:{}".format(sum(SRD_v) / NUM_TEST, min(SRD_v), max(SRD_v), SRD_v_std),
                  file=f)
            print("GUM_3(v) : avg:{}, m:{}, M:{}, std:{}".format(sum(G3_v) / NUM_TEST, min(G3_v), max(G3_v), G3_v_std),
                  file=f)
            print("GUM_5(v) : avg:{}, m:{}, M:{}, std:{}".format(sum(G5_v) / NUM_TEST, min(G5_v), max(G5_v), G5_v_std),
                  file=f)
            print("GUM_7(v) : avg:{}, m:{}, M:{}, std:{}".format(sum(G7_v) / NUM_TEST, min(G7_v), max(G7_v), G7_v_std),
                  file=f)
            print("GUM_10(v): avg:{}, m:{}, M:{}, std:{}" .format(sum(G10_v)/NUM_TEST, min(G10_v), max(G10_v), G10_v_std),
                  file=f)
            print("GUM_15(v): avg:{}, m:{}, M:{}, std:{}".format(sum(G15_v) / NUM_TEST, min(G15_v), max(G15_v),
                                                                 G15_v_std),
                  file=f)
            print("GUM_20(v): avg:{}, m:{}, M:{}, std:{}".format(sum(G20_v) / NUM_TEST, min(G20_v), max(G20_v),
                                                                 G20_v_std),
                  file=f)
            print("GUM_30(v): avg:{}, m:{}, M:{}, std:{}".format(sum(G30_v) / NUM_TEST, min(G30_v), max(G30_v),
                                                                 G30_v_std),
                  file=f)
            print("\n", file=f)
            print("OPA(t)   : avg:{}, m:{}, M:{}, std:{}".format(sum(OPA_t) / NUM_TEST, min(OPA_t), max(OPA_t), OPA_t_std),
                  file=f)
            print("RL(t)    : avg:{}, m:{}, M:{}, std:{}".format(sum(RL_t) / NUM_TEST, min(RL_t), max(RL_t), RL_t_std),
                  file=f)
            print("RLS(t)   : avg:{}, m:{}, M:{}, std:{}".format(sum(RLS_t) / NUM_TEST, min(RLS_t), max(RLS_t), RLS_t_std),
                  file=f)
            print("SRD(t)   : avg:{}, m:{}, M:{}, std:{}".format(sum(SRD_t) / NUM_TEST, min(SRD_t), max(SRD_t), SRD_t_std),
                  file=f)
            print("GUM_3(t) : avg:{}, m:{}, M:{}, std:{}".format(sum(G3_t) / NUM_TEST, min(G3_t), max(G3_t), G3_t_std),
                  file=f)
            print("GUM_5(t) : avg:{}, m:{}, M:{}, std:{}".format(sum(G5_t) / NUM_TEST, min(G5_t), max(G5_t), G5_t_std),
                  file=f)
            print("GUM_7(t) : avg:{}, m:{}, M:{}, std:{}".format(sum(G7_t) / NUM_TEST, min(G7_t), max(G7_t), G7_t_std),
                  file=f)
            print("GUM_10(t): avg:{}, m:{}, M:{}, std:{}".format(sum(G10_t) / NUM_TEST, min(G10_t), max(G10_t), G10_t_std),
                  file=f)
            print("GUM_15(t): avg:{}, m:{}, M:{}, std:{}".format(sum(G15_t) / NUM_TEST, min(G15_t), max(G15_t),
                                                                 G15_t_std),
                  file=f)
            print("GUM_20(t): avg:{}, m:{}, M:{}, std:{}".format(sum(G20_t) / NUM_TEST, min(G20_t), max(G20_t),
                                                                 G20_t_std),
                  file=f)
            print("GUM_30(t): avg:{}, m:{}, M:{}, std:{}".format(sum(G30_t) / NUM_TEST, min(G30_t), max(G30_t),
                                                                 G30_t_std),
                  file=f)