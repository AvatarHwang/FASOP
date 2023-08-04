"""
Portions of this code adapted from the 'AMP' project (https://github.com/DachengLi1/AMP). 
@article{li2022amp,
  title={AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness},
  author={Li, Dacheng and Wang, Hongyi and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2210.07297},
  year={2022}
}
"""

import time

import os
import numpy as np

import torch

from amp_utils import amp_no_placement_strategy
from estimate import FASOP
from device_placement import device_placement, get_all_cluster_combinations, device_placement_all
from model_config import get_model_config

import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--heterogeneous", action='store_true', help="True if you want to run heterogeneous experiments (default: False)")
parser.add_argument("--type", type=str, default="gpt2XL")
parser.add_argument("--gpu_per_node", type=int, default=4)
parser.add_argument("--precision", type=int, default=16)
parser.add_argument("--iter", type=int, default=12_500_000, help="number of iterations for each experiment (default: 1)")
parser.add_argument("--pareto", action='store_true', help="True if you want to run pareto experiments (default: False)")
parser.add_argument("--add_exp_name", type=str, default="")
parser.add_argument("--exhaustive", action='store_true', help="True if you want to run exhaustive search for model partitioning (default: False)")
args = parser.parse_args()

if args.exhaustive:
    print("type parallelization strategy you want to search")
    mbs_exhaustive = int(input("mbs: "))
    gpu_type_list_exhaustive = input("gpu_type_list (example:A100,A10,A10,A10) :").split(',')
    tp_exhaustive = int(input("tp: "))
    dp_exhaustive = int(input("dp: "))
    pp_exhaustive = int(input("pp: "))


if args.pareto and args.gpu_per_node ==4 :
    print("Pareto experiments should use 8 GPUs per node, so we will use 8 GPUs per node")
    args.gpu_per_node = 8

time_s = time.time()
# number of GPU per node, number of nodes
gpu_per_node = args.gpu_per_node

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'tdpp/FASOP/main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)


cluster_info = {} # a100:4 : a10:28    8 x nodes

A100 = [torch.tensor([40 * 1e9]).float(), torch.tensor([1840 * 1e9]).float()]
A10 = [torch.tensor([40 * 1e9]).float(), torch.tensor([252 * 1e9]).float()]

if args.pareto:
    A100 = [torch.tensor([400 * 1e9]).float(), torch.tensor([1840 * 1e9]).float()]
    A10 = [torch.tensor([100 * 1e9]).float(), torch.tensor([252 * 1e9]).float()]

cluster_combinations = get_all_cluster_combinations(args.type, args.pareto, args.heterogeneous)
want_simulate = [] 

for cluster_info in cluster_combinations:
    num_node = len(cluster_info.keys())
    
    n_a100 = 0
    for i in range(num_node):
        if cluster_info[i] == '1':
            n_a100+=1
    n_a10 = num_node - n_a100

    D = device_placement(n_a100, n_a10)
    #if args.pareto is False and args.heterogeneous is True:
    #    assert len(D) != 1, "Stochastic bug: try to run the code few more times!"

    model_config, gbs, exp_name = get_model_config(args.type, args.precision, args.heterogeneous, args.pareto)
    exp_name = exp_name + args.add_exp_name

    # remove cache directory from last run
    if os.path.exists(os.path.join(home_path, "tmp")):
        for root, dirs, files in os.walk(os.path.join(home_path, "tmp")):
            for f in files:
                os.unlink(os.path.join(root, f))

    for d in D:
        print(d)
        node_type = []
        for i in range(len(d)):
            if d[i] == 'A':
                node_type.append('p4d.24xlarge')
                n_a100+=1
                d[i] = A100
                
            elif d[i] == 'B':
                node_type.append('g5.24xlarge')
                d[i] = A10
            else:
                assert False, "Unknown node type"

        model = FASOP(model_config, exp_name, A100, A10, len(cluster_info))
               
        known = None

        # Estimating best configurations
        while True:
            ret = amp_no_placement_strategy(M=gpu_per_node, N=num_node, gbs=gbs, known=known, num_layers=model_config["num_layers"])
            if ret is None:
                break
            else:
                h, w, mbs, known = ret
                parallel_dim = {"tp_deg": torch.ones(1,)*h, "dp_deg": torch.ones(1,)*w, "pp_deg": torch.ones(1,)*(gpu_per_node*num_node/(h*w))}
                fake_config = np.ones((gpu_per_node,num_node)) * (-1)
                model_args = (fake_config, gbs, mbs, d, model_config, parallel_dim)
                if args.exhaustive:
                    mbs = mbs_exhaustive
                    gpu_type_list = gpu_type_list_exhaustive
                    tp = tp_exhaustive
                    dp = dp_exhaustive
                    pp = pp_exhaustive
                    parallel_dim = {"tp_deg": torch.ones(1,)*int(tp), "dp_deg": torch.ones(1,)*int(dp), "pp_deg": torch.ones(1,)*int(pp)} 
                    model_args = (fake_config, gbs, mbs, d, model_config, parallel_dim)
                    exhaustive_args = {"exhaustive": True, "gpu_type_lst": gpu_type_list}
                else:
                    exhaustive_args = {"exhaustive": False, "gpu_type_lst": None}

                with torch.no_grad():
                    rank_map, partition, cost, pipecost, dp_side_cost, all_reduce_embedding_cost, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem = model(model_args, node_type, exhaustive_args)
                
                for k in parallel_dim:
                    parallel_dim[k] = int(parallel_dim[k].item())

                price_per_sec_1 = 32.7726 / 3600
                price_per_sec_2 = 8.144 / 3600
                if args.pareto:
                    price_per_sec_2 = 9.773 / 3600

                price_per_sec = price_per_sec_1*n_a100 + price_per_sec_2 * n_a10
                price_per_step = price_per_sec * cost.item() # price per second * second per step 
                pretrain_cost = price_per_step * args.iter
                want_simulate.append((mbs, h, w,(gpu_per_node*num_node/(h*w)), node_type, partition, cost.item(), pipecost.item(), dp_side_cost.item(), all_reduce_embedding_cost, price_per_step, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem, pretrain_cost))

print(f"Finished {time.time() - time_s}")

sorted_settings = sorted(want_simulate, key = lambda kv: kv[6])

df = pd.DataFrame(sorted_settings, columns = ['mbs','tp','dp','pp','node placement','partition','estimated time (step/s)','pipeline time','DP all-reduce time','embedding layer all-reduce time','price_per_step','is_oom','oom_gpumem','is_zero_oom','zerooom_gpumem', 'train_cost'])

# first remove existing csv file
if os.path.exists(f"{os.path.join(dir_path, exp_name)}.csv"):
    os.remove(f"{os.path.join(dir_path, exp_name)}.csv")
df.to_csv(f"{os.path.join(dir_path, exp_name)}.csv", index=False)

print("csv file saved at: ", f"{os.path.join(dir_path, exp_name)}.csv")