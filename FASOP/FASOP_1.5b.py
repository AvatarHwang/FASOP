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
from device_placement import device_placement, get_all_cluster_combinations
from model_config import get_model_config

import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--heterogeneous", action='store_true', help="True if you want to run heterogeneous experiments (default: False)")
parser.add_argument("--type", type=str, default="gpt2XL")
parser.add_argument("--gpu_per_node", type=int, default=4)
parser.add_argument("--precision", type=int, default=16)
parser.add_argument("--pareto", action='store_true', help="True if you want to run pareto experiments (default: False)")
args = parser.parse_args()

if args.pareto and args.gpu_per_node ==4 :
    print("Pareto experiments should use 8 GPUs per node")
    print("would you like to continue? (y/n)")
    ans = input()
    if ans == 'n':
        exit()
    if ans == 'y':
        args.gpu_per_node = 8

time_s = time.time()
# number of GPU per node, number of nodes
gpu_per_node = args.gpu_per_node

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'tdpp/FASOP/main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)


cluster_info = {} # a100:4 : a10:28    8 x nodes

A100 = [torch.tensor([40 * 1e9]).float(), torch.tensor([230 * 8 * 1e9]).float()] # HGX NVLink 600GB/s -> 230 GB/s
A10 = [torch.tensor([40 * 1e9]).float(), torch.tensor([252 * 1e9]).float()]

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
    print(f"type D {type(D)}")
    
    if args.pareto is False and args.heterogeneous is True:
        assert len(D) != 1, "Stochastic bug: try to run the code few more times!"

    model_config, gbs, exp_name = get_model_config(args.type, args.precision, args.heterogeneous, args.pareto)

    # remove cache directory from last run
    if os.path.exists(os.path.join(home_path, "tmp")):
        for root, dirs, files in os.walk(os.path.join(home_path, "tmp")):
            for f in files:
                os.unlink(os.path.join(root, f))

    for d in D:
        print(d)
        node_type = []
        if n_a100>0:
            for i in range(num_node):
                if d[i] == 1:
                    node_type.append('p4d.24xlarge')
                    n_a100+=1
                else:
                    node_type.append('g5.24xlarge')
            for i in range(len(d)):
                if d[i] == 0:
                    d[i] = A10
                else:
                    d[i] = A100
        else:
            for i in range(num_node):
                node_type.append('g5.24xlarge')
                d[i] = A10
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

                with torch.no_grad():
                    rank_map, partition, cost, pipecost, dp_side_cost, all_reduce_embedding_cost, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem = model(model_args, node_type)
                
                for k in parallel_dim:
                    parallel_dim[k] = int(parallel_dim[k].item())

                if d[0][1] == torch.tensor([4800 * 1e9]).float():
                    price_per_s_1 = 32.7726 / 3600
                else:
                    price_per_s_1 = 5.672 / 3600
                price_per_s_2 = 5.672 / 3600

                price_per_s = price_per_s_1 + price_per_s_2 * (num_node - 1)
                price_per_step = price_per_s * cost.item() # price per second * second per step 
                want_simulate.append((mbs, h, w,(gpu_per_node*num_node/(h*w)), node_type, partition, cost.item(), pipecost.item(), dp_side_cost.item(), all_reduce_embedding_cost, price_per_step, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem))

print(f"Finished {time.time() - time_s}")

sorted_settings = sorted(want_simulate, key = lambda kv: kv[6])

df = pd.DataFrame(sorted_settings, columns = ['mbs','tp','dp','pp','node placement','partition','estimated time (step/s)','pipeline time','DP all-reduce time','embedding layer all-reduce time','price_per_step','is_oom','oom_gpumem','is_zero_oom','zerooom_gpumem'])

# first remove existing csv file
if os.path.exists(f"{os.path.join(dir_path, exp_name)}.csv"):
    os.remove(f"{os.path.join(dir_path, exp_name)}.csv")
df.to_csv(f"{os.path.join(dir_path, exp_name)}.csv", index=False)

print("csv file saved at: ", f"{os.path.join(dir_path, exp_name)}.csv")