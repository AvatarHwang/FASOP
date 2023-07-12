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
from estimate import FASOP, EstimatePeakMemory

import argparse

from device_placement import device_placement

parser = argparse.ArgumentParser()
parser.add_argument("--gbs", type=int, default=64)
parser.add_argument("--exp_name", type=str, default="het_cluster")
parser.add_argument("--model_config", type=str, default="gpt2XL")
parser.add_argument("--hidden_size", type=int, default=1600)
parser.add_argument("--sequence_length", type=int, default=1024)
parser.add_argument("--num_layers", type=int, default=48)
parser.add_argument("--vocab_size", type=int, default=50257)
parser.add_argument("--type", type=str, default="gpt2XL")
parser.add_argument("--gpu_per_node", type=int, default=4)
parser.add_argument("--num_attention_heads", type=int, default=16)
parser.add_argument("--precision", type=int, default=16)
parser.add_argument("--iter", type=int, default=12_500_000)
parser.add_argument("--budget", type=int, default=600_000_000_000)
parser.add_argument("--num_a100_node", type=int, default=4)
parser.add_argument("--num_a10_node", type=int, default=4)
args = parser.parse_args()

time_s = time.time()
# number of GPU per node, number of nodes
gpu_per_node = args.gpu_per_node

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'tdpp/FASOP/main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

a100_node_comm = [torch.tensor([400 * 1e9]).float(), torch.tensor([4800 * 1e9]).float()]
a10_node_comm = [torch.tensor([100 * 1e9]).float(), torch.tensor([252 * 1e9]).float()]

cluster_combinations=[]
num_c = 0
for num_a100 in range(0, args.num_a100_node+1):
    for num_a10 in range(0, args.num_a10_node+1):
        cluster = {}
        for i in range(num_a100+num_a10):
            cluster[i] = '0'
        for i in range(num_a100):
            cluster.update({i:'1'})
        if len(cluster.keys())>0:
            cluster_combinations.append(cluster)
        num_c += 1

print(f"Number of clusters combinations: {num_c}")

want_simulate = [] 

exp_name = f"pareto"
record_file = f"{os.path.join(dir_path, exp_name)}.txt"


model_config = {"hidden_size": torch.tensor([int(args.hidden_size)]).float(), 
                "sequence_length": torch.tensor([1024]).float(), 
                "num_layers": torch.tensor([48]).float(), 
                "vocab_size":torch.tensor([50257]).float(),
                "num_attention_heads": torch.tensor([16]).float(),
                "type":args.type,
                "precision":torch.tensor([int(args.precision)]).float()} 

config_h = int((model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())

gbs = int(args.gbs)

with open(record_file, "a") as fp:
    fp.write(f"{model_config}\n")                
    fp.write(f"gbs:{gbs}\n") 

total_count = 0

for cluster_info in cluster_combinations:
    num_node = len(cluster_info.keys())
    n_a100=0
    for i in range(num_node):
        if cluster_info[i] == '1':
            n_a100+=1
    n_a10 = num_node - n_a100
    print(f"n_a100: {n_a100}, n_a10: {n_a10}")

    print(f"cluster info: {cluster_info}")
    D = device_placement(n_a100, n_a10)

    for d in D:
        print(d)
        node_type = []
        if n_a100>0:
            for i in range(num_node):
                if d[i] == 0:
                    node_type.append('p4d.24xlarge')
                else:
                    node_type.append('g5.24xlarge')
            for i in range(num_node):
                if d[i] == 0:
                    d[i] = a100_node_comm
                else:
                    d[i] = a10_node_comm
        else:
            for i in range(num_node):
                node_type.append('g5.24xlarge')
            for i in range(num_node):
                d[i] = a10_node_comm

        if num_node == n_a100:
            model = FASOP(model_config, exp_name, a100_node_comm, a100_node_comm, len(cluster_info))
        else:
            model = FASOP(model_config, exp_name, a100_node_comm, a10_node_comm, len(cluster_info))
                
        known = None

        # Estimating best configurations
        while True:
            ret = amp_no_placement_strategy(M=gpu_per_node, N=num_node, gbs=gbs, known=known, num_layers=config_n)
            if ret is None:
                break
            else:
                h, w, mbs, known = ret
                parallel_dim = {"tp_deg": torch.ones(1,)*h, "dp_deg": torch.ones(1,)*w, "pp_deg": torch.ones(1,)*(gpu_per_node*num_node/(h*w))}
                fake_config = np.ones((gpu_per_node,num_node)) * (-1)

                model_args = (fake_config, gbs, mbs, d, model_config, parallel_dim)    

                with torch.no_grad():
                    rank_map, partition, cost, pipecost, dp_side_cost, all_reduce_embedding_cost = model(model_args, node_type)
                
                for k in parallel_dim:
                    parallel_dim[k] = int(parallel_dim[k].item())

                price_per_s_1 = 32.7726 / 3600
                price_per_s_2 = 8.144 / 3600

                price_per_s = price_per_s_1 * n_a100 + price_per_s_2 * n_a10
                price_per_step = price_per_s * cost.item() # price per second * second per step 
                price = args.iter * price_per_step # 12.5M steps * price per step

                if price <= args.budget:
                    want_simulate.append((mbs,'*', parallel_dim,'*', (n_a100, n_a10),'*', partition,'*', cost.item(),'*', pipecost.item(),'*', dp_side_cost.item(),'*', all_reduce_embedding_cost,'*', price))
                total_count += 1

print(f"Finished {time.time() - time_s}, total count: {total_count}")

sorted_settings = sorted(want_simulate, key = lambda kv: kv[-9])
with open(record_file, "a") as fp:
    for item in sorted_settings:
        fp.write(f"rank {sorted_settings.index(item)}: {item}")
        fp.write("\n")

print("file saved at: ", record_file)