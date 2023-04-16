import time

import os
import numpy as np

import torch
from torch import optim as optim

from sa import amp_no_placement_strategy
from cost_het_cluster import HetGPT

import argparse

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
args = parser.parse_args()

time_s = time.time()
# number of GPU per node, number of nodes
gpu_per_node = args.gpu_per_node

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'tdpp/HetGPT/main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

a100 = [torch.tensor([400 * 1e9]).float(), torch.tensor([4800 * 1e9]).float()]
a10 = [torch.tensor([100 * 1e9]).float(), torch.tensor([252 * 1e9]).float()]

cluster_combinations=[]
count = 0
for i in [2,4,8,16]:
    for j in range(0, i):
        cluster = {}
        n_a100 = j
        n_a10 = i-j
        for k in range(i):
            if k < n_a100:
                cluster[k] = a100
            else:
                cluster[k] = a10
        cluster_combinations.append(cluster)
        count += 1
print("count: ",count)
assert 1==2
want_simulate = [] 

time_stamp = int(time.time())
exp_name = f"het_cluster"
record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"

# remove cache directory from last run
if os.path.exists(os.path.join(home_path, "tmp")):
    for root, dirs, files in os.walk(os.path.join(home_path, "tmp")):
        for f in files:
            os.unlink(os.path.join(root, f))

# save this name to env
os.environ["amp_log_path"] = record_file

for cluster_info in cluster_combinations:
    num_node = len(cluster_info.keys())
    gpu_of_cluster = []
    n_a100=0
    for i in range(num_node):
        if cluster_info[i][1] == torch.tensor([4800 * 1e9]).float():
            gpu_of_cluster.append('p4d.24xlarge')
            n_a100+=1
        else:
            gpu_of_cluster.append('g5.24xlarge')
    n_a10 = num_node - n_a100

    model_config = {"hidden_size": torch.tensor([int(args.hidden_size)]).float(), 
                    "sequence_length": torch.tensor([1024]).float(), 
                    "num_layers": torch.tensor([48]).float(), 
                    "vocab_size":torch.tensor([50257]).float(),
                    "num_attention_heads": torch.tensor([16]).float(),
                    "type":args.type,
                    "precision":torch.tensor([int(args.precision)]).float()} # egi: add precision argument

    config_h = int((model_config["hidden_size"]).item())
    config_n = int(model_config["num_layers"].item())
    # time_stamp = int(time.time())
    # exp_name = f"het_cluster"
    # record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"

    gbs = int(args.gbs)
    if num_node == n_a100:
        model = HetGPT(model_config, exp_name, cluster_info[0], cluster_info[0], len(cluster_info))
    else:
        model = HetGPT(model_config, exp_name, cluster_info[0], cluster_info[n_a100], len(cluster_info))
    assert (gbs % gpu_per_node == 0) and (gbs % num_node == 0), "global batch size is too irrgular"

    with open(record_file, "a") as fp:
        fp.write(f"{model_config}\n")                
        fp.write(f"gbs:{gbs}\n")                
    known = None

    # Estimating best configurations
    while True:
        ret = amp_no_placement_strategy(M=gpu_per_node, N=num_node, gbs=gbs, known=known)
        if ret is None:
            break
        else:
            h, w, mbs, known = ret
            parallel_dim = {"tp_deg": torch.ones(1,)*h, "dp_deg": torch.ones(1,)*w, "pp_deg": torch.ones(1,)*(gpu_per_node*num_node/(h*w))}
            fake_config = np.ones((gpu_per_node,num_node)) * (-1)
            model_args = (fake_config, gbs, mbs, cluster_info, model_config, parallel_dim)    

            with torch.no_grad():
                rank_map, partition, cost, pipecost, dp_side_cost, all_reduce_embedding_cost = model(model_args)
            
            
            for k in parallel_dim:
                parallel_dim[k] = int(parallel_dim[k].item())

            price_per_s_1 = 32.7726 / 3600
            price_per_s_2 = 8.144 / 3600

            price_per_s = price_per_s_1 * n_a100 + price_per_s_2 * n_a10
            price_per_step = price_per_s * cost.item() # price per second * second per step 

            want_simulate.append((mbs,'*', parallel_dim,'*', gpu_of_cluster,'*', partition,'*', cost.item(),'*', pipecost.item(),'*', dp_side_cost.item(),'*', all_reduce_embedding_cost,'*', price_per_step))

print(f"Finished {time.time() - time_s}")

sorted_settings = sorted(want_simulate, key = lambda kv: kv[-9])
with open(record_file, "a") as fp:
    for item in sorted_settings:
        fp.write(f"rank {sorted_settings.index(item)}: {item}")
        fp.write("\n")