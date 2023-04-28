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
dir_path = os.path.join(home_path, 'tdpp/FASOP/main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)


cluster_info4 = {} # a100:4 : a10:28    8 x nodes
cluster_info7 = {} # a100:4, a10:60     16 x nodes

cluster_info4[0] = [torch.tensor([40 * 1e9]).float(), torch.tensor([4800 * 1e9]).float()]
for i in range(1, 8):
    cluster_info4[i] = [torch.tensor([40 * 1e9]).float(), torch.tensor([252 * 1e9]).float()]

cluster_info7[0] = [torch.tensor([40 * 1e9]).float(), torch.tensor([4800 * 1e9]).float()]
for i in range(1,16):
    cluster_info7[i] = [torch.tensor([40 * 1e9]).float(), torch.tensor([252 * 1e9]).float()]

cluster_combinations = [cluster_info4, cluster_info7]
want_simulate = [] 

for cluster_info in cluster_combinations:
    num_node = len(cluster_info.keys())
    gpu_of_cluster = []
    for i in range(num_node):
        if cluster_info[i][1] == torch.tensor([4800 * 1e9]).float():
            gpu_of_cluster.append('p4d.24xlarge')
        else:
            gpu_of_cluster.append('g5.12xlarge')

    model_config = {"hidden_size": torch.tensor([int(args.hidden_size)]).float(), 
                    "sequence_length": torch.tensor([int(args.sequence_length)]).float(), 
                    "num_layers": torch.tensor([int(args.num_layers)]).float(), 
                    "vocab_size":torch.tensor([int(args.vocab_size)]).float(),
                    "num_attention_heads": torch.tensor([int(args.num_attention_heads)]).float(),
                    "type":args.type,
                    "precision":torch.tensor([int(args.precision)]).float()} # egi: add precision argument

    config_h = int((model_config["hidden_size"]).item())
    config_n = int(model_config["num_layers"].item())
    exp_name = f"gpt1.5b"
    record_file = f"{os.path.join(dir_path, exp_name)}.txt"

    # remove cache directory from last run
    if os.path.exists(os.path.join(home_path, "tmp")):
        for root, dirs, files in os.walk(os.path.join(home_path, "tmp")):
            for f in files:
                os.unlink(os.path.join(root, f))

    # save this name to env
    os.environ["amp_log_path"] = record_file

    gbs = int(args.gbs)
    model = FASOP(model_config, exp_name, cluster_info[0], cluster_info[1], len(cluster_info))
    assert (gbs % gpu_per_node == 0) and (gbs % num_node == 0), "global batch size is too irrgular"

    # with open(record_file, "a") as fp:
    #     fp.write(f"{model_config}\n")                
    #     fp.write(f"gbs:{gbs}\n")                
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

            if cluster_info[0][1] == torch.tensor([4800 * 1e9]).float():
                price_per_s_1 = 32.7726 / 3600
            else:
                price_per_s_1 = 5.672 / 3600
            price_per_s_2 = 5.672 / 3600

            price_per_s = price_per_s_1 + price_per_s_2 * (num_node - 1)
            price_per_step = price_per_s * cost.item() # price per second * second per step 
            want_simulate.append((mbs,'*', parallel_dim,'*', gpu_of_cluster,'*', partition,'*', cost.item(),'*', pipecost.item(),'*', dp_side_cost.item(),'*', all_reduce_embedding_cost,'*', price_per_step))

print(f"Finished {time.time() - time_s}")

sorted_settings = sorted(want_simulate, key = lambda kv: kv[-9])
with open(record_file, "a") as fp:
    for item in sorted_settings:
        fp.write(f"rank {sorted_settings.index(item)}: {item}")
        fp.write("\n")