# Script to reproduce homogeneous setting results

import math
import time
from collections import defaultdict
import operator
import random
import os
import copy

from tqdm import tqdm

import numpy as np

import torch
from torch import optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sa import amp_no_placement_strategy
from cost_het_cluster import HetGPT
from amp_utils import simulate, to_float_torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gbs", type=int, default=32)
parser.add_argument("--exp_name", type=str, default="het_cluster")
parser.add_argument("--model_config", type=str, default="gpt2XL")
parser.add_argument("--hidden_size", type=int, default=1600)
parser.add_argument("--sequence_length", type=int, default=2048)
parser.add_argument("--num_layers", type=int, default=48)
parser.add_argument("--vocab_size", type=int, default=51200)
parser.add_argument("--type", type=str, default="gpt2XL")
parser.add_argument("--gpu_per_node", type=int, default=4)
parser.add_argument("--num_node", type=int, default=4)
args = parser.parse_args()

# cluster information

time_s = time.time()
# number of GPU per node, number of nodes
gpu_per_node = args.gpu_per_node
num_node = args.num_node

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'tdpp/HetGPT/main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

cluster_info = {}

# inter-node bandwidth, intra-node bandwidth
cluster_info[0] = [torch.tensor([40 * 1e9 / 32]).float(), torch.tensor([252 * 1e9 / 32]).float()]
cluster_info[1] = [torch.tensor([40 * 1e9 / 32]).float(), torch.tensor([252 * 1e9 / 32]).float()]
cluster_info[2] = [torch.tensor([40 * 1e9 / 32]).float(), torch.tensor([126 * 1e9 / 32]).float()]
cluster_info[3] = [torch.tensor([40 * 1e9 / 32]).float(), torch.tensor([126 * 1e9 / 32]).float()]

model_config = {"hidden_size": torch.tensor([int(args.hidden_size)]).float(), 
                "sequence_length": torch.tensor([2048]).float(), 
                "num_layers": torch.tensor([48]).float(), 
                "vocab_size":torch.tensor([51200]).float(),
                "type":args.type}

config_h = int((model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())
time_stamp = int(time.time())
exp_name = f"het_cluster"
record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"
simulate_dir = os.path.join(home_path, "amp_simulate")
if not os.path.exists(simulate_dir):
    os.mkdir(simulate_dir)

# remove cache directory from last run
if os.path.exists(os.path.join(home_path, "tmp")):
    for root, dirs, files in os.walk(os.path.join(home_path, "tmp")):
        for f in files:
            os.unlink(os.path.join(root, f))

# save this name to env
os.environ["amp_log_path"] = record_file

global_bs = int(args.gbs)
model = HetGPT(model_config, exp_name)
assert (global_bs % gpu_per_node == 0) and (global_bs % num_node == 0), "global batch size is too irrgular"

want_simulate = [] 
feasible = {}

with open(record_file, "a") as fp:
    fp.write(f"{model_config}\n")                
    fp.write(f"gbs:{global_bs}\n")                
known = None
iter_count = 0

# Estimating best configurations
while True:
    ret = amp_no_placement_strategy(M=gpu_per_node, N=num_node, gbs=global_bs, known=known)
    if ret is None:
        break
    else:
        h, w, mbs, known = ret
        oth = {"mp_deg": torch.ones(1,)*h, "dp_deg": torch.ones(1,)*w, "pp_deg": torch.ones(1,)*(gpu_per_node*num_node/(h*w))}
        fake_config = np.ones((gpu_per_node,num_node)) * (-1)
        model_args = (fake_config, global_bs, mbs, cluster_info, model_config, oth)    
        
        with torch.no_grad():
            rank_map, partition, cost, pipecost, dp_side_cost = model(model_args)
        
        want_simulate.append(((mbs, oth, rank_map, partition), cost, pipecost, dp_side_cost))
    iter_count += 1
    if iter_count % 10 == 0:
        print(f"AMP finishes {iter_count} iterations")
time_e = time.time()
print(f"AMP finishes without placement in {iter_count} iterations in {time_e - time_s}")

sorted_settings = sorted(want_simulate, key = lambda kv: kv[1])
with open(record_file, "a") as fp:
    for item in sorted_settings:
        fp.write(f"rank {sorted_settings.index(item)}: {item}")
        fp.write("\n")