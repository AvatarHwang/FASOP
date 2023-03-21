from collections import defaultdict
import time
import json
import copy

import subprocess
import sys
import os

import torch
import torch.nn as nn
import numpy as np

from amp_utils import rank2axis, axis2rank, get_host
from pipe import pipe_ast, pipe_cost, pipe_uniform

home_dir = os.environ['HOME'] 
workdir_path = os.path.join(home_dir, "AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")
example_path = os.path.join(workdir_path, "examples")
sys.path.append(workdir_path)
sys.path.append(example_path)

class HetGPT(nn.Module):
    def __init__(self, model_config, exp_name):
        
        super().__init__()
        self.model_config = model_config
        self.exp_name = "init_" + exp_name 
        self.model_type = model_config["type"]
        assert self.model_type == "gpt2XL" 
        self.init_param()
        
    def init_param(self):
        h = float(self.model_config["hidden_size"].item())
        n = float(self.model_config["num_layers"].item())
        s = float(self.model_config["sequence_length"].item())
        v = float(self.model_config["vocab_size"].item())
 
        config_h = int((self.model_config["hidden_size"]).item())
        config_n = int(n)

        json_path = os.path.join(example_path, "ds_config.json")

        self.profile_cost1 = {}
        self.profile_cost2 = {}
        for mp_size in [1,2,4]:
            # known_cost directory stores the real forward time with correponding model parallel degree.
            known_record = f"known_cost/{self.model_type}_A100_{mp_size}" 
            cur_profile_cost1 = 3 * np.load(f"{known_record}.npy")
            known_record = f"known_cost/{self.model_type}_A10_{mp_size}"
            cur_profile_cost2 = 3 * np.load(f"{known_record}.npy")

            self.profile_cost1[str(mp_size)] = cur_profile_cost1
            self.profile_cost2[str(mp_size)] = cur_profile_cost2
        
    def forward(self, args):
        model_type = self.model_type
        config, bs, micro_bs, cluster_info, model_config, oth = args
        amp_config = {"profile_cost1" : self.profile_cost1,
                      "profile_cost2" : self.profile_cost2}
        rank_map, partition, amp_pred, pipecost, dp_side_cost = predict(config, bs, micro_bs, cluster_info, model_config, amp_config, oth)
        return rank_map, partition, amp_pred, pipecost, dp_side_cost
        
# pipeline communication cost, return shape: (L-1, pp-1)
def get_cost_c(cluster_info, model_config, parallel_config, amp_config, dp_index=0):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    
    _layer = ["embed2h"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")
  
    _layer.extend(["noop"])

    _num_layer = len(_layer)
      
    # build layer activation lookup table. Noop exatly has the same activation as the previous op.
    # Leave bs factor outside.
    layer_volume = []
    last_volume = torch.zeros(1,)
    for i in range(_num_layer):
        layer_type = _layer[i]
        if layer_type == "embed2h" or layer_type == "transformer_layer":
            last_volume = bs * s * h
            layer_volume.append(last_volume)
        elif layer_type == "embed2v":
            last_volume = bs * s * v / mp
            layer_volume.append(last_volume)
        elif layer_type == "noop":
            layer_volume.append(last_volume)
        else:
            raise RuntimeError("Unknown layer type.")
            
    # Build communication cost between pipeline stages by looking up the cluster information
    cost_c = torch.zeros((int(dp.item()), _num_layer-1, int(pp.item()-1)))
    for i in range(int(dp.item())):    
        for j in range(int(pp.item()-1)):
            # get the slowest mp gpu connection
            slowest_bandwidth = np.inf
            for k in range(int(mp.item())):    
                rank_cur = axis2rank(axis=(j,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                rank_peer = axis2rank(axis=(j+1,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                node_peer = rank_node_map[int(rank_peer.item())]
                
                if node_cur != node_peer: 
                    cur_bandwidth = min(cluster_info[node_cur][0], cluster_info[node_peer][0])
                else:
                    cur_bandwidth = cluster_info[node_cur][1]
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth
            for k in range(_num_layer-1):
                cost_c[i][k][j] = layer_volume[k]  / slowest_bandwidth
            
    cost_c = torch.mean(cost_c, dim=0)
    print(len(cost_c))
    return cost_c, _layer

# execution cost for one layer, return shape (L,)
def get_cost_e(cluster_info, model_config, parallel_config, profile_cost):    

    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]

    _layer = ["embed2h"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")
    _layer.extend(["noop"])

    _num_layer = len(_layer)
            
    cost_e = np.zeros((int(dp.item()), _num_layer))

    for i in range(int(dp.item())):
        assert _num_layer == len(profile_cost["1"]), "predicted number of layers not equal to actual"
        
        # cost_e in the main result is equivalent to using profile_cost.
        for layer_id in range(_num_layer):
            layer_type = _layer[layer_id]
            cur_layer = bs * profile_cost[str(int(mp.item()))][layer_id]
                
            cost_e[i][layer_id] = cur_layer
    
    cost_e = torch.from_numpy(np.stack(cost_e, axis=0))            
    cost_e = torch.mean(cost_e, dim=0)
    return cost_e

def dp_cost(config, cluster_info, model_config, parallel_config, amp_config, partition):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    
    _layer = ["embed2h"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")    
    _layer.extend(["noop"])
    _num_layer = len(_layer)
        
    # First translate to deepspeed partition form
    ds_partition = [0]
    print(f"partition: {partition}")
    for i in range(len(partition)):
        ds_partition.append(ds_partition[-1]+partition[i])
    print(ds_partition, _num_layer)
    assert ds_partition[-1] == _num_layer
    assert len(ds_partition) == pp + 1
                
    # should be per-dp_group time
    max_dp = torch.zeros(1,)
    for i in range(int(pp.item())):
        for j in range(int(mp.item())): 
            slowest = float("inf")
            for k in range(int(dp.item())):
                rank_cur = axis2rank(axis=(i,k,j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                rank_next = axis2rank(axis=(i,(k+1)%(dp.item()),j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_next = rank_node_map[int(rank_next.item())]       
                if node_cur == node_next:
                    connectivity = cluster_info[node_cur][1]
                else:
                    connectivity = min(cluster_info[node_cur][0], cluster_info[node_next][0])
                        
            slowest = min(slowest, connectivity)
            dp_const = 2 * (dp-1) / (dp * slowest)
            dp_const = torch.tensor([dp_const])
                
            param_count = torch.zeros(1,)
            counted = False
            for layer_id in range(ds_partition[i], ds_partition[i+1]):
                layer_type = _layer[layer_id]
                if layer_type == "embed2h" or layer_type == "embed2v":
                    if not counted:
                        counted = True
                        param_count += 164249600
                elif layer_type == "transformer_layer":
                    param_count += 24 * h ** 2 / mp
                    param_count += 3200 * 2 /mp
                elif layer_type == "noop":
                    pass
                else:
                    raise RuntimeError("Unknown layer type.")
                        
            #print(f"dp: {dp_const} and param {param_count}")
            cur_dp = dp_const * param_count
            if cur_dp > max_dp:
                max_dp = cur_dp
                
    return ds_partition, max_dp

def predict(config, bs, mbs, cluster_info, model_config, amp_config, oth):
    L = model_config["num_layers"]
    cost = torch.zeros(1,)
    M, N = config.shape
    config = np.asarray(config)
       
    if np.all(config == -1):
        rank_map = defaultdict(list)
        rank_node_map = dict()

        m = oth["mp_deg"]
        n = oth["dp_deg"]
        pp = oth["pp_deg"]                   
        
        # infer a GPU rank map                
        counter = 0 
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                rank_map[j].append(counter)
                rank_node_map[counter] = j
                counter += 1
    
    # valid config, inferred from sa 
    else:
        config = torch.from_numpy(config)
        pp = torch.max(config).float()
        
        # infer rank_map: given node name, returns the global mapped rank(int) in (pp, dp, mp) order
        # rank_node_map: given rank, returns the node
        rank_map = defaultdict(list)
        rank_node_map = dict()
    
        if pp >= (L + 2):
            print(f"early return with pp={pp}, L={L}")
            return None, None, torch.tensor([float("inf")])
           
        m = oth["mp_deg"]
        n = oth["dp_deg"]
        pp = oth["pp_deg"]                  
        
        rank_counter = np.zeros(int(pp.item()))
            
        # infer a GPU rank map                    
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                cur_pp = int(config[k][j] - 1)
                rank_map[j].append(int((rank_counter[cur_pp] + cur_pp * m * n).item()))
                rank_node_map[int((rank_counter[cur_pp] + cur_pp * m * n).item())] = j
                rank_counter[cur_pp] += 1
            
    # infer number of micro-batch size B
    # mbs = gbs / dp
    B = bs / (n * mbs)
    mbs = bs / (n * B)
            
    parallel_config = {"mp" : m, "dp" : n, "pp" : pp, "micro_bs" : mbs, "rank_map" : rank_map, "rank_node_map": rank_node_map}
        
    cost_e1 = get_cost_e(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, profile_cost=amp_config["profile_cost1"])
    cost_e2 = get_cost_e(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, profile_cost=amp_config["profile_cost2"])
    cost_c, layer_type = get_cost_c(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, amp_config=amp_config)
        

    # TODO: use cost_e1 and cost_e2
    print("pp", pp)
    partition, stage_latency = pipe_ast(len(cost_e1), np.asarray(cost_e1), np.asarray(cost_e2), np.asarray(cost_c), int(pp.item()), int(B.item()), N)
    
    print(f"amp gives partition: {partition}")
    # TODO: use cost_e1 and cost_e2
    pipecost = pipe_cost(pp, B, stage_latency)
        
    # translate to ds form, add data parallelism cost
    ds_partition, dp_side_cost = dp_cost(config, cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, 
                        amp_config=amp_config, partition=partition)
       
    cost = pipecost + dp_side_cost

    oom_flag = EstimatePeakMemory(partition, model_config, parallel_config, layer_type)

    return rank_map, ds_partition, cost, pipecost, dp_side_cost
    

def EstimatePeakMemory(partition, model_config, parallel_config, layer_type):
    hidden_size = model_config["hidden_size"]
    v = model_config["vocab_size"]
    seq_len = model_config["sequence_length"]
    num_head = model_config["num_attention_heads"]
    tp_degree = parallel_config["mp"]
    mbs = parallel_config["micro_bs"]
    param_count = []
    activation_memory = []
    pipeline_buffer_memory = []
    layer_index = 0

    for stage in partition:
        param=0
        avtivation = 0
        pipeline_buffer = 0
        for i in range(stage):
            if layer_type[i] == "embed2h" or "embed2v":
                param += 164249600
                avtivation += mbs * seq_len * hidden_size
            elif layer_type[i] == "transformer_layer":
                param += 24 * hidden_size ** 2 / tp_degree
                param += 3200 * 2 # LayerNorm
                avtivation += 9 * mbs * seq_len * hidden_size + mbs * seq_len ** 2 * num_head # reference: https://arxiv.org/pdf/2110.05722.pdf
                if i == 0:
                    pipeline_buffer += mbs * seq_len * hidden_size # BLH
            elif layer_type[i] == "noop":
                pass
            layer_index += 1
        param_count.append(param * 16 / 8 / 1024 / 1024 / 1024) # Translate into GB
        activation_memory.append(avtivation * 16 / 8 / 1024 / 1024 / 1024)
        pipeline_buffer_memory.append(pipeline_buffer * 16 / 8 / 1024 / 1024 / 1024)

    # get peak memory
    peak_memory = []
    for i in range(len(partition)):
        peak_memory.append(param_count[i] + activation_memory[i] + pipeline_buffer_memory[i])
    peak_memory = max(peak_memory)

    gpu_memory = 24
    if peak_memory > gpu_memory:
        print("peak memory is larger than gpu memory")
        print(f"peak memory is {peak_memory} GB")
        return False
    else:
        print("peak memory is smaller than gpu memory")
        print(f"peak memory is {peak_memory} GB")
        return True

            