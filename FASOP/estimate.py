"""
Portions of this code adapted from the 'AMP' project (https://github.com/DachengLi1/AMP). 
@article{li2022amp,
  title={AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness},
  author={Li, Dacheng and Wang, Hongyi and Xing, Eric and Zhang, Hao},
  journal={arXiv preprint arXiv:2210.07297},
  year={2022}
}
"""

from collections import defaultdict

import os

import torch
import torch.nn as nn
import numpy as np

from amp_utils import axis2rank
from pipe import minmax, schedule, get_stage_latency, explain_minmax, dynamic_programming, exhaustive_partition
from device_placement import get_gpu_for_stage

import copy

home_dir = os.environ['HOME'] 

class FASOP(nn.Module):
    def __init__(self, model_config, exp_name, cluster_info0, cluster_info1, num_node):
        
        super().__init__()
        self.model_config = model_config
        self.exp_name = "init_" + exp_name 
        self.model_type = model_config["type"]
        self.cluster_info0 = cluster_info0
        self.cluster_info1 = cluster_info1
        self.init_param()
        self.num_node = num_node
        
    def init_param(self):
        h = float(self.model_config["hidden_size"].item())
        n = float(self.model_config["num_layers"].item())
        s = float(self.model_config["sequence_length"].item())
        v = float(self.model_config["vocab_size"].item())
 
        config_h = int((self.model_config["hidden_size"]).item())
        config_n = int(n)

        self.profile_cost_A100 = {}
        self.profile_cost_A10 = {}
        for mp_size in [1,2,4]:
            # known_cost directory stores the real forward time with correponding model parallel degree.
            known_record = f"known_cost/{self.model_type}_A100_{mp_size}" 
            profile_cost_a100 = 3 * np.load(f"{known_record}.npy")
            self.profile_cost_A100[str(mp_size)] = profile_cost_a100

            known_record = f"known_cost/{self.model_type}_A10_{mp_size}"
            profile_cost_a10 = 3 * np.load(f"{known_record}.npy")
            self.profile_cost_A10[str(mp_size)] = profile_cost_a10
        
    def forward(self, args, node_type, exhaustive):
        config, bs, micro_bs, cluster_info, model_config, oth = args
        amp_config = {"profile_cost_a100" : self.profile_cost_A100,
                      "profile_cost_a10" : self.profile_cost_A10}
        rank_map, partition, amp_pred, pipecost, dp_side_cost, all_reduce_embedding_cost, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem = predict(config, bs, micro_bs, cluster_info, model_config, amp_config, oth, node_type, exhaustive)
        return rank_map, partition, amp_pred, pipecost, dp_side_cost, all_reduce_embedding_cost, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem
        
# pipeline communication cost, return shape: (L-1, pp-1)
def get_cost_c(cluster_info, model_config, parallel_config, amp_config, dp_index=0, _layer=None):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    bs = parallel_config["micro_bs"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    
    precision = torch.ones(1)*16 # TODO: support fp32, should be args.precision

    _num_layer = len(_layer)

    if pp == 1:
        return torch.zeros(int(n.item())), _layer
      
    # build layer activation lookup table.
    layer_volume = []
    last_volume = torch.zeros(1,)
    for i in range(_num_layer):
        layer_type = _layer[i]
        if layer_type == "embedding_layer" or layer_type == "transformer_layer" or layer_type == "encoder" or layer_type == "decoder" or layer_type == "post_process":
            last_volume = bs * s * h
            layer_volume.append(last_volume)
        else:
            # unrecognized layer type
            raise ValueError(f"Unrecognized layer type: {layer_type}")
            
    # Build communication cost between pipeline stages by looking up the cluster information
    cost_c = torch.zeros((int(dp.item()), _num_layer, int(pp.item()-1)))
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
                    cur_bandwidth = cur_bandwidth / int(dp.item()) / int(mp.item()) / 1.05
                else:
                    cur_bandwidth = cluster_info[node_cur][1] / 3.5
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth     
            for k in range(_num_layer):
                cost_c[i][k][j] = layer_volume[k] * precision / slowest_bandwidth
    cost_c = torch.max(cost_c, dim=0)
    return cost_c.values, _layer

# execution cost for one layer, return shape (L,)
def get_cost_e(is_a100, model_config, parallel_config, profile_cost, _layer=None, model_type=None):    
    n = model_config["num_layers"]
    bs = parallel_config["micro_bs"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]

    _num_layer = len(_layer)
            
    cost_e = np.zeros((int(dp.item()), _num_layer))

    for i in range(int(dp.item())):
        assert _num_layer == len(profile_cost["1"]), "predicted number of layers not equal to actual"
        
        # cost_e in the main result is equivalent to using profile_cost.
        embedding_layer_counted = False
        for layer_id in range(_num_layer):
            layer_type = _layer[layer_id]
            #################################################################
            #                      GPT
            #################################################################
            if model_type == "gpt2XL":
                if layer_type == "embedding_layer" or layer_type == "post_process":
                    if is_a100 is False: # A10 GPU
                        cur_layer = bs * profile_cost[str(int(mp.item()))][layer_id]
                    else: # A100 GPU
                        cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                elif layer_type == "transformer_layer":
                    cur_layer = bs * profile_cost[str(int(mp.item()))][layer_id]
                else:
                    cur_layer = 0
                cost_e[i][layer_id] = cur_layer
            #################################################################
            #                        Bert
            #################################################################
            elif model_type == "bert":
                if is_a100 is False: # A10 GPU
                    if layer_type == "embedding_layer":
                        if bs < 4:
                            cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                        else:
                            cur_layer = bs/4 * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    elif layer_type == "transformer_layer":
                        if bs < 8:
                            cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                        else:
                            cur_layer = bs/4 * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    elif layer_type == "post_process":
                        cur_layer = bs * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    else:
                        assert False, "layer type not recognized"
                else: # A100 GPU
                    if layer_type == "embedding_layer":
                        if bs < 8:
                            cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                        elif bs==8:
                            cur_layer = 1.385 * profile_cost[str(int(mp.item()))][layer_id]
                        elif bs==16:
                            cur_layer = 1.385 * 1.589 * profile_cost[str(int(mp.item()))][layer_id]
                        elif bs==32:
                            cur_layer = 1.385 * 1.589 * 1.565 * profile_cost[str(int(mp.item()))][layer_id]
                        else:
                            cur_layer = 1.385 * 1.589 * 1.565 * bs / 32.0 * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    elif layer_type == "transformer_layer":
                        if bs < 8:
                            cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                        elif bs==8:
                            cur_layer = 1.236 * profile_cost[str(int(mp.item()))][layer_id]
                        elif bs==16:
                            cur_layer = 1.235 * 1.658 * profile_cost[str(int(mp.item()))][layer_id]
                        else:
                            cur_layer = 1.235 * 1.658 * bs / 16.0 * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    elif layer_type == "post_process":
                        cur_layer = bs * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    else:
                        assert False, "layer type not recognized"
            #################################################################
            #                        T5
            #################################################################
            elif model_type == "T5":
                # A10 GPU
                if is_a100 is False: # A10 GPU
                    if layer_type == "encoder":
                        if bs == 1:
                            cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                        elif bs == 2:
                            cur_layer = 1.140 * profile_cost[str(int(mp.item()))][layer_id]
                        else:
                            cur_layer = 1.140 * bs / 2.0 * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    elif layer_type == "decoder":
                        if bs == 1:
                            cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                        elif bs == 2:
                            cur_layer = 1.094 * profile_cost[str(int(mp.item()))][layer_id]
                        else:
                            cur_layer = 1.094 * bs / 2.0 * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    elif layer_type == "embedding_layer":
                        if bs == 1:
                            cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                        elif bs == 2:
                            cur_layer = 1.2 * profile_cost[str(int(mp.item()))][layer_id]
                        elif bs == 4:
                            cur_layer = 1.2 * 1.2 * profile_cost[str(int(mp.item()))][layer_id]
                        elif bs == 8:
                            cur_layer = 1.2 * 1.2 * 1.4 * profile_cost[str(int(mp.item()))][layer_id]
                        elif bs == 16:
                            cur_layer = 1.2 * 1.2 * 1.4 * 1.6 * profile_cost[str(int(mp.item()))][layer_id]
                        else:
                            cur_layer = bs / 16.0 * 1.2 * 1.2 * 1.4 * 1.6 * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                # A100 GPU
                else:
                    if layer_type == "encoder":
                        if bs < 4:
                            cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                        elif bs==4:
                            cur_layer = 1.173 * profile_cost[str(int(mp.item()))][layer_id]
                        elif bs==8:
                            cur_layer = 1.173 * 1.489 * profile_cost[str(int(mp.item()))][layer_id]
                        else:
                            cur_layer = 1.173 * 1.489 * bs / 8.0 * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    elif layer_type == "decoder":
                        if bs < 4:
                            cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                        elif bs == 4:
                            cur_layer = 1.142 * profile_cost[str(int(mp.item()))][layer_id]
                        elif bs == 8:
                            cur_layer = 1.142 * 1.683 * profile_cost[str(int(mp.item()))][layer_id]
                        else:
                            cur_layer = 1.142 * 1.683 * bs / 8.0 * profile_cost[str(int(mp.item()))][layer_id]
                        cost_e[i][layer_id] = cur_layer
                    elif layer_type == "embedding_layer":
                        if embedding_layer_counted is False: # Encoder embedding layer
                            if bs == 1:
                                cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 2:
                                cur_layer = 1.386 * profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 4:
                                cur_layer = 1.386 * 1.244 * profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 8:
                                cur_layer = 1.386 * 1.244 * 1.356 * profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 16:
                                cur_layer = 1.386 * 1.244 * 1.356 * 1.055 * profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 32:
                                cur_layer = 1.386 * 1.244 * 1.356 * 1.055 * 1.368 * profile_cost[str(int(mp.item()))][layer_id]
                            else:
                                cur_layer = 1.386 * 1.244 * 1.356 * 1.055 * 1.368 * bs / 32.0 * profile_cost[str(int(mp.item()))][layer_id]
                            cost_e[i][layer_id] = cur_layer
                        if embedding_layer_counted is True: # Decoder embedding layer
                            if bs == 1:
                                cur_layer = profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 2:
                                cur_layer = 1.126 * profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 4:
                                cur_layer = 1.126 * 1.080 * profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 8:
                                cur_layer = 1.126 * 1.080 * 1.090 * profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 16:
                                cur_layer = 1.126 * 1.080 * 1.090 * 1.596 * profile_cost[str(int(mp.item()))][layer_id]
                            elif bs == 32:
                                cur_layer = 1.126 * 1.080 * 1.090 * 1.596 * 1.529 * profile_cost[str(int(mp.item()))][layer_id]
                            else:
                                cur_layer = 1.126 * 1.080 * 1.090 * 1.596 * 1.529 * bs / 32.0 * profile_cost[str(int(mp.item()))][layer_id]
                            cost_e[i][layer_id] = cur_layer
                    else:
                        assert False, "layer type not recognized"
            else:
                assert False, "model type not recognized"

    cost_e = torch.from_numpy(np.stack(cost_e, axis=0))            
    cost_e = torch.mean(cost_e, dim=0)
    return cost_e


def cost_all_reduce_embedding(model_config, cluster_info, parallel_config, gpu_per_node):
    precision = 16 # TODO: support fp32, should be args.precision
    tp_degree = int(parallel_config["mp"].item())
    dp_degree = int(parallel_config["dp"].item())
    pp_degree = int(parallel_config["pp"].item())
    rank_node_map = parallel_config["rank_node_map"]
    hidden_size = int(model_config["hidden_size"].item())
    vocab_size = int(model_config["vocab_size"].item())

    if pp_degree>1:
        # Get communication bandwidth between pipeline stage 0 and -1
        for i in range(dp_degree):    
            # get the slowest mp gpu connection
            slowest_bandwidth = np.inf
            for k in range(tp_degree):    
                rank_cur = axis2rank(axis=(0,i,k), mp_deg=tp_degree, dp_deg=dp_degree, pp_deg=pp_degree)
                rank_peer = axis2rank(axis=(pp_degree-1,i,k), mp_deg=tp_degree, dp_deg=dp_degree, pp_deg=pp_degree)
                node_cur = rank_node_map[rank_cur]
                node_peer = rank_node_map[rank_peer]
                
                if node_cur != node_peer: # use inter-node bandwidth
                    cur_bandwidth = min(cluster_info[node_cur][0], cluster_info[node_peer][0])
                else: # use intra-node bandwidth
                    cur_bandwidth = cluster_info[node_cur][1]
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth
        
        # if dp_degree<gpu_per_node, we assume the bandwidth is shared by all dp_degree
        # else, we assume the bandwidth is shared by all gpu_per_node
        band_width = slowest_bandwidth/min(dp_degree, gpu_per_node) 
        embedding_syn_cost = 2*(2-1)*(hidden_size*vocab_size*precision)/(2*band_width)/tp_degree
        return embedding_syn_cost.item()
    else:
        return 0
        

def dp_cost(config, cluster_info, model_config, parallel_config, amp_config, partition, _layer=None, gpu_per_node=4, gpu_type_lst=None):
    h = model_config["hidden_size"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    
    _num_layer = len(_layer)
        
    # First translate to deepspeed partition form
    ds_partition = [0]
    for i in range(len(partition)):
        ds_partition.append(ds_partition[-1]+partition[i])
    assert ds_partition[-1] == _num_layer
    assert len(ds_partition) == pp + 1

    counted = False
    # debug
    # print(f"tp: {mp.item()}, dp: {dp.item()}, pp: {pp.item()}")
    
    param_count = 0    
    for layer_id in range(ds_partition[0], ds_partition[1]):
        layer_type = _layer[layer_id]
        if layer_type == "embedding_layer" or layer_type == "post_process":
            if not counted:
                counted = True
                param_count += (h*v)
        elif layer_type == "transformer_layer" or layer_type == "encoder" or layer_type == "decoder":
            param_count += ((12 * h ** 2)+20800) / mp
    
    # print(f" param_count: {param_count}")
    
    # Get communication bandwidth of pipeline stage 0
    dp_cost_list = []
    for i in range(int(pp.item())):
        for j in range(int(mp.item())):
            bandwidth_lst = []
            for k in range(int(dp.item())):
                rank_cur = axis2rank(axis=(0,k,j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                rank_next = axis2rank(axis=(0,(k+1)%(dp.item()),j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_next = rank_node_map[int(rank_next.item())]

                if node_cur == node_next:
                    connectivity = cluster_info[node_cur][1]
                else:
                    connectivity = min(cluster_info[node_cur][0], cluster_info[node_next][0])
                bandwidth_lst.append(connectivity)
        # get slowest of bandwidth
        bandwidth = min(bandwidth_lst)

        # Inter-node bandwidth share
        if int(mp.item())*int(dp.item()) >= gpu_per_node and int(dp.item())>1:
            bandwidth = bandwidth / int(mp.item())
        
        # Intra-node bandwidth share
        elif int(mp.item())*int(dp.item()) < gpu_per_node and int(dp.item())>1 and int(mp.item()) > 1:
            if gpu_type_lst[i] == "A10":
                bandwidth = bandwidth / (gpu_per_node/int(dp.item()))
    
        # All-reduce cost: 2(n-1)M / nB
        precision = 16 #TODO: precision should be args.precision
        dp_cost_list.append(2 * (int(dp.item()) - 1) * (param_count * precision) / (int(dp.item()) * bandwidth))
    
    # print(f"bandwidth: {bandwidth}")
    # print(f"dp_cost_list: {dp_cost_list}")
        
    return ds_partition, dp_cost_list


def predict(config, gbs, mbs, cluster_info, model_config, amp_config, oth, node_type, exhaustive):
    L = int(model_config["num_layers"])
    model_type = model_config["type"]
    cost = torch.zeros(1,)
    M, N = config.shape
    config = np.asarray(config)
       
    if np.all(config == -1):
        rank_map = defaultdict(list)
        rank_node_map = dict()

        tp_degree = oth["tp_deg"]
        dp_degree = oth["dp_deg"]
        pp_degree = oth["pp_deg"]                   
        
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
           
        tp_degree = oth["tp_deg"]
        dp_degree = oth["dp_deg"]
        pp_degree = oth["pp_deg"]                  
        
        rank_counter = np.zeros(int(pp.item()))
            
        # infer a GPU rank map                    
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                cur_pp = int(config[k][j] - 1)
                rank_map[j].append(int((rank_counter[cur_pp] + cur_pp * tp_degree * dp_degree).item()))
                rank_node_map[int((rank_counter[cur_pp] + cur_pp * tp_degree * dp_degree).item())] = j
                rank_counter[cur_pp] += 1
            
    num_mb = gbs / (dp_degree * mbs)
            
    parallel_config = {"mp" : tp_degree, "dp" : dp_degree, "pp" : pp_degree, "micro_bs" : mbs, "rank_map" : rank_map, "rank_node_map": rank_node_map}
    pp_degree = int(pp_degree.item())
    _layer = get_layer_type(model_type=model_type, n=L, pp=pp_degree)

    cost_e_a100 = get_cost_e(is_a100=True, 
                        model_config=model_config, parallel_config=parallel_config, profile_cost=amp_config["profile_cost_a100"], _layer=_layer, model_type=model_type)
    cost_e_a10 = get_cost_e(is_a100=False, 
                        model_config=model_config, parallel_config=parallel_config, profile_cost=amp_config["profile_cost_a10"], _layer=_layer, model_type=model_type)
    cost_c, layer_type = get_cost_c(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, amp_config=amp_config, _layer=_layer)

    if "T5" not in model_type:
        gpu_type_lst = get_gpu_for_stage(pp_degree, N, node_type)
        if exhaustive["exhaustive"] is False:
            partition, stage_comp_time_lst, _, _, stage_for_send_time_lst, stage_back_send_time_lst  = minmax(len(cost_e_a100), np.asarray(cost_e_a100), np.asarray(cost_e_a10), np.asarray(cost_c), pp_degree, gpu_type_lst)
        else:
            partition, stage_comp_time_lst, _, _, stage_for_send_time_lst, stage_back_send_time_lst  = explain_minmax(len(cost_e_a100), np.asarray(cost_e_a100), np.asarray(cost_e_a10), np.asarray(cost_c), pp_degree, exhaustive["gpu_type_lst"])
            #partition, stage_comp_time_lst, _, _, stage_for_send_time_lst, stage_back_send_time_lst = exhaustive_partition(len(cost_e_a100), np.asarray(cost_e_a100), np.asarray(cost_e_a10), np.asarray(cost_c), pp_degree, exhaustive["gpu_type_lst"])
            #partition, stage_comp_time_lst, _, _, stage_for_send_time_lst, stage_back_send_time_lst = dynamic_programming(len(cost_e_a100), np.asarray(cost_e_a100), np.asarray(cost_e_a10), np.asarray(cost_c), pp_degree, num_mb, exhaustive["gpu_type_lst"])
            assert False, "Done!"
        pipecost_last, stage_wise_cost_lst = schedule(pp_degree, 
                                                    num_mb, stage_comp_time_lst, 
                                                    stage_for_send_time_lst, 
                                                    stage_back_send_time_lst)
        is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem  = EstimatePeakMemory(partition, model_config, parallel_config, layer_type, cluster_info, gbs, num_mb)
    else: # If the model is T5
        # get gpu type for each stage
        gpu_type_lst = get_gpu_for_stage(pp_degree, N, node_type)

        if pp_degree>1:
            PP_C = []
            for pp_encoder in range(1, min(pp_degree, int(len(cost_e_a100)/2))):
                pp_decoder = pp_degree - pp_encoder
                if pp_decoder <= L/2:
                    PP_C.append([pp_encoder, pp_decoder])
            pipecost_last = 1000000
            if exhaustive["exhaustive"] is True:
                gpu_type_lst = exhaustive["gpu_type_lst"]
                max_t = 100000.0
                for pp_c in PP_C:
                    pp_en, pp_de = pp_c
                    partition_en, stage_comp_time_lst_en, _, _, stage_for_send_time_lst_en, stage_back_send_time_lst_en = exhaustive_partition(int(len(cost_e_a100)/2),
                                                        np.asarray(cost_e_a100), 
                                                        np.asarray(cost_e_a10), 
                                                        np.asarray(cost_c), 
                                                        pp_en, 
                                                        gpu_type_lst[:pp_en])
                    partition_de, stage_comp_time_lst_de, _, _, stage_for_send_time_lst_de, stage_back_send_time_lst_de = exhaustive_partition(int(len(cost_e_a100)/2),
                                                        np.asarray(cost_e_a100),
                                                        np.asarray(cost_e_a10),
                                                        np.asarray(cost_c),
                                                        pp_de,
                                                        gpu_type_lst[pp_en:])
                    partition = partition_en + partition_de
                    stage_latency = get_stage_latency(partition, cost_e_a100, cost_e_a10, cost_c, gpu_type_lst)
                    stage_time_lst_temp = [stage.get_stage_time() for stage in stage_latency]
                    stage_comp_time_lst_temp = [stage.get_comp_time() for stage in stage_latency]
                    stage_for_send_time_lst_temp = [stage.get_for_send_time() for stage in stage_latency]
                    stage_back_send_time_lst_temp = [stage.get_back_send_time() for stage in stage_latency]
                    pipecost_last_temp, stage_wise_cost_lst_temp = schedule(pp_degree, num_mb, stage_comp_time_lst_temp, stage_for_send_time_lst_temp, stage_back_send_time_lst_temp)
                    if max_t > pipecost_last_temp:
                        max_t = pipecost_last_temp
                        partition_last=partition[:]
                print("partition_last", partition_last, "max_t", max_t)
                assert False, "Done!"
                
            for pp_c in PP_C:
                pp_en, pp_de = pp_c
                partition_en, stage_comp_time_lst_en, _, _, stage_for_send_time_lst_en, stage_back_send_time_lst_en = minmax(int(len(cost_e_a100)/2), 
                                                    np.asarray(cost_e_a100), 
                                                    np.asarray(cost_e_a10), 
                                                    np.asarray(cost_c), 
                                                    pp_en, 
                                                    gpu_type_lst[:pp_en])
                partition_de, stage_comp_time_lst_de, _, _, stage_for_send_time_lst_de, stage_back_send_time_lst_de = minmax(int(len(cost_e_a100)/2),
                                                    np.asarray(cost_e_a100), 
                                                    np.asarray(cost_e_a10), 
                                                    np.asarray(cost_c), 
                                                    pp_de,
                                                    gpu_type_lst[pp_en:])
                
                partition_temp = partition_en + partition_de

                # re-minmax
                stage_latency = get_stage_latency(partition_temp, cost_e_a100, cost_e_a10, cost_c, gpu_type_lst)
                stage_time_lst_temp = [stage.get_stage_time() for stage in stage_latency]
                stage_comp_time_lst_temp = [stage.get_comp_time() for stage in stage_latency]
                stage_for_send_time_lst_temp = [stage.get_for_send_time() for stage in stage_latency]
                stage_back_send_time_lst_temp = [stage.get_back_send_time() for stage in stage_latency]

                pipecost_last_temp, stage_wise_cost_lst_temp = schedule(pp_degree, num_mb, stage_comp_time_lst_temp, stage_for_send_time_lst_temp, stage_back_send_time_lst_temp)
                
                if pipecost_last_temp < pipecost_last:
                    pipecost_last = pipecost_last_temp
                    partition = partition_temp
                    stage_comp_time_lst = stage_comp_time_lst_temp
                    stage_for_send_time_lst = stage_for_send_time_lst_temp
                    stage_back_send_time_lst = stage_back_send_time_lst_temp
                    stage_wise_cost_lst = stage_wise_cost_lst_temp
                    
                is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem  = EstimatePeakMemory(partition, model_config, parallel_config, layer_type, cluster_info, gbs, num_mb) # use gpu_type_lst
        else:
            partition, stage_comp_time_lst, _, _, stage_for_send_time_lst, stage_back_send_time_lst = minmax(int(len(cost_e_a100)),
                                                    np.asarray(cost_e_a100), 
                                                    np.asarray(cost_e_a10), 
                                                    np.asarray(cost_c), 
                                                    pp_degree,
                                                    gpu_type_lst)
            is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem  = EstimatePeakMemory(partition, model_config, parallel_config, layer_type, cluster_info, gbs, num_mb) # use gpu_type_lst

            pipecost_last, stage_wise_cost_lst = schedule(pp_degree, num_mb, stage_comp_time_lst, stage_for_send_time_lst, stage_back_send_time_lst)
        
    # translate to ds form, add data parallelism cost
    _, dp_cost_list = dp_cost(config, cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, 
                        amp_config=amp_config, partition=partition, _layer=_layer, gpu_per_node=M, gpu_type_lst=gpu_type_lst)
    
    if model_type != "T5":
        all_reduce_embedding_cost = cost_all_reduce_embedding(model_config, cluster_info, parallel_config, M)
    else:
        all_reduce_embedding_cost = 0

    end2end_stage_latency=[]
    for i in range(len(stage_wise_cost_lst)):
        end2end_stage_latency.append(stage_wise_cost_lst[i] + dp_cost_list[i])
    cost_last = max(end2end_stage_latency) + all_reduce_embedding_cost

    max_latency = max(end2end_stage_latency)
    max_latency_index = end2end_stage_latency.index(max_latency)
    
    dp_side_cost_last = dp_cost_list[max_latency_index]
    if exhaustive["exhaustive"] is True:
        print(f"partition: {partition}")
        print(f"estimated time / pp latency / dp cost / allreduce embedding")
        print(f"{cost_last.item():.4f}, {pipecost_last.item():.4f}, {dp_side_cost_last.item():.4f}, {all_reduce_embedding_cost:.4f}")
        assert False, "Done!"
    
    return rank_map, partition, cost_last, pipecost_last, dp_side_cost_last, all_reduce_embedding_cost, is_oom, oom_gpumem, is_zero_oom, zerooom_gpumem
    

def EstimatePeakMemory(partition, model_config, parallel_config, layer_type, cluster_info, gbs, num_mb):
    h = model_config["hidden_size"] 
    v = model_config["vocab_size"]
    s = model_config["sequence_length"]
    a = model_config["num_attention_heads"]
    tp = parallel_config["mp"]
    pp = parallel_config["pp"] 
    dp = parallel_config["dp"]
    b = parallel_config["micro_bs"]
    gbs = gbs
    num_mb = num_mb
    N = len(cluster_info)
    
    
    memory = []
    memory_zero = []
    p = pp
    if num_mb > pp:
        p = pp
    else:
        p = num_mb
        # if num_mb < 1:
        #     p = 
        # else:
        #     p = num_mb
    # print(f"h: {h} v: {v} s: {s} a: {a} tp: {tp.item()} dp: {dp.item()} pp: {pp.item()} b: {b} N: {N} p: {p}")        
    # print(f"layer_type: {layer_type}")
    # print(f"partition: {partition}")
    st = 0
    en = 0
    for j, stage in enumerate(partition):
        st = 0 + en 
        en += stage
        param_count = 0 # unit: bytes
        activation = 0 # unit: bytes
        # pipeline_buffer = 0
        for i in range(st, en):
            if layer_type[i] == "embedding_layer" :
                param_count += ( h * v ) / tp
                activation += ( s * b * h * pp * p) / tp
                # activation += ( s ) / tp
            elif layer_type[i] == "transformer_layer" or layer_type[i] == "encoder" or layer_type[i] == "decoder":
                param_count += ( 12 * h ** 2 ) / tp
                activation += (s * b * p * h ) * (10 + ( 24 / tp ) + 5 * (a * s) / (h * tp) )  # tensor + sequence 
                # if i == 0:
                    # pipeline_buffer += mbs * seq_len * h # BLH
            elif layer_type[i] == "post_process":
                param_count += ( h * v ) / tp
                activation += (4 * s * b * h ) / tp + ( 4 * s * b * v ) / tp
            # print(f" [{j}]: layer_type: {layer_type[i]} param_count:{param_count.item()} activation: {activation.item()}")
        major = param_count * 18
        major_zero = param_count * (6 + int(12 / dp))
        #print(f"major: {major} major_zero: {major_zero}")
        memory.append((major + activation) / 1024 /1024 /1024)
        memory_zero.append((major_zero + activation) / 1024 / 1024 /1024)
    

    oom = False
    oom_zero = False
    error_percent=1.10
    # oom_gpumem = 0.0
    # zerooom_gpumem = 0.0
    oom_gpumem = max(memory)
    zerooom_gpumem = max(memory_zero)
    # debug    
    # print(f"partition size: {len(partition)}, \n partition: {partition}")
    # print(f"cluster size: {len(cluster_info)}, \n cluster_info: {cluster_info}")
    # print(f"memory size: {len(memory)}, oom_gpumem: {oom_gpumem}, \n {memory}")
    # print(f"memory zero size: {len(memory_zero)}, zerooom_gpumem: {zerooom_gpumem}, \n {memory_zero}")
    
    # print(f"oom_gpumem: {oom_gpumem}, \n {memory}")
    # print(f"zerooom_gpumem: {zerooom_gpumem}, \n {memory_zero}")
    
    for i in range(len(partition)):
        if len(partition) > N:
            a = int(len(partition) / N)
            j = int(i / a)
        elif len(partition) == N:
            j = i
        else:
            j = None

        if j is not None:
            # print(f"cluster_info: {cluster_info[j]}")
            if cluster_info[j][1] == torch.tensor([230 * 8 * 1e9]).float():
                memory_max = 39.59
            else:
                memory_max = 22.20
        else:
            s = i * int(N/len(partition))
            e = (i + 1) * int(N/len(partition)) -1
            for j in range(s, e):
                if cluster_info[j][1] == torch.tensor([230 * 8 * 1e9]).float():
                    memory_max = 39.59
                else:
                    memory_max = 22.20    
        
        # print(f"memory_max: {memory_max}")
        if (memory[i] * error_percent) > memory_max:
            oom = True
            oom_gpumem = memory[i] * error_percent
        
        if (memory_zero[i]) > memory_max:
            oom_zero = True
            zerooom_gpumem = memory_zero[i] * error_percent
    # debug              
    # print(f"is oom: {oom}")
    # print(f"is zero oom: {oom_zero}")
    
    # print(f"is oom_gpumem: {oom_gpumem}")
    # print(f"is zerooom_gpumem: {zerooom_gpumem}")
                
    return oom, oom_gpumem, oom_zero, zerooom_gpumem


def get_layer_type(model_type, n, pp):
    _layer = ["embedding_layer"]
    if model_type != "T5":
        for i in range(n):
            _layer.append("transformer_layer")
        _layer.append("post_process")

    else:
        for i in range(int(n/2)):
            _layer.append("encoder")
        _layer.append("embedding_layer")
        for i in range(int(n/2)):
            _layer.append("decoder")

    return _layer