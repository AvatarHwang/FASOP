from collections import defaultdict

import os

import torch
import torch.nn as nn
import numpy as np

from amp_utils import axis2rank
from pipe import pipe_ast, pipe_cost, get_stage_latency

home_dir = os.environ['HOME'] 

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

        

        self.profile_cost1 = {}
        self.profile_cost2 = {}
        for mp_size in [1,2,4]:
            # known_cost directory stores the real forward time with correponding model parallel degree.
            known_record = f"known_cost/{self.model_type}_A100_{mp_size}" 
            cur_profile_cost1 = 3 * np.load(f"{known_record}.npy")
            cur_profile_cost1 = np.append(cur_profile_cost1, cur_profile_cost1[0])
            known_record = f"known_cost/{self.model_type}_A10_{mp_size}"
            cur_profile_cost2 = 3 * np.load(f"{known_record}.npy")
            cur_profile_cost2 = np.append(cur_profile_cost2, cur_profile_cost2[0])

            self.profile_cost1[str(mp_size)] = cur_profile_cost1
            self.profile_cost2[str(mp_size)] = cur_profile_cost2
        
        if 1==2: #args.profile_per_mbs
            for mbs in [1, 2, 4, 8, 16]:
                for mp_size in [1,2,4]:
                    # known_cost directory stores the real forward time with correponding model parallel degree.
                    known_record = f"known_cost/{self.model_type}_A100_{mp_size}_mbs{mbs}" 
                    cur_profile_cost1 = 2 * np.load(f"{known_record}.npy")
                    known_record = f"known_cost/{self.model_type}_A10_{mp_size}_mbs{mbs}"
                    cur_profile_cost2 = 2 * np.load(f"{known_record}.npy")

                    self.profile_cost1[str(mp_size)+'-'+str(mbs)] = cur_profile_cost1
                    self.profile_cost2[str(mp_size)+'-'+str(mbs)] = cur_profile_cost2
        
    def forward(self, args):
        model_type = self.model_type
        config, bs, micro_bs, cluster_info, model_config, oth = args
        amp_config = {"profile_cost1" : self.profile_cost1,
                      "profile_cost2" : self.profile_cost2}
        oom_flag, rank_map, partition, amp_pred, pipecost, dp_side_cost, all_reduce_embedding_cost = predict(config, bs, micro_bs, cluster_info, model_config, amp_config, oth)
        return oom_flag, rank_map, partition, amp_pred, pipecost, dp_side_cost, all_reduce_embedding_cost
        
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
    
    precision = 16 # TODO: support fp32, should be args.precision

    _layer = ["embedding_layer"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")

    if int(pp.item()) > 1:
        _layer.append("embedding_layer")
    else:
        _layer.append("None")

    _num_layer = len(_layer)

    if pp == 1:
        return torch.zeros(int(n.item())), _layer
      
    # build layer activation lookup table.
    layer_volume = []
    last_volume = torch.zeros(1,)
    for i in range(_num_layer):
        layer_type = _layer[i]
        if layer_type == "embedding_layer" or layer_type == "transformer_layer":
            last_volume = bs * s * h
            layer_volume.append(last_volume)
        else:
            layer_volume.append(last_volume)
            
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
                    slowest_bandwidth = slowest_bandwidth
            for k in range(_num_layer-1):
                cost_c[i][k][j] = layer_volume[k] * precision / slowest_bandwidth
    # cost_c = torch.mean(cost_c, dim=0) 
    cost_c = torch.max(cost_c, dim=0) # max is reasonable, since we are using the slowest connection
    return cost_c.values, _layer

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

    _layer = ["embedding_layer"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")
    if pp>1:
        _layer.extend(["embedding_layer"])
    else:
        _layer.extend(["None"])

    _num_layer = len(_layer)
            
    cost_e = np.zeros((int(dp.item()), _num_layer))

    for i in range(int(dp.item())):
        assert _num_layer == len(profile_cost["1"]), "predicted number of layers not equal to actual"
        
        # cost_e in the main result is equivalent to using profile_cost.
        for layer_id in range(_num_layer):
            layer_type = _layer[layer_id]
            if layer_type == "embedding_layer":
                cur_layer = profile_cost[str(int(mp.item()))][layer_id]
            elif layer_type == "transformer_layer":
                cur_layer = bs * profile_cost[str(int(mp.item()))][layer_id]
            else:
                cur_layer = 0
            cost_e[i][layer_id] = cur_layer
    
    cost_e = torch.from_numpy(np.stack(cost_e, axis=0))            
    cost_e = torch.mean(cost_e, dim=0)
    return cost_e

def cost_all_reduce_embedding(model_config, cluster_info, parallel_config, gpu_per_node):
    precision = 16 # TODO: support fp32, should be args.precision
    tp_degree = int(parallel_config["mp"].item())
    dp_degree = int(parallel_config["dp"].item())
    pp_degree = int(parallel_config["pp"].item())
    rank_map = parallel_config["rank_map"]
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
        embedding_syn_cost = 2*(2-1)*(hidden_size*vocab_size*precision)/(2*band_width) #TODO: caculate the size of embedding, TODO: clearify if we should divide by tp_degree
        return embedding_syn_cost
    else:
        return 0
        


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
    
    _layer = ["embedding_layer"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")
    if pp>1:
        _layer.extend(["embedding_layer"])    
    else:
        _layer.extend(["None"])
    _num_layer = len(_layer)
        
    # First translate to deepspeed partition form
    ds_partition = [0]
    for i in range(len(partition)):
        ds_partition.append(ds_partition[-1]+partition[i])
    assert ds_partition[-1] == _num_layer
    assert len(ds_partition) == pp + 1

    counted = False
    param_count = 0    
    for layer_id in range(ds_partition[0], ds_partition[1]):
        layer_type = _layer[layer_id]
        if layer_type == "embedding_layer":
            if not counted:
                counted = True
                param_count += 164_249_600
        elif layer_type == "transformer_layer":
            param_count += 24 * h ** 2 / mp
            param_count += 3200 * 2 /mp
    
    # Get communication bandwidth of pipeline stage 0
    for j in range(int(mp.item())):
        bandwidth = []
        for k in range(int(dp.item())):
            rank_cur = axis2rank(axis=(0,k,j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
            node_cur = rank_node_map[int(rank_cur.item())]
            rank_next = axis2rank(axis=(0,(k+1)%(dp.item()),j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
            node_next = rank_node_map[int(rank_next.item())]

            if node_cur == node_next:
                connectivity = cluster_info[node_cur][1]
            else:
                connectivity = min(cluster_info[node_cur][0], cluster_info[node_next][0])
            bandwidth.append(connectivity)
    # get slowest of bandwidth
    bandwidth = min(bandwidth)

    # All-reduce cost: 2(n-1)M / nB
    precision = 32 #TODO: precision should be args.precision
    dp_cost = 2 * (int(dp.item()) - 1) * (param_count * precision) / (int(dp.item()) * bandwidth)
                
    return ds_partition, dp_cost

def predict(config, gbs, mbs, cluster_info, model_config, amp_config, oth):
    L = model_config["num_layers"]
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
    
        if pp >= (L + 2):
            print(f"early return with pp={pp}, L={L}")
            return None, None, torch.tensor([float("inf")])
           
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
            
    # infer number of micro-batch size B
    # mbs = gbs / dp
    num_mb = gbs / (dp_degree * mbs)
    mbs = gbs / (dp_degree * num_mb)
            
    parallel_config = {"mp" : tp_degree, "dp" : dp_degree, "pp" : pp_degree, "micro_bs" : mbs, "rank_map" : rank_map, "rank_node_map": rank_node_map}
        
    cost_e1 = get_cost_e(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, profile_cost=amp_config["profile_cost1"])
    cost_e2 = get_cost_e(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, profile_cost=amp_config["profile_cost2"])
    cost_c, layer_type = get_cost_c(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, amp_config=amp_config)
        
    partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst  = pipe_ast(len(cost_e1), np.asarray(cost_e1), np.asarray(cost_e2), np.asarray(cost_c), int(pp_degree.item()), int(num_mb.item()), N, M, dp_degree)
    
    print(f"Step 1 partition: {partition}")

    pipecost_last, _ = pipe_cost(pp_degree, num_mb, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst)
    # translate to ds form, add data parallelism cost
    ds_partition_last, dp_side_cost_last = dp_cost(config, cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, 
                        amp_config=amp_config, partition=partition)
    
    all_reduce_embedding_cost = cost_all_reduce_embedding(model_config, cluster_info, parallel_config, M)

    cost_last = pipecost_last + dp_side_cost_last + all_reduce_embedding_cost

    if pp_degree>1:
        while(1):
            min_latency = min(stage_time_lst)
            min_latency_index = stage_time_lst.index(min_latency)
            if partition[0] <= 2:
                break
            partition[min_latency_index] += 1
            partition[0] -= 1

            # update stage_latency
            pp_degree = int(pp_degree)
            pp_per_node = int(pp_degree / N)

            # update stage_latency
            stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, pp_per_node, M, dp_degree)

            stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
            stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
            stage_time_lst = [stage.get_stage_time() for stage in stage_latency]

            pipecost, _ = pipe_cost(pp_degree, num_mb, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst)       
            # translate to ds form, add data parallelism cost
            ds_partition, dp_side_cost = dp_cost(config, cluster_info=cluster_info, 
                                model_config=model_config, parallel_config=parallel_config, 
                                amp_config=amp_config, partition=partition)               
            cost_current = pipecost + dp_side_cost + all_reduce_embedding_cost
            print("HetGPT changed partition: ", partition, "cost: ", cost_current.item(), "cost_last: ", cost_last.item())
            if cost_current < cost_last:
                ds_partition_last = ds_partition
                pipecost_last = pipecost
                dp_side_cost_last = dp_side_cost
                cost_last = cost_current
                print("continue to change partition")
            else:
                print("Total latency increased, revert to previous partition.")
                partition[min_latency_index] -= 1
                partition[0] += 1
                print("HetGPT reverted partition: ", partition, "cost: ", cost_current.item(), "cost_last: ", cost_last.item())
                break
    oom_flag = EstimatePeakMemory(partition, model_config, parallel_config, layer_type)

    return oom_flag, rank_map, partition, cost_last, pipecost_last, dp_side_cost_last, all_reduce_embedding_cost
    

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
            if layer_type[i] == "embedding_layer":
                param += 164249600
                avtivation += mbs * seq_len * hidden_size
            elif layer_type[i] == "transformer_layer":
                param += 24 * hidden_size ** 2 / tp_degree
                param += 3200 * 2 # LayerNorm
                avtivation += 9 * mbs * seq_len * hidden_size + mbs * seq_len ** 2 * num_head # reference: https://arxiv.org/pdf/2110.05722.pdf
                if i == 0:
                    pipeline_buffer += mbs * seq_len * hidden_size # BLH
            else:
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

    gpu_memory = 24 #TODO: get gpu memory from args
    if peak_memory > gpu_memory:
        print(f"peak memory is larger than gpu memory. MBS: {mbs.item()}, TP: {tp_degree.item()}, PP: {parallel_config['pp'].item()}, partition: {partition}")
        print(f"peak memory is {peak_memory} GB")
        return False
    else:
        print("peak memory is smaller than gpu memory")
        print(f"peak memory is {peak_memory} GB")
        return True

            