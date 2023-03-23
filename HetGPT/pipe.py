import copy
import time

import torch
import numpy as np

from amp_utils import rank2axis, axis2rank, get_host

def pipe_ast(num_layer, cost_e1, cost_e2, cost_c, pp_degree, num_mb, num_node):
    time_s = time.time()

    pp_per_node = pp_degree // num_node
    num_balanced_layer = num_layer // pp_degree
    if pp_degree ==2:
        num_balanced_layer = 24
    if pp_degree ==1:
        num_balanced_layer = 48
    partition = []
    max_latency = 1000000
    last_max_latency = 1000000
    for i in range(pp_degree):
        partition.append(num_balanced_layer)
    partition[0] += 1
    partition[-1] += 1

    print("initial partition", partition)

    while(1):
        if pp_degree>=4:
            stage_latency = []
            for i in range(pp_degree):
                if i < pp_per_node:
                    if i == 0:
                        stage_latency.append(np.sum(cost_e1[:partition[i]]))
                    else:
                        stage_latency.append(np.sum(cost_e1[sum(partition[:i]):sum(partition[:i+1])] \
                        + cost_c[sum(partition[:i])][i-1]))
                else:
                    stage_latency.append(np.sum(cost_e2[sum(partition[:i]):sum(partition[:i+1])] \
                    + cost_c[sum(partition[:i])][i-1]))
            
            # get index of max and value
            print("stage_latency", stage_latency)
            temp_max_latency = max(stage_latency)

            if temp_max_latency <= max_latency:
                max_latency = temp_max_latency
                last_max_latency = temp_max_latency
                max_latency_index = stage_latency.index(max_latency)

                min_latency = min(stage_latency)
                min_latency_index = stage_latency.index(min_latency)

                if partition[max_latency_index] == 1:
                    break
                if max_latency_index == 0 and partition[max_latency_index] == 2:
                    break
                partition[max_latency_index] -= 1
                partition[min_latency_index] += 1
                print("changed partition", partition)
            else:
                break
        else:
            stage_latency = []
            if pp_degree ==1:
                partition=[50]
                stage_latency = [min(np.sum(cost_e1), np.sum(cost_e2))]
                break
            if pp_degree == 2:
                stage_latency.append(np.sum(cost_e1[:partition[0]]))
                stage_latency.append(np.sum(cost_e2[partition[0]:sum(partition[:2])]) + cost_c[partition[0]][0])
                temp_max_latency = max(stage_latency)
                if temp_max_latency <= max_latency:
                    max_latency = temp_max_latency
                    print("max_latency", max_latency)
                    print("last_max_latency", last_max_latency)
                    last_max_latency = temp_max_latency
                    max_latency_index = stage_latency.index(max_latency)

                    min_latency = min(stage_latency)
                    min_latency_index = stage_latency.index(min_latency)
                    if partition[max_latency_index] == 1:
                        break
                    partition[max_latency_index] -= 1
                    partition[min_latency_index] += 1
                else:
                    break

    print(f"pipe_ast used {time.time()-time_s} seconds with {num_layer} layers and {pp_degree} stages.")
    return partition, stage_latency


def pipe_uniform(L, pp):
    each = L // pp
    remain = L - pp * each
    ret = [each]
    for i in range(pp-1):
        ret.append(each)
    for i in range(remain):
        ret[i] += 1

    return ret, None

def pipe_cost(pp_degree, num_mb, stage_latency):
    max_latency = max(stage_latency)
    cost = (num_mb-1) * max_latency
    cost += sum(stage_latency)
    return cost

def dp_cost(config, cluster_info,model_config, parallel_config, amp_config, partition):
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
    i=1
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
                    param_count += h * v / mp
            elif layer_type == "transformer_layer":
                param_count += 12 * h ** 2 / mp
            elif layer_type == "noop":
                pass
            else:
                raise RuntimeError("Unknown layer type.")
                    
        cur_dp = dp_const * param_count
        if cur_dp > max_dp:
            max_dp = cur_dp
                
    return ds_partition, max_dp