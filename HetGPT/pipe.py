import copy
import time

import torch
import numpy as np

from amp_utils import axis2rank

def pipe_ast(num_layer, cost_e1, cost_e2, cost_c, pp_degree, num_mb, num_node, gpu_per_node, dp_degree):
    time_s = time.time()

    pp_per_node = pp_degree // num_node
    num_balanced_layer = num_layer // pp_degree
    if pp_degree ==2:
        num_balanced_layer = 24
    if pp_degree ==1:
        num_balanced_layer = 48
    partition = []
    max_latency = 1000000
    for i in range(pp_degree):
        partition.append(num_balanced_layer)
    partition[0] += 1
    partition[-1] += 1

    print("initial partition is", partition)

    while(1):
        stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, pp_per_node, gpu_per_node, dp_degree)

        if pp_degree == 1 or pp_per_node < 1:
            break
            
        # get index of max and value
        print("stage_latency", stage_latency)
        last_max_latency = max(stage_latency)

        if last_max_latency <= max_latency:
            max_latency = last_max_latency
            max_latency_index = stage_latency.index(max_latency)

            min_latency = min(stage_latency)
            min_latency_index = stage_latency.index(min_latency)

            if partition[max_latency_index] == 1:
                break
            if (max_latency_index == 0 or max_latency_index == pp_degree-1) and partition[max_latency_index] == 2:
                break
            partition[max_latency_index] -= 1
            partition[min_latency_index] += 1
            print("changed partition", partition)
        else:
            print("latency is not decreasing, break")
            partition[max_latency_index] += 1
            partition[min_latency_index] -= 1
            break

    print(f"pipe_ast used {time.time()-time_s} seconds with {num_layer} layers and {pp_degree} stages.")
    return partition, stage_latency#TODO:stage_comp, stage_comm # stage-1 dim


def pipe_cost(pp_degree, num_mb, stage_latency):
    print(f"sum of stage_latency is {sum(stage_latency)}")

    max_latency = max(stage_latency)
    print(f"max_latency is {max_latency}")
    cost = (num_mb-1) * max_latency
    print(f"num_mb is {num_mb}, cost is {cost}")
    cost += sum(stage_latency)
    print(f"cost is {cost}")

    # cost = sum(stage_latency)
    # last_stage_latency = stage_latency[-1]
    # cost += (num_mb.item()-1)*last_stage_latency

    return cost



def dp_cost(config, cluster_info,model_config, parallel_config, amp_config, partition):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
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


def get_stage_latency(partition, cost_e1, cost_e2, cost_c, pp_per_node, gpu_per_node, dp_degree):
    stage_latency = []
    num_bw_share = min(gpu_per_node, dp_degree)
    num_stage = len(partition)
    if num_stage==1:
        return [sum(cost_e2)]
    for stage in range(num_stage):
        num_layer_til_last_stage = sum(partition[:stage])
        num_layer_til_cur_stage = sum(partition[:stage+1])
        if pp_per_node >=1:
            if stage < pp_per_node:
                if stage == 0:
                    stage_latency.append(sum(cost_e1[:partition[0]]))
                else:
                    num_layer_til_last_stage = sum(partition[:stage])
                    num_layer_til_cur_stage = sum(partition[:stage+1])
                    stage_latency.append(sum(cost_e1[num_layer_til_last_stage:num_layer_til_cur_stage]) \
                                        + 2*cost_c[sum(partition[:stage])][stage-1]*num_bw_share)
            else:
                stage_latency.append(sum(cost_e2[num_layer_til_last_stage:num_layer_til_cur_stage]) \
                                    + 2*cost_c[sum(partition[:stage])][stage-1]*num_bw_share)
        else:
            if num_stage ==1:
                stage_latency = [max(np.sum(cost_e1), np.sum(cost_e2))]
            else:
                stage_latency.append(sum(cost_e2[num_layer_til_last_stage:num_layer_til_cur_stage])\
                                    + 2*cost_c[sum(partition[:stage])][stage-1])
        if stage == num_stage-1:
            # Substract the activation communication cost from the last stage
            # since the backward of the last stage does not need to communicate
            stage_latency[-1] -= cost_c[sum(partition[:stage])][stage-1]

    return stage_latency

