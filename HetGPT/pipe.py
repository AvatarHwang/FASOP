import copy
import time

import torch
import numpy as np

from amp_utils import rank2axis, axis2rank, get_host

def pipe_ast(L, cost_e, cost_c, k, B):
    time_dp_s = time.time()
    possible = [0]
    
    for i in range(1, L+1):
        ptr = 0
        while ptr + i <= L:
            possible.append(sum(cost_e[ptr:ptr+i]))
            ptr += 1
    
    possible = sorted(list(set(possible)))
    trace = []
    for i in range(L):
        outer = []
        for j in range(k):
            inner = []
            for m in range(len(possible)):
                inner.append(([],np.infty))
            outer.append(inner)
        trace.append(outer)
    
    for i in range(L):
        for j in range(k):
            for m in range(len(possible)):
                if i+1 <= j: # invalid
                    pass
                else:
                    if j == 0: # base case: 0 cut
                        cur_sum = sum(cost_e[:i+1])
                        assert cur_sum in possible
                        trace[i][j][m] = ([i+1], (B-1) * max(0, cur_sum - possible[m]))
                    else:
                        cost_best = np.infty
                        S_best = []
                        for cut in range(j-1, i):
                            cur_sum = sum(cost_e[cut+1:i+1])
                            assert cur_sum in possible
                            S, cost_ = trace[cut][j-1][possible.index(max(cur_sum, possible[m]))]
                            cost_ += (B-1) * max(0, cur_sum - possible[m])
                            cost_ += cost_c[cut][j-1]
                            if cost_ < cost_best:
                                cost_best = cost_ - cost_c[cut][j-1]
                                S_ = copy.deepcopy(S)
                                S_.append(i-cut)
                                S_best = S_
                        trace[i][j][m] = (S_best, cost_best)
                        
    time_dp_used = time.time() - time_dp_s
    
    # add each stage cost at the end 
    S, cost = trace[L-1][k-1][0]
    cost += np.sum(cost_e)
    print(f"pipe_ast used {round(time_dp_used,2)} seconds with {L} layers and {k} stages.")
    return (S, cost)

def pipe_ds(L, cost_e, cost_c, k, B):
    per_stage = L // k
    s = [int(per_stage.item())] * (int(k.item())-1)
    s.append(int(L.item())-sum(s))
    p = [s[0]-1]
    
    for i in range(1, int(k.item())):
        p.append(p[i-1] + s[i])
    lens = torch.reshape(torch.sum(cost_e[:p[0]+1]), (-1,1))
    
    for i in range(len(s)-1):
        lens = torch.cat([lens,torch.reshape(torch.sum(cost_e[p[i]+1:p[i+1]+1]), (-1,1))])
        
    max_l = torch.max(lens)
    cost = (B-1) * max_l
    for i in range(int(k.item())-1):
        cost += cost_c[p[i]][i]
    cost += torch.sum(cost_e)
    return s, cost

def pipe_gpt2(L, pp):
    each = L // pp
    remain = L - pp * each
    start = 2
    ret = [start + each]
    for i in range(pp-1):
        ret.append(each)
    for i in range(remain):
        ret[i] += 1
    ret[-1] += 4
    #print(f"-----------{ret}. {L}, {pp}")
    return ret, None

def pipe_uniform(L, pp):
    #print("using a uniform")
    each = L // pp
    remain = L - pp * each
    ret = [each]
    for i in range(pp-1):
        ret.append(each)
    for i in range(remain):
        ret[i] += 1
    # print(f"pipe uniform returns {ret}")
    # print(f"-----------{ret}. {L}, {pp}")
    return ret, None

def pipe_cost(L, cost_e, cost_c, k, B, partition):
    s = partition
    p = [s[0]-1]
    
    for i in range(1, int(k.item())):
        p.append(p[i-1] + s[i])
    lens = torch.reshape(torch.sum(cost_e[:p[0]+1]), (-1,1))
    #print(f"calculating cost: {cost_e} {cost_c} {k} {B} {partition}")
    for i in range(len(s)-1):
        lens = torch.cat([lens,torch.reshape(torch.sum(cost_e[p[i]+1:p[i+1]+1]), (-1,1))])
        
    max_l = torch.max(lens)
    cost = (B-1) * max_l
    for i in range(int(k.item())-1):
        cost += cost_c[p[i]][i]
    cost += torch.sum(cost_e)
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
    #_layer = ["embed2h", "noop"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")
    
    _layer.extend(["noop"])
    #_layer.extend(["noop","noop", "embed2v", "noop"])
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
                        param_count += h * v / mp
                elif layer_type == "transformer_layer":
                    param_count += 12 * h ** 2 / mp
                elif layer_type == "noop":
                    pass
                else:
                    raise RuntimeError("Unknown layer type.")
                        
            #print(f"dp: {dp_const} and param {param_count}")
            cur_dp = dp_const * param_count
            if cur_dp > max_dp:
                max_dp = cur_dp
                
    return ds_partition, max_dp