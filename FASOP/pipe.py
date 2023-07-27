import torch
import time
import numpy as np
import copy

from stage import PPGroup


class Stage:
    def __init__(self):
        self.comm_time = 0.
        self.comp_time = 0.
        self.for_send_time = 0.
        self.back_send_time = 0.

    def set_comp_time(self, comp_time):
        self.comp_time = comp_time

    def set_comm_time(self, comm_time):
        self.comm_time = comm_time
    
    def set_for_send_time(self, for_send_time):
        self.for_send_time = for_send_time
    
    def set_back_send_time(self, back_send_time):
        self.back_send_time = back_send_time

    def get_comp_time(self):
        return self.comp_time
    
    def get_comm_time(self):
        return self.comm_time
    
    def get_for_send_time(self):
        return self.for_send_time

    def get_back_send_time(self):
        return self.back_send_time

    def get_stage_time(self):
        return self.comm_time+self.comp_time


def minmax(num_layer, cost_e1, cost_e2, cost_c, pp_degree, gpu_type_lst):


    num_balanced_layer = num_layer // pp_degree
    partition = []
    for i in range(pp_degree):
        partition.append(num_balanced_layer)
    rest = int(num_layer - (num_balanced_layer * pp_degree))
    for i in range(rest):
        partition[i-1] += 1

    partition_history = []
    partition_history.append(partition[:])

    last_max_latency = 1000000
    counted = False
    while(1):
        stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
        stage_time_lst = [stage.get_stage_time() for stage in stage_latency]

        max_latency = max(stage_time_lst)
        if max_latency > last_max_latency:
            if counted:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
                stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
                break
        if max_latency == last_max_latency:
            if counted and partition in partition_history[:-1]:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
                stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
                break
        last_max_latency = max_latency
        max_latency_index = stage_time_lst.index(max_latency)

        min_latency = min(stage_time_lst)
        min_latency_index = stage_time_lst.index(min_latency)

        if (max_latency_index == 0 or max_latency_index == pp_degree-1) and partition[max_latency_index] == 2:
            if counted:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
            break
        if partition[max_latency_index]>1:
            partition[max_latency_index] -= 1
            partition[min_latency_index] += 1
            counted=True
            partition_history.append(partition[:])
        else: # no layers to substract
            break
    
    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst


def explain_minmax(num_layer, cost_e1, cost_e2, cost_c, pp_degree, gpu_type_lst):

    num_balanced_layer = num_layer // pp_degree
    partition = []
    for i in range(pp_degree):
        partition.append(num_balanced_layer)
    rest = int(num_layer - (num_balanced_layer * pp_degree))
    for i in range(rest):
        partition[i-1] += 1

    print(f"\ngpu type list: {gpu_type_lst}")
    print(f"initial partition: {partition}")
    partition_history = []
    partition_history.append(partition[:])

    last_max_latency = 1000000
    counted = False
    while(1):
        stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
        stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
        print(stage_time_lst)

        max_latency = max(stage_time_lst)
        if max_latency > last_max_latency:
            if counted:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
                stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
                print(f"Final partition: {partition}")
                break
        if max_latency == last_max_latency:
            if counted and partition in partition_history[:-1]:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
                stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
                print(f"Final partition: {partition}")
                break
        last_max_latency = max_latency
        max_latency_index = stage_time_lst.index(max_latency)

        min_latency = min(stage_time_lst)
        min_latency_index = stage_time_lst.index(min_latency)

        if (max_latency_index == 0 or max_latency_index == pp_degree-1) and partition[max_latency_index] == 2:
            if counted:
                partition[max_latency_index] += 1
                partition[min_latency_index] -= 1
                print(f"Final partition: {partition}")
            break
        if partition[max_latency_index]>1:
            partition[max_latency_index] -= 1
            partition[min_latency_index] += 1
            counted=True
            partition_history.append(partition[:])
    
    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]
    print("partition history", partition_history)

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst


def get_stage_latency(partition, cost_e_a100, cost_e_a10, cost_c, gpu_type_lst):
    
    num_bw_share = 1 # which should be caculated in get_cost_c considering PCIe
    num_stage = len(partition)

    stage_latency = [Stage() for _ in range(num_stage)]

    if num_stage==1:
        if gpu_type_lst[0] == 'A10':
            stage_latency[0].set_comp_time(sum(cost_e_a10))
            return stage_latency
        elif gpu_type_lst[0] == 'A100':
            stage_latency[0].set_comp_time(sum(cost_e_a100))
            return stage_latency
        else:
            assert False, "gpu type is not recognized"
    
    for stage in range(num_stage):
        num_layer_til_last_stage = sum(partition[:stage])
        num_layer_til_cur_stage = sum(partition[:stage+1])
        node_idx = stage
        if gpu_type_lst[node_idx] == 'A100':
            cost_e=cost_e_a100
        elif gpu_type_lst[node_idx] == 'A10':
            cost_e=cost_e_a10
        else:
            assert False, "gpu type is not recognized"

        if stage == 0:
            stage_latency[stage].set_comp_time(sum(cost_e[:num_layer_til_cur_stage]))
            stage_latency[stage].set_for_send_time((cost_c[sum(partition[:stage])][stage]*num_bw_share).item())
        elif stage == num_stage-1:
            stage_latency[stage].set_comp_time(sum(cost_e[num_layer_til_last_stage:num_layer_til_cur_stage]))
            stage_latency[stage].set_back_send_time((cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())
        else:
            stage_latency[stage].set_comp_time(sum(cost_e[num_layer_til_last_stage:num_layer_til_cur_stage]))
            stage_latency[stage].set_comm_time((cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())
            
    return stage_latency



def schedule(pp_degree, num_mb, stage_comp_time_lst, stage_for_send_time_lst, stage_back_send_time_lst):

    ppgroup_cfg = {"num_mb": None,
                   "pp_degree": None,
                   "stage_comp_time_lst": stage_comp_time_lst,
                   "stage_for_send_time_lst": stage_for_send_time_lst,
                   "stage_back_send_time_lst": stage_back_send_time_lst
                   }

    if isinstance(num_mb, torch.Tensor):
        ppgroup_cfg["num_mb"] = int(num_mb.item())
    else:
        ppgroup_cfg["num_mb"] = num_mb
    
    if isinstance(pp_degree, torch.Tensor):
        ppgroup_cfg["pp_degree"] = int(pp_degree.item())
    else:
        ppgroup_cfg["pp_degree"] = pp_degree

    if ppgroup_cfg["pp_degree"] == 1:
        cost = num_mb * sum(stage_comp_time_lst)

    else:    
        my_pp_group = PPGroup(**ppgroup_cfg)
        
        my_pp_group.simulate_full_pipeline()
        cost = my_pp_group.get_pipe_cost()

    if not isinstance(cost, torch.Tensor):
        cost = torch.tensor(cost)

    if ppgroup_cfg["pp_degree"] == 1:
        stage_wise_cost_lst = [cost]
    else:
        stage_wise_cost_lst = my_pp_group.get_stagewise_end_time_lst()

    return cost, stage_wise_cost_lst


def dynamic_programming(L, cost_e_a100, cost_e_a10, cost_c, pp, num_mb, gpu_type_list):
    """
    Model partitioning method coded by AMP
    converted output for FASOP
    """
    time_dp_s = time.time()
    possible = [0]

    if pp==1:
        if gpu_type_list[0]=="A100":
            cost_e = cost_e_a100
        else:
            cost_e = cost_e_a10
        S = [L]
        stage_latency = get_stage_latency(S, cost_e_a100, cost_e_a10, cost_c, gpu_type_list)
        stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
        stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
        stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
        stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
        stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]

        return S, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst

    else:
        for stage in range(pp):
            if gpu_type_list[stage]=='A100':
                cost_e = cost_e_a100
            elif gpu_type_list[stage]=='A10':
                cost_e = cost_e_a10
            else:
                raise ValueError(f"Unrecognized gpu type: {gpu_type_list[stage]}")
            for i in range(1, L+1):
                ptr = 0
                while ptr + i <= L:
                    possible.append(sum(cost_e[ptr:ptr+i]))
                    ptr += 1
        
        possible = sorted(list(set(possible)))
        trace = []
        for i in range(L):
            outer = []
            for j in range(pp):
                inner = []
                for m in range(len(possible)):
                    inner.append(([],np.infty))
                outer.append(inner)
            trace.append(outer)
        
        for i in range(L): # num layer
            for j in range(pp): 
                if gpu_type_list[j]=='A100':
                    cost_e = cost_e_a100
                elif gpu_type_list[j]=='A10':
                    cost_e = cost_e_a10
                else:
                    raise ValueError(f"Unrecognized gpu type: {gpu_type_list[j]}")
                for m in range(len(possible)):
                    if i+1 <= j: # invalid
                        pass
                    else:
                        if j == 0: # base case: 0 cut
                            cur_sum = sum(cost_e[:i+1])
                            trace[i][j][m] = ([i+1], (num_mb-1) * max(0, cur_sum - possible[m]))
                        else:
                            cost_best = np.infty
                            S_best = []
                            for cut in range(j-1, i):
                                cur_sum = sum(cost_e[cut+1:i+1])
                                assert cur_sum in possible
                                S, cost_ = trace[cut][j-1][possible.index(max(cur_sum, possible[m]))]
                                cost_ += (num_mb-1) * max(0, cur_sum - possible[m])
                                cost_ += cost_c[cut][j-1]
                                if cost_ < cost_best:
                                    cost_best = cost_ - cost_c[cut][j-1]
                                    S_ = copy.deepcopy(S)
                                    S_.append(i-cut)
                                    S_best = S_
                            trace[i][j][m] = (S_best, cost_best)
                            
        time_dp_used = time.time() - time_dp_s
        
        # add each stage cost at the end 
        S, cost = trace[L-1][pp-1][0]
        print(f"dynamic programming used {round(time_dp_used,2)} seconds with {L} layers and {pp} stages.")
        
        stage_latency = get_stage_latency(S, cost_e_a100, cost_e_a10, cost_c, gpu_type_list)
        stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
        stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
        stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
        stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
        stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]

        return S, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst