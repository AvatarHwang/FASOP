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

    partition_history = []
    partition_history.append(partition[:])

    last_max_latency = 1000000
    counted = False
    while(1):
        stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
        stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
        print(f"partition : {partition}\n stage_time_lst : {stage_time_lst}")

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


def get_max_submodel(num_layer, cost_e1, cost_e2, cost_c, pp_degree, gpu_type_lst):

    partition = [14, 5, 5, 5, 5, 5, 5, 6]
    print(f"partition : {partition}")
    stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
    
    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]

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


def exhaustive_partition(num_layer, cost_e1, cost_e2, cost_c, pp_degree, gpu_type_lst):

    s_time = time.time()
    P = compositions(num_layer, pp_degree)
    max_latency = np.inf
    for p in P:
        cur_latency = get_stage_latency(p, cost_e1, cost_e2, cost_c, gpu_type_lst)
        stage_time_lst = [stage.get_comp_time() for stage in cur_latency]
        
        if max(stage_time_lst) < max_latency:
            partition = p[:]
            stage_latency = cur_latency
            max_latency = max(stage_time_lst)

    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]
    print(f"exhaustive_partition: {time.time()-s_time:.4f} sec")
    print(f"partition: {partition}")

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst
    
        
from itertools import permutations
def compositions(n, k):
    def inner(n, k):
        if k == 1:
            yield (n,)
        else:
            for i in range(1, n):
                for rest in inner(n-i, k-1):
                    yield (i,) + rest
    return list(inner(n, k))

def dynamic_programming(L, cost_e_a100, cost_e_a10, cost_c, pp, num_mb, gpu_type_list):
    """
    Model partitioning method coded by AMP
    converted output for FASOP
    """
    time_dp_s = time.time()

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

    possible = [0]
    for cost_e in [cost_e_a10, cost_e_a100]:
        for i in range(1, L+1):
            ptr = 0
            while ptr + i <= L:
                possible.append(sum(cost_e[ptr:ptr+i]))
                ptr += 1

    cost_e = [cost_e_a100 if gpu_type_list[i]=="A100" else cost_e_a10 for i in range(pp)]

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
    
    for i in range(L):
        for j in range(pp):
            for m in range(len(possible)):
                if i+1 <= j: # invalid
                    pass
                else:
                    if j == 0: # base case: 0 cut
                        comp_t = sum(cost_e[j][:i+1])
                        comp_t = max(comp_t, possible[m])
                        #cost_ = (num_mb-1) * max(0, cur_sum - possible[m])
                        stage_time = comp_t * (num_mb-1)
                        trace[i][j][m] = ([i+1], stage_time)
                    else:
                        cost_best = np.infty
                        S_best = []
                        for cut in range(j-1, i):
                            comp_t = sum(cost_e[j][cut+1:i+1])
                            S, past_stage_time = trace[cut][j-1][possible.index(max(comp_t, possible[m]))]
                            cur_stage_time = comp_t
                            cur_stage_time += cost_c[cut][j-1]
                            if j != pp-1:
                                cur_stage_time += cost_c[cut][j]
                            cur_stage_time = cur_stage_time * (num_mb-1)
                            stage_time = max(cur_stage_time, past_stage_time)
                            if stage_time < cost_best:
                                cost_best = stage_time
                                S_ = copy.deepcopy(S)
                                S_.append(i-cut)
                                S_best = S_
                        trace[i][j][m] = (S_best, cost_best)
                            
    time_dp_used = time.time() - time_dp_s
    
    # add each stage cost at the end 
    S, cost = trace[L-1][pp-1][0]
    print(f"dynamic programming used {round(time_dp_used,2)} seconds with {L} layers and {pp} stages.")
    print(f"S: {S}, cost: {cost}")
    
    stage_latency = get_stage_latency(S, cost_e_a100, cost_e_a10, cost_c, gpu_type_list)
    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]

    return S, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst



def ILP(num_layer, cost_e1, cost_e2, cost_c, pp_degree, gpu_type_lst, num_mb):
    s_time = time.time()
    import docplex.mp

    from docplex.mp.model import Model
    
    # get cplex solver
    m = Model(name='partitioning')
    
    # create variables (#layers for each stage)
    layers_per_stage = []
    for i in range(pp_degree):
        layers_per_stage.append(m.integer_var(name=f'stage_{i}'))
    
    # add constraints
    m.add_constraint(m.sum(layers_per_stage) == num_layer-2)

    # make a list of cost_e, which is a list of cost_e1 or cost_e2 for each stage
    cost_e =[]
    for i in range(pp_degree):
        if gpu_type_lst[i] == 'A100':
            cost_e.append(cost_e1)
        elif gpu_type_lst[i] == 'A10':
            cost_e.append(cost_e2)
        else:
            assert False, "gpu type is not recognized"

    # calculate embedding and communication time for each stage, which is a constant
    embedding_and_comm_time = []
    for i in range(pp_degree):
        if i == 0:
            embedding_and_comm_time.append(cost_e[i][0])
        elif i == pp_degree-1:
            embedding_and_comm_time.append(cost_e[i][-1])
        else:
            embedding_and_comm_time.append(0)
        embedding_and_comm_time[i] += cost_c[0][i] # 0 because comm volume is always same for Transformers. TODO: change this for other models
        if i != 0 and i != pp_degree-1:
            embedding_and_comm_time[i] += cost_c[0][i-1]

    # m.minimize(m.max(layers_per_stage[i]*cost_e[i][1]+embedding_and_comm_time[i] for i in range(pp_degree)))
    # If you'd like to minimize pipeline latency not the max latency, use the following line instead of the above line.
    m.minimize((num_mb-1)*m.max(layers_per_stage[i]*cost_e[i][1]+embedding_and_comm_time[i] \
                for i in range(pp_degree)) + \
                m.sum(layers_per_stage[i]*cost_e[i][1]+embedding_and_comm_time[i] \
                for i in range(pp_degree)))
    m.print_information()
    
    while m.solve():
        s = m.solution
        partition = []
        for i in range(pp_degree):
            partition.append(int(s.get_value(layers_per_stage[i])))
            if i == 0 or i == pp_degree-1:
                partition[i] += 1
        print(f" solution: {partition}")
        m.add_constraint(m.max(layers_per_stage[i]*cost_e[i][1]+embedding_and_comm_time[i] for i in range(pp_degree)) \
                        <= 0.99 * m.max(int(s.get_value(layers_per_stage[i]))*cost_e[i][1]+embedding_and_comm_time[i] for i in range(pp_degree)))
        # m.add_constraint(m.sum(layers_per_stage[i]*cost_e[i][1]+embedding_and_comm_time[i] for i in range(pp_degree)) \
        #                 <= 0.99 * m.sum(int(s.get_value(layers_per_stage[i]))*cost_e[i][1]+embedding_and_comm_time[i] for i in range(pp_degree)))

    print(f"ILP: {time.time()-s_time:.4f} sec")

    stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
    
    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]
    stage_for_send_time_lst = [stage.get_for_send_time() for stage in stage_latency]
    stage_back_send_time_lst = [stage.get_back_send_time() for stage in stage_latency]

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst, stage_for_send_time_lst, stage_back_send_time_lst