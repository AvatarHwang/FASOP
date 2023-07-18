import torch

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


def minmax(num_layer, cost_e1, cost_e2, cost_c, pp_degree, num_node, gpu_per_node, gpu_type_lst):

    num_balanced_layer = num_layer // pp_degree
    partition = []
    for i in range(pp_degree):
        partition.append(num_balanced_layer)
    rest = int(num_layer - (num_balanced_layer * pp_degree))
    for i in range(rest):
        partition[i-1] += 1

    last_max_latency = 1000000
    counted = False
    while(1):
        stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, gpu_type_lst)
        stage_time_lst = [stage.get_stage_time() for stage in stage_latency]

        max_latency = max(stage_time_lst)

        if max_latency >= last_max_latency:
            if counted:
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
        cost_e = cost_e_a100
        for i in range(len(gpu_type_lst)):
            if gpu_type_lst[i] == 'A10':
                stage_latency[0].set_comp_time(sum(cost_e_a10))
                return stage_latency
        stage_latency[0].set_comp_time(sum(cost_e_a100))
        return stage_latency
    
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
            stage_latency[stage].set_comp_time(sum(cost_e[:partition[0]]))
            stage_latency[stage].set_for_send_time((cost_c[sum(partition[:stage])][stage]*num_bw_share).item())
        elif stage == num_stage-1:
            stage_latency[stage].set_comp_time(sum(cost_e[num_layer_til_last_stage:num_layer_til_cur_stage]))
            stage_latency[stage].set_back_send_time((cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())
        else:
            stage_latency[stage].set_comp_time(sum(cost_e[num_layer_til_last_stage:num_layer_til_cur_stage]))
            stage_latency[stage].set_comm_time(2*(cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())
            
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