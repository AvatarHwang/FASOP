import copy
import time

import torch
import numpy as np

from amp_utils import axis2rank
from stage import PPGroup

class Stage:

    def __init__(self):
        self.comm_time = 0.
        self.comp_time = 0.

    # def __repr__(self) -> str:
    #     string = "stage has comp time: " + str(self.set_comp_time)
    #     string += " stage has comm time: " + str(self.set_comm_time)
    #     return string

    def set_comp_time(self, comp_time):
        self.comp_time = comp_time

    def set_comm_time(self, comm_time):
        self.comm_time = comm_time
    
    def get_comp_time(self):

        return self.comp_time
    
    def get_comm_time(self):
        
        return self.comm_time
    
    def get_stage_time(self):

        return self.comm_time+self.comp_time

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
    if pp_degree == 32:
        partition = [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2]
    partition[0] += 1
    partition[-1] += 1

    print("initial partition is", partition)

    while(1):
        stage_latency = get_stage_latency(partition, cost_e1, cost_e2, cost_c, pp_per_node, gpu_per_node, gpu_per_node*num_node, pp_degree)
        stage_time_lst = [stage.get_stage_time() for stage in stage_latency]

        if pp_degree == 1 or pp_per_node < 1:
            break
            
        # get index of max and value

        # print("stage_latency", stage_latency)
        print("stage_latency", stage_time_lst)

        last_max_latency = max(stage_time_lst)
        # last_max_latency = max(stage_latency)

        if last_max_latency <= max_latency:
            max_latency = last_max_latency
            max_latency_index = stage_time_lst.index(max_latency)
            # max_latency_index = stage_latency.index(max_latency)

            min_latency = min(stage_time_lst)
            min_latency_index = stage_time_lst.index(min_latency)

            # min_latency = min(stage_latency)
            # min_latency_index = stage_latency.index(min_latency)

            if partition[max_latency_index] == 1:
                break
            if (max_latency_index == 0 or max_latency_index == pp_degree-1) and partition[max_latency_index] == 2:
                break
            partition[max_latency_index] -= 1
            partition[min_latency_index] += 1
            #print("changed partition", partition)
        else:
            #print("latency is not decreasing, break")
            partition[max_latency_index] += 1
            partition[min_latency_index] -= 1
            break

    print(f"pipe_ast used {time.time()-time_s} seconds with {num_layer} layers and {pp_degree} stages.")
    
    stage_time_lst = [stage.get_stage_time() for stage in stage_latency]
    stage_comp_time_lst = [stage.get_comp_time() for stage in stage_latency]
    stage_comm_time_lst = [stage.get_comm_time() for stage in stage_latency]

    return partition, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst
    # return partition, stage_latency#TODO:stage_comp, stage_comm # stage-1 dim


def pipe_cost(pp_degree, num_mb, stage_comp_time_lst, stage_comm_time_lst, stage_time_lst):

    ppgroup_cfg = {"num_mb": None,
                   "pp_degree": None,
                   "stage_comp_time_lst": stage_comp_time_lst,
                   "p2p_time_lst": stage_comm_time_lst
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

    cost = torch.tensor(cost)

    print("estimated pipeline latency:", cost.item())

    if ppgroup_cfg["pp_degree"] == 1:
        stage_wise_cost_lst = [cost]
    else:
        stage_wise_cost_lst = my_pp_group.get_stagewise_end_time_lst()

    return cost, stage_wise_cost_lst


def get_stage_latency(partition, cost_e1, cost_e2, cost_c, pp_per_node, gpu_per_node, world_size, pp_degree):
    
    # stage_latency = []
    #num_bw_share = min(gpu_per_node, world_size/pp_degree)
    num_bw_share = 1 # which should be caculated in get_cost_c considering PCIe
    num_stage = len(partition)

    stage_latency = [Stage() for _ in range(num_stage)]

    if num_stage==1:
        stage_latency[0].set_comp_time(sum(cost_e2))
        return stage_latency
        # return [sum(cost_e2)]
    
    for stage in range(num_stage):
        num_layer_til_last_stage = sum(partition[:stage])
        num_layer_til_cur_stage = sum(partition[:stage+1])
        if pp_per_node >=1:
            if stage < pp_per_node:
                if stage == 0:
                    stage_latency[stage].set_comp_time(sum(cost_e1[:partition[0]]))
                    # stage_latency.append(sum(cost_e1[:partition[0]]))
                else:
                    num_layer_til_last_stage = sum(partition[:stage])
                    num_layer_til_cur_stage = sum(partition[:stage+1])
                    
                    stage_latency[stage].set_comp_time(sum(cost_e1[num_layer_til_last_stage:num_layer_til_cur_stage]))
                    stage_latency[stage].set_comm_time(2*(cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())

                    # stage_latency.append(sum(cost_e1[num_layer_til_last_stage:num_layer_til_cur_stage]) \
                    #                     + 2*cost_c[sum(partition[:stage])][stage-1]*num_bw_share)
            else:
                stage_latency[stage].set_comp_time(sum(cost_e2[num_layer_til_last_stage:num_layer_til_cur_stage]))
                stage_latency[stage].set_comm_time(2*(cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())

                # stage_latency.append(sum(cost_e2[num_layer_til_last_stage:num_layer_til_cur_stage]) \
                #                     + 2*cost_c[sum(partition[:stage])][stage-1]*num_bw_share)
        else:
            # if num_stage ==1:
            #     stage_latency = [max(np.sum(cost_e1), np.sum(cost_e2))]
            # else:
            #     stage_latency.append(sum(cost_e2[num_layer_til_last_stage:num_layer_til_cur_stage])\
            #                         + 2*cost_c[sum(partition[:stage])][stage-1])

            stage_latency[stage].set_comp_time(sum(cost_e2[num_layer_til_last_stage:num_layer_til_cur_stage]))
            stage_latency[stage].set_comm_time((cost_c[sum(partition[:stage])][stage-1]*num_bw_share).item())

        # if stage == num_stage-1:
        #     # Substract the activation communication cost from the last stage
        #     # since the backward of the last stage does not need to communicate
        #     stage_latency[-1] -= cost_c[sum(partition[:stage])][stage-1]

        if stage == num_stage-1:
            # Substract the activation communication cost from the last stage
            # since the backward of the last stage does not need to communicate
            stage_comm_time_last = stage_latency[-1].get_comm_time()
            stage_comm_time_last -= cost_c[sum(partition[:stage])][stage-1].item()
            stage_latency[-1].set_comm_time(stage_comm_time_last)

    return stage_latency

