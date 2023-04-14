class StageTime:
    
    def __init__(self, num_mb, comp_time, for_send_time, back_send_time):
        self.num_mb = num_mb
        self.forward_time = comp_time/3
        self.backward_time = 2*comp_time/3
        self.send_for_time = for_send_time
        self.send_back_time = back_send_time

        self.for_compute_end_time_lst = [0. for i in range(self.num_mb)]

        self.for_send_end_time_lst = [0. for i in range(self.num_mb)]

        self.for_recv_end_time_lst = [0. for i in range(self.num_mb)]

        self.back_compute_end_time_lst = [0. for i in range(self.num_mb)]

        self.back_send_end_time_lst = [0. for i in range(self.num_mb)]

        self.back_recv_end_time_lst = [0. for i in range(self.num_mb)]

    def get_for_compute_end_time_lst_elem(self, i):
        a = self.for_compute_end_time_lst[i]
        return a

    def set_for_compute_end_time_lst_elem(self, i, cost):
        self.for_compute_end_time_lst[i] = cost

    def get_for_send_end_time_lst_elem(self, i):
        a = self.for_send_end_time_lst[i]
        return a

    def set_for_send_end_time_lst_elem(self, i, cost):
        self.for_send_end_time_lst[i] = cost

    def get_for_recv_end_time_lst_elem(self, i):
        a = self.for_recv_end_time_lst[i]
        return a

    def set_for_recv_end_time_lst_elem(self, i, cost):
        self.for_recv_end_time_lst[i] = cost

    def get_back_compute_end_time_lst_elem(self, i):
        a = self.back_compute_end_time_lst[i]
        return a

    def set_back_compute_end_time_lst_elem(self, i, cost):
        self.back_compute_end_time_lst[i] = cost

    def get_back_send_end_time_lst_elem(self, i):
        a = self.back_send_end_time_lst[i]
        return a

    def set_back_send_end_time_lst_elem(self, i, cost):
        self.back_send_end_time_lst[i] = cost

    def get_back_recv_end_time_lst_elem(self, i):
        a = self.back_recv_end_time_lst[i]
        return a

    def set_back_recv_end_time_lst_elem(self, i, cost):
        self.back_recv_end_time_lst[i] = cost

class PPGroup:
    def __init__(self, num_mb, pp_degree, stage_comp_time_lst, stage_for_send_time_lst, stage_back_send_time_lst):
        self.num_mb = num_mb
        self.pp_degree = pp_degree
        self.stage_comp_time_lst = stage_comp_time_lst
        self.stage_for_send_time_lst = stage_for_send_time_lst
        self.stage_back_send_time_lst = stage_back_send_time_lst
        self.__set_stage_time_lst()
        self.__compute_num_warmup_mb()
        self.__compute_num_remain_mb()
    
    def __set_stage_time_lst(self):
        self.stage_time_lst = []
        for i in range(self.pp_degree):
            self.stage_time_lst.append(StageTime(self.num_mb,
                                                 self.stage_comp_time_lst[i],
                                                 self.stage_for_send_time_lst[i],
                                                 self.stage_back_send_time_lst[i]
                                                 )
                                       )
    
    def __compute_num_warmup_mb(self):
        self.num_warmup_mb_lst = []
        for i in range(self.pp_degree):
            num_warmup_mb = min(self.pp_degree-i-1, self.num_mb)
            self.num_warmup_mb_lst.append(num_warmup_mb)

    def __compute_num_remain_mb(self):
        self.num_remain_mb = []
        for i in range(self.pp_degree):
            self.num_remain_mb.append(self.num_mb-self.num_warmup_mb_lst[i])
    
    def __simulate_warmup_forward(self):
        for j in range(self.num_mb):
            
            for i in range(self.pp_degree):        
                num_warmup_mb = self.num_warmup_mb_lst[i]
                if j > num_warmup_mb-1:
                    continue

                stage_time = self.stage_time_lst[i]
                next_stage_time = None
                if i < self.pp_degree-1:
                    next_stage_time = self.stage_time_lst[i+1]

                forward_time = stage_time.forward_time
                send_for_time = stage_time.send_for_time
                if i == 0:
                    if j == 0:
                        stage_time.set_for_compute_end_time_lst_elem(j,forward_time)
                        send_end_time = forward_time + send_for_time
                        stage_time.set_for_send_end_time_lst_elem(j,send_end_time)
                        next_stage_time.set_for_recv_end_time_lst_elem(j, send_end_time)

                    else:
                        forward_start_time = stage_time.get_for_send_end_time_lst_elem(j-1)
                        forward_end_time = forward_start_time + forward_time
                        stage_time.set_for_compute_end_time_lst_elem(j, forward_end_time)

                        prev_mb_for_end_time_next_stage = next_stage_time.get_for_compute_end_time_lst_elem(j-1)
                        send_start_time = max(forward_end_time, prev_mb_for_end_time_next_stage)
                        send_end_time = send_start_time + send_for_time
                        stage_time.set_for_send_end_time_lst_elem(j, send_end_time)
                        next_stage_time.set_for_recv_end_time_lst_elem(j, send_end_time)

                else:
                    if j == 0:
                        recv_end_time = stage_time.get_for_recv_end_time_lst_elem(j)
                        forward_end_time = recv_end_time + forward_time
                        stage_time.set_for_compute_end_time_lst_elem(j, forward_end_time)
                        
                        if next_stage_time is not None:
                            send_end_time = forward_end_time + send_for_time
                            stage_time.set_for_send_end_time_lst_elem(j, send_end_time)
                            next_stage_time.set_for_recv_end_time_lst_elem(j, send_end_time)
                    
                    else:
                        prev_send_for_end_time = stage_time.get_for_send_end_time_lst_elem(j-1)
                        recv_end_time = stage_time.get_for_recv_end_time_lst_elem(j)
                        forward_start_time = max(prev_send_for_end_time, recv_end_time)

                        forward_end_time = forward_start_time + forward_time
                        stage_time.set_for_compute_end_time_lst_elem(j, forward_end_time)

                        prev_mb_for_end_time_next_stage = next_stage_time.get_for_compute_end_time_lst_elem(j-1)
                        send_start_time = max(forward_end_time, prev_mb_for_end_time_next_stage)
                        send_end_time = send_start_time + send_for_time
                        stage_time.set_for_send_end_time_lst_elem(j,send_end_time)
                        next_stage_time.set_for_recv_end_time_lst_elem(j, send_end_time)

    def __simulate_1f1b(self):

        for j in range(self.num_mb):
            forward_mb_id = j
            backward_mb_id = j

            for i in range(self.pp_degree):
                stage_time = self.stage_time_lst[i]
                
                num_warmup_mb = self.num_warmup_mb_lst[i]
                if forward_mb_id < num_warmup_mb:
                    continue

                next_stage_time = None
                if i!=self.pp_degree-1:
                    next_stage_time = self.stage_time_lst[i+1]

                forward_time = stage_time.forward_time
                send_for_time = stage_time.send_for_time

                # first 1f1b
                if forward_mb_id == num_warmup_mb:
                    prev_comp_id = num_warmup_mb-1
                    
                    # has warmup forward
                    if prev_comp_id >= 0:
                        prev_send_end_time = stage_time.get_for_send_end_time_lst_elem(prev_comp_id)
                        recv_end_time = stage_time.get_for_recv_end_time_lst_elem(forward_mb_id)
                        forward_start_time = max(prev_send_end_time, recv_end_time)
                        forward_end_time = forward_start_time + forward_time
                        stage_time.set_for_compute_end_time_lst_elem(forward_mb_id, forward_end_time)

                        if next_stage_time is not None:
                            prev_mb_forward_end_time_next_stage = next_stage_time.get_for_compute_end_time_lst_elem(prev_comp_id)
                            send_start_time = max(prev_mb_forward_end_time_next_stage, forward_end_time)
                            send_end_time = send_start_time + send_for_time
                            stage_time.set_for_send_end_time_lst_elem(forward_mb_id, send_end_time)
                            next_stage_time.set_for_recv_end_time_lst_elem(forward_mb_id,send_end_time)

                    # has no warmup forward
                    else:
                        prev_send_end_time = 0.
                        recv_end_time = stage_time.get_for_recv_end_time_lst_elem(forward_mb_id)
                        forward_start_time = max(prev_send_end_time, recv_end_time)
                        forward_end_time = forward_start_time + forward_time
                        stage_time.set_for_compute_end_time_lst_elem(forward_mb_id, forward_end_time)
                        
                        if next_stage_time is not None:
                            prev_mb_forward_end_time_next_stage = 0.
                            send_start_time = max(prev_mb_forward_end_time_next_stage, forward_end_time)
                            send_end_time = send_start_time + send_for_time
                            stage_time.set_for_send_end_time_lst_elem(forward_mb_id, send_end_time)
                            next_stage_time.set_for_recv_end_time_lst_elem(forward_mb_id, send_end_time)

                else:
                    prev_comp_id = forward_mb_id-num_warmup_mb-1
                    prev_send_end_time = stage_time.get_back_send_end_time_lst_elem(prev_comp_id)
                    recv_end_time = stage_time.get_for_recv_end_time_lst_elem(forward_mb_id)
                    forward_start_time = max(prev_send_end_time, recv_end_time)
                    forward_end_time = forward_start_time + forward_time
                    stage_time.set_for_compute_end_time_lst_elem(forward_mb_id, forward_end_time)
                    
                    if next_stage_time is not None:
                        prev_mb_backward_end_time_next_stage = next_stage_time.get_back_compute_end_time_lst_elem(prev_comp_id+1)
                        send_start_time = max(prev_mb_backward_end_time_next_stage, forward_end_time)
                        send_end_time = send_start_time + send_for_time
                        stage_time.set_for_send_end_time_lst_elem(forward_mb_id, send_end_time)
                        next_stage_time.set_for_recv_end_time_lst_elem(forward_mb_id,send_end_time)

            for i in reversed(range(self.pp_degree)):
                stage_time = self.stage_time_lst[i]
                
                num_warmup_mb = self.num_warmup_mb_lst[i]
                if backward_mb_id > self.num_mb-num_warmup_mb-1 :
                    continue

                prev_stage_time = None
                if i!=0:
                    prev_stage_time = self.stage_time_lst[i-1]
                
                backward_time = stage_time.backward_time
                send_back_time = stage_time.send_back_time

                prev_comp_id = backward_mb_id + num_warmup_mb

                prev_send_end_time = 0.
                if prev_stage_time is not None:
                    prev_send_end_time = stage_time.get_for_send_end_time_lst_elem(prev_comp_id)
                prev_forward_end_time = stage_time.get_for_compute_end_time_lst_elem(prev_comp_id)
                grad_recv_end_time = stage_time.get_back_recv_end_time_lst_elem(backward_mb_id)

                backward_start_time = max(prev_send_end_time, grad_recv_end_time, prev_forward_end_time)
                backward_end_time = backward_start_time + backward_time
                stage_time.set_back_compute_end_time_lst_elem(backward_mb_id, backward_end_time)
                
                if prev_stage_time is not None:

                    if backward_mb_id == self.num_mb-num_warmup_mb-1:
                        prev_mb_back_end_time_prev_stage = prev_stage_time.get_back_compute_end_time_lst_elem(backward_mb_id-1)
                        send_start_time = max(prev_mb_back_end_time_prev_stage, backward_end_time)

                        send_end_time = send_start_time + send_back_time
                        stage_time.set_back_send_end_time_lst_elem(backward_mb_id, send_end_time)

                        prev_stage_time.set_back_recv_end_time_lst_elem(backward_mb_id, send_end_time)

                    else:
                        prev_stage_num_warmup_mb = self.num_warmup_mb_lst[i-1]
                        prev_mb_for_prev_stage_id = backward_mb_id + prev_stage_num_warmup_mb
                        prev_mb_for_end_time_prev_stage = prev_stage_time.get_for_compute_end_time_lst_elem(prev_mb_for_prev_stage_id)
                        send_start_time = max(prev_mb_for_end_time_prev_stage, backward_end_time)

                        send_end_time = send_start_time + send_back_time
                        stage_time.set_back_send_end_time_lst_elem(backward_mb_id, send_end_time)

                        prev_stage_time.set_back_recv_end_time_lst_elem(backward_mb_id, send_end_time)

    def __simulate_cooldown_backward(self):
        for j in range(self.num_mb):
            for i in reversed(range(self.pp_degree)):
                stage_time = self.stage_time_lst[i]
                
                backward_mb_id = j
                num_warmup_mb = self.num_warmup_mb_lst[i]

                if backward_mb_id < self.num_mb-num_warmup_mb:
                    continue

                prev_stage_time = None
                if i!=0:
                    prev_stage_time = self.stage_time_lst[i-1]

                backward_time = stage_time.backward_time
                send_back_time = stage_time.send_back_time

                prev_comp_id = backward_mb_id-1
                prev_comp_end_time = stage_time.get_back_compute_end_time_lst_elem(prev_comp_id)

                recv_end_time = stage_time.get_back_recv_end_time_lst_elem(backward_mb_id)

                backward_start_time = max(prev_comp_end_time, recv_end_time)
                backward_end_time = backward_start_time + backward_time
                stage_time.set_back_compute_end_time_lst_elem(backward_mb_id, backward_end_time)

                if prev_stage_time is not None:
                    prev_mb_back_end_time_prev_stage = prev_stage_time.get_back_compute_end_time_lst_elem(prev_comp_id)
                    send_start_time = max(backward_end_time, prev_mb_back_end_time_prev_stage)
                    send_end_time = send_start_time + send_back_time
                    stage_time.set_back_send_end_time_lst_elem(backward_mb_id, send_end_time)
                    prev_stage_time.set_back_recv_end_time_lst_elem(backward_mb_id, send_end_time)

    def simulate_full_pipeline(self):
        self.__simulate_warmup_forward()
        self.__simulate_1f1b()
        self.__simulate_cooldown_backward()
    
    def print_stagewise_end_time(self):
        for i in range(self.pp_degree):
            stage_time = self.stage_time_lst[i]
            last_comp_time = stage_time.get_back_compute_end_time_lst_elem(self.num_mb-1)
            print(i, last_comp_time)

    def get_pipe_cost(self):
        stage_time_0 = self.stage_time_lst[0]
        pipe_cost = stage_time_0.get_back_compute_end_time_lst_elem(self.num_mb-1)
        return pipe_cost
    
    def get_stagewise_end_time_lst(self):
        stage_wise_end_time_lst = []
        for i in range(self.pp_degree):
            stage_time = self.stage_time_lst[i]
            last_comp_time = stage_time.get_back_compute_end_time_lst_elem(self.num_mb-1)
            stage_wise_end_time_lst.append(last_comp_time)
        return stage_wise_end_time_lst