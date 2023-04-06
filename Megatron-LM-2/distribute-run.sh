#!/bin/bash

NODE_RANK=$1
NNODES=2
NPROC_PER_NODE=2
MASTER_ADDR=192.168.120.87


DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port 6000"

python -m torch.distributed.launch $DISTRIBUTED_ARGS distribute-test.py