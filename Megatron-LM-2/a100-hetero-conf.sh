#!/bin/bash
NPROC_PER_NODE=4
NNODES=8
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1
TENSOR_MP_SIZE=2
DP_SIZE=1
PIPELINE_MP_SIZE=16
PARTITION="4-3-3-3-3-3-3-3-3-3-3-3-3-3-3-2"