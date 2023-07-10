#!/bin/bash

CONTAINER_IMAGE_PATH="$HOME/tdpp/image/nvcr.io+nvidia+pytorch+23.03-py3.sqsh"
CONTAINER_PATH="/scratch/enroot/$UID/data/megatron-latest"
CONTAINER_NAME="megatron-latest"

HOMOGENEOUS_CLUSTER=true
MODEL="T5" # Bert / GPT / T5
NPROC_PER_NODE=1
NNODES=1
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
GLOBAL_BATCH_SIZE=1
MICRO_BATCH_SIZE=1
TENSOR_MP_SIZE=1
DP_SIZE=1
PIPELINE_MP_SIZE=1 #PARTITION="1-1" # It may not work for T5.
NSYS=false
PROFILE=false # 
MASTER_PORT=6777
RELOAD_CONTAINER=false

# set model specific arguments
echo "MODEL : $MODEL"
if [ $MODEL == "Bert" ]; then
        HIDDEN_SIZE=1024
        NUM_LAYERS=1
elif [ $MODEL == "GPT" ]; then
        HIDDEN_SIZE=1600
        NUM_LAYERS=48
elif [ $MODEL == "T5" ]; then
        HIDDEN_SIZE=1024
        NUM_LAYERS=12
else
        echo error: invalid model argument MODEL only "xl" or "small" is allowed
        return 1
fi

# set profiling arguments
if $PROFILE; then
        echo "Profiling"
        PROFILE_ARGS="--timing-log-level 2 \
                      --timing-log-option all"
else
        echo "Not profiling"
        PROFILE_ARGS=""
fi