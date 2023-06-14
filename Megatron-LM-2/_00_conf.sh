#!/bin/bash

CONTAINER_IMAGE_PATH="$HOME/tdpp/image/nvcr.io+nvidia+pytorch+23.03-py3.sqsh"
CONTAINER_PATH="/scratch/enroot/$UID/data/megatron-latest"
CONTAINER_NAME="megatron-latest"

HOMOGENEOUS_CLUSTER=false
MODEL='xl' # 'small' or 'xl'
NPROC_PER_NODE=4
NNODES=8
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1
TENSOR_MP_SIZE=1
DP_SIZE=4
PIPELINE_MP_SIZE=8
PARTITION="0-11-8-8-8-8-5-0"
NSYS=false
PROFILE=true # 
MASTER_PORT=6787
RELOAD_CONTAINER=false

# set model specific arguments
echo "MODEL size: $MODEL"
if [ $MODEL == "xl" ]; then
        HIDDEN_SIZE=1600
        NUM_LAYERS=48
elif [ $MODEL == "small" ]; then
        HIDDEN_SIZE=1024
        NUM_LAYERS=24
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