#!/bin/bash

CONTAINER_IMAGE_PATH="$HOME/tdpp/image/nvcr.io+nvidia+pytorch+23.03-py3.sqsh"
CONTAINER_PATH="/scratch/enroot/$UID/data/megatron-latest"
CONTAINER_NAME="megatron-latest"

HOMOGENEOUS_CLUSTER=true
MODEL="T5" # Bert / GPT / T5
NPROC_PER_NODE=4
NNODES=4
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
TENSOR_MP_SIZE=1
DP_SIZE=4
PIPELINE_MP_SIZE=4
PARTITION="13-12-13-12"
NSYS=false
PROFILE=true 
MASTER_PORT=6778
RELOAD_CONTAINER=false

PIPELINE_MODEL_PARALLEL_SPLIT_RANK=2 # Where the encoder ends within the pipeline group

# set model specific arguments
echo "MODEL : $MODEL"
if [ $MODEL == "Bert" ]; then
        HIDDEN_SIZE=1024
        NUM_LAYERS=1
elif [ $MODEL == "GPT" ]; then
        HIDDEN_SIZE=1600
        NUM_LAYERS=48
elif [ $MODEL == "T5" ]; then
        HIDDEN_SIZE=512
        NUM_LAYERS=48
        ENCODER_NUM_LAYERS=$((NUM_LAYERS / 2))
        DECODER_NUM_LAYERS=$((NUM_LAYERS / 2))
else
        echo error: invalid model argument MODEL only "Bert/GPT/T5" is allowed
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