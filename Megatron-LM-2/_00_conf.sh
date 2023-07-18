#!/bin/bash

CONTAINER_IMAGE_PATH="$HOME/tdpp/image/nvcr.io+nvidia+pytorch+23.03-py3.sqsh"
CONTAINER_PATH="/scratch/enroot/$UID/data/megatron-latest"
CONTAINER_NAME="megatron-latest"

HOMOGENEOUS_CLUSTER=true
MODEL="gpt2" # Bert / gpt2 / T5
NPROC_PER_NODE=4
NNODES=2
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1
TENSOR_MP_SIZE=1
DP_SIZE=4
PIPELINE_MP_SIZE=2
PARTITION="12-12-12-12"
NSYS=false
PROFILE=false 
MASTER_PORT=6000
RELOAD_CONTAINER=false
ZERO=false

PIPELINE_MODEL_PARALLEL_SPLIT_RANK=6 # Where the encoder ends within the pipeline group

# set model specific arguments
echo "MODEL : $MODEL"
if [ $MODEL == "Bert" ]; then
        HIDDEN_SIZE=1024
        NUM_LAYERS=24
elif [ $MODEL == "gpt2" ]; then
        HIDDEN_SIZE=1600
        NUM_LAYERS=48
elif [ $MODEL == "T5" ]; then
        HIDDEN_SIZE=1024
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

# set zero arguments
if $ZERO; then
        echo "ZERO ON"
        ZERO_ARGS="--use-distributed-optimizer"
else
        echo "ZERO OFF"
        ZERO_ARGS=""
fi