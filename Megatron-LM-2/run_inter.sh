#!/bin/bash

NODE_RANK=$1
MASTER_ADDR=$2
NPROC_PER_NODE=$3
NNODES=$4
GLOBAL_BATCH_SIZE=$5
MICRO_BATCH_SIZE=$6
TENSOR_MP_SIZE=$7
DP_SIZE=$8
PIPELINE_MP_SIZE=$9
PARTITION=${10}

WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NNODES: $NNODES"
echo "GLOBAL_BATCH_SIZE*MICRO_BATCH_DIM: $GLOBAL_BATCH_SIZE"
echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "TENSOR_MP_SIZE: $TENSOR_MP_SIZE"
echo "DP_SIZE: $DP_SIZE"
echo "PIPELINE_MP_SIZE: $PIPELINE_MP_SIZE"
echo "PARTITION: $PARTITION"


DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port 6787"

VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt

cp -r /root/indexmap/* /root/Megatron-LM

ls /root/Megatron-LM/my-gpt2_text_document*

DATA_PATH=/root/Megatron-LM/my-gpt2_text_document
MODEL_ARGS="--num-layers 48 \
        --hidden-size 1600 \
        --num-attention-heads 16 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --lr 0.00015 \
        --train-iters 50 \
        --lr-decay-iters 320000 \
        --lr-decay-style cosine \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --lr-warmup-fraction .01 \
        --fp16 \
        --balance $PARTITION" 

# OUTPUT_ARGS="--log-interval 1 \
#              --timing-log-level 2 \
#              --timing-log-option all \
#              --save-interval 100 \
#              --eval-interval 100 \
#              --eval-iters 10"

# OUTPUT_ARGS="--log-interval 10 \
#             --timing-log-level 2 \
#             --timing-log-option all \
#             --save-interval 100 \
#             --eval-interval 100 \
#             --eval-iters 10"

OUTPUT_ARGS="--log-interval 10 \
              --save-interval 100 \
              --eval-interval 100 \
              --eval-iters 10"

# TENSORBOARD_ARGS="--tensorboard-dir /root/Megatron-LM/tensorboard \
#                 --tensorboard-log-interval 10"


hostname
OMP_NUM_THREADS=4 python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
                $MODEL_ARGS \
                $OUTPUT_ARGS \
                --data-path $DATA_PATH \
                --tensor-model-parallel-size $TENSOR_MP_SIZE \
                --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
                --DDP-impl local \
                --no-async-tensor-model-parallel-allreduce \
                --no-gradient-accumulation-fusion \
                --split 100,0,0 
