#!/bin/bash

NODE_RANK=$1
MASTER_ADDR=$2
NPROC_PER_NODE=4
NNODES=4
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))

MICRO_BATCH_DIM=1
TENSOR_MP_SIZE=1
PIPELINE_MP_SIZE=4
DP_SIZE=$((WORLD_SIZE/PIPELINE_MP_SIZE/TENSOR_MP_SIZE))

GLOBAL_BATCH_SIZE=$((32*MICRO_BATCH_DIM))
MICRO_BATCH_SIZE=$((GLOBAL_BATCH_SIZE/DP_SIZE/MICRO_BATCH_DIM/PIPELINE_MP_SIZE))
echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "GLOBAL_BATCH_SIZE*MICRO_BATCH_DIM: $GLOBAL_BATCH_SIZE"

DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port 6000"

VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
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
        --balance 12-12-12-12" 

OUTPUT_ARGS="--log-interval 10 \
             --save-interval 100 \
             --eval-interval 100 \
             --eval-iters 10"

# TENSORBOARD_ARGS="--tensorboard-dir /root/Megatron-LM/tensorboard \
#                 --tensorboard-log-interval 10"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
                $MODEL_ARGS \
                $OUTPUT_ARGS \
                --data-path $DATA_PATH \
                --tensor-model-parallel-size $TENSOR_MP_SIZE \
                --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
                --DDP-impl local \
                --no-async-tensor-model-parallel-allreduce \
                --no-gradient-accumulation-fusion \
                --split 100,0,0