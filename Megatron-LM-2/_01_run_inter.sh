#!/bin/bash

hostname

NODE_RANK=$1
MASTER_ADDR=$2

. _00_conf.sh

# copy data indexmap from nfs mount dir to local dir
cp -r /root/indexmap/* /root/Megatron-LM

echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NNODES: $NNODES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "GLOBAL_BATCH_SIZE: $GLOBAL_BATCH_SIZE"
echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "TENSOR_MP_SIZE: $TENSOR_MP_SIZE"
echo "DP_SIZE: $DP_SIZE"
echo "PIPELINE_MP_SIZE: $PIPELINE_MP_SIZE"
echo "PARTITION: $PARTITION"
echo "MODEL: $MODEL"
echo "HIDDEN_SIZE: $HIDDEN_SIZE"
echo "NUM_LAYERS: $NUM_LAYERS"
echo "PROFILE: $PROFILE"
echo "PROFILE_ARGS: $PROFILE_ARGS"

DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

DATA_PATH=/root/Megatron-LM/my-gpt2_text_document
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt

MODEL_ARGS="--num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
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

OUTPUT_ARGS="--log-interval 10 \
              --save-interval 100 \
              --eval-interval 100 \
              --eval-iters 10 $PROFILE_ARGS"


RUN_TORCH_SCRIPT=$(cat << EOF
python -m torch.distributed.launch $DISTRIBUTED_ARGS _00_pretrain_gpt.py \
                $MODEL_ARGS \
                $OUTPUT_ARGS \
                --data-path $DATA_PATH \
                --tensor-model-parallel-size $TENSOR_MP_SIZE \
                --pipeline-model-parallel-size $PIPELINE_MP_SIZE \
                --DDP-impl local \
                --no-async-tensor-model-parallel-allreduce \
                --no-gradient-accumulation-fusion \
                --split 100,0,0 
EOF
)

function run_torch() {
        OMP_NUM_THREADS=4 $RUN_TORCH_SCRIPT
}

function run_torch_with_nsys() {
        OMP_NUM_THREADS=4 nsys profile -t cuda,nvtx \
                --delay=5 \
                -o ../log-nsys/$SLURM_JOB_ID \
                --export=sqlite \
                -f true \
                $RUN_TORCH_SCRIPT
}

function run_torch_with_ncu(){
        OMP_NUM_THREADS=4 ncu --target-processes all --nvtx \
        -o ../log-ncu/$REPORT \
                
}

# if NSYS is true, then run nsys
if $NSYS; then
        echo "Run torch  with nsys"
        run_torch_with_nsys
else
        echo "Run torch"
        run_torch
fi