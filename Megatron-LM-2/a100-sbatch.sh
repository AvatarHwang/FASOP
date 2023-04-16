#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx
#SBATCH --gres=gpu:hgx:4
#SBATCH --cpus-per-task=28
#SBATCH -o ./log2/%j.sbatch.%N.out         
#SBATCH -e ./log2/%j.sbatch.%N.err         

#************************************************************
GRES="gpu:hgx:4"
NPROC_PER_NODE=4
NNODES=4
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=1
TENSOR_MP_SIZE=1
DP_SIZE=2
PIPELINE_MP_SIZE=8
PARTITION="6-6-6-6-6-6-6-6"
#************************************************************

cd $HOME/tdpp/Megatron-LM-2
mkdir -p ./log2/$SLURM_JOB_ID
NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
MASTER_HOST=`echo $NODE_LIST | awk '{print $1}'`
MASTER_ADDR=`cat /etc/hosts | grep $MASTER_HOST | awk '{print $1}'`
echo $NODE_LIST
echo $MASTER_ADDR

ENROOT_SCRIPT=$(cat <<EOF
CONTAINER_PATH="/scratch/enroot/\$UID/data/megatron-latest"

rm -rf /scratch/enroot/\$UID/data/megatron-latest

if [ -d "\$CONTAINER_PATH" ] ; then 
    echo "container exist";
else
    enroot create -n megatron-latest \$HOME/tdpp/image/nvcr.io+nvidia+pytorch+23.03-py3.sqsh ;
fi

hostnode=\`hostname -s\`

/usr/local/bin/gpustat -i > \$HOME/tdpp/Megatron-LM-2/log2/\$SLURM_JOB_ID/\$hostnode.gpu &

enroot start --root \
            --rw \
            -m \$HOME/tdpp/Megatron-LM-2:/root/Megatron-LM \
            megatron-latest \
            bash -c "cd /root/Megatron-LM/ && sh run_inter.sh 0
EOF
)


srun --partition=$SLURM_JOB_PARTITION \
      --gres=$GRES \
      --cpus-per-task=12 \
      -o ./log2/%j/%N.out \
      -e ./log2/%j/%N.err \
      bash -c "$ENROOT_SCRIPT $MASTER_ADDR $NPROC_PER_NODE $NNODES $GLOBAL_BATCH_SIZE $MICRO_BATCH_SIZE $TENSOR_MP_SIZE $DP_SIZE $PIPELINE_MP_SIZE $PARTITION \" " 