#!/bin/bash
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=1              
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=14
##SBATCH --nodelist=n051,n052,n053
#SBATCH -o ../log2/%j.sbatch.%N.out         
#SBATCH -e ../log2/%j.sbatch.%N.err         

#************************************************************
GRES="gpu:a10:4"
. a100-hetero-conf.sh
#************************************************************

cd $HOME/tdpp/Megatron-LM-2
mkdir -p ../log2/$SLURM_JOB_ID
NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
echo $NODE_LIST
MASTER_ADDR=192.168.120.60
echo $MASTER_ADDR
ENROOT_SCRIPT=$(cat <<EOF
CONTAINER_PATH="/scratch/enroot/\$UID/data/megatron-latest"

rm -rf /scratch/enroot/\$UID/data/megatron-latest

if [ -d "\$CONTAINER_PATH" ] ; then 
    echo "container exist";
else
    enroot create -n megatron-latest \$HOME/tdpp/image/nvcr.io+nvidia+pytorch+23.03-py3.sqsh ;
fi


NODE_LIST=\`scontrol show hostnames \$SLURM_JOB_NODELIST\`
node_array=(\$NODE_LIST)
length=\${#node_array[@]}
hostnode=\`hostname -s\`
for (( index = 0; index < length ; index++ )); do
    node=\${node_array[\$index]}
    if [ \$node == \$hostnode ]; then
        local_rank=\$index
        local_rank=\$((local_rank+1))
    fi
done 

/usr/local/bin/gpustat -i > \$HOME/tdpp/log2/\$SLURM_JOB_ID/\$hostnode.gpu &

enroot start --root \
            --rw \
            -m \$HOME/tdpp/Megatron-LM-2:/root/Megatron-LM-2 \
            -m \$HOME/tdpp/log2:/root/log2 \
            -m \$HOME/tdpp/$INDEXMAP_DIR:/root/indexmap \
            megatron-latest \
            bash -c "cp -r /root/Megatron-LM-2 /root/Megatron-LM && cd /root/Megatron-LM/ && sh run_inter.sh \$local_rank

EOF
)


srun --partition=$SLURM_JOB_PARTITION \
      --gres=$GRES \
      --cpus-per-task=12 \
      -o ../log2/%j/%N.out \
      -e ../log2/%j/%N.err \
      bash -c "$ENROOT_SCRIPT $MASTER_ADDR $NPROC_PER_NODE $NNODES $GLOBAL_BATCH_SIZE $MICRO_BATCH_SIZE $TENSOR_MP_SIZE $DP_SIZE $PIPELINE_MP_SIZE $PARTITION \" " 
