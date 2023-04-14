#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=hgx
#SBATCH --nodelist=n050
#SBATCH --gres=gpu:hgx:4
#SBATCH --cpus-per-task=28
#SBATCH -o ./log2/%j.sbatch.%N.out         # STDOUT
#SBATCH -e ./log2/%j.sbatch.%N.err         # STDERR

#************************************************************
MASTER_HOST=n050
GRES="gpu:hgx:4"
START_RANK=0
#************************************************************

cd $HOME/tdpp/Megatron-LM-2
mkdir -p ./log2/$SLURM_JOB_ID
MASTER_ADDR=`cat /etc/hosts | grep $MASTER_HOST | awk '{print $1}'`
echo "MASTER_ADDR:$MASTER_ADDR"

ENROOT_SCRIPT=$(cat <<EOF
CONTAINER_PATH="/scratch/enroot/\$UID/data/megatron-latest"
if [ -d "\$CONTAINER_PATH" ] ; then 
    echo "container exist";
else
    enroot create \$HOME/tdpp/image/megatron-latest.sqsh ;
fi

NODE_LIST=\`scontrol show hostnames \$SLURM_JOB_NODELIST\`
node_array=(\$NODE_LIST)
length=\${#node_array[@]}
hostnode=\`hostname -s\`
last_rank=\$((length + $START_RANK ))
for (( index = $START_RANK; index <  \$last_rank; index++ )); do
    array_index=\$((\$index - $START_RANK))
    node=\${node_array[\$array_index]}
    if [ \$node == \$hostnode ]; then
        local_rank=\$index
    fi
done 

/usr/local/bin/gpustat -i > \$HOME/tdpp/Megatron-LM-2/log2/\$SLURM_JOB_ID/\$hostnode.gpu &

enroot start --root \
            --rw \
            -m \$HOME/tdpp/Megatron-LM-2:/root/Megatron-LM megatron-latest \
            bash -c "cd /root/Megatron-LM/ && sh run_inter.sh \$local_rank
EOF
)

srun --partition=$SLURM_JOB_PARTITION \
      --gres=$GRES \
      --cpus-per-task=10 \
      -o ./log2/%j/%N.out \
      -e ./log2/%j/%N.err \
      bash -c "$ENROOT_SCRIPT $MASTER_ADDR\" " 
