#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1              
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:1
#SBATCH --exclude=n051,n052,n053,n054,n055,n066,n067,n068,n069,n070
#SBATCH --cpus-per-task=14
#SBATCH -o ../log2/%j.sbatch.%N.out         
#SBATCH -e ../log2/%j.sbatch.%N.err

#************************************************************
GRES="gpu:a10:1"
. _00_conf.sh
#************************************************************

cd $HOME/tdpp/Megatron-LM-2

mkdir -p ../log2/$SLURM_JOB_ID


function get_master_adress(){
    NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
    MASTER_HOST=`echo $NODE_LIST | awk '{print $1}'`
    MASTER_ADDR=`cat /etc/hosts | grep $MASTER_HOST | awk '{print $1}'`
}
get_master_adress
echo MASTER_ADDR:$MASTER_ADDR
echo CONTAINER_PATH:$CONTAINER_PATH



INIT_CONTAINER_SCRIPT=$(cat <<EOF
    
    if $RELOAD_CONTAINER ; then
        rm -rf $CONTAINER_PATH
    fi

    if [ -d "$CONTAINER_PATH" ] ; then 
        echo "container exist";
    else
        enroot create -n $CONTAINER_NAME $CONTAINER_IMAGE_PATH ;
    fi

EOF
)

MONITOR_GPU_SCRIPT=$(cat <<EOF
    hostnode=\`hostname -s\`
    /usr/local/bin/gpustat -i > $HOME/tdpp/log2/$SLURM_JOB_ID/\$hostnode.gpu &
EOF
)

ENROOT_SCRIPT="rm -rf /root/Megatron-LM && \
                cp -r /root/Megatron-LM-2 /root/Megatron-LM  && \
                cd /root/Megatron-LM/ && \
                bash _01_run_inter.sh"


SRUN_SCRIPT=$(cat <<EOF

    $INIT_CONTAINER_SCRIPT

    $MONITOR_GPU_SCRIPT

    NODE_LIST=\`scontrol show hostnames \$SLURM_JOB_NODELIST\`
    node_array=(\$NODE_LIST)
    length=\${#node_array[@]}
    hostnode=\`hostname -s\`
    for (( index = 0; index < length ; index++ )); do
        node=\${node_array[\$index]}
        if [ \$node == \$hostnode ]; then
            LOCAL_RANK=\$index
        fi
    done 

    enroot start --root \
                --rw \
                -m $HOME/tdpp/Megatron-LM-2:/root/Megatron-LM-2 \
                -m $HOME/tdpp/log2:/root/log2 \
                -m $HOME/tdpp/log-ncu:/root/log-ncu \
                -m $HOME/tdpp/log-nsys:/root/log-nsys \
                -m $HOME/tdpp/$MODEL:/root/indexmap \
                $CONTAINER_NAME \
                bash -c "$ENROOT_SCRIPT \$LOCAL_RANK $MASTER_ADDR"
EOF
)


srun --partition=$SLURM_JOB_PARTITION \
      --gres=$GRES \
      --cpus-per-task=12 \
      -o ../log2/%j/%N.out \
      -e ../log2/%j/%N.err \
      bash -c "$SRUN_SCRIPT"

