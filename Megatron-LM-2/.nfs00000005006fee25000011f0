#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu2
#SBATCH --nodelist=n076
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=28
#SBATCH -o ./log2/SLURM.%N.%j.out       
#SBATCH -e ./log2/SLURM.%N.%j.err        
#************************************************************
# GRES="gpu:a10:4"
#************************************************************

# set container
gate_node='gate1'
image_path='/home2/eung0/tdpp/image/megatron-latest.sqsh'

# parse image, container name
image_name=${image_path##*/}
container_name=${image_name%%.*}

# check old container in calculatation node and remove.
container_path="/scratch/enroot/$UID/data/$container_name"
test -d $container_path && 
    rm -rf $container_path

# cp -r $HOME/jupyter-pytorch/sshkeys $container_path/

# connect gate:$gate_port and node:$node_porls


user=`whoami`
gate_port=`ssh $user@$gate_node "ruby -e 'require \"socket\"; puts Addrinfo.tcp(\"\", 0).bind {|s| s.local_address.ip_port }'"`
gate_port=`echo $gate_port | awk '{print \$1;}'`
node_port=`ruby -e 'require "socket"; puts Addrinfo.tcp("", 0).bind {|s| s.local_address.ip_port }'`
ssh $user@$gate_node -R $gate_port:localhost:$node_port -fN "while sleep 100; do; done"&

# print job info
echo "start at:" `date`
echo "node: $HOSTNAME"
echo "gate: $gate_node"
echo "container_name: $container_name"
echo "node_port: $node_port"
echo "gate_port: $gate_port"
echo "jobid: $SLURM_JOB_ID"

# cd $HOME/tdpp/Megatron-LM-2
# mkdir -p ./log2/$SLURM_JOB_ID
NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
MASTER_HOST=`echo $NODE_LIST | awk '{print $1}'`
MASTER_ADDR=`cat /etc/hosts | grep $MASTER_HOST | awk '{print $1}'`
echo $NODE_LIST
echo $MASTER_ADDR

# create new container folder
enroot create -n $container_name $image_path

# CONTAINER_PATH="/scratch/enroot/$UID/data/megatron-latest"
# if [ -d $CONTAINER_PATH ] ; then 
#     echo "container exist";
# else
#     enroot create $HOME/tdpp/image/megatron-latest.sqsh ;
# fi

# get local rank
# NODE_LIST=`scontrol show hostnames $SLURM_JOB_NODELIST`
# node_array=($NODE_LIST)
# length=${#node_array[@]}
# hostnode=`hostname -s`
# for (( index = 0; index < length ; index++ )); do
#     node=${node_array[$index]}
#     if [ $node == $hostnode ]; then
#         local_rank=$index
#     fi
# done 

#/usr/local/bin/gpustat -i > $HOME/tdpp/Megatron-LM-2/log2/$SLURM_JOB_ID/$hostnode.gpu &

enroot start \
            --rw \
            -m $HOME/tdpp:/tdpp \
            $container_name \
            python -m jupyter lab /tdpp --ip=0.0.0.0 --port $node_port --allow-root --no-browser &
sleep 60 &&
echo 'start megatron-lm container' &&
enroot_string=`enroot list -f` &&
get_string="${enroot_string#*$container_name}" &&
JPID="${get_string%% jupy*}" &&
echo $JPID &&
# enroot exec $JPID mkdir $HOME/.ssh && # home 폴더에 .ssh가 만들어진 container를 사용한다면(사용 후 추가 패키지가 저장된 container) 이 명령어는 사용하지 않기
# enroot exec $JPID conda init # &&
enroot exec $JPID cp /sshkeys/authorized_keys /sshkeys/id_rsa /sshkeys/config   $HOME/.ssh &&
enroot exec $JPID /ssh/sshd -D -p 4567

# EOF
# )


# srun --partition=$SLURM_JOB_PARTITION \
#       --gres=$GRES \
#       --cpus-per-task=10 \
#       -o ./log2/%j/%N.out \
#       -e ./log2/%j/%N.err \
#       bash -c "$ENROOT_SCRIPT $MASTER_ADDR\" " 
