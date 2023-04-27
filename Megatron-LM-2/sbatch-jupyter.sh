#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=hgx
#SBATCH --nodelist=n050
#SBATCH --gres=gpu:hgx:4
#SBATCH --cpus-per-task=28
#SBATCH -o ./SLURM.%N.%j.out         # STDOUT
#SBATCH -e ./SLURM.%N.%j.err         # STDERR     

#************************************************************
conda deactivate
# set container
gate_node='gate1'
image_path="$HOME/tdpp/image/megatron-latest.sqsh"

# parse image, container name
image_name=${image_path##*/}
container_name=${image_name%%.*}

# check old container in calculatation node and remove.
container_path="/scratch/enroot/$UID/data/$container_name"

if [ -d "$container_path" ] ; then 
    echo "container exist";
else
    enroot create -n $container_name $image_path ;
fi

# connect gate:$gate_port and node:$node_port
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

echo 'start enroot container'
export ENROOT_MOUNT_HOME=y 

enroot start \
    --root \
    --rw \
    -m $HOME/tdpp/Megatron-LM-2:/root/Megatron-LM \
    $container_name \
    /opt/conda/bin/python -m jupyter lab /root --ip=0.0.0.0 --port $node_port --allow-root --no-browser

