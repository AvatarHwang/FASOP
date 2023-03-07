

## I. setup

- `$HOME/tdpp` 경로에 tdpp 폴더를 위치시킵니다.
- `$HOME/tdpp/image/megatron-latest.sqsh` 경로에 `megatron-latest.sqsh` 파일을 위치시킵니다.
- `$HOME/tdpp/Megatron-LM-2/log2` 경로에 `log2` 폴더를 생성합니다.
- `$HOME/tdpp/Megatron-LM-2/log` 경로에 `log` 폴더를 생성합니다.


## II. run
### 1. run by sbatch

sbatch.sh 파일과 run_inter.sh 파일을 수정한 후 sbatch 커맨드를 통해서 job을 실행합니다.

- `$HOME/tdpp/Megatron-LM-2/sbatch.sh` 파일을 적절히 수정합니다.
    ```bash
    #!/bin/bash
    #SBATCH --nodes=4
    #SBATCH --ntasks-per-node=1
    #SBATCH --partition=gpu2
    #SBATCH --nodelist=n063,n064,n065,n066
    #SBATCH --gres=gpu:a10:4,gpu:a10:4,gpu:a10:4,gpu:a10:4
    #SBATCH --cpus-per-task=28
    #SBATCH -o ./log2/%j.sbatch.%N.out         # STDOUT
    #SBATCH -e ./log2/%j.sbatch.%N.err         # STDERR
    ```

- `$HOME/tdpp/Megatron-LM-2/run-inter.sh` 파일을 적절히 수정합니다.
    ```bash
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
    ```

- `sbatch`
    ```bash
    $ sbatch sbatch.sh
    ```

### 2. run by srun
`srun` 커맨드를 통해서 개별 노드에 직접 접속하여 job을 실행할 수 있습니다.

```bash
PARTITION=gpu2
GRES=gpu:a10:4
NODE=n062
NODE_RANK=0
MASTER_ADDR=

srun --cpus-per-task 28 -p $PARTITION --nodelist $NODE --gres=$GRES --pty bash
container_path="/scratch/enroot/$UID/data/megatron-latest"
test -d $container_path && 
    rm -rf $container_path
enroot create $HOME/tdpp/image/megatron-latest.sqsh
SCRIPT_NAME='run_inter.sh'
enroot start --root \
            --rw \
            -m $HOME/tdpp/Megatron-LM:/root/Megatron-LM megatron-latest \
            bash -c "cd /root/Megatron-LM/ &&\
                    sh $SCRIPT_NAME $NODE_RANK $MASTER_ADDR"
```


## III. report

실행 로그는 다음 경로에 저장됩니다.
- sbatch 로그: `$HOME/tdpp/Megatron-LM-2/log2/JOBID.sbatch.NODE.out, err`
- srun 로그: `$HOME/tdpp/Megatron-LM-2/log2/JOBID/NODE.out, err`
- gpu 사용량 로그: `$HOME/tdpp/Megatron-LM-2/log2/JOBID/NODE-gpu.log`

job을 실행한 후 다음 명령어를 통해서 gpu 사용량, 노드별 std out 등을 확인할 수 있습니다.

```bash
tail -f $HOME/tdpp/Megatron-LM-2/log2/JOBID/NODE.out
tail -f $HOME/tdpp/Megatron-LM-2/log2/JOBID/NODE-gpu.log
```