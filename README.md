

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


## IV. Profile with Torch Profiler

아래 경로에서 training.py가 오리지널 파일인지, torch profiler가 작동하는 파일인지 확인합니다.

Megatron-LM-2/megatron/

메가트론을 실행합니다.
프로그램이 종료되면 Megatron-LM-2/log/ 경로에 생성된 로그파일들을 텐서보드를 사용해 확인합니다.
생성된 파일들 중에 마스터 노드에서 실행된 파일 딱 하나만 다른 경로(예:temp)로 옮겨 확인해야 distributed 탭을 확인 가능합니다.
```bash
tensorboard --logdir=./log/temp --bind_all
```

이미 생성된 log파일들은 아래 경로에 rank 별로 존재합니다.

`/home/soonyear/profile/12_A10-4_RTX3090/`

먼저 computation 시간과 communication 시간을 확인합니다. 확인은 Normal - view - Distributed 에서 확인 가능합니다.
두번째로는 TP comm.시간과 DP comm.시간을 확인합니다. 확인은 trace에서 가능합니다.
자세한 방법은 문의하세요.

## V. Profile layer by layer execution time

작성예정