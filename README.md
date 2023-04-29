# FASOP: Fast yet Accurate Automatic Search for Optimal Parallelization of a Transformer on Heterogeneous GPU Clusters

This repository contains FASOP, a framework that automates the process of finding the optimal degrees of parallelism and model partitioning for Transformer-based models on heterogeneous GPU clusters. FASOP accurately estimates pipelining latency and GPU communications, enabling it to find configurations that minimize the cost of GPU clusters while satisfying training time constraints, or configurations that minimize training time while meeting cost constraints. FASOP supports a variety of Transformer-based models and uses advanced algorithms and techniques to rapidly and accurately estimate device configurations.

-----

# Usage

(1) Finding Optimal Parallel Strategy for GPT on Heterogeneous GPU Clusters.
(2) As results of FASOP, you can launch practical distributed learning using Megatron-LM.

## Reproducing the Experiments from [FASOP: Fast yet Accurate Automatic Search for Optimal Parallelization of a Transformer on Heterogeneous GPU Clusters]

To reproduce the experiments from [FASOP: Fast yet Accurate Automatic Search for Optimal Parallelization of a Transformer on Heterogeneous GPU Clusters], follow these steps:

### I. Install the necessary dependencies for FASOP. 

- FASOP requires a CPU for estimation tasks.
- We recommend to create conda environment for test of reproducibility.
- Python 3.8+, PyTorch 1.7+, numpy, CUDA 11.0+ (if you have a gpu)

#링크 마지막에 확인
    ```bash
    cd ~/workspace
    git clone https://github.com/{git_id}/FASOP
    conda create -name fasop python=3.8
    conda activate fasop
    conda install numpy
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
    ```

### II. Reproducing Experiment 4.1: Finding Optimal Parallel Strategy for GPT on Heterogeneous GPU Clusters
To reproduce Experiment 4.1, which involves finding the optimal parallel strategy for the GPT 3.5m model and 1.5b on heterogeneous GPU clusters, follow the steps below. The python codes should be located in the 'FASOP' directory of the FASOP repository. 
- To reproduce GPT-2 345m experiment, run `FASOP_345m.py`.

    ```bash
    python FASOP_345m.py
    ```

- To reproduce GPT-2 1.5b experiment, run `FASOP_1.5b.py`.

    ```bash
    python FASOP_1.5b.py
    ```

To reproduce Experiment 4.2, which involves finding the optimal parallel strategy for the GPT 1.5b model on virtual AWS cluster.
 - To reproduce Experiment 4.2, run `FASOP_pareto.py`.

    ```bash
    python FASOP_pareto.py
    ```
    
### III. Report    
Find the results of the experiment.    
FASOP will output a summary of the optimal parallel strategy for your model on your heterogeneous GPU cluster, including any estimated training time, cost, and other relevant metrics, in a text file. 

#링크 마지막에 확인
- output directory location: `~/workspace/FASOP/FASOP/main_logs`

    ```bash
    main_logs
    |- gpt345m.txt
    |- gpt1.5b.txt
    |- pareto.txt
    ```    
    
- The results file will contain the following fields, separated by ('\*'):
`rank`, `mbs`, `tp degree`, `dp degree`, `pp degree`, `node_info`, `partition`, `estimated_time`, `pipeline time`, `time of DP`, `time of reducing embedding layers`, `$/step`.

- example for the result of `FASOP_1.5b.py` located as `~/workspace/FASOP/FASOP/main_logs/gpt1.5b.txt`
    ```bash
    rank 0: (1, '*', {'tp_deg': 1, 'dp_deg': 4, 'pp_deg': 16}, '*', ['p4d.24xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge'], '*', [6, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], '*', 0.982886552810669, '*', 0.8530580251371566, '*', 0.0011705760844051838, '*', tensor([0.1287]), '*', 0.0321765932649374)
    rank 1: (1, '*', {'tp_deg': 2, 'dp_deg': 2, 'pp_deg': 16}, '*', ['p4d.24xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge'], '*', [5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], '*', 1.059550404548645, '*', 1.026912873923363, '*', 0.00047297601122409105, '*', tensor([0.0322]), '*', 0.0346863250019749)
    rank 2: (1, '*', {'tp_deg': 1, 'dp_deg': 8, 'pp_deg': 8}, '*', ['p4d.24xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge'], '*', [8, 7, 6, 6, 6, 6, 6, 5], '*', 1.149800419807434, '*', 0.8142246340005697, '*', 0.20691776275634766, '*', tensor([0.1287]), '*', 0.03764082470983267)
    rank 3: (1, '*', {'tp_deg': 2, 'dp_deg': 4, 'pp_deg': 8}, '*', ['p4d.24xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge', 'g5.12xlarge'], '*', [7, 7, 6, 6, 6, 6, 6, 6], '*', 1.2146559953689575, '*', 0.943166779941091, '*', 0.2071603238582611, '*', tensor([0.0643]), '*', 0.03976399087772767)
    ...
    ```


## Run modified Megatron-LM

### I. Environment

We ran our experiment in an environment where slurm and enroot were installed. However, if you prefer, it is also possible to perform the same test smoothly without slurm by using Docker.

- slurm version: 20.11.4
- enroot version: 3.4.0
- container image: `nvcr.io/nvidia/pytorch:23.04-py3`

### II. Prepare Wikipedia Training Dataset
Download Wikipedia data: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
then extract the text with https://github.com/attardi/wikiextractor

### III. run
#### 1. run without slurm.
You can run Megatron without relying on Slurm and Enroot. If Docker is installed on every node, you can carry out multi-GPU training across different architectures using the provided script. Just keep in mind that the node with the high-performance GPU should be designated as the master node.
```
$ cd ~

#링크 마지막에 확인
$ git clone https://github.com/{git_id}/FASOP

$ docker run --gpus all \
    -it \
    -p 6787:6787 \
    --mount type=bind,source="$HOME/FASOP/Megatron-LM-2",target=/root/Megatron-LM-2 \
    --mount type=bind,source="$HOME/FASOP/log2", target=/root/log2 \
    --mount type=bind,source="$HOME/FASOP/$INDEXMAP_DIR",target=/root/indexmap \
    nvcr.io/nvidia/pytorch:23.04-py3 bash

(in container)# bash run_inter $NODE_RANK \
                    $MASTER_ADDR \
                    $NPROC_PER_NODE \
                    $NNODES \
                    $GLOBAL_BATCH_SIZE \
                    $MICRO_BATCH_SIZE \
                    $TENSOR_MP_SIZE \
                    $DP_SIZE \
                    $PIPELINE_MP_SIZE \
                    $PARTITION > /root/log2/$NODE_RANK.out

```

#### 2. run with slurm and enroot.

If you use Slurm and Enroot, you can easily run jobs on multiple nodes. To start the training process, you first need to adjust the desired training conditions in the `hetero-conf.sh` file. Afterwards, you can run the master and slave jobs by executing the `./submit-hetero.sh `script.

#링크 마지막에 확인
```
$ cd ~

$ git clone https://github.com/{git_id}/FASOP

$ cd ./FASOP/Megatron-LM-2

$ vim ./hetero-conf.sh

$ ./submit-hetero.sh
```


## References
<a id="1">[1]</a> 
- Li, Dacheng, et al. "AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness." arXiv preprint arXiv:2210.07297 (2022). [the paper link](https://arxiv.org/abs/2210.07297)

<a id="2">[2]</a> 
- Narayanan, Deepak, et al. "Efficient large-scale language model training on gpu clusters using megatron-lm." Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2021. [the paper link](https://dl.acm.org/doi/abs/10.1145/3458817.3476209)

<a id="2">[3]</a> 
@misc{Wikiextractor2015,
  author = {Giusepppe Attardi},
  title = {WikiExtractor},
  year = {2015},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/attardi/wikiextractor}}
}

