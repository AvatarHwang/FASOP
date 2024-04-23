# FASOP: Fast yet Accurate Automated Search for Optimal Parallelization of Transformers on Heterogeneous GPU Clusters

This repository contains FASOP, a framework that automates the process of finding the optimal degrees of parallelism and model partitioning for Transformer-based models on heterogeneous GPU clusters. FASOP accurately estimates pipelining latency and GPU communications, enabling it to find configurations that minimize the cost of GPU clusters while satisfying training time constraints, or configurations that minimize training time while meeting cost constraints. FASOP supports a variety of Transformer-based models and uses advanced algorithms and techniques to rapidly and accurately estimate device configurations.

-----

# Usage

This repository includes the FASOP framework, which can be used for the following two purposes:

(1) Finding Optimal Parallel Strategy for GPT on Heterogeneous GPU Clusters.
(2) Launching practical distributed learning using Megatron-LM based on the results from FASOP.

## Reproducing the Experiments from [FASOP: Fast yet Accurate Automated Search for Optimal Parallelization of Transformers on Heterogeneous GPU Clusters]

To reproduce the experiments from [FASOP: Fast yet Accurate Automated Search for Optimal Parallelization of Transformers on Heterogeneous GPU Clusters], follow these steps:

### I. Install the necessary dependencies for FASOP. 

FASOP requires a CPU for estimation tasks. We recommend creating a conda environment for the test of reproducibility. Ensure that you have installed the following dependencies:
- Python 3.9
- PyTorch 2.0
- NumPy 

To prepare the necessary dependencies for FASOP, follow these steps:

- Clone the FASOP repository to your local machine:
 
    ```
    $ cd ~
    $ git clone https://github.com/AvatarHwang/FASOP
    ```
    
- Create a conda environment named `fasop` with Python 3.9:

    ```
    $ conda create -n fasop python=3.9
    ```
    
- Activate the `fasop` environment:

    ```
    $ conda activate fasop
    ```
    
- Install the `numpy, pandas` package:

    ```
    $ conda install numpy pandas
    ```
    
- Install PyTorch 2.0

    ```    
    $ conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```

### II. Reproducing Experiment 4.2: Training Throughput

To inspect the parallel strategies used, execute FASOP.py with the --type argument set to the desired model (bert, gpt2XL, or T5) and the --heterogeneous flag.

Example command for BERT:
    
    
    $ python FASOP.py --type bert --heterogeneous
    
    

Reproducing the Experiment
The experiment can be reproduced by adding the --pareto flag. Here is an example using the gpt2XL model:

    
    
    $ python FASOP.py --heterogeneous --pareto
    
    
    
### III. Report    

FASOP will generate a summary of the optimal parallel strategy for the chosen model on your heterogeneous GPU cluster. This summary includes estimated training time, cost, and other relevant metrics. The results are saved in a text file located in the ~/FASOP/main_logs directory.

The directory structure of the output folder is as follows:

- output directory location: `~/FASOP/main_logs/`

    ```bash
    main_logs
    |- bert.csv
    |- bert_heterogeneous.csv
    |- T5.csv
    |- T5_heterogeneous.csv
    |- gpt2.csv
    |- gpt2_heterogeneous.csv
    |- gpt2_heterogeneous_pareto.csv
    ```    
    
- The results file will contain the following fields, separated by ('\*'):
`mbs`, `tp`, `dp`, `pp`, `node placement`, `num_a100`, `num_a10`, `partition`, `estimated time (s/step)`, `pipeline time`, `DP all-reduce time`, `embedding layer all-reduce time`, `is_oom`, `oom_gpumem`, `is_fsdp_oom`, `fsdpoom_gpumem`, `train_cost`.

- example for the result of `FASOP.py --type bert` located as `~/FASOP/main_logs/bert.csv`. this is sorted by steptime in ascending order.
    ```bash
    4,1,16,1.0,"['g5.24xlarge', 'g5.24xlarge', 'g5.24xlarge', 'g5.24xlarge']",[26],0.95458984375,0.7042821049690247,0.25030770897865295,0.0,0.006016037326388889,False,tensor([9.0552]),False,tensor([5.3309])
    8,1,16,1.0,"['g5.24xlarge', 'g5.24xlarge', 'g5.24xlarge', 'g5.24xlarge']",[26],0.95458984375,0.7042821049690247,0.25030770897865295,0.0,0.006016037326388889,False,tensor([12.5239]),False,tensor([8.7996])
    16,1,16,1.0,"['g5.24xlarge', 'g5.24xlarge', 'g5.24xlarge', 'g5.24xlarge']",[26],0.95458984375,0.7042821049690247,0.25030770897865295,0.0,0.006016037326388889,False,tensor([19.4614]),False,tensor([15.7371])
    ...
    ```


## Run modified Megatron-LM

The following steps provide instructions on how to set up and run the modified Megatron-LM code used in our experiments.

### I. Environment

Our experiements were run using the following environment:

- slurm version: 20.11.4
- enroot version: 3.4.0
- container image: `nvcr.io/nvidia/pytorch:23.04-py3` see [ngc catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

However, it is also possible to run the experiments without Slurm and Enroot using Docker.

### II. Prepare Wikipedia Training Dataset
To prepare the Wikipedia training dataset, follow these steps:
- Download Wikipedia data from https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2.
- Extract the text using WikiExtractor tool from https://github.com/attardi/wikiextractor.

### III. Setup Model Configuration

In the `_00_conf.sh` file, you can adjust the model by modifying the `MODEL_ARGS` value. It's important to note that the `gpt2xl`, `Bert`, and `T5` models have different `--num-layers`, `--hidden-size`, etc., so you need to carefully set these parameters accordingly.

### IV. Running the modified Megatron-LM Code

There are two ways to run the modified Megatron-LM code: with or without Slurm and Enroot.

#### 1. Running Without Slurm and Enroot

To run Megatron without relying on Slurm and Enroot, you can use the provided Docker script. Follow these steps:

```
$ cd ~

$ cd FASOP/Megatron-LM-2/

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

#### 2. Running With Slurm and Enroot

If you use Slurm and Enroot, you can easily run jobs on multiple nodes. To start the training process, follow these steps:

- Navigate to the Megatron-LM-2 directory:

```
$ cd ~

$ cd FASOP/Megatron-LM-2
```

- Edit the `_00_conf.sh` file to adjust the desired training configurations.

```

$ vim ./_00_conf.sh

```

- Run the `_03_summit.sh` script to start the master `_02_hetero_master_job.sh` and slave `_02_hetero_slave_job.sh` jobs:

```

$ ./_03_summit.sh

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

