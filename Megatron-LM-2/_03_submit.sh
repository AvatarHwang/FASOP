#!/bin/bash

. _00_conf.sh

if  $HOMOGENEOUS_CLUSTER  ; then
    sbatch _02_homo_job.sh
else 
    sbatch _02_hetero_master_job.sh
    sbatch _02_hetero_slave_job.sh
fi