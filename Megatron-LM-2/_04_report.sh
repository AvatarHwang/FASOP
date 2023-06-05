#!/bin/bash
jobid=$1 # jobid
type=$2  # out, err, gpu

if [ $type == "out" ]; then
    cat ../log2/$jobid/*.$type
elif [ $type == "err" ]; then
    cat ../log2/$jobid/*.$type
elif [ $type == "gpu" ]; then
    cat ../log2/$jobid/*.$type
else
    echo "type error"
fi