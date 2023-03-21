#!/bin/bash
jobid=$1 # jobid
type=$2  # out, err, gpu

if [ $type == "out" ]; then
    watch -n 1 "tail -n 10 ./log2/$jobid/*.$type"
elif [ $type == "err" ]; then
    watch -n 1 "tail -n 8 ./log2/$jobid/*.$type"
elif [ $type == "gpu" ]; then
    watch -n 1 "tail -n 5 ./log2/$jobid/*.$type"
else
    echo "type error"
fi