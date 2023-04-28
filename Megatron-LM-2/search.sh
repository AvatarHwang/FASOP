#!/bin/bash

for i in `grep -R "PARTITION: 8-6-7-3" ../log2/* | awk -F':' '{print $1}' | awk -F'/' '{print $3}'  | awk  '!a[$0]++'` ;
 do 
 ./get-result.sh $i | grep "NNODES: 8" -A 30 
 done 