#!/bin/bash
jobid=$1 # jobid

echo "******************* last output message *******************"
ls ../log2/$jobid/*.out
tail -n 3 ../log2/$jobid/*.out 
echo `cat ../log2/$jobid/*.out | wc -l` lines..

echo "******************* last error message *******************"
ls ../log2/$jobid/*.err
tail -n 1 ../log2/$jobid/*.err 
echo `cat ../log2/$jobid/*.err | wc -l` lines..

echo "******************* last gpu message *******************"
ls ../log2/$jobid/*.gpu
tail -n 4 ../log2/$jobid/*.gpu
echo `cat ../log2/$jobid/*.gpu | wc -l` lines..


echo "******************* job info $jobid *******************"
items=`ls ../log2/$jobid/*.out`
echo items: `ls ../log2/$jobid`
arr=($items)
if [   ${#arr[@]} -eq 1 ]; then
    cat $items | grep NODE_RANK -A 8
else
    cat ${arr[1]} | grep NODE_RANK -A 8
fi


echo "******************* iteration time *******************"
for num in {1..5}
do
num="$num"0
var=`cat ../log2/$jobid/*.out | grep "iteration       $num/"`
echo  $var | awk -F'iteration \\(ms\\)\\:' '{print $2}' | awk -v val=$num '{print "iteration " val " : " $1/1000 " s"}'
done