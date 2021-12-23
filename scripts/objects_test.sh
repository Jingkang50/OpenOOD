#!/bin/bash

# sh scripts/objects_test.sh
DIR_NAME='objects_baseline'
# METHOD='mds' # [msp, odin, ebo, llfs, hlff]
GPU=1
CPU=1
node=73
jobname=fsood

for METHOD in msp odin ebo llfs hlff mds
do
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 --cpus-per-task=${CPU} --ntasks-per-node=${GPU} --kill-on-bad-exit=1 \
--job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python test.py \
    --config configs/test/objects_${METHOD}.yml \
    --checkpoint ../report/${DIR_NAME}/best.ckpt \
    --csv_path ../report/${DIR_NAME}/${DIR_NAME}_${METHOD}_nearest.csv &
done
