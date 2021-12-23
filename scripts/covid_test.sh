#!/bin/bash

# sh scripts/covid_test.sh
DIR_NAME='covid_baseline'
METHOD='mds' # [msp, odin, ebo, mds, llfs, hlff]
GPU=1
CPU=1
node=73
jobname=fsood

srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 --cpus-per-task=${CPU} --ntasks-per-node=${GPU} --kill-on-bad-exit=1 \
--job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python test.py \
    --config configs/test/covid_${METHOD}.yml \
    --checkpoint ../report/${DIR_NAME}/best.ckpt \
    --csv_path ../report/${DIR_NAME}/${DIR_NAME}_${METHOD}.csv &
