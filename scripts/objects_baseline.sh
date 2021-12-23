#!/bin/bash
# sh scripts/objects_baseline.sh

OUTPUT_DIR='../report/objects_baseline'
GPU=1
CPU=1
node=73
jobname=fsood

srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 --cpus-per-task=${CPU} --ntasks-per-node=${GPU} --kill-on-bad-exit=1 \
--job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python train.py \
  --config configs/train/objects_baseline.yml \
  --output_dir ${OUTPUT_DIR} &

#python test.py \
#    --config configs/test/digits_msp.yml \
#    --checkpoint ${OUTPUT_DIR}/best.ckpt \
#    --csv_path ${OUTPUT_DIR}/results.csv
