#!/bin/bash
# sh scripts/digits_baseline.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 --cpus-per-task=${CPU} --ntasks-per-node=${GPU} --kill-on-bad-exit=1 \
--job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python tools/train.py \
  --config configs/datasets/mnist_datasets.yml configs/train/digits_baseline.yml \



#python test.py \
#    --config configs/test/digits_msp.yml \
#    --checkpoint ${OUTPUT_DIR}/best.ckpt \
#    --csv_path ${OUTPUT_DIR}/results.csv
