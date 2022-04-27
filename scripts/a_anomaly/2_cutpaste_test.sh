#!/bin/bash
# sh scripts/_get_started/1_mnist_test.sh

GPU=1
CPU=1
node=68
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/mvtec/bottle.yml \
configs/networks/CutPaste.yml \
configs/pipelines/test/test_cutpaste.yml \
configs/preprocessors/cutpaste_preprocessor.yml
