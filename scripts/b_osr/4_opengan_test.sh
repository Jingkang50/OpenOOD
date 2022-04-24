#!/bin/bash
# sh scripts/_get_started/-1_openGan_test.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/tin.yml \
configs/datasets/digits/mnist_ood.yml \
configs/networks/openGan.yml \
configs/pipelines/test/test_openGan.yml \
configs/postprocessors/msp.yml \
--num_workers 8
