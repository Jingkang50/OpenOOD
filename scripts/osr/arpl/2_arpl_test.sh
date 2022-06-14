#!/bin/bash
# sh scripts/c_ood/0_mnist_test_ood_msp.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/digits/mnist.yml \
configs/datasets/digits/mnist_ood.yml \
configs/networks/arpl_net.yml \
configs/pipelines/test/test_arpl.yml \
configs/postprocessors/msp.yml \
--num_workers 8 \
--mark 0
