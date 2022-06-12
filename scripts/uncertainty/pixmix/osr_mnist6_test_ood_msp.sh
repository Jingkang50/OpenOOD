#!/bin/bash
# sh scripts/uncertainty/pixmix/osr_mnist6_test_ood_msp.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed1.yml \
configs/datasets/osr_mnist6/mnist6_seed1_ood.yml \
configs/networks/lenet.yml \
configs/pipelines/test/test_osr.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
--num_workers 8 \
--network.checkpoint 'results/osr_mnist6_seed1_lenet_base_e100_lr0.1_pixmix/best.ckpt' \
--mark osr_mnist6_pixmix
