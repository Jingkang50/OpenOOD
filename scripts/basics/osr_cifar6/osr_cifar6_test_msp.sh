#!/bin/bash
# sh scripts/basics/osr_cifar6/osr_cifar6_test_msp.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/osr_cifar6/cifar6_seed1.yml \
configs/datasets/osr_cifar6/cifar6_seed1_osr.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_osr.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/msp.yml \
--num_workers 8 \
--network.checkpoint './results/cifar6_seed1_resnet18_32x32_base_e100_lr0.1_default/best.ckpt'
