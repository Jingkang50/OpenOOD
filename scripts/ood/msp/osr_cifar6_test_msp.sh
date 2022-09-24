#!/bin/bash
# sh scripts/ood/msp/osr_cifar6_test_msp.sh

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
configs/datasets/osr_cifar6/cifar6_seed1.yml \
configs/preprocessors/base_preprocessor.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_acc.yml \
--num_workers 8 \
--network.checkpoint './results/checkpoints/osr/cifar6_seed1.ckpt'
