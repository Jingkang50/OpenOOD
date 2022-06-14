#!/bin/bash
# sh scripts/d_uncertainty/0_tempscaling.sh

# mnist
# GPU=1
# CPU=1
# node=73
# jobname=openood

# PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
# python main.py \
# --config configs/datasets/digits/mnist.yml \
# configs/datasets/digits/mnist_fsood.yml \
# configs/networks/lenet.yml \
# configs/pipelines/test/test_fsood.yml \
# configs/postprocessors/temperature_scaling.yml \
# configs/preprocessors/base_preprocessor.yml \
# --num_workers 8 \
# --network.checkpoint ./results/mnist_lenet_base_e100_lr0.1/best_epoch86_acc0.9920.ckpt \
# --mark 0 \
# --exp_name mnist_lenet_base_e100_lr0.1_test_fsood_temperature_scaling


# cifar10
GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/datasets/objects/cifar10_fsood.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/test/test_fsood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/temperature_scaling.yml \
--num_workers 8 \
--mark 0 \
--network.checkpoint ./results/cifar10_resnet18_32x32_base_e100_lr0.1/best.ckpt \
--exp_name cifar10_resnet18_32x32_base_e100_lr0.1_test_fsood_temperature_scaling
