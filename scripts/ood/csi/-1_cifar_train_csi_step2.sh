#!/bin/bash
# sh scripts/c_ood/-1_cifar_train_csi_step2.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \
'''
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/networks/csinet.yml \
configs/pipelines/train/train_csi.yml \
configs/preprocessors/base_preprocessor.yml \
--optimizer.num_epochs 2 \
--dataset.train.batch_size 128 \
--force_merge True \
--mode csi_step2 \
--network.pretrained True \
--network.checkpoint './results/cifar10_csinet_csi_csi_step1_e2_lr0.1/best.ckpt'

'''
#Mnist Lenet:
'''
python main.py \
--config configs/datasets/digits/mnist.yml \
configs/networks/csinet.yml \
configs/pipelines/train/train_csi.yml \
configs/preprocessors/base_preprocessor.yml
'''
#cifar10 resnet:
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/networks/csinet.yml \
configs/pipelines/train/train_csi.yml \
configs/preprocessors/base_preprocessor.yml \
