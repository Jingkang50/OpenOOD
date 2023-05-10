#!/bin/bash
# sh scripts/ood/mcd/cifar100_train_mcd.sh

GPU=1
CPU=1
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_oe.yml \
    configs/networks/mcd_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_mcd.yml \
    --network.backbone.name resnet18_32x32 \
    --network.pretrained False \
    --dataset.image_size 32 \
    --optimizer.num_epochs 100 \
    --num_workers 8 \
    --seed 0
