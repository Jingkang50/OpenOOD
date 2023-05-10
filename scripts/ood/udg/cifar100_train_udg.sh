#!/bin/bash
# sh scripts/ood/udg/cifar100_train_udg.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
-w SG-IDC1-10-51-2-${node} \
python -m pdb -c continue main.py \
--config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_oe.yml \
    configs/networks/udg_net.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/pipelines/train/baseline.yml \
    configs/pipelines/train/train_udg.yml \
    --dataset.train.dataset_class UDGDataset \
    --dataset.oe.dataset_class UDGDataset \
    --network.backbone.name resnet18_32x32 \
    --network.pretrained False \
    --seed 0
