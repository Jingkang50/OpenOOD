#!/bin/bash
# sh scripts/basics/cifar100/train_cifar100.sh

#GPU=1
#CPU=1
#node=73
#jobname=openood

#PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} \
#-w SG-IDC1-10-51-2-${node} \
python  main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/networks/resnet18_32x32.yml \
configs/preprocessors/base_preprocessor.yml \
configs/pipelines/train/baseline.yml \
--network.pretrained False \
--dataset.image_size 32 \
--optimizer.num_epochs 100 \
--num_workers 4 \
--mark 4
