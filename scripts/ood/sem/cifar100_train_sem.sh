#!/bin/bash
# sh scripts/ood/sem/cifar100_train_sem.sh

#GPU=1
#CPU=1
#node=79
#jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} \
#-w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/networks/resnet18_32x32.yml \
configs/preprocessors/base_preprocessor.yml \
configs/pipelines/train/train_sem.yml \
--optimizer.num_epochs 100 \
--network.pretrained False \
--network.checkpoint ./results/mnist_0408_3/mnist_lenet_base_e100_lr0.1/best_epoch77_acc0.9940.ckpt \
--num_workers 8
