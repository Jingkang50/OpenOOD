#!/bin/bash
# sh scripts/ad/dsvdd/cifar100_train_dsvdd.sh


PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/cifar100/cifar100.yml \
configs/datasets/cifar100/cifar100_ood.yml \
configs/pipelines/train/train_dsvdd.yml \
configs/networks/resnet18_32x32.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/dsvdd.yml \
--optimizer.num_epochs 100 \
--network.pretrained True \
--network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1/best.ckpt'
