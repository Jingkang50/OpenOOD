#!/bin/bash
# sh scripts/d_uncertainty/1_cifar10_pixmix_train.sh

GPU=1
CPU=1
node=73
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
-w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/cifar10.yml \
configs/networks/resnet18_32x32.yml \
configs/pipelines/train/train_cutmix.yml \
configs/preprocessors/base_preprocessor.yml \
--dataset.image_size 32 \
--optimizer.num_epochs 100 \
--num_workers 4 \
--trainer.trainer_args.cutmix_prob 0.5

