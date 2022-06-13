#!/bin/bash
# sh scripts/ad/draem/mnist_train_draem.sh

GPU=1
CPU=1
node=30
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/osr_cifar6/cifar6_seed1.yml \
configs/datasets/osr_cifar6/cifar6_seed1_ood.yml \
configs/networks/draem.yml \
configs/pipelines/train/train_draem.yml \
configs/preprocessors/draem_preprocessor.yml \
configs/postprocessors/draem.yml \
--evaluator.name ad \
--dataset.train.batch_size 64 \
--num_workers 8 \
--optimizer.num_epochs 100
