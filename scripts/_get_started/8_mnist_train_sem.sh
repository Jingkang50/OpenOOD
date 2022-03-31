#!/bin/bash
# sh scripts/_get_started/8_mnist_train_sem.sh

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
--config configs/datasets/digits/mnist.yml \
configs/datasets/digits/benchmark_digits.yml \
configs/networks/lenet.yml \
configs/pipelines/train/train_sem.yml \
--network.checkpoint ./results/mnist_lenet_base_e100_lr0.1/best_epoch100_acc0.996.ckpt \
--dataset.image_size 28 \
--network.name lenet \
--num_workers 8
