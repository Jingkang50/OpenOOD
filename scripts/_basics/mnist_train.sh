#!/bin/bash
# sh scripts/others/0_mnist_mixup.sh

GPU=1
CPU=1
node=69
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
-w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/digits/mnist.yml \
configs/networks/lenet.yml \
configs/pipelines/train/baseline.yml \
--dataset.image_size 28 \
--network.name lenet \
--optimizer.num_epochs 100 \
--num_workers 0
