#!/bin/bash
# sh scripts/d_uncertainty/3_mnist_dropout_train.sh


# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \
# -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/digits/mnist.yml \
configs/networks/lenet.yml \
configs/preprocessors/base_preprocessor.yml \
configs/pipelines/train/train_dropout.yml \
--optimizer.num_epochs 10 \
--num_workers 8 \
--output_dir ./results/lenet_dropout_pretrained \
