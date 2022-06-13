#!/bin/bash
# sh scripts/ad/cutpaste/osr_test_ood_cutpaste.sh

GPU=1
CPU=1
jobname=openood

PYTHONPATH='.':$PYTHONPATH \
srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
--kill-on-bad-exit=1 --job-name=${jobname} \
python main.py \
--config configs/datasets/mnist/mnist.yml \
configs/datasets/mnist/mnist_ood.yml \
configs/networks/cutpaste.yml \
configs/pipelines/test/test_cutpaste.yml \
configs/preprocessors/cutpaste_preprocessor.yml \
configs/postprocessors/cutpaste.yml \
--network.backbone.name lenet \
--num_workers 8 \
--network.pretrained True \
--network.checkpoint "results/mnist_projectionNet_cutpaste_e100_lr0.03/best.ckpt"
