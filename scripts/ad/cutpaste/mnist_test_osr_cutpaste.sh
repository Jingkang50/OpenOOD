#!/bin/bash
# sh scripts/ad/cutpaste/mnist_test_osr_cutpaste.sh

# GPU=1
# CPU=1
# node=68
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \

python main.py \
--config configs/datasets/osr_mnist6/mnist6_seed1.yml \
configs/datasets/osr_mnist6/mnist6_seed1_ood.yml \
configs/networks/cutpaste.yml \
configs/pipelines/test/test_cutpaste.yml \
configs/preprocessors/cutpaste_preprocessor.yml \
configs/postprocessors/cutpaste.yml \
--network.backbone.name lenet \
--num_workers 8 \
--network.checkpoint 'results/osr_mnist6_seed1_projectionNet_cutpaste_e100_lr0.03/best.ckpt' \
--network.backbone.pretrained False \
--evaluator.name osr
