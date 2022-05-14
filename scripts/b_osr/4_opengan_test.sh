#!/bin/bash
# sh scripts/b_osr/4_opengan_test.sh

# GPU=1
# CPU=1
# node=36
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
#srun -p dsta --mpi=pmi2 --gres=gpu:${GPU} -n1 \
#--cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
#--kill-on-bad-exit=1 --job-name=${jobname} -w SG-IDC1-10-51-2-${node} \
python main.py \
--config configs/datasets/objects/tin.yml \
configs/datasets/digits/mnist_ood.yml \
configs/networks/opengan.yml \
configs/pipelines/test/test_opengan.yml \
configs/postprocessors/opengan.yml \
--num_workers 8 \
--network.backbone.pretrained True \
--network.backbone.checkpoint './results/tin_resnet18_32x32_base_e200_lr0.125/best.ckpt' \
--evaluator.name ood
