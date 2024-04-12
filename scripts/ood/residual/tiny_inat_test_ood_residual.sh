#!/bin/bash
# sh scripts/ood/residual/cifar10_test_ood_residual.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

PYTHONPATH='.':$PYTHONPATH \
# srun -p mediasuper -x SZ-IDC1-10-112-2-17 --gres=gpu:${GPU} \
# --cpus-per-task=${CPU} --ntasks-per-node=${GPU} \
# --kill-on-bad-exit=1 --job-name=${jobname} \

python main.py \
--config configs/datasets/tiny_inet/tiny_inat.yml \
configs/datasets/tiny_inet/tiny_inat_ood.yml \
configs/networks/convnext_base.yml \
configs/pipelines/test/test_hc_ood.yml \
configs/preprocessors/base_preprocessor.yml \
configs/postprocessors/residual.yml \
--num_workers 1 \
--network.checkpoint 'pretrained_models/checkpoint-best.pth' \
--mark 0
